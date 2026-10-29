"""Single-target Z2 math, explicit scope, checkpoint isolation and launchers."""
import copy
from contextlib import redirect_stdout
import io
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from torch.utils.data import DataLoader

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'system'))
from utils import target_z2
from utils.multi_group_proto import Features,z2_loss
from utils.head_snapshot import tensor_fingerprint
from common.experiment import rng_state,set_rng
from test_control_population import assert_rng
from flcore.trainmodel.models import Hyper_CNN_512
from flcore.clients.clientTargetProj import clientTargetProj
from utils.projection_variants import aggregate_projection_weighting,aggregate_softmax_only


def options(**changes):
    args=dict(algorithm='FedTargetProj',model_family='Decom_CNN-5-512',target_proj_mode='avg',target_head_mode='shared',
        target_local_schedule='joint',target_prox_mu=0.,is_regular=1,regular_lamda=.001,target_z2_lambda=1.,
        target_z2_start_round=2,target_z2_feature='base_output',target_z2_diagnostics=1,num_clients=20,
        exp_name='test',save_folder_name='temp',h5_result_root='h5_results',final_model_root='final_models',
        training_log_root='training_logs',run_log_dir='')
    args.update(changes);return SimpleNamespace(**args)


def client(model,data,**changes):
    c=clientTargetProj.__new__(clientTargetProj);c.args=options(**changes)
    c.device='cpu';c.id=0;c.role='Client_0';c.save_folder_name='unused';c.learning_rate=.005
    c.local_epochs=1;c.batch_size=2;c.train_slow=False;c.loss=torch.nn.CrossEntropyLoss()
    c.train_time_cost=dict(num_rounds=0,total_cost=0.);c._load_model=lambda:model
    c.load_train_data=lambda:DataLoader(data,batch_size=2,shuffle=True,drop_last=False,num_workers=0)
    return c


class TargetZ2Tests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        if torch.cuda.is_available():torch.cuda.get_rng_state_all()

    def test_validation_and_zero_names_do_not_restrict_existing_combinations(self):
        for mode in target_z2.MODES:target_z2.validate(options(target_proj_mode=mode))
        for changes in [dict(target_z2_lambda=-1),dict(target_z2_lambda=float('nan')),dict(target_z2_lambda=float('inf')),
                        dict(target_z2_start_round=1),dict(target_z2_start_round=2.5),dict(target_z2_feature='bad'),
                        dict(algorithm='FedMultiGroupProj'),dict(target_proj_mode='feedback_avg'),dict(target_proj_mode='dwa_soft'),
                        dict(target_proj_mode='group_avg_only'),dict(target_local_schedule='fedrep'),dict(target_head_mode='local'),
                        dict(target_prox_mu=.1),dict(target_head_l2_lambda=.01),dict(model_family='CNN-512'),
                        dict(meta_c0_split='split.json'),dict(meta_collect_snapshots=True),dict(join_ratio=.5),dict(head_diag_collect=True)]:
            with self.subTest(changes=changes),self.assertRaises(ValueError):target_z2.validate(options(**changes))
        off=options(target_z2_lambda=0,algorithm='other',model_family='other',target_proj_mode='feedback_avg',target_prox_mu=.1)
        before=copy.deepcopy(vars(off));target_z2.validate(off);target_z2.ensure_names(off)
        self.assertEqual(vars(off),before)
        args=options();target_z2.ensure_names(args);before=copy.deepcopy(vars(args));target_z2.ensure_names(args)
        self.assertEqual(vars(args),before)

    def test_math_feature_negative_values_and_backbone_gradient(self):
        model=Hyper_CNN_512(n_kernels=2,num_classes=100,ratio_LR=.15);x=torch.randn(3,3,32,32)
        expected=model.base(x).clone();self.assertTrue((expected<0).any())
        with Features(model,'base_output') as capture:logits=model(x);h=capture.get()
        self.assertTrue(torch.equal(h,expected));self.assertTrue(torch.equal(logits,model(x)))
        loss,n=z2_loss(h);self.assertEqual(n,3);self.assertTrue(torch.equal(loss,expected.square().mean()))
        loss.backward();self.assertIsNone(model.fc3.weight.grad)
        self.assertTrue(any(p.grad is not None and p.grad.abs().sum()>0 for p in model.base.parameters()))
        z=torch.randn(3,7,dtype=torch.float64,requires_grad=True);reference=z.detach().clone().requires_grad_(True)
        z2_loss(z)[0].backward();reference.square().mean().backward();self.assertTrue(torch.equal(z.grad,reference.grad))

    def test_zero_and_first_round_equal_frozen_train_and_evaluation_objective_unchanged(self):
        source=subprocess.check_output(['git','show','89d442c:system/flcore/clients/clientTargetProj.py'],cwd=ROOT,text=True)
        old={'__name__':'fixed_89d442c_client'};exec(compile(source,'89d442c/clientTargetProj.py','exec'),old)
        model=Hyper_CNN_512(n_kernels=2,num_classes=100,ratio_LR=.15)
        data=[(torch.randn(3,32,32),torch.tensor(i)) for i in range(3)]
        with redirect_stdout(io.StringIO()),patch('flcore.clients.clientTargetProj.save_item'),patch.dict(old,save_item=lambda *a:None):
            start=rng_state();reference=copy.deepcopy(model);old['clientTargetProj'].train(client(reference,data,target_z2_lambda=0),current_round=0);end=rng_state()
            for coefficient,loop in [(0,0),(0,1),(1,0)]:
                trial=copy.deepcopy(model);c=client(trial,data,target_z2_lambda=coefficient);set_rng(start)
                with patch('utils.multi_group_proto.Features',side_effect=AssertionError('No zero/first-round hook')):c.train(loop)
                self.assertEqual(tensor_fingerprint(trial.state_dict()),tensor_fingerprint(reference.state_dict()));assert_rng(self,end)
            c=client(copy.deepcopy(model),data)
            with patch('utils.multi_group_proto.Features',side_effect=AssertionError('Evaluation must not add Z2')):
                ce,f=c._objective(c._load_model(),torch.stack([x for x,y in data]),torch.tensor([y for x,y in data]))
                self.assertTrue(torch.isfinite(ce+f))

    def test_active_client_one_forward_per_batch_and_log_toggle_exact(self):
        model=Hyper_CNN_512(n_kernels=2,num_classes=100,ratio_LR=.15)
        data=[(torch.randn(3,32,32),torch.tensor(i)) for i in range(3)];start=rng_state();outputs=[]
        for diagnostic in (1,0):
            trial=copy.deepcopy(model);c=client(trial,data,target_z2_diagnostics=diagnostic);set_rng(start)
            with redirect_stdout(io.StringIO()),patch('flcore.clients.clientTargetProj.save_item'),patch.object(trial,'forward',wraps=trial.forward) as count:
                c.train(1);self.assertEqual(count.call_count,2)
            self.assertEqual(c.last_z2_train_stats['effective_lambda'],1.)
            self.assertEqual(c.last_z2_train_stats['train_sample_visits'],3)
            self.assertGreater(c.last_z2_train_stats['train_mean_raw_z2'],0.)
            self.assertFalse(trial.base._forward_hooks);self.assertFalse(trial.fc3._forward_pre_hooks)
            outputs.append((tensor_fingerprint(trial.state_dict()),rng_state()))
        self.assertEqual(outputs[0][0],outputs[1][0]);assert_rng(self,outputs[0][1])

    def test_hook_cleanup_on_exception_and_no_checkpoint_write(self):
        model=Hyper_CNN_512(n_kernels=2,num_classes=100,ratio_LR=.15)
        data=[(torch.randn(3,32,32),torch.tensor(0))];c=client(model,data)
        before=tensor_fingerprint(model.state_dict())
        with patch('flcore.clients.clientTargetProj.save_item') as save,patch.object(c,'_objective',side_effect=RuntimeError('forced')):
            with self.assertRaisesRegex(RuntimeError,'forced'):c.train(1)
            save.assert_not_called()
        self.assertFalse(model.base._forward_hooks);self.assertFalse(model.fc3._forward_pre_hooks)
        self.assertEqual(before,tensor_fingerprint(model.state_dict()))

    def test_fixed_snapshot_softmax_weights_identical_with_conflict(self):
        base={'w':torch.tensor([5.,2.],dtype=torch.float64)};posts=[{'w':base['w']+torch.tensor(v,dtype=torch.float64)} for v in ([1.,0.],[-2.,3.],[2.,1.])]
        uploads=lambda:iter((i,[.2,.3,.5][i],p) for i,p in enumerate(posts))
        plain=aggregate_softmax_only(base,posts[0],uploads,0,.125)
        projected=aggregate_projection_weighting(base,posts[0],uploads,0,'projection_softmax',.125)
        self.assertEqual([r['aggregation_weight'] for r in plain[2]],[r['aggregation_weight'] for r in projected[2]])
        self.assertFalse(torch.equal(plain[0]['w'],projected[0]['w']))

    def test_launcher_forwarding_and_rejection(self):
        command=[sys.executable,str(ROOT/'system/run_target_proj.py'),'--dry-run','--modes','softmax_only','projection_softmax',
                 '--projection_self_weight','.125','--target_z2_lambda','1','--device','cpu']
        result=subprocess.run(command,cwd=ROOT,text=True,capture_output=True);self.assertEqual(result.returncode,0,result.stderr)
        self.assertEqual(result.stdout.count('--target_z2_lambda 1.0'),2)
        self.assertEqual(result.stdout.count('--target_z2_start_round 2'),2)
        zero=subprocess.run([sys.executable,str(ROOT/'system/run_target_proj.py'),'--dry-run','--modes','avg',
            '--target_z2_lambda','0','--target_z2_start_round','3','--device','cpu'],cwd=ROOT,text=True,capture_output=True)
        self.assertEqual(zero.returncode,0,zero.stderr);self.assertIn('--target_z2_lambda 0.0',zero.stdout)
        self.assertIn('--target_z2_start_round 3',zero.stdout)
        bad=subprocess.run([sys.executable,str(ROOT/'system/run_target_proj.py'),'--dry-run','--modes','feedback_avg','--target_z2_lambda','1'],cwd=ROOT,text=True,capture_output=True)
        self.assertNotEqual(bad.returncode,0);self.assertIn('Nonzero target_z2_lambda',bad.stderr)

    def test_three_complete_commands_no_placeholders_and_matched_configuration(self):
        commands=[shlex.split(row.strip()[1:-1]) for row in (ROOT/'system/target_z2_commands.sh').read_text().splitlines() if row.strip().startswith('"python ')]
        self.assertEqual(len(commands),3)
        for command,mode in zip(commands,target_z2.MODES):
            self.assertNotIn('$',' '.join(command))
            for flag,value in [('--target_proj_mode',mode),('--target_z2_lambda','1.0'),('--target_z2_start_round','2'),
                               ('--target_z2_feature','base_output'),('-gr','100'),('-ls','5'),('-nc','20'),('--projection_softmax_tau','0.20')]:
                self.assertEqual(command[command.index(flag)+1],value)
            if mode!='avg':self.assertEqual(command[command.index('--projection_self_weight')+1],'0.125')
            else:self.assertNotIn('--projection_self_weight',command)


if __name__=='__main__':unittest.main()

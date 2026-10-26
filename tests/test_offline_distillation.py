"""Fixed teacher selection, exact zero-KD regression, source/cache/output isolation."""
import copy
import io
import json
import shutil
import shlex
from pathlib import Path
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
from torch.nn import functional as F

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'system'))
from common.experiment import rng_state,set_rng,sha256
from common.data_loading import make_data_loader
from utils.cnn512_control import seed_control
from utils.multi_group_svd import model_capacity,load_checkpoint
from utils.head_snapshot import tensor_fingerprint
from utils.offline_distillation import (select_teachers,teacher_target,kd_loss,freeze_teachers,teacher_identity,
    assert_frozen,teacher_cache,train_mode,observe,IndexedData,forward_teachers,training_rng_digest)
from utils.offline_distillation_io import load_sources,read_data,Bundle,digest_json
from flcore.clients.clientTargetProj import clientTargetProj
from flcore.trainmodel.models import Hyper_CNN_512
from offline_distillation import parser,run,validate
from compare_offline_distillation import compare_runs
from test_control_population import assert_rng


def fixture(root,epochs=2,kd_lambda=0):
    root=Path(root);(root/'models').mkdir();d=root/'data/Cifar100/pat_20'
    generator=np.random.default_rng(79)
    for split,n in [('train',5),('test',3)]:
        (d/split).mkdir(parents=True)
        np.savez(d/split/'2.npz',data=dict(x=generator.standard_normal((n,3,32,32)).astype('float32'),y=np.arange(n)%100))
    (d/'config.json').write_text(json.dumps(dict(num_clients=4,num_classes=100,partition='pat')))
    paths=[]
    full=Hyper_CNN_512(in_features=3,num_classes=100,n_kernels=1,ratio_LR=1.)
    for k in range(3):
        m=Hyper_CNN_512(in_features=3,num_classes=100,n_kernels=1,ratio_LR=.15)
        path=root/'models'/f'm{k}.pt';paths.append(str(path))
        torch.save(dict(model_class='flcore.trainmodel.models.Hyper_CNN_512',model_family='Decom_CNN-5-512',
            model_kwargs=dict(in_features=3,num_classes=100,n_kernels=1,ratio_LR=.15,input_size=32),
            state_dict=m.state_dict(),state_schema={n:list(p.shape) for n,p in m.state_dict().items()},
            capacity=model_capacity(m,dict(full.named_parameters())),role='ordinary_post_local_low_rank',prediction_checkpoint=True,
            client_id=2,model_id=k,assigned_model_id=1,completed_train_count=3,loop_round=2,class_order=list(range(100))),path)
    return parser().parse_args(['--checkpoints',*paths,'--client-id','2','--assigned-model-id','1','--completed-round','3',
        '--data-root',str(root/'data'),'--device','cpu','--epochs',str(epochs),'--batch-size','2','--test-batch-size','2',
        '--teacher-batch-size','2','--seed','7','--kd-lambda',str(kd_lambda),'--cache-dir',str(root/'cache'),
        '--output-dir',str(root/'outputs'),'--verify-cache'])


class OfflineDistillationTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        if torch.cuda.is_available(): torch.cuda.get_rng_state_all()

    def test_true_label_selection_and_deterministic_tie_priority(self):
        probs=torch.tensor([
            [[.6,.2,.2],[.2,.7,.1],[.3,.6,.1]],
            [[.1,.3,.6],[.7,.2,.1],[.6,.2,.2]],
            [[.1,.2,.7],[.3,.5,.2],[.6,.2,.2]],
            [[.5,.3,.2],[.5,.3,.2],[.4,.3,.3]],
            [[.1,.3,.6],[.5,.3,.2],[.5,.3,.2]]],dtype=torch.float64)
        selected,true=select_teachers(probs.log(),torch.zeros(5,dtype=torch.int64))
        self.assertEqual(selected.tolist(),[0,1,2,0,1])
        # log -> softmax round-trip has ordinary float64 rounding (one ulp).
        torch.testing.assert_close(true,probs[:,:,0],rtol=0,atol=2*torch.finfo(torch.float64).eps)
        target=teacher_target(probs.log().requires_grad_(),'kl_dp',2.,selected)
        self.assertFalse(target.requires_grad)
        torch.testing.assert_close(target,(probs.log()[torch.arange(5),selected]/2).softmax(-1))

    def test_mean_probabilities_KL_direction_temperature_and_single_T_squared(self):
        logits=torch.tensor([[[2.,0.,1.],[8.,0.,0.],[0.,1.,0.]],[[0.,2.,1.],[0.,4.,1.],[2.,0.,4.]]],dtype=torch.float64,requires_grad=True)
        target=teacher_target(logits,'kd_mean',2.)
        expected=((logits[:,1]/2).softmax(-1)+(logits[:,2]/2).softmax(-1))/2
        torch.testing.assert_close(target,expected)
        self.assertGreater(float((target-(logits[:,1:].mean(1)/2).softmax(-1)).abs().max()),.05)
        student=torch.tensor([[.1,.4,.7],[.7,.2,.3]],dtype=torch.float64,requires_grad=True)
        raw=kd_loss(student,target,2.)
        manual=(target*(target.log()-(student/2).log_softmax(-1))).sum(-1).mean()
        torch.testing.assert_close(raw,manual,atol=1e-14,rtol=0)
        weighted=1.7*4*raw;torch.testing.assert_close(weighted,manual*6.8)
        weighted.backward();self.assertIsNone(logits.grad);self.assertTrue(torch.isfinite(student.grad).all())
        own=teacher_target(logits.detach(),'kl_dp',2.,torch.zeros(2,dtype=torch.int64))
        self.assertGreater(float(kd_loss(student.detach(),own,2.)),0.)  # A0 selection is NOT skipped.

    def test_zero_lambda_all_modes_exact_original_SGD_batches_and_rng(self):
        with tempfile.TemporaryDirectory() as folder,redirect_stdout(io.StringIO()):
            a=fixture(folder);bundle=Bundle(None);entries,source=load_sources(a,bundle);data,record=read_data(a,bundle,source)
            teachers=[e['model'] for e in entries];freeze_teachers(teachers,a.device);before=teacher_identity(teachers)
            source_hashes=[sha256(p) for p in a.checkpoints];state=rng_state()
            cache=teacher_cache(teachers,data['train'],source,record['files']['train'],a);assert_rng(self,state)
            seed_control(a);initial=rng_state();results=[]
            for mode in ('continue_ce','kd_mean','kl_dp'):
                r=train_mode(mode,teachers,data,cache,a,initial);results.append(r)
            for r in results[1:]:
                self.assertEqual(r['batches'],results[0]['batches']);self.assertEqual(r['final_rng_sha256'],results[0]['final_rng_sha256'])
                for n,p in r['student'].state_dict().items(): self.assertTrue(torch.equal(p,results[0]['student'].state_dict()[n]),n)
            # Independent ordinary SGD with the repository's CE+F objective and clipping.
            set_rng(initial);reference=copy.deepcopy(teachers[0]).requires_grad_(True);reference.train()
            opt=torch.optim.SGD(reference.parameters(),lr=a.lr)
            obj=SimpleNamespace(loss=torch.nn.CrossEntropyLoss(),args=SimpleNamespace(is_regular=1,regular_lamda=a.f_lambda))
            loader=make_data_loader(IndexedData(data['train']),a.batch_size,shuffle=True,drop_last=False,num_workers=0)
            for _ in range(a.epochs):
                for x,y,ids in loader:
                    opt.zero_grad();ce,frob=clientTargetProj._objective(obj,reference,x,y)
                    (ce+frob).backward();torch.nn.utils.clip_grad_norm_(reference.parameters(),a.grad_clip);opt.step()
            for n,p in reference.state_dict().items(): self.assertTrue(torch.equal(p,results[0]['student'].state_dict()[n]),n)
            self.assertEqual(training_rng_digest(),results[0]['final_rng_sha256'])
            assert_frozen(teachers,before);self.assertEqual(source_hashes,[sha256(p) for p in a.checkpoints])
            self.assertNotEqual(tensor_fingerprint(reference.state_dict()),source['checkpoints'][0]['state_sha256'])
            # Classifier is trained jointly; it is not reset or frozen.
            self.assertFalse(torch.equal(reference.fc3.weight,teachers[0].fc3.weight))

    def test_cache_content_identity_indices_and_no_rng_advance(self):
        with tempfile.TemporaryDirectory() as folder:
            a=fixture(folder);b=Bundle(None);entries,source=load_sources(a,b);data,record=read_data(a,b,source)
            models=[e['model'] for e in entries];freeze_teachers(models,a.device);state=rng_state()
            first=teacher_cache(models,data['train'],source,record['files']['train'],a);assert_rng(self,state)
            second=teacher_cache(models,data['train'],source,record['files']['train'],a);assert_rng(self,state)
            self.assertTrue(second['report']['hit']);self.assertEqual(second['report']['direct_verification']['max_absolute_error'],0.)
            self.assertTrue(torch.equal(first['logits'],second['logits']))
            path=Path(first['report']['path']);bad=torch.load(path,weights_only=False);bad['logits'][0,0,0]+=1;torch.save(bad,path)
            with self.assertRaisesRegex(ValueError,'content/index'): teacher_cache(models,data['train'],source,record['files']['train'],a)
            assert_rng(self,state)

    def test_checkpoint_round_client_capacity_alias_class_and_membership_checks(self):
        with tempfile.TemporaryDirectory() as folder:
            a=fixture(folder);b=Bundle(None);original=Path(a.checkpoints[0]).read_bytes()
            for case in ('round','client','capacity','alias','class_order','assigned'):
                d=torch.load(io.BytesIO(original),weights_only=False)
                if case=='round':d['completed_train_count']=4
                if case=='client':d['client_id']=0
                if case=='capacity':d['capacity']['rank_rate']=.3
                if case=='alias':d['state_dict']['head.1.weight']=d['state_dict']['head.1.weight'].clone()+1
                if case=='class_order':d['class_order']=list(reversed(range(100)))
                if case=='assigned':d['assigned_model_id']=2
                torch.save(d,a.checkpoints[0])
                with self.subTest(case=case),self.assertRaises(ValueError):load_sources(a,b)
                Path(a.checkpoints[0]).write_bytes(original)

    def test_real_CNN_all_modes_export_fixed_epoch_and_immutable_inputs(self):
        with tempfile.TemporaryDirectory() as folder,redirect_stdout(io.StringIO()):
            a=fixture(folder,kd_lambda=1);hashes=[sha256(p) for p in a.checkpoints]
            before=rng_state()
            with patch.object(Hyper_CNN_512,'recover_larger_model',side_effect=AssertionError('No recovery in offline KD')), \
                    patch.object(Hyper_CNN_512,'decom_larger_model',side_effect=AssertionError('No SVD in offline KD')):
                result=run(a)
            assert_rng(self,before)
            out=Path(a.output_dir);self.assertEqual(len(result['comparison']),3)
            for mode in ('continue_ce','kd_mean','kl_dp'):
                rows=json.loads((out/mode/'metrics.json').read_text())
                self.assertEqual([r['epoch'] for r in rows],[0,1,2])
                self.assertEqual(rows[-1]['optimizer_steps'],3);self.assertEqual(rows[-1]['sample_visits'],5)
                model,payload=load_checkpoint(out/mode/'Student_A.pt')
                self.assertEqual(model.ratio_LR,.15);self.assertEqual(payload['model_id'],1)
                self.assertEqual(payload['offline_epochs'],2)
                self.assertEqual(tensor_fingerprint(model.state_dict()),rows[-1]['student_state_sha256'])
            self.assertEqual(hashes,[sha256(p) for p in a.checkpoints])
            self.assertEqual(len(list(out.rglob('Student_A.pt'))),3)
            combined=compare_runs([out],Path(folder)/'comparison')
            self.assertEqual([r['mode'] for r in combined],['continue_ce','kd_mean','kl_dp'])
            bad=out/'kl_dp/manifest.json';manifest=json.loads(bad.read_text());manifest['effective']['temperature']=3
            bad.write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError,'Runs differ'):compare_runs([out],Path(folder)/'bad_comparison')

    def test_invalid_parameters_and_no_silent_data_fallback(self):
        with tempfile.TemporaryDirectory() as folder:
            args=fixture(folder)
            for key,value in [('lr',0),('temperature',float('inf')),('kd_lambda',float('nan')),
                              ('f_lambda',-1),('grad_clip',-1),('batch_size',0),('seed',-1)]:
                candidate=copy.deepcopy(args);setattr(candidate,key,value)
                with self.subTest(key=key),self.assertRaises(ValueError):validate(candidate)
            entries,source=load_sources(args,Bundle(None))
            bad=Path(folder)/'data/Cifar100/pat_20/train/2.npz';bad.unlink()
            with self.assertRaisesRegex(FileNotFoundError,'no synthetic fallback'):read_data(args,Bundle(None),source)

    def test_original_dataset_fingerprint_and_selected_shards_are_verified(self):
        from common.client_data import audit_image_clients
        with tempfile.TemporaryDirectory() as folder:
            args=fixture(folder);directory=Path(folder)/'data/Cifar100/pat_20'
            for client in (0,1,3):
                for split in ('train','test'):shutil.copyfile(directory/split/'2.npz',directory/split/f'{client}.npz')
            entries,source=load_sources(args,Bundle(None))
            original=audit_image_clients(SimpleNamespace(algorithm='Offline',data_root=args.data_root,dataset=args.dataset,
                data_subdir=args.data_subdir,data_formats='npz_only',num_clients=4,num_classes=100))
            keys=('client_id','split','sha256','decoded_x_sha256','decoded_y_sha256','count')
            fingerprint=digest_json(dict(config_sha256=original['config_sha256'],files=[{k:r[k] for k in keys} for r in original['files']]))
            source['group']=dict(context=dict(num_clients=4,data_fingerprint=fingerprint))
            data,record=read_data(args,Bundle(None),source)
            self.assertEqual(record['verified_source_data_fingerprint'],fingerprint)
            path=directory/'train/2.npz';changed=np.load(path,allow_pickle=True)['data'].tolist();changed['x'][0,0,0,0]+=1
            np.savez(path,data=changed)
            with self.assertRaisesRegex(ValueError,'grouping source fingerprint'):read_data(args,Bundle(None),source)
            original_path=Path(folder)/'original_manifest.json';original_path.write_text(json.dumps(original));args.data_manifest=str(original_path)
            with self.assertRaisesRegex(ValueError,'train fingerprint'):read_data(args,Bundle(None),source)

    def test_commands_contain_three_complete_independent_fixed_budget_modes(self):
        lines=(ROOT/'system/offline_distillation_commands.sh').read_text().splitlines()
        commands=[shlex.split(line[1:-1].replace('${DISTILL_CLIENT_ID}','0').replace('${DISTILL_ASSIGNED_MODEL_ID}','0'))
                  for line in lines if line.startswith('"python ')]
        self.assertEqual(len(commands),3)
        parsed=[parser().parse_args(command[3:]) for command in commands]
        self.assertEqual([p.mode for p in parsed],['continue_ce','kd_mean','kl_dp'])
        self.assertEqual([p.device_id for p in parsed],[0,1,2])
        self.assertEqual(len({p.output_dir for p in parsed}),3)
        for p in parsed:
            self.assertEqual((p.epochs,p.batch_size,p.lr,p.grad_clip,p.f_lambda,p.temperature,p.kd_lambda),(5,16,.005,10,.001,2,1))

    def test_read_only_evaluation_restores_params_buffers_modes_and_rng(self):
        class Mutating(torch.nn.Module):
            def __init__(self):
                super().__init__();self.fc=torch.nn.Linear(3,2);self.register_buffer('counter',torch.zeros(()))
            def forward(self,x): self.counter.add_(1);return self.fc(x[:,:,0,0])
        model=Mutating().train();data=[(torch.zeros(3,32,32),torch.tensor(0))]
        a=SimpleNamespace(device='cpu',test_batch_size=1)
        initial=tensor_fingerprint(model.state_dict());state=rng_state();observe(model,data,a)
        self.assertEqual(initial,tensor_fingerprint(model.state_dict()));self.assertTrue(model.training);assert_rng(self,state)


if __name__=='__main__':unittest.main()

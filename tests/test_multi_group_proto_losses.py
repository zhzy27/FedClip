"""Exact raw-feature objectives, original forward hooks, and old-path parity."""
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
import torch.nn.functional as F

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'system'))
from test_multi_group_proto import fixture
from test_multi_group_svd import options
from test_control_population import assert_rng
from common.experiment import rng_state,set_rng
from utils.head_snapshot import tensor_fingerprint
from utils.multi_group_proto import Features,prototype_loss,z2_loss,prepare_round,decode_packet,Session
from utils.multi_group_proto_options import protocol
from flcore.trainmodel.models import Hyper_CNN_512


class AuxiliaryTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        if torch.cuda.is_available():torch.cuda.get_rng_state_all()

    def test_both_feature_positions_one_forward_and_no_inplace_relu_corruption(self):
        model=Hyper_CNN_512(n_kernels=2,num_classes=100,ratio_LR=.15).eval()
        x=torch.randn(3,3,32,32);base=model.base(x).clone();before=tensor_fingerprint(model.state_dict())
        self.assertTrue((base<0).any())
        for position,expected in [('base_output',base),('classifier_input',base.relu())]:
            with patch.object(model.base,'forward',wraps=model.base.forward) as counter:
                with Features(model,position) as capture:output=model(x);h=capture.get()
                self.assertEqual(counter.call_count,1)
            self.assertTrue(torch.equal(h,expected));self.assertTrue(torch.equal(output,model(x)))
            self.assertEqual(before,tensor_fingerprint(model.state_dict()))
            self.assertFalse(model.base._forward_hooks);self.assertFalse(model.fc3._forward_pre_hooks)

    def test_raw_mse_reduction_target_detach_and_scale(self):
        z=torch.tensor([[3.,4.],[-2.,6.],[4.,8.]],dtype=torch.float64,requires_grad=True)
        p=torch.tensor([[1.,0.],[0.,1.],[1.,0.]],dtype=torch.float64,requires_grad=True)
        y=torch.arange(3);mask=torch.tensor([True,True,False])
        loss,n=prototype_loss(z,y,p,mask,1e-12,'mse',2.)
        expected=((z[:2]-2*p[:2].detach())**2).sum()/4
        self.assertEqual(n,2);self.assertTrue(torch.equal(loss,expected));loss.backward()
        self.assertIsNone(p.grad);torch.testing.assert_close(z.grad[:2],2*(z[:2]-2*p[:2].detach())/4,rtol=0,atol=0)
        self.assertTrue(torch.equal(z.grad[2],torch.zeros(2,dtype=torch.float64)))
        normalized=F.mse_loss(F.normalize(z[:2],dim=1),2*p[:2].detach())
        self.assertNotEqual(float(loss),float(normalized))

    def test_z2_exact_mean_square_and_gradient_no_labels_or_prototypes(self):
        z=torch.randn(4,7,dtype=torch.float64,requires_grad=True)
        loss,n=z2_loss(z);self.assertEqual(n,4);self.assertTrue(torch.equal(loss,z.square().mean()))
        loss.backward();reference=z.detach().clone().requires_grad_(True);reference.square().mean().backward()
        self.assertTrue(torch.equal(z.grad,reference.grad))
        torch.testing.assert_close(z.grad,2*z/28,rtol=5e-16,atol=1e-17)  # Different arithmetic order only.
        with self.assertRaises(FloatingPointError):z2_loss(torch.full((2,3),float('inf')))
        with self.assertRaises(FloatingPointError):z2_loss(torch.full((2,3),3e38))

    def test_normalized_mse_identity_detects_wrong_raw_mse_substitution(self):
        a=F.normalize(torch.randn(9,11,dtype=torch.float64),dim=1);b=F.normalize(torch.randn_like(a),dim=1)
        torch.testing.assert_close(F.mse_loss(a,b),2/11*(1-F.cosine_similarity(a,b)).mean(),rtol=1e-14,atol=1e-14)

    def test_mse_and_z2_backbone_gradient_not_classifier_and_invalid_zero(self):
        for kind in ('mse','z2'):
            model=Hyper_CNN_512(n_kernels=2,num_classes=100,ratio_LR=.15)
            p=F.normalize(torch.randn(100,512),dim=1).requires_grad_(True)
            with Features(model,'base_output') as capture:model(torch.randn(2,3,32,32));h=capture.get()
            loss,n=z2_loss(h) if kind=='z2' else prototype_loss(h,torch.tensor([0,1]),p,torch.ones(100,dtype=torch.bool),1e-12,kind)
            loss.backward();self.assertIsNone(p.grad);self.assertIsNone(model.fc3.weight.grad)
            self.assertTrue(any(v.grad is not None and v.grad.abs().sum()>0 for v in model.base.parameters()))
        h=torch.randn(2,4,requires_grad=True);p=torch.randn(3,4,requires_grad=True)
        zero,n=prototype_loss(h,torch.tensor([0,1]),p,torch.zeros(3,dtype=torch.bool),1e-12,'mse')
        zero.backward();self.assertEqual(n,0);self.assertTrue(torch.equal(h.grad,torch.zeros_like(h)))

    def test_z2_bypasses_every_prototype_operation_and_one_forward_per_SGD_step(self):
        with tempfile.TemporaryDirectory() as folder,redirect_stdout(io.StringIO()):
            server=fixture(folder);server.args.multi_proto_loss='z2';server.args.multi_proto_feature='base_output'
            for client in server.clients:
                client.set_parameters(client.download_servers,1)
                self.assertIsNone(client.proto_download_local_states)
            state=rng_state()
            with patch('utils.multi_group_proto.collect_client',side_effect=AssertionError('Z2 must not collect')),\
                 patch('utils.multi_group_proto.merge_statistics',side_effect=AssertionError('Z2 must not merge')),\
                 patch('utils.multi_group_proto.encode_packet',side_effect=AssertionError('Z2 must not communicate')):
                prepare_round(server);assert_rng(self,state)
                c=server.clients[0]
                with patch.object(c.model_set[0],'forward',wraps=c.model_set[0].forward) as counter:c.train()
                self.assertEqual(counter.call_count,c.train_stats[0]['train_batches'])
            self.assertIsNone(server.proto_latest_packet);self.assertFalse(server.proto_source_records)
            self.assertFalse(server.proto_collection_records);self.assertFalse(server.proto_upload_records)
            self.assertTrue(all(r['aux_effective_loss']=='z2' and r['prototype_library_sha256'] is None for r in c.train_stats))
            self.assertTrue(all(r['auxiliary_valid_ratio']==1. and r['prototype_valid_ratio'] is None for r in c.train_stats))
            self.assertEqual(server.proto_round_records[0]['statistics_upload_bytes'],0)

    def test_default_cosine_matches_fixed_d213477_active_train_exactly(self):
        source=subprocess.run(['git','show','d213477:system/flcore/clients/clientMultiGroupSVD.py'],cwd=ROOT,capture_output=True,text=True,check=True).stdout
        auxiliary=subprocess.run(['git','show','d213477:system/utils/multi_group_proto.py'],cwd=ROOT,capture_output=True,text=True,check=True).stdout
        old={'__name__':'fixed_d213477_proto'};exec(compile(auxiliary,'d213477/multi_group_proto.py','exec'),old)
        namespace={'__name__':'fixed_d213477_train'};exec(compile(source,'d213477/clientMultiGroupSVD.py','exec'),namespace)
        with tempfile.TemporaryDirectory() as folder,redirect_stdout(io.StringIO()):
            server=fixture(folder);prepare_round(server);trial=server.clients[0];reference=copy.deepcopy(trial);state=rng_state()
            with patch('utils.multi_group_proto.Session',old['Session']):namespace['clientMultiGroupSVD'].train(reference)
            expected=rng_state();set_rng(state);trial.train();assert_rng(self,expected)
            for k,m in enumerate(trial.model_set):
                self.assertEqual(tensor_fingerprint(m.state_dict()),tensor_fingerprint(reference.model_set[k].state_dict()))
                self.assertEqual(tensor_fingerprint(trial.full_uploads[k]),tensor_fingerprint(reference.full_uploads[k]))

    def test_loss_diagnostics_toggle_exact_all_three_modes(self):
        for kind in ('cosine','mse','z2'):
            with self.subTest(kind=kind),tempfile.TemporaryDirectory() as folder,redirect_stdout(io.StringIO()):
                server=fixture(folder);server.args.multi_proto_loss=kind;server.args.multi_proto_feature='base_output'
                other=copy.deepcopy(server);other.args.multi_proto_diagnostics=0
                state=rng_state();prepare_round(server);set_rng(state);prepare_round(other)
                self.assertEqual(server.proto_latest_packet,other.proto_latest_packet)
                start=rng_state();server.clients[0].train();expected=rng_state();set_rng(start);other.clients[0].train();assert_rng(self,expected)
                for a,b in zip(server.clients[0].model_set,other.clients[0].model_set):self.assertEqual(tensor_fingerprint(a.state_dict()),tensor_fingerprint(b.state_dict()))
                for row in server.clients[0].train_stats:
                    self.assertIsNotNone(row['training_mean_feature_l2']);self.assertEqual(row['preclip_gradient_observed_steps'],row['train_batches'])
                self.assertTrue(all(r['training_mean_feature_l2'] is None for r in other.clients[0].train_stats))

    def test_zero_coefficients_skip_all_auxiliary_work_and_match_d213477(self):
        source=subprocess.run(['git','show','d213477:system/flcore/clients/clientMultiGroupSVD.py'],cwd=ROOT,capture_output=True,text=True,check=True).stdout
        namespace={'__name__':'fixed_d213477_zero'};exec(compile(source,'d213477/clientMultiGroupSVD.py','exec'),namespace)
        with tempfile.TemporaryDirectory() as folder,redirect_stdout(io.StringIO()):
            server=fixture(folder,coefficient=0);reference=copy.deepcopy(server.clients[0]);reference.args.multi_proto_mode='none'
            state=rng_state();namespace['clientMultiGroupSVD'].train(reference);expected=rng_state()
            for kind in ('cosine','mse','z2'):
                trial=copy.deepcopy(server);trial.args.multi_proto_loss=kind;trial.args.multi_proto_feature='base_output';set_rng(state)
                with patch('utils.multi_group_proto.Features',side_effect=AssertionError('Zero must not capture features')),\
                     patch('utils.multi_group_proto.collect_client',side_effect=AssertionError('Zero must not collect')):
                    prepare_round(trial);trial.clients[0].train()
                assert_rng(self,expected)
                for k,m in enumerate(trial.clients[0].model_set):
                    self.assertEqual(tensor_fingerprint(m.state_dict()),tensor_fingerprint(reference.model_set[k].state_dict()))
                    self.assertEqual(tensor_fingerprint(trial.clients[0].full_uploads[k]),tensor_fingerprint(reference.full_uploads[k]))

    def test_clip_statistics_use_the_original_single_clip_return_per_step(self):
        boundary=SimpleNamespace(active=True,args=SimpleNamespace(multi_proto_diagnostics=1),
            clip_norm_sum=0.,clip_steps=0,clipped_steps=0)
        norm=torch.tensor(9.999999046325684,dtype=torch.float32)
        self.assertTrue(10/(float(norm)+1e-6)<1.)
        Session.observe_clip(boundary,norm,10.)
        self.assertEqual(boundary.clipped_steps,0)  # Actual float32 coefficient rounds to exactly one.
        with tempfile.TemporaryDirectory() as folder,redirect_stdout(io.StringIO()):
            server=fixture(folder);server.args.multi_proto_loss='z2';server.args.local_grad_clip=1e-5;prepare_round(server)
            original=torch.nn.utils.clip_grad_norm_;observed=[]
            def record(*args,**kwargs):
                norm=original(*args,**kwargs);observed.append(float(norm));return norm
            with patch('torch.nn.utils.clip_grad_norm_',record):server.clients[0].train()
            offset=0
            for row in server.clients[0].train_stats:
                count=row['train_batches'];norms=observed[offset:offset+count];offset+=count
                self.assertEqual(row['preclip_gradient_norm_mean'],sum(norms)/count)
                self.assertEqual(row['gradient_clipped_steps'],sum(1e-5/(n+1e-6)<1 for n in norms))
                self.assertEqual(row['gradient_clip_trigger_ratio'],1.)
            self.assertEqual(offset,len(observed))

    def test_collection_position_matches_student_and_rejects_stale_position(self):
        with tempfile.TemporaryDirectory() as folder,redirect_stdout(io.StringIO()):
            server=fixture(folder);server.args.multi_proto_loss='mse';server.args.multi_proto_feature='base_output'
            prepare_round(server);values,meta=decode_packet(server.proto_latest_packet,'library')
            self.assertEqual(meta['feature_position'],'base_output');self.assertTrue((values['prototypes']<0).any())
            server.args.multi_proto_feature='classifier_input'
            with self.assertRaisesRegex(ValueError,'feature position'):server.clients[0].train()

    def test_options_reject_ignored_target_norm_invalid_values_and_nondefault_z2_collection(self):
        for change in [dict(multi_proto_target_norm=0),dict(multi_proto_target_norm=float('nan')),dict(multi_proto_target_norm=float('inf')),
                       dict(multi_proto_target_norm=2),dict(multi_proto_loss='bad'),dict(multi_proto_feature='bad'),
                       dict(multi_proto_loss='z2',multi_proto_batch_size=2)]:
            with self.subTest(change=change),self.assertRaises(ValueError):options(multi_proto_mode='group_priority',multi_proto_lambda=1,**change)
        args=options(multi_proto_mode='group_priority',multi_proto_lambda=1,multi_proto_loss='mse',multi_proto_target_norm=2)
        self.assertEqual(protocol(args)['target_norm'],2)

    def test_formal_four_commands_101_rounds_same_group_feature_and_budget(self):
        lines=(ROOT/'system/multi_group_proto_loss_commands.sh').read_text().splitlines()
        commands=[shlex.split(line[1:-1]) for line in lines if line.startswith('"python ')]
        self.assertEqual(len(commands),4)
        group=commands[0][commands[0].index('--multi_group_publish')+1]
        for command,kind,coefficient in zip(commands,('cosine','cosine','mse','z2'),('0','1','1','1')):
            for flag,value in [('-gr','100'),('-ls','5'),('-nsm','3'),('-nc','20'),('--multi_proto_feature','base_output'),
                               ('--multi_proto_loss',kind),('--multi_proto_lambda',coefficient)]:
                self.assertEqual(command[command.index(flag)+1],value)
            if coefficient=='1':self.assertEqual(command[command.index('--multi_group_file')+1],group)


if __name__=='__main__':unittest.main()

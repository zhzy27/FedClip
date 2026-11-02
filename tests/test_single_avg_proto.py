"""New K1 protocol: raw last-epoch means, timing, paired streams and full/SVD."""
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

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'system'))
from test_multi_group_svd import options as svd_options
from test_control_population import assert_rng
from common.experiment import rng_state,set_rng
from utils import single_avg_proto as p
from utils.multi_group_svd import initial_model,model_spec
from utils.multi_group_proto import Features,decode_packet,encode_packet
from flcore.clients.clientMultiGroupProj import clientMultiGroupProj
from flcore.clients.clientMultiGroupSVD import clientMultiGroupSVD
from flcore.trainmodel.models import Hyper_CNN_512


def options(**changes):
    a=dict(multi_avg_proto_ablation=True,num_server_models=1,multi_group_mode='group_avg_only',multi_group_weight=1.,
        multi_proto_mode='global_last_epoch',multi_proto_loss='mse',multi_proto_lambda=1.,multi_proto_feature='base_output',
        multi_proto_target_type='raw_mean',multi_svd_init='low_rank_recovered',multi_svd_init_rank_rate=.15)
    a.update(changes);return svd_options(**a)


class AvgProtoTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        if torch.cuda.is_available():torch.cuda.get_rng_state_all()

    def test_equal_client_means_not_sample_weighted_zero_mean_valid_and_missing(self):
        a=dict(client_id=0,source_completed_train_count=1,sums=torch.tensor([[2.,0.],[0.,0.],[0.,0.]],dtype=torch.float64),counts=torch.tensor([1,3,0]))
        b=dict(client_id=1,source_completed_train_count=1,sums=torch.tensor([[0.,18.],[0.,0.],[0.,0.]],dtype=torch.float64),counts=torch.tensor([9,0,0]))
        values,ids=p.equal_client_means([b,a],3,2,1)
        self.assertTrue(torch.equal(values['prototypes'],torch.tensor([[1.,1.],[0.,0.],[0.,0.]],dtype=torch.float64)))
        self.assertEqual(values['valid'].tolist(),[True,True,False]);self.assertEqual(ids,[[0,1],[0],[]])
        self.assertEqual(values['samples'].tolist(),[10,3,0]);self.assertEqual(values['contributors'].tolist(),[2,1,0])
        with self.assertRaises(ValueError):p.equal_client_means([a,b],3,2,2)

    def test_raw_mse_gradient_missing_classes_and_no_target_gradient(self):
        z=torch.tensor([[2.,-3.],[8.,4.]],dtype=torch.float64,requires_grad=True)
        bank=torch.tensor([[1.,-1.],[0.,0.]],dtype=torch.float64,requires_grad=True)
        loss,n=p.global_mse(z,torch.tensor([0,1]),bank,torch.tensor([True,False]))
        self.assertEqual(n,1);self.assertEqual(float(loss),1.25)
        loss.backward();self.assertTrue(torch.equal(z.grad,torch.tensor([[.5,-1.],[0.,0.]],dtype=torch.float64)))
        self.assertIsNone(bank.grad)
        zero,n=p.global_mse(z,torch.tensor([0,1]),bank,torch.zeros(2,dtype=torch.bool));self.assertEqual(n,0)
        self.assertEqual(float(zero),0.)
        with self.assertRaises(FloatingPointError):p.global_mse(z*float('nan'),torch.tensor([0,1]),bank,torch.ones(2,dtype=torch.bool))

    def test_eight_configuration_validation_and_same_initial_mother(self):
        reference=None;before=rng_state()
        for svd,f,clip,proto in [(1,1,10,0),(1,1,10,1),(1,0,10,0),(1,0,10,1),(0,0,10,0),(0,0,10,1),(1,1,0,0),(1,1,0,1)]:
            a=options(multi_svd=bool(svd),is_regular=f,local_grad_clip=clip,multi_proto_lambda=proto)
            model=initial_model(a);assert_rng(self,before)
            if reference is None:reference=copy.deepcopy(model.state_dict())
            for n,value in model.state_dict().items():self.assertTrue(torch.equal(value,reference[n]),n)
            self.assertEqual(model.ratio_LR,1.);self.assertEqual(model.fc3.in_features,512)
            self.assertEqual(model_spec(a)['model_class'],'flcore.trainmodel.models.Hyper_CNN_512')
        for changes in [dict(num_server_models=3),dict(multi_group_mode='group_projection_softmax'),dict(multi_proto_loss='z2'),
            dict(multi_svd=False,is_regular=1),dict(multi_group_file='old.json'),dict(multi_proto_target_norm=2.),
            dict(multi_proto_lambda=-1.),dict(multi_proto_lambda=float('nan')),dict(multi_kd_mode='full')]:
            with self.subTest(changes=changes),self.assertRaises(ValueError):options(**changes)

    def test_shared_training_final_epoch_only_zero_diagnostics_and_no_clip_mutation(self):
        data=[(torch.randn(3,32,32),torch.tensor(i%2)) for i in range(3)]
        with tempfile.TemporaryDirectory() as tmp,redirect_stdout(io.StringIO()):
            a=options(multi_svd=False,is_regular=0,local_epochs=3,batch_size=2,local_grad_clip=0)
            a.device='cpu';a.save_folder_name_full=tmp
            model=Hyper_CNN_512(n_kernels=2,num_classes=100,ratio_LR=.15);model.recover_larger_model();model.ratio_LR=1.
            initial_rng=rng_state();states=[]
            for coefficient,diag in [(0,1),(0,0),(1,1)]:
                set_rng(initial_rng);args=copy.deepcopy(a);args.multi_proto_lambda=coefficient;args.multi_proto_diagnostics=diag
                c=clientMultiGroupProj(args,0,3,3,initial_model=model,train_slow=False,send_slow=False)
                c.set_parameters([model],0);p.prepare_round(SimpleNamespace(avg_proto_packet=None,clients=[c],args=args))
                source=[];expected=p.LastEpochStats(100,512)
                original=c._move_batch if hasattr(c,'_move_batch') else None
                # Hook actual original forward output; no added model evaluation.
                counter=[0]
                def capture(module,inputs,output):
                    counter[0]+=1
                    if counter[0]>4:source.append(output.detach().clone())
                h=c.model_set[0].base.register_forward_hook(capture)
                with patch('flcore.clients.clientMultiGroupProj.read_client_data',return_value=data),\
                     patch.object(torch.nn.utils,'clip_grad_norm_',side_effect=AssertionError('Unclipped must never call clip function')),\
                     patch.object(Hyper_CNN_512,'frobenius_decay',side_effect=AssertionError('F disabled must not execute')):
                    c.train()
                h.remove();self.assertEqual(counter[0],6);self.assertEqual(c.train_stats[0]['train_batches'],6)
                self.assertEqual(c.train_stats[0]['regularization_loss'],0.)
                if coefficient:
                    values,meta=decode_packet(c.avg_proto_local,'global_last_epoch_local')
                    self.assertEqual(int(values['counts'].sum()),3)
                    self.assertEqual(float(values['sums'].sum()),float(torch.cat(source).double().sum()))
                    self.assertFalse(values['sums'].requires_grad)
                else:self.assertIsNone(c.avg_proto_local)
                states.append((copy.deepcopy(c.model_set[0].state_dict()),rng_state()))
            for state,random in states[1:]:
                for n,value in state.items():self.assertTrue(torch.equal(value,states[0][0][n]),n)
                set_rng(random);assert_rng(self,states[0][1])

    def test_pre_relu_feature_and_previous_round_receipt_checks(self):
        m=Hyper_CNN_512(n_kernels=2,num_classes=100,ratio_LR=1.)
        x=torch.randn(2,3,32,32)
        expected=m.base(x).detach().clone()
        with Features(m,'base_output') as capture:m(x);self.assertTrue(torch.equal(capture.get(),expected))
        a=options();c=SimpleNamespace(args=a,download_round=1)
        tensor=dict(prototypes=torch.zeros(100,512,dtype=torch.float64),valid=torch.ones(100,dtype=torch.bool))
        bank=encode_packet('global_last_epoch_library',tensor,dict(source_completed_train_count=1,
            feature_position='base_output',aggregation='equal_client_class_means'))
        s=SimpleNamespace(avg_proto_packet=bank,args=a,clients=[c],loop_round=1)
        p.prepare_round(s);self.assertEqual(c.avg_proto_receipt['source_completed_train_count'],1)
        s.loop_round=2
        with self.assertRaises(ValueError):p.prepare_round(s)

    def test_off_matches_frozen_old_SVD_CE_F_given_same_new_protocol_batches(self):
        source=subprocess.check_output(['git','show','bdd821c:system/flcore/clients/clientMultiGroupSVD.py'],cwd=ROOT,text=True)
        namespace={'__name__':'fixed_bdd821c_client'};exec(compile(source,'bdd821c/clientMultiGroupSVD.py','exec'),namespace)
        torch.manual_seed(8);model=Hyper_CNN_512(n_kernels=2,num_classes=100,ratio_LR=.15)
        model.recover_larger_model();model.ratio_LR=1.
        generator=torch.Generator().manual_seed(33)
        data=list(zip(torch.randn(3,3,32,32,generator=generator),torch.tensor([0,1,0])))
        initial=rng_state();results=[]
        with tempfile.TemporaryDirectory() as directory,redirect_stdout(io.StringIO()):
            for legacy in (True,False):
                a=options(multi_proto_mode='none',multi_proto_lambda=0.,multi_proto_target_type='fixed_norm',
                    local_epochs=2,batch_size=2,local_grad_clip=10)
                a.device='cpu';a.save_folder_name_full=directory
                c=clientMultiGroupSVD(a,0,3,3,initial_model=model,train_slow=False,send_slow=False)
                set_rng(initial);c.set_parameters([model],0)
                p.prepare_round(SimpleNamespace(avg_proto_packet=None,clients=[c],args=a))
                with patch('flcore.clients.clientMultiGroupProj.read_client_data',return_value=data),patch.object(torch,'save'):
                    if legacy:namespace['clientMultiGroupSVD'].train(c)
                    else:c.train()
                results.append((copy.deepcopy(c.model_set[0].state_dict()),c.full_uploads[0],rng_state()))
            for group in (0,1):
                for name,value in results[0][group].items():self.assertTrue(torch.equal(value,results[1][group][name]),name)
            set_rng(results[1][2]);assert_rng(self,results[0][2])

    def test_commands_have_only_intended_factor_differences(self):
        text=(ROOT/'system/single_avg_proto_commands.sh').read_text()
        commands=[shlex.split(line.strip().strip('"')) for line in text.splitlines() if line.strip().startswith('"python')]
        self.assertEqual(len(commands),8)
        for index,c in enumerate(commands):
            for key,value in [('-nsm','1'),('-nc','20'),('-gr','100'),('-ls','5'),('-lr','0.005'),('-lbs','16'),
                ('--multi_svd_init','low_rank_recovered'),('--multi_svd_init_rank_rate','0.15'),
                ('--multi_group_mode','group_avg_only'),('--multi_group_sizes','20'),('--multi_group_weight','1.0')]:
                self.assertEqual(c[c.index(key)+1],value)
            self.assertEqual('--multi_svd' in c,index not in (4,5))
            self.assertEqual(c[c.index('-is_regular')+1],'0' if index in (2,3,4,5) else '1')
            self.assertEqual(c[c.index('--local_grad_clip')+1],'0' if index in (6,7) else '10')
            self.assertEqual(c[c.index('--multi_proto_lambda')+1],'1.0' if index%2 else '0.0')
            self.assertNotIn('--multi_group_file',c)


if __name__=='__main__':unittest.main()

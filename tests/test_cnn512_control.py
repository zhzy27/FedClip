"""Full CNN shared construction, actual SGD paths and unchanged aggregation."""
import ast
import copy
from contextlib import redirect_stdout
import io
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'system'))
from common.models import BaseHeadCNN, CNN_512, create_full_model, initial_full_model, copy_full_parameters
from common.experiment import rng_state, set_rng
from flcore.clients.clientTargetProj import clientTargetProj
from flcore.clients.clientfedfews import clientFedFewS
from flcore.clients.clientbase import save_item, load_item
from utils.target_projection import aggregate_target_updates, EPS
from utils.projection_variants import aggregate_projection_weighting
from utils.cnn512_control import validate_control


class CNNControlTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)

    def test_original_cnn_constructor_forward_rng_and_old_pickle_path(self):
        source=subprocess.check_output(['git','show','2af9cd2:system/flcore/trainmodel/models.py'],cwd=ROOT).decode('utf-8')
        tree=ast.parse(source)
        nodes=[n for n in tree.body if isinstance(n,ast.ClassDef) and n.name in ('BaseHeadCNN','CNN_512')]
        scope=dict(nn=torch.nn,torch=torch)
        exec(compile(ast.Module(body=nodes,type_ignores=[]),'oldCNN','exec'),scope)
        torch.manual_seed(77)
        before=torch.get_rng_state()
        old=scope['CNN_512'](out_dim=100)
        after=torch.get_rng_state()
        torch.set_rng_state(before)
        new=CNN_512(out_dim=100)
        self.assertTrue(torch.equal(after,torch.get_rng_state()))
        self.assertEqual(list(old.state_dict()),list(new.state_dict()))
        for n,p in old.state_dict().items():
            self.assertTrue(torch.equal(p,new.state_dict()[n]),n)
        x=torch.randn(2,3,32,32)
        self.assertTrue(torch.equal(old(x),new(x)))
        self.assertEqual(new.base[7].weight.shape,(2000,800))
        from flcore.trainmodel.models import CNN_512 as LegacyCNN
        self.assertIs(LegacyCNN,CNN_512)
        with tempfile.TemporaryDirectory() as directory:
            # Old pickle global lookup resolves the existing public alias.
            cls=CNN_512
            module=cls.__module__
            try:
                cls.__module__='flcore.trainmodel.models'
                torch.save(new,Path(directory)/'old.pt')
            finally:
                cls.__module__=module
            self.assertTrue(torch.equal(torch.load(Path(directory)/'old.pt',weights_only=False)(x),old(x)))

    def test_seeded_factory_is_identical_and_does_not_consume_rng(self):
        args=SimpleNamespace(model_family='CNN-512',num_classes=100,cnn512_control=True,seed=0)
        before=rng_state()
        a=initial_full_model(args)
        torch.manual_seed(999)
        consumed=torch.get_rng_state()
        b=initial_full_model(args)
        self.assertTrue(torch.equal(consumed,torch.get_rng_state()))
        self.assertTrue(all(torch.equal(p,q) for p,q in zip(a.parameters(),b.parameters())))
        self.assertFalse(any(hasattr(a,n) for n in ['ratio_LR','recover_larger_model','decom_larger_model']))
        set_rng(before)
        wrong=torch.nn.Linear(2,2)
        with self.assertRaises(ValueError): copy_full_parameters(a,wrong)

    def test_actual_target_and_few_single_step_match_with_active_clipping(self):
        torch.manual_seed(0)
        model=CNN_512(out_dim=100)
        # Large input ensures clipping really activates.
        x,y=torch.randn(4,3,32,32)*20,torch.tensor([0,1,2,3])
        expected=copy.deepcopy(model)
        opt=torch.optim.SGD(expected.parameters(),lr=.005)
        ce=torch.nn.functional.cross_entropy(expected(x),y); ce.backward()
        norm=torch.nn.utils.clip_grad_norm_(expected.parameters(),10.)
        self.assertGreater(float(norm),10.)
        opt.step()
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            c=clientTargetProj.__new__(clientTargetProj)
            c.args=SimpleNamespace(target_local_schedule='joint', target_prox_mu=0., target_prox_scope='helpers',
                target_client_id=0,is_regular=0, model_family='CNN-512',local_grad_clip=10.)
            c.role,c.save_folder_name,c.id,c.device='Client_0',directory,0,'cpu'
            c.learning_rate,c.local_epochs,c.train_slow=.005,1,False
            c.loss=torch.nn.CrossEntropyLoss(); c.train_time_cost=dict(num_rounds=0,total_cost=0.)
            c.load_train_data=lambda:[(x,y)]
            save_item(model,c.role,'model',directory)
            c.train()
            actual=load_item(c.role,'model',directory)
            for k in [1,3]:
                f=clientFedFewS.__new__(clientFedFewS)
                f.args=SimpleNamespace(local_grad_clip=10.,cnn512_control=True)
                f.K,f.device,f.local_epochs,f.train_samples=k,'cpu',1,4
                f.loss=torch.nn.CrossEntropyLoss()
                f.model_set=[copy.deepcopy(model) for _ in range(k)]
                f.optimizer_set=[torch.optim.SGD(m.parameters(),lr=.005) for m in f.model_set]
                f.load_train_data=lambda:[(x,y)]
                f.train_time_cost=dict(num_rounds=0,total_cost=0.)
                f.train()
                for m in [actual]+f.model_set:
                    for n,p in expected.named_parameters():
                        self.assertTrue(torch.equal(p,dict(m.named_parameters())[n]),n)
                self.assertEqual(f.last_normal_train_stats['train_batches'],k)

    def test_avg_math_and_projection_existing_kernel_against_independent_formula(self):
        g={'head.weight':torch.tensor([1.,2.],dtype=torch.float64)}
        delta=[torch.tensor([1.,0.],dtype=torch.float64),torch.tensor([-2.,3.],dtype=torch.float64),torch.tensor([1.,2.],dtype=torch.float64)]
        posts=[{'head.weight':g['head.weight']+d} for d in delta]
        weights=[.1,.3,.6]
        uploads=lambda:iter([(i,weights[i],p) for i,p in enumerate(posts)])
        avg,_,_=aggregate_target_updates(g,posts[0],uploads(),0,'avg')
        expected=sum(weights[i]*posts[i]['head.weight'] for i in range(3))
        self.assertTrue(torch.equal(avg['head.weight'],expected))
        actual,_,rows,_,_=aggregate_projection_weighting(g,posts[0],uploads,0,'projection_softmax',.125)
        cos=torch.tensor([torch.dot(d,delta[0])/(d.norm()*delta[0].norm()) for d in delta[1:]])
        q=torch.softmax(cos/.2,dim=0)
        alpha=[.125]+(.875*q).tolist()
        expected=g['head.weight']+.125*delta[0]
        for i in [1,2]:
            coeff=min(0.,float(torch.dot(delta[i],delta[0])))/(float(delta[0].square().sum())+EPS)
            expected+=alpha[i]*(delta[i]-coeff*delta[0])
        torch.testing.assert_close(actual['head.weight'],expected,atol=1e-14,rtol=1e-14)
        self.assertTrue(rows[1]['conflict'])

    def test_clip_parameter_validation_and_full_model_restrictions(self):
        args=SimpleNamespace(algorithm='FedFewS',model_family='FedAvgCNN-plain',local_grad_clip=None)
        validate_control(args)  # No new defaults for the reproduction entry.
        self.assertIsNone(args.local_grad_clip)
        for value in [-1,float('nan'),float('inf')]:
            args.local_grad_clip=value
            with self.assertRaises(ValueError): validate_control(args)

    def test_control_protocol_is_opt_in_and_rejects_mixed_settings(self):
        values=dict(algorithm='FedTargetProj',model_family='CNN-512',dataset='Cifar100',num_classes=100,
            num_clients=20,target_client_id=0,seed=0,cnn512_control=True,target_proj_mode='projection_softmax',
            projection_self_weight=.125, local_grad_clip=None,join_ratio=1.,random_join_ratio=False,
            client_drop_rate=0.,train_slow_rate=0.,send_slow_rate=0.,auto_break=False,learning_rate_decay=False,
            resume=False,few_shot=0,is_regular=0,target_prox_mu=0.,meta_c0_split='',meta_collect_snapshots=False,
            target_head_mode='shared',target_local_schedule='joint',head_diag_collect=False,time_select=False,
            local_learning_rate=.005,batch_size=16,local_epochs=5,eval_gap=1,control_test_batch_size=16,
            data_root='existing',data_subdir='pat_20',data_formats='npz_only')
        a=SimpleNamespace(**values)
        validate_control(a)
        self.assertEqual((a.local_grad_clip,a.test_batch_size,a.data_formats),(10.,16,'npz_only'))
        self.assertFalse(a.target_prox_experiment)
        for field,value in [('target_prox_mu',.001),('is_regular',1),('target_head_mode','local'),
            ('target_proj_mode','feedback_avg'),('projection_self_weight',1.05),('local_epochs',0),
            ('data_root',None),('data_formats','unknown'),('model_family','CNN-5-512'),('seed',-1)]:
            bad=dict(values);bad[field]=value
            with self.subTest(field=field), self.assertRaises(ValueError): validate_control(SimpleNamespace(**bad))

    def test_few_named_full_model_branch_keeps_dual_weight_increment(self):
        from flcore.servers.serverfedfews import FedFewS
        s=FedFewS.__new__(FedFewS)
        s.args=SimpleNamespace(model_family='CNN-512')
        s.K,s.mu=3,.01
        def model(value):
            m=torch.nn.Linear(2,1).double()
            with torch.no_grad():
                for p in m.parameters(): p.fill_(value)
            return m
        old=[1.,2.,-3.]
        s.global_model_set=[model(v) for v in old]
        losses=np.array([[.1,.2,.3],[.3,.1,.2],[.2,.3,.1]],dtype=np.float32)
        s.clients=[SimpleNamespace(id=i,train_model_losses=losses[i]) for i in range(3)]
        s.uploaded_ids=[2,0,1]
        counts=np.array([2,5,11])
        s.uploaded_weights=[counts[i]/counts.sum() for i in s.uploaded_ids]
        s.uploaded_models=[[model(old[k]+(i+1)*(k+1)*.2) for k in range(3)] for i in s.uploaded_ids]
        saved=copy.deepcopy(s.uploaded_models)
        exp=np.exp(-losses[s.uploaded_ids].astype(float)*np.array(s.uploaded_weights)[:,None]/s.mu)
        w=exp/exp.sum(axis=1,keepdims=True)
        alpha=1/exp.sum(axis=1); alpha/=alpha.sum()
        s.aggregate_parameters()
        for k in range(3):
            expected=old[k]+sum(alpha[r]*w[r,k]*(cid+1)*(k+1)*.2 for r,cid in enumerate(s.uploaded_ids))
            for p in s.global_model_set[k].parameters():
                torch.testing.assert_close(p,torch.full_like(p,expected),atol=2e-14,rtol=0)
        for a,b in zip(saved,s.uploaded_models):
            self.assertTrue(all(torch.equal(p,q) for m,n in zip(a,b) for p,q in zip(m.parameters(),n.parameters())))


if __name__=='__main__': unittest.main()

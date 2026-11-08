"""Classwise ranks, sample-index alignment, score isolation and CE/F gradients."""
import copy
from contextlib import redirect_stdout
import io
from pathlib import Path
import random
import subprocess
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'system'))
from utils import target_curriculum as cur
from flcore.clients.clientTargetProj import clientTargetProj
from common.experiment import rng_state,set_rng


def assert_rng(test,before,after):
    # Compare captured states directly. Do not query/initialize extra CUDA cards.
    test.assertEqual(before['python'],after['python'])
    test.assertEqual(before['numpy'][0],after['numpy'][0])
    np.testing.assert_array_equal(before['numpy'][1],after['numpy'][1])
    test.assertEqual(before['numpy'][2:],after['numpy'][2:])
    test.assertTrue(torch.equal(before['torch'],after['torch']))
    test.assertEqual(len(before['cuda']),len(after['cuda']))
    for x,y in zip(before['cuda'],after['cuda']):test.assertTrue(torch.equal(x,y))


def options(**changes):
    d=dict(algorithm='FedTargetProj',model_family='Decom_CNN-5-512',target_proj_mode='projection_softmax',
        target_head_mode='shared',target_local_schedule='joint',target_client_id=2,num_clients=4,
        target_curriculum_mode='loss_rank',target_curriculum_strength=.5,target_curriculum_warmup_rounds=1,
        target_curriculum_diagnostics=1,is_regular=1,regular_lamda=.001,seed=17,local_grad_clip=10.)
    d.update(changes);return SimpleNamespace(**d)


class Toy(torch.nn.Module):
    def __init__(self):
        super().__init__();self.weight_v=torch.nn.Parameter(torch.randn(3,4)*.1);self.bias=torch.nn.Parameter(torch.zeros(4))
        self.register_buffer('probe',torch.zeros(1));self.ratio_LR=.5;self.trace=[];self.f_calls=0;self.score_forwards=0
        self.random_reads=False;self.fail_score=False
    def forward(self,x):
        if self.training:self.trace.append(x[:,0].tolist())
        else:self.score_forwards+=1
        self.probe.add_(1.)
        if self.random_reads:random.random();np.random.rand();torch.rand(2)
        if self.fail_score and not self.training:raise RuntimeError('score error')
        return x@self.weight_v+self.bias
    def frobenius_decay(self):self.f_calls+=1;return self.weight_v.square().sum()


def client(model,data,cid=1,epochs=3,**kw):
    c=clientTargetProj.__new__(clientTargetProj);c.args=options(**kw);c.device='cpu';c.id=cid;c.role=f'Client_{cid}'
    c.save_folder_name='unused';c.learning_rate=.005;c.local_epochs=epochs;c.batch_size=2;c.train_samples=len(data)
    c.train_slow=False;c.loss=torch.nn.CrossEntropyLoss();c.train_time_cost=dict(num_rounds=0,total_cost=0.)
    c._load_model=lambda:model;c.load_train_data=lambda:DataLoader(data,batch_size=2,shuffle=True,drop_last=False,num_workers=0)
    return c


class CurriculumTests(unittest.TestCase):
    def test_launcher_forwarding_and_formal_101_round_commands(self):
        import shlex
        for mode in ('none','loss_rank','random_rank'):
            result=subprocess.run([sys.executable,str(ROOT/'system/run_target_proj.py'),'--dry-run',
                '--rounds','100','--modes','projection_softmax','--projection_self_weight','.125',
                '--target_curriculum_mode',mode,'--target_curriculum_strength','.5',
                '--target_curriculum_warmup_rounds','10'],capture_output=True,text=True)
            self.assertEqual(result.returncode,0,result.stderr)
            self.assertIn('--target_curriculum_mode '+mode,result.stdout)
            self.assertIn('--target_curriculum_warmup_rounds 10',result.stdout)
            self.assertIn('-gr 100',result.stdout)
        result=subprocess.run([sys.executable,str(ROOT/'system/run_target_proj.py'),'--dry-run',
            '--modes','avg','--target_curriculum_mode','loss_rank'],capture_output=True,text=True)
        self.assertNotEqual(result.returncode,0)
        commands=[shlex.split(line.strip().strip('"')) for line in (ROOT/'system/target_curriculum_commands.sh').read_text().splitlines() if line.strip().startswith('"')]
        self.assertEqual(len(commands),3)
        for did,(mode,command) in enumerate(zip(('none','loss_rank','random_rank'),commands)):
            for key,value in [('-gr','100'),('-did',str(did)),('-ls','5'),('--target_curriculum_warmup_rounds','10'),
                ('--target_curriculum_mode',mode),('--projection_self_weight','0.125'),('--local_grad_clip','10')]:
                self.assertEqual(command[command.index(key)+1],value)

    def setUp(self):
        torch.set_num_threads(2);torch.manual_seed(12)
        self.data=list(zip(torch.randn(5,3),torch.tensor([0,0,1,1,2])))

    def test_validation_and_warmup(self):
        cur.validate(options());self.assertFalse(cur.active(options(target_curriculum_warmup_rounds=10),1,9))
        self.assertTrue(cur.active(options(target_curriculum_warmup_rounds=10),1,10));self.assertFalse(cur.active(options(),2,100))
        bad=[dict(target_curriculum_strength=v) for v in (-.1,1.,float('nan'),float('inf'))]+[
            dict(target_curriculum_warmup_rounds=-1),dict(target_curriculum_mode='bad'),dict(target_curriculum_diagnostics=3),
            dict(target_proj_mode='projection_history_softmax'),dict(algorithm='FedMultiGroupProj'),dict(target_head_mode='local'),
            dict(target_local_schedule='fedrep'),dict(target_prox_mu=.1),dict(target_z2_lambda=1),dict(target_c0_replay_mode='batch_pair'),
            dict(meta_c0_split='x'),dict(multi_proto_mode='group_priority'),dict(join_ratio=.5),dict(target_client_id=4)]
        for change in bad:
            with self.subTest(change=change),self.assertRaises(ValueError):cur.validate(options(**change))
        for off in [dict(target_curriculum_mode='none'),dict(target_curriculum_strength=0.)]:
            cur.validate(options(**off,algorithm='Other',target_prox_mu=1.,target_local_schedule='fedrep'))

    def test_class_ties_random_multisets_and_weights(self):
        scores=torch.tensor([1.,1.,3.,8.,8.,4.]);labels=torch.tensor([0,0,0,1,1,2])
        ranks=cur.class_ranks(scores,labels)
        torch.testing.assert_close(ranks,torch.tensor([1/3,1/3,5/6,.5,.5,.5],dtype=torch.float64),rtol=0,atol=1e-15)
        for c in labels.unique():self.assertAlmostEqual(float(ranks[labels==c].mean()),.5)
        before=rng_state();randomized=cur.permute_class_ranks(ranks,labels,2,3,4);assert_rng(self,before,rng_state())
        self.assertTrue(torch.equal(randomized,cur.permute_class_ranks(ranks,labels,2,3,4)))
        for c in labels.unique():self.assertTrue(torch.equal(ranks[labels==c].sort()[0],randomized[labels==c].sort()[0]))
        self.assertEqual([cur.epoch_beta(.5,e,5) for e in range(1,6)],[.5,.375,.25,.125,0.])
        self.assertEqual(cur.epoch_beta(.5,1,1),0.)
        for e in range(1,6):
            weights=cur.sample_weights(ranks,cur.epoch_beta(.5,e,5));self.assertTrue((weights>0).all())
            for c in labels.unique():self.assertAlmostEqual(float(weights[labels==c].mean()),1.)
        self.assertTrue(torch.equal(cur.sample_weights(ranks,0.),torch.ones_like(ranks)))
        with self.assertRaises(ValueError):cur.class_ranks(torch.tensor([float('nan')]),torch.tensor([0]))

    def test_scoring_protects_state_modes_rng_on_success_and_failure(self):
        m=Toy();m.random_reads=True;c=client(m,self.data)
        state=copy.deepcopy(m.state_dict());modes=[x.training for x in m.modules()];before=rng_state()
        with patch.object(torch.cuda,'get_rng_state_all',side_effect=AssertionError('all GPUs forbidden')):
            ranks,stats=cur.score_download(c,m,c.load_train_data(),3)
        self.assertEqual(stats['score_samples'],5);self.assertEqual(stats['score_batches'],3);self.assertEqual(m.f_calls,0)
        for n,p in m.state_dict().items():self.assertTrue(torch.equal(p,state[n]))
        self.assertEqual(modes,[x.training for x in m.modules()]);assert_rng(self,before,rng_state())
        m.fail_score=True
        with self.assertRaisesRegex(RuntimeError,'score error'):cur.score_download(c,m,c.load_train_data(),3)
        for n,p in m.state_dict().items():self.assertTrue(torch.equal(p,state[n]))
        self.assertEqual(modes,[x.training for x in m.modules()]);assert_rng(self,before,rng_state())

    def test_disabled_target_zero_warmup_equal_frozen_original(self):
        old={};source=subprocess.check_output(['git','show','34a3331:system/flcore/clients/clientTargetProj.py'],cwd=ROOT,text=True)
        exec(compile(source,'34a3331/clientTargetProj.py','exec'),old)
        template=Toy();initial=rng_state();results=[]
        for kind in ('legacy','none','zero','warmup','target'):
            set_rng(initial);model=copy.deepcopy(template)
            kw={}
            if kind in ('legacy','none'):kw['target_curriculum_mode']='none'
            if kind=='zero':kw['target_curriculum_strength']=0.
            if kind=='warmup':kw['target_curriculum_warmup_rounds']=10
            c=client(model,self.data,cid=2 if kind=='target' else 1,**kw)
            with patch('flcore.clients.clientTargetProj.save_item'),redirect_stdout(io.StringIO()),\
                 patch.object(cur,'score_download',side_effect=AssertionError('no score work')):
                if kind=='legacy':old['clientTargetProj'].train(c,3)
                else:c.train(3)
            self.assertEqual(model.f_calls,9);self.assertEqual(model.score_forwards,0)
            results.append((copy.deepcopy(model.state_dict()),rng_state(),model.trace))
        for state,rng,trace in results[1:]:
            for n,p in state.items():self.assertTrue(torch.equal(p,results[0][0][n]),n)
            assert_rng(self,results[0][1],rng);self.assertEqual(trace,results[0][2])

    def test_active_sampler_diagnostics_and_E1_match_original(self):
        template=Toy();template.random_reads=True;initial=rng_state();results=[]
        for diagnostics in (0,1):
            set_rng(initial);m=copy.deepcopy(template);c=client(m,self.data,target_curriculum_diagnostics=diagnostics)
            with patch('flcore.clients.clientTargetProj.save_item'),redirect_stdout(io.StringIO()):c.train(3)
            self.assertEqual(m.f_calls,9);self.assertEqual(c.last_normal_train_stats['train_batches'],9)
            self.assertEqual(sum(r['score_samples'] for r in c.last_curriculum_rows),5)
            self.assertEqual(sum(r['train_sample_visits'] for r in c.last_curriculum_rows),15)
            self.assertEqual(c.last_curriculum_rows[-1]['weight_min'],1.)
            self.assertEqual(c.last_curriculum_rows[-1]['weight_max'],1.)
            results.append((copy.deepcopy(m.state_dict()),rng_state(),m.trace))
        for n,p in results[0][0].items():self.assertTrue(torch.equal(p,results[1][0][n]))
        assert_rng(self,results[0][1],results[1][1]);self.assertEqual(results[0][2],results[1][2])
        one=[]
        for mode in ('none','loss_rank'):
            set_rng(initial);m=copy.deepcopy(template);c=client(m,self.data,epochs=1,target_curriculum_mode=mode)
            with patch('flcore.clients.clientTargetProj.save_item'),redirect_stdout(io.StringIO()):c.train(3)
            one.append((copy.deepcopy(m.state_dict()),rng_state(),m.trace))
        for n,p in one[0][0].items():self.assertTrue(torch.equal(p,one[1][0][n]),n)
        assert_rng(self,one[0][1],one[1][1]);self.assertEqual(one[0][2],one[1][2])

    def test_independent_weighted_SGD_formula_F_once_and_sample_denominator(self):
        initial=Toy();before=rng_state();actual=copy.deepcopy(initial);expected=copy.deepcopy(initial)
        c=client(actual,self.data,epochs=2,local_grad_clip=0.)
        set_rng(before)
        with patch('flcore.clients.clientTargetProj.save_item'),redirect_stdout(io.StringIO()):c.train(3)
        actual_rng=rng_state();set_rng(before)
        ref=client(expected,self.data,epochs=2,local_grad_clip=0.);original=ref.load_train_data()
        ranks,_=cur.score_download(ref,expected,original,3)
        loader=cur.indexed_training_loader(original);optimizer=torch.optim.SGD(expected.parameters(),lr=.005);expected.train()
        for epoch in (1,2):
            beta=cur.epoch_beta(.5,epoch,2)
            for x,y,ids in loader:
                optimizer.zero_grad();logits=expected(x)
                if beta:
                    values=torch.nn.functional.cross_entropy(logits,y,reduction='none')
                    # Independent per-example scalar construction, explicit 1/B.
                    terms=[(1+beta*(1-2*float(ranks[int(i)])))*values[j] for j,i in enumerate(ids)]
                    ce=torch.stack(terms).sum()/len(y)
                else:ce=ref.loss(logits,y)
                f=.001*expected.frobenius_decay();(ce+f).backward();optimizer.step()
        for n,p in actual.named_parameters():torch.testing.assert_close(p,dict(expected.named_parameters())[n],rtol=1e-6,atol=1e-8)
        self.assertEqual(actual.f_calls,6);self.assertEqual(expected.f_calls,6);self.assertEqual(actual.trace,expected.trace)
        assert_rng(self,actual_rng,rng_state())

    @unittest.skipUnless(torch.cuda.is_available(),'CUDA is not available')
    def test_active_CUDA_scoring_and_diagnostic_isolation(self):
        # Exercise the actual CUDA CE/weights/backward path, using toy TRAIN data.
        template=Toy().to('cuda');initial=rng_state();results=[]
        for diagnostics in (0,1):
            set_rng(initial);model=copy.deepcopy(template)
            c=client(model,self.data,target_curriculum_diagnostics=diagnostics);c.device='cuda'
            with patch('flcore.clients.clientTargetProj.save_item'),redirect_stdout(io.StringIO()):c.train(3)
            self.assertTrue(all(torch.isfinite(p).all() for p in model.parameters()))
            results.append((copy.deepcopy(model.state_dict()),rng_state(),model.trace))
        for n,p in results[0][0].items():self.assertTrue(torch.equal(p,results[1][0][n]),n)
        assert_rng(self,results[0][1],results[1][1]);self.assertEqual(results[0][2],results[1][2])


if __name__=='__main__':unittest.main()

"""Paired-batch gradients, private sampling, disabled paths and CLI contracts."""
import copy
from contextlib import redirect_stdout
import io
from pathlib import Path
import shlex
import subprocess
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT/'system'))
from utils import target_c0_replay as replay
from flcore.clients.clientTargetProj import clientTargetProj
from common.experiment import rng_state, set_rng
from test_control_population import assert_rng as assert_current_rng
from utils.target_head import preserved_head_rng


def assert_rng(test, before, after):
    with preserved_head_rng():
        set_rng(after)
        assert_current_rng(test, before)


def options(**changes):
    values = dict(algorithm='FedTargetProj', model_family='Decom_CNN-5-512', target_proj_mode='projection_softmax',
        target_head_mode='shared', target_local_schedule='joint', target_prox_mu=0., target_client_id=2, num_clients=4,
        is_regular=1, regular_lamda=.001, target_c0_replay_mode='batch_pair', target_c0_replay_weight=.5,
        target_c0_replay_diagnostics=0, seed=17)
    values.update(changes); return SimpleNamespace(**values)


class Toy(torch.nn.Module):
    def __init__(self):
        super().__init__(); self.fc=torch.nn.Linear(3, 4); self.f_calls=0; self.forwards=0

    def forward(self, x):
        self.forwards += 1; return self.fc(x)

    def frobenius_decay(self):
        self.f_calls += 1; return self.fc.weight.square().sum()


def client(model, own, cid=1, **changes):
    c=clientTargetProj.__new__(clientTargetProj); c.args=options(**changes)
    c.device='cpu'; c.dataset='Synthetic'; c.id=cid; c.role=f'Client_{cid}'; c.few_shot=0
    c.save_folder_name='unused'; c.learning_rate=.005; c.local_epochs=2; c.batch_size=2; c.train_samples=len(own)
    c.train_slow=False; c.loss=torch.nn.CrossEntropyLoss(); c.train_time_cost=dict(num_rounds=0,total_cost=0.)
    c._load_model=lambda:model
    c.load_train_data=lambda:DataLoader(own,batch_size=2,shuffle=True,drop_last=False,num_workers=0)
    return c


class ReplayTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        if torch.cuda.is_available(): torch.cuda.get_rng_state_all()
        generator=torch.Generator().manual_seed(31)
        self.own=list(zip(torch.randn(5,3,generator=generator),torch.arange(5)%4))
        self.source=list(zip(torch.randn(3,3,generator=generator),torch.arange(3)%4))

    def run_client(self,c):
        with patch('flcore.clients.clientTargetProj.save_item'), redirect_stdout(io.StringIO()):
            c.train(3)

    def test_validation_and_disabled_combinations(self):
        replay.validate(options())
        for changes in [dict(target_c0_replay_weight=v) for v in (-1,1.1,float('nan'),float('inf'))]+[
            dict(target_c0_replay_mode='bad'),dict(target_c0_replay_diagnostics=2),dict(target_client_id=4),
            dict(algorithm='FedMultiGroupProj'),dict(target_proj_mode='avg'),dict(target_proj_mode='feedback_avg'),
            dict(target_local_schedule='fedrep'),dict(target_head_mode='local'),dict(model_family='CNN-512'),
            dict(target_prox_mu=.01),dict(target_head_l2_lambda=.1),dict(target_z2_lambda=1),
            dict(meta_c0_split='split'),dict(meta_collect_snapshots=True),dict(join_ratio=.9),
            dict(random_join_ratio=True),dict(few_shot=1),dict(client_drop_rate=.1)]:
            with self.subTest(changes=changes), self.assertRaises(ValueError): replay.validate(options(**changes))
        for changes in (dict(target_c0_replay_mode='none'),dict(target_c0_replay_weight=0)):
            args=options(**changes,algorithm='Other',target_local_schedule='fedrep',target_prox_mu=1)
            before=vars(args).copy(); replay.validate(args); replay.ensure_names(args)
            self.assertEqual(vars(args),before)

    def test_private_stream_reproducible_round_local_wrap_and_train_only(self):
        c=client(Toy(),self.own)
        calls=[]
        def read(dataset,cid,is_train,**kwargs):
            calls.append((dataset,cid,is_train)); return self.source
        state=rng_state()
        with patch('flcore.clients.clientbase.read_client_data',side_effect=read):
            a,b=replay.TargetBatchStream(c,3),replay.TargetBatchStream(c,3)
            batches_a=[a.next()[1].tolist() for _ in range(7)]
            batches_b=[b.next()[1].tolist() for _ in range(7)]
            self.assertEqual(batches_a,batches_b); self.assertEqual(a.wraps,3)
            self.assertEqual(sum(map(len,batches_a)),11)
            other=replay.TargetBatchStream(c,4)
            self.assertNotEqual(a.seed,other.seed)
            c.id=3; self.assertNotEqual(a.seed,replay.TargetBatchStream(c,3).seed)
        assert_rng(self,state,rng_state())
        self.assertEqual(calls,[('Synthetic',2,True)]*4)

    def test_paired_loss_full_gradient_one_step_F_once_short_batches_and_sampling(self):
        torch.manual_seed(93); original=Toy(); actual=copy.deepcopy(original); expected=copy.deepcopy(original)
        c=client(actual,self.own); state=rng_state(); expected_trace=[]; actual_trace=[]
        with patch('flcore.clients.clientbase.read_client_data',return_value=self.source) as reader:
            target=replay.TargetBatchStream(c,3)
            optimizer=torch.optim.SGD(expected.parameters(),lr=.005)
            ce_own=ce_target=reg_steps=total_steps=0.; nown=ntarget=0
            loss=torch.nn.CrossEntropyLoss()
            for _ in range(2):
                for x,y in c.load_train_data():
                    expected_trace.append(y.tolist()); tx,ty=target.next(); optimizer.zero_grad()
                    a=loss(expected(x),y); b=loss(expected(tx),ty); f=.001*expected.frobenius_decay()
                    total=.5*a+.5*b+f; total.backward()
                    torch.nn.utils.clip_grad_norm_(expected.parameters(),10.); optimizer.step()
                    ce_own+=float(a)*len(y); ce_target+=float(b)*len(ty)
                    reg_steps+=float(f); total_steps+=float(total); nown+=len(y); ntarget+=len(ty)
            expected_rng=rng_state(); set_rng(state)
            native=c.load_train_data
            class Traced(DataLoader):
                def __iter__(self):
                    for x,y in super().__iter__(): actual_trace.append(y.tolist()); yield x,y
            c.load_train_data=lambda:Traced(self.own,batch_size=2,shuffle=True,drop_last=False,num_workers=0)
            with patch.object(torch.optim.SGD,'step',autospec=True,side_effect=torch.optim.SGD.step) as step:
                self.run_client(c); self.assertEqual(step.call_count,6)
            self.assertEqual(reader.call_count,2)
        for n,p in actual.named_parameters(): self.assertTrue(torch.equal(p,dict(expected.named_parameters())[n]),n)
        assert_rng(self,expected_rng,rng_state()); self.assertEqual(actual_trace,expected_trace)
        self.assertEqual(actual.f_calls,6); self.assertEqual(actual.forwards,12)
        s=c.last_c0_replay_train_stats
        self.assertEqual((s['own_sample_visits'],s['c0_sample_visits'],s['own_batches'],s['c0_batches']),(10,9,6,6))
        self.assertEqual((s['optimizer_steps'],s['local_epochs'],s['replay_iterator_wraps']),(6,2,2))
        self.assertEqual(c.train_samples,5); self.assertEqual(s['aggregation_train_samples'],5)
        for key,value in dict(train_mean_ce_own=ce_own/nown,train_mean_ce_c0=ce_target/ntarget,
                              train_mean_weighted_frobenius=reg_steps/6,train_mean_total_loss=total_steps/6).items():
            self.assertEqual(s[key],value)

    def test_formula_gradient_two_sources_and_single_regularizer(self):
        torch.manual_seed(2); m=Toy(); x,y=next(iter(DataLoader(self.own,batch_size=2))); tx,ty=next(iter(DataLoader(self.source,batch_size=2)))
        c=client(m,self.own); own,f=c._objective(m,x,y); target=c.loss(m(tx),ty)
        params=tuple(m.parameters()); a=.5
        g_own=torch.autograd.grad(own,params,retain_graph=True)
        g_target=torch.autograd.grad(target,params,retain_graph=True)
        g_f=torch.autograd.grad(f,params,retain_graph=True,allow_unused=True)
        actual=torch.autograd.grad((1-a)*own+a*target+f,params)
        for g,go,gt,gf in zip(actual,g_own,g_target,g_f):
            torch.testing.assert_close(g,(1-a)*go+a*gt+(gf if gf is not None else 0),rtol=1e-6,atol=1e-8)
        self.assertEqual(m.f_calls,1)

    def test_target_and_disabled_paths_no_extra_source_read_forward_or_rng(self):
        source={}; text=subprocess.check_output(['git','show','1abe63f:system/flcore/clients/clientTargetProj.py'],cwd=ROOT,text=True)
        exec(compile(text,'1abe63f/clientTargetProj.py','exec'),source)
        torch.manual_seed(42); template=Toy(); initial=rng_state(); results=[]
        for kind in ('legacy','omitted','none','zero','target'):
            set_rng(initial); model=copy.deepcopy(template)
            c=client(model,self.own,cid=2 if kind=='target' else 1)
            if kind=='omitted':del c.args.target_c0_replay_mode;del c.args.target_c0_replay_weight
            if kind=='none':c.args.target_c0_replay_mode='none'
            if kind=='zero':c.args.target_c0_replay_weight=0
            with patch('flcore.clients.clientbase.read_client_data',side_effect=AssertionError('unexpected target TRAIN read')),\
                 patch('flcore.clients.clientTargetProj.save_item'),redirect_stdout(io.StringIO()):
                if kind=='legacy':
                    with patch.dict(source,save_item=lambda *a:None):source['clientTargetProj'].train(c,3)
                else:c.train(3)
            results.append((copy.deepcopy(model.state_dict()),rng_state()))
            self.assertEqual(model.forwards,6);self.assertEqual(model.f_calls,6)
            if kind=='target':
                stats=c.last_c0_replay_train_stats
                self.assertEqual((stats['effective_weight'],stats['c0_batches'],stats['c0_sample_visits']),(0,0,0))
        for state,rng in results[1:]:
            for n,p in state.items():self.assertTrue(torch.equal(p,results[0][0][n]),n)
            assert_rng(self,results[0][1],rng)

    def test_diagnostics_readonly_and_sampler_does_not_change_own_order(self):
        torch.manual_seed(112); initial_model=Toy(); initial=rng_state(); results=[]
        for diag in (0,1):
            set_rng(initial); model=copy.deepcopy(initial_model); c=client(model,self.own,target_c0_replay_diagnostics=diag)
            with patch('flcore.clients.clientbase.read_client_data',return_value=self.source):self.run_client(c)
            results.append((copy.deepcopy(model.state_dict()),rng_state()))
        for n,p in results[0][0].items():self.assertTrue(torch.equal(p,results[1][0][n]))
        assert_rng(self,results[0][1],results[1][1])
        # This CNN has no dropout/augmentation. The extra independent loader never advances global sampling RNG.
        set_rng(initial); sequences=[]
        for extra in (False,True):
            set_rng(initial); c=client(Toy(),self.own); own_rng=rng_state(); batches=[]
            with patch('flcore.clients.clientbase.read_client_data',return_value=self.source):
                stream=replay.TargetBatchStream(c,3) if extra else None
                set_rng(own_rng)
                for _ in range(2):
                    for x,y in c.load_train_data():
                        batches.append(y.tolist())
                        if stream:stream.next()
            sequences.append((batches,rng_state()))
        self.assertEqual(sequences[0][0],sequences[1][0]);assert_rng(self,sequences[0][1],sequences[1][1])

    def test_launcher_forwarding_and_complete_formal_command(self):
        command=[sys.executable,'system/run_target_proj.py','--dry-run','--modes','projection_softmax',
                 '--projection_self_weight','.125','--target_c0_replay_mode','batch_pair','--target_c0_replay_weight','.5']
        result=subprocess.run(command,cwd=ROOT,text=True,capture_output=True)
        self.assertEqual(result.returncode,0,result.stderr)
        self.assertIn('--target_c0_replay_mode batch_pair',result.stdout)
        self.assertIn('--target_c0_replay_weight 0.5',result.stdout)
        self.assertIn('replay_t0_pair0p5',result.stdout)
        nonzero_target=subprocess.run(command+['--target_client_id','2'],cwd=ROOT,text=True,capture_output=True)
        self.assertEqual(nonzero_target.returncode,0,nonzero_target.stderr)
        self.assertIn('--target_client_id 2',nonzero_target.stdout)
        self.assertIn('replay_t2_pair0p5',nonzero_target.stdout)
        bad=subprocess.run(command+['--target_z2_lambda','1'],cwd=ROOT,text=True,capture_output=True)
        self.assertNotEqual(bad.returncode,0)
        text=(ROOT/'system/target_c0_replay_commands.sh').read_text(encoding='utf-8')
        line=next(s.strip().strip('"') for s in text.splitlines() if s.strip().startswith('"python'))
        self.assertNotIn('$',line); tokens=shlex.split(line)
        for key,value in {'-gr':'100','-ls':'5','--projection_self_weight':'0.125','--target_c0_replay_weight':'0.5',
                          '--target_c0_replay_mode':'batch_pair','--target_z2_lambda':'0'}.items():
            self.assertEqual(tokens[tokens.index(key)+1],value)


if __name__=='__main__':unittest.main()

"""Paired replay uses real original SVD/F/SGD; no source fixture performance claims."""
import copy
from contextlib import redirect_stdout
import io
import shlex
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'system'))
from test_single_avg_proto import options
from utils import svd_r6_replay as d,fc1_tail_diagnostic as old,single_avg_proto as proto
from utils.multi_group_svd import recover_parameters
from utils.cnn512_reporting import model_digest
from flcore.trainmodel.models import Hyper_CNN_512


def args(**kw):
    a=options(multi_proto_mode='none',multi_proto_lambda=0.,multi_proto_loss='mse',device='cpu',
        multi_svd_diagnostics=0,multi_svd_init_rank_rate=.9,local_epochs=2,batch_size=2,**kw)
    a.offline_loop_round=5;a.initial_sha256='a'*64;a.data_fingerprint='b'*64
    a.multi_svd_spectral_mode='none';a.multi_svd_five_diagnostics=0
    a.multi_svd_avg_train_samples=[2+i%3 for i in range(a.num_clients)]
    return a


class ReplayTests(unittest.TestCase):
    def setUp(self):torch.set_num_threads(2);torch.manual_seed(52)

    def test_pair_uses_three_svds_raw_exact_shared_prefix_buffers_and_optimizer(self):
        a=args();mother=Hyper_CNN_512(n_kernels=2,ratio_LR=1.);mother.register_buffer('sentinel',torch.tensor([3.]))
        before=model_digest(mother);rng=old.rng_fingerprint(old.rng_snapshot(mother))
        native=torch.linalg.svd
        with redirect_stdout(io.StringIO()),patch.object(torch.linalg,'svd',wraps=native) as called:
            raw,repaired,rows=d.paired_download(mother,.9,a,0)
        self.assertEqual(called.call_count,3);self.assertEqual(len(rows),3)
        normal=copy.deepcopy(mother)
        with d.protected_rng([mother]),redirect_stdout(io.StringIO()):normal.decom_larger_model(.9);normal.ratio_LR=.9
        for name,value in raw.state_dict().items():self.assertTrue(torch.equal(value,normal.state_dict()[name]),name)
        self.assertTrue(torch.equal(raw.sentinel,repaired.sentinel));self.assertIs(raw.base[7],raw.fc1)
        self.assertIs(repaired.head[1],repaired.fc3)
        opt=torch.optim.SGD(repaired.parameters(),lr=.005)
        self.assertEqual([id(p) for p in repaired.parameters()],[id(p) for g in opt.param_groups for p in g['params']])
        self.assertEqual(before,model_digest(mother));self.assertEqual(rng,old.rng_fingerprint(old.rng_snapshot(mother)))

    def test_epoch_observation_isolated_from_original_training_and_exception_cleanup(self):
        from flcore.clients.clientTargetProj import clientTargetProj
        a=args();mother=Hyper_CNN_512(n_kernels=2,ratio_LR=1.)
        data=[(torch.randn(3,32,32),torch.tensor(i%2)) for i in range(3)]
        with redirect_stdout(io.StringIO()):initial,_,_=d.paired_download(mother,.9,a,0,False)
        state=old.rng_snapshot(initial);results=[];states=[];original=clientTargetProj._objective;clip=torch.nn.utils.clip_grad_norm_
        with tempfile.TemporaryDirectory(dir='E:/' if Path('E:/').exists() else None) as folder,redirect_stdout(io.StringIO()):
            for collect in (False,True):
                model=copy.deepcopy(initial);old.restore_rng(state)
                result,c=d.train_branch(model,mother,a,0,data,data,folder,collect=collect)
                self.assertEqual(result['training']['train_batches'],4);self.assertEqual(result['training']['processed_train_samples'],6)
                self.assertIsNone(c.avg_proto_local);self.assertEqual(sum(r['optimizer_steps'] for r in result['epochs']),4 if collect else 0)
                if collect:
                    self.assertEqual({r['epoch'] for r in result['endpoints']},{0,1,2})
                    self.assertTrue(all(0<=r['clip_trigger_fraction']<=1 for r in result['epochs']))
                results.append(result);states.append(copy.deepcopy(model.state_dict()))
            self.assertEqual(results[0]['batches'],results[1]['batches'])
            self.assertEqual(results[0]['final_rng_sha256'],results[1]['final_rng_sha256'])
            for name,value in states[0].items():self.assertTrue(torch.equal(value,states[1][name]),name)
            self.assertIs(clientTargetProj._objective,original);self.assertIs(torch.nn.utils.clip_grad_norm_,clip)
            model=copy.deepcopy(initial)
            c=old.local_client(model,mother,a,0,None,None,data,folder)
            observer=d.ReplayObserver(c,data,data)
            with self.assertRaisesRegex(RuntimeError,'injected'):
                with observer:raise RuntimeError('injected')
            self.assertNotIn('step',c.optimizer_set[0].__dict__)
            self.assertIs(clientTargetProj._objective,original);self.assertIs(torch.nn.utils.clip_grad_norm_,clip)

    def test_original_ordered_avg_unequal_samples_and_r7_ordinary(self):
        from utils.group_projection import aggregate_group_updates
        a=args();mother=Hyper_CNN_512(n_kernels=2,ratio_LR=1.);values={n:p.detach().clone() for n,p in mother.named_parameters()}
        uploads={i:{n:p+i*.001 for n,p in values.items()} for i in range(3)};samples={0:1,1:2,2:6};order=[2,0,1]
        factory=lambda:iter((i,samples[i]/9,uploads[i]) for i in order)
        expected=aggregate_group_updates(values,factory,{i:n/9 for i,n in samples.items()},[0,1,2],'group_avg_only')[0]
        combined,receipt=d.aggregate_once(mother,uploads,samples,order)
        for name,p in combined.named_parameters():self.assertTrue(torch.equal(p,expected[name]),name)
        self.assertEqual(receipt['order'],order)
        with patch.object(d.spectral,'all_tail_spectrum',side_effect=AssertionError('R7 must never repair')),redirect_stdout(io.StringIO()):
            local,other,records=d.paired_download(combined,.9,a,0,False)
        self.assertIsNone(other);self.assertEqual(records,[])
        self.assertEqual(local.fc1.rank,90)

    def test_weighted_scopes_comparisons_and_fixed_epoch_trends(self):
        rows=[]
        for index,branch in enumerate(d.BRANCHES):
            phases=['R6_local']+([] if branch=='L_raw' else ['R7_pre_ordinary'])
            for phase in phases:
                for epoch in (range(3) if phase=='R6_local' else [0]):
                    for cid in range(3):
                        for split in ('train','test'):
                            rows.append(dict(branch=branch,phase=phase,epoch=epoch,client_id=cid,split=split,
                                samples=cid+1,correct=cid,ce=1+index/10+(.1*epoch if split=='test' else -.1*epoch)))
        metrics=d.summarize_endpoints(rows,3,[0,2],0)
        allrow=next(r for r in metrics if r['evaluation_scope']=='all');self.assertEqual(allrow['accuracy'],.5)
        fixed=next(r for r in metrics if r['evaluation_scope']=='fixed_large');self.assertEqual(fixed['accuracy'],.5)
        pairs=d.comparisons(metrics,2);self.assertEqual(len(pairs),54)
        trends=d.fitting_trends(metrics,2);self.assertTrue(all(r['train_CE_down_test_CE_up'] for r in trends))
        with self.assertRaises(ValueError):d.summarize_endpoints(rows+[rows[0]],3,[0,2],0)

    def test_source_protocol_validation_rejects_wrong_round_capacity_and_training(self):
        a=args();a.global_rounds=4;a.local_epochs=5;a.batch_size=16;a.local_learning_rate=.005;a.local_grad_clip=10.;a.regular_lamda=.001
        a.multi_svd_aggregation='full_avg';a.multi_svd_init='low_rank_recovered';a.multi_svd_init_rank_rate=.9
        def manifest(a):
            return dict(effective=copy.deepcopy(vars(a)),algorithm='FedMultiGroupProj',actual_train_count=5,
                server_model_count=1,client_count=20,aggregation_mode='group_avg_only',
                svd_spectral=dict(policy=d.spectral.policy(a)),environment={k:False for k in d.BACKEND},
                optimizer=dict(name='SGD',lr=.005,momentum=0.,weight_decay=0.,gradient_norm_cap=10.))
        m=manifest(a);self.assertEqual(d.validate_manifest(m,'h',5).local_epochs,5)
        for key,value in [('local_epochs',4),('local_grad_clip',0),('multi_proto_lambda',1),('target_head_mode','local'),
            ('multi_svd_spectral_mode','all_tail'),('seed',-1),('join_ratio',.5),('multi_fc1_tail_scale_strength',1)]:
            bad=copy.deepcopy(m);bad['effective'][key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):d.validate_manifest(bad,'h',5)
        bad=copy.deepcopy(m);bad['actual_train_count']=4
        with self.assertRaises(ValueError):d.validate_manifest(bad,'h',5)
        a.multi_svd_effective_ratios=[.9];a.multi_svd_effective_assignment=[0]*20
        self.assertEqual(d.client_rates(d.validate_manifest(manifest(a),'l',5)),[.9]*20)
        with self.assertRaises(ValueError):d.validate_manifest(manifest(a),'h',5)

    def test_complete_command_parser_and_fixed_protocol(self):
        from svd_r6_replay import parser
        lines=[line.strip()[1:-1] for line in (ROOT/'system/svd_r6_replay_commands.sh').read_text().splitlines() if line.strip().startswith('"')]
        self.assertEqual(len(lines),1);words=shlex.split(lines[0]);self.assertEqual(words[3],'-did');self.assertNotIn('$',lines[0])
        cli=parser().parse_args(words[3:]);self.assertEqual(cli.device_id,0);self.assertEqual(cli.source_round,5)
        self.assertEqual(cli.focus_client_ids,'0,5,10,15');self.assertEqual(cli.data_subdir,'pat_20')
        self.assertTrue(cli.h_run.endswith('/H'));self.assertTrue(cli.hplus_run.endswith('/Hplus'));self.assertTrue(cli.l_run.endswith('/L'))


if __name__=='__main__':unittest.main()

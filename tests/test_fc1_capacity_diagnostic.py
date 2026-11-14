"""Offline capacity branches, unchanged SGD, probe isolation and rollback scope."""
import copy
from contextlib import redirect_stdout
import io
from pathlib import Path
import shlex
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
import torch.nn.functional as F

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'system'))
from fc1_capacity_diagnostic import parser
from utils import fc1_capacity_diagnostic as d,fc1_tail_diagnostic as old,multi_svd_spectral as sp,single_avg_proto as proto
from utils import multi_group_svd as svd
from flcore.trainmodel.models import Hyper_CNN_512,FactorizedLinear
from test_single_avg_proto import options as base_options


def options(**kw):
    settings=dict(multi_proto_mode='none',multi_proto_lambda=0.,multi_svd_init_rank_rate=.9,
        multi_svd_spectral_mode='next_tier',multi_svd_spectral_layers='fc1',multi_svd_spectral_max_gain=1.5,
        local_epochs=5,batch_size=16,local_learning_rate=.005,regular_lamda=.001,local_grad_clip=10.,num_clients=20,device='cpu',multi_svd_diagnostics=0)
    settings.update(kw);args=base_options(**settings)
    args.offline_loop_round=101;args.initial_sha256='a'*64;args.data_fingerprint='b'*64
    return args


def model_and_evidence():
    args=options();mother=Hyper_CNN_512(n_kernels=2,num_classes=100,ratio_LR=1.)
    evidence,receipt=d.common_fc1(mother,args,[15,25,90]);return mother,args,evidence


class CapacityTests(unittest.TestCase):
    def setUp(self):torch.set_num_threads(2);torch.random.default_generator.manual_seed(93)

    def test_shared_once_svd_prefix_other_layers_actual_ranks_optimizer_and_rng(self):
        args=options();mother=Hyper_CNN_512(n_kernels=2,num_classes=100,ratio_LR=1.)
        mother.register_buffer('test_buffer',torch.tensor([3.]))
        before=old.rng_fingerprint(old.rng_snapshot(mother));native=torch.linalg.svd
        with patch.object(torch.linalg,'svd',wraps=native) as calls:evidence,receipt=d.common_fc1(mother,args,[15,25,90])
        self.assertEqual(calls.call_count,1);self.assertEqual(receipt['k'],15);self.assertEqual(receipt['b'],25)
        for rate in (.9,.25):
            with patch.object(torch.linalg,'svd',wraps=native) as calls:template=d.client_template(mother,rate)
            self.assertEqual(calls.call_count,2)
            branches=[d.rank_branch(template,evidence,r) for r in [15,25,90]]
            for model in branches:
                d.verify_prefix_and_other_state(branches[0],model)
                self.assertEqual(model.ratio_LR,rate);self.assertIs(model.base[7],model.fc1)
                self.assertIs(model.head[1],model.fc3)
                self.assertEqual((model.conv2.rank,model.fc2.rank),(round(10*rate),round(512*rate)))
                optimizer=torch.optim.SGD(model.parameters(),lr=.005)
                self.assertEqual([id(p) for p in model.parameters()],[id(p) for g in optimizer.param_groups for p in g['params']])
        self.assertEqual(before,old.rng_fingerprint(old.rng_snapshot(mother)))

    def test_direct_standard_download_matches_native_fc1_rank_and_source_spectrum(self):
        args=options();mother=svd.initial_model(args)
        before=old.rng_fingerprint(old.rng_snapshot(mother));evidence,_=d.common_fc1(mother,args,[120,200,720])
        for cid,rate,rank in [(0,.9,720),(18,.25,200)]:
            model=d.rank_branch(d.client_template(mother,rate),evidence,rank);normal=copy.deepcopy(mother)
            with old.read_only(mother),redirect_stdout(io.StringIO()):sp.decompose(normal,rate,args,cid,0,101)
            normal.ratio_LR=rate
            for name,value in model.state_dict().items():self.assertTrue(torch.equal(value,normal.state_dict()[name]),name)
        self.assertEqual(before,old.rng_fingerprint(old.rng_snapshot(mother)))

    def test_diagnostics_and_intermediate_endpoints_match_original_trainer_parameters_batches_rng(self):
        mother,args,evidence=model_and_evidence();args.batch_size=2
        data=[(torch.randn(3,32,32),torch.tensor(i%3)) for i in range(3)]
        template=d.client_template(mother,.9);initial=d.rank_branch(template,evidence,25)
        ranges=d.slot_ranges(sp.boundary_ranks(mother,args)['fc1']);start=old.rng_snapshot(initial)
        results=[];states=[]
        with tempfile.TemporaryDirectory(dir='E:/' if Path('E:/').exists() else None) as temp,redirect_stdout(io.StringIO()):
            for enabled in (True,False):
                model=copy.deepcopy(initial);old.restore_rng(start)
                result,client=d.train_branch(model,mother,args,0,data,data,temp,evidence,ranges,enabled)
                self.assertEqual(result['training']['train_batches'],10);self.assertEqual(result['training']['processed_train_samples'],15)
                self.assertEqual({r['epoch'] for r in result['endpoints']},{0,1,5});self.assertIsNone(client.avg_proto_local)
                results.append(result);states.append(copy.deepcopy(model.state_dict()))
            model=copy.deepcopy(initial);old.restore_rng(start)
            client=old.local_client(model,mother,args,0,None,None,data,temp);client.svd_spectral_latest=[]
            loader=old.EpochLoader(data,args,0,101);client.load_train_data=lambda:loader;proto.train(client)
            for state in states:
                for name,value in model.state_dict().items():self.assertTrue(torch.equal(value,state[name]),name)
            self.assertEqual(loader.batches,results[0]['batches']);self.assertEqual(results[0]['batches'],results[1]['batches'])
            self.assertEqual(results[0]['final_rng_sha256'],results[1]['final_rng_sha256'])
            self.assertEqual(results[0]['final_rng_sha256'],old.rng_fingerprint(old.rng_snapshot(model)))
            for key in ('ce_loss','regularization_loss','total_loss','raw_mse','pre_clip_grad_norm_mean_per_step','clip_trigger_fraction'):
                self.assertEqual(results[0]['training'][key],client.train_stats[0][key],key)
        self.assertEqual(len(results[0]['updates']),12);self.assertEqual(len(results[0]['first_step']),3)
        self.assertEqual(results[1]['updates'],[]);self.assertTrue(all(r['epoch'] in (0,1,5) for r in results[1]['endpoints']))

    def test_first_CE_probe_segment_gradients_actual_total_clip_and_exception_cleanup(self):
        class Mini(torch.nn.Module):
            def __init__(self):super().__init__();self.fc1=FactorizedLinear(6,8,5/6).double()
            def forward(self,x):return self.fc1(x)
        model=Mini();x=torch.randn(3,6,dtype=torch.float64);y=torch.tensor([0,2,5]);ranges=[(0,2),(2,4),(4,5)]
        evidence=dict(P=torch.eye(8,dtype=torch.float64)[:,:6],Qh=torch.eye(6,dtype=torch.float64))
        client=SimpleNamespace(model_set=[model],optimizer_set=[torch.optim.SGD(model.parameters(),lr=.005)],
            loss=torch.nn.CrossEntropyLoss(),args=SimpleNamespace(local_grad_clip=.1),local_epochs=5)
        w=(model.fc1.weight_u@model.fc1.weight_v).detach().requires_grad_()
        G=torch.autograd.grad(F.cross_entropy(F.linear(x,w,model.fc1.bias.detach()),y),w)[0]
        u=model.fc1.weight_u.detach().clone();v=model.fc1.weight_v.detach().clone()
        observer=d.Observer(client,evidence,ranges,[],[]);optimizer=client.optimizer_set[0];original_step=optimizer.step
        with observer:
            optimizer.zero_grad();ce=client.loss(model(x),y)
            self.assertTrue(all(p.grad is None for p in model.parameters()))
            (ce+.001*model.fc1.frobenius_loss()).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),.1);optimizer.step()
        for (lo,hi),row,actual in zip(ranges,observer.signal,observer.gradients):
            self.assertAlmostEqual(row['CE_U_gradient_frobenius'],float((G@v[lo:hi].T).norm()),places=12)
            self.assertAlmostEqual(row['CE_V_gradient_frobenius'],float((u[:,lo:hi].T@G).norm()),places=12)
            self.assertAlmostEqual(actual['U_grad_norm_post_clip'],actual['U_grad_norm_pre_clip']*actual['clip_coefficient'],places=12)
        self.assertEqual(optimizer.step,original_step);self.assertNotIn('step',optimizer.__dict__)
        try:
            with d.Observer(client,evidence,ranges,[],[]):raise RuntimeError('injected')
        except RuntimeError:pass
        self.assertFalse(model.fc1._forward_hooks);self.assertFalse(model.fc1._forward_pre_hooks)
        for p in model.parameters():self.assertFalse(p._backward_hooks)
        self.assertIsInstance(client.loss,torch.nn.CrossEntropyLoss)

    def test_rollback_only_original_tail_both_factors_and_keeps_trained_state(self):
        mother,args,evidence=model_and_evidence();model=d.rank_branch(d.client_template(mother,.25),evidence,90)
        initial=d.factor_state(model)
        with torch.no_grad():
            for p in model.parameters():p.add_(.02)
        before=copy.deepcopy(model.state_dict());rng=old.rng_fingerprint(old.rng_snapshot(model))
        back=d.rollback_tail(model,initial,25)
        self.assertTrue(torch.equal(back.fc1.weight_u[:,25:],initial['weight_u'][:,25:]))
        self.assertTrue(torch.equal(back.fc1.weight_v[25:,:],initial['weight_v'][25:,:]))
        self.assertTrue(torch.equal(back.fc1.weight_u[:,:25],model.fc1.weight_u[:,:25]))
        self.assertTrue(torch.equal(back.fc1.weight_v[:25,:],model.fc1.weight_v[:25,:]))
        for name,p in model.named_parameters():
            if name not in ('fc1.weight_u','fc1.weight_v'):self.assertTrue(torch.equal(p,dict(back.named_parameters())[name]),name)
        for name,value in before.items():self.assertTrue(torch.equal(value,model.state_dict()[name]),name)
        self.assertEqual(rng,old.rng_fingerprint(old.rng_snapshot(model)))
        with self.assertRaises(ValueError):d.rollback_tail(model,initial,90)

    def test_rank_delta_and_capacity_summary_use_same_epoch_and_test_sample_weights(self):
        rows=[]
        for cid,n in [(0,1),(10,9)]:
            for rank in (120,200,720):
                for epoch in (0,1,5):
                    correct=min(n,rank//200+int(epoch==5))
                    rows.append(dict(client_id=cid,rank_rate=.9,fc1_rank=rank,epoch=epoch,split='test',samples=n,correct=correct,accuracy=correct/n,ce=rank/1000))
        pairs=d.paired_comparisons(rows,[0,10],dict(k=120,b=200,actual_ranks=[120,200,720]))
        self.assertEqual(len(pairs),4);self.assertEqual(pairs[0]['epoch_0_higher_minus_lower_test_accuracy'],1.)
        summary=d.capacity_summary(rows);point=next(r for r in summary if r['fc1_rank']==200 and r['epoch']==0)
        self.assertEqual(point['accuracy'],.2);self.assertEqual(point['samples'],10)

    def test_manifest_protocol_rejects_other_sources_and_parser_missing_inputs(self):
        args=options();manifest=dict(effective=vars(args),algorithm='FedMultiGroupProj',server_model_count=1,client_count=20,
            actual_train_count=101,aggregation_mode='group_avg_only',optimizer=dict(name='SGD',lr=.005,momentum=0.,weight_decay=0.,gradient_norm_cap=10.),
            svd_spectral=dict(policy=sp.policy(args)))
        self.assertEqual(d.validate_manifest(manifest,101).num_clients,20)
        for key,value in [('multi_proto_mode','global_last_epoch'),('multi_svd_init_rank_rate',.15),
            ('multi_svd_spectral_mode','single_point'),('multi_svd_spectral_layers','conv2,fc1,fc2'),
            ('multi_svd_spectral_max_gain',2.),('multi_svd_aggregation','rank_partition'),('multi_fc1_tail_scale_strength',1.)]:
            bad=copy.deepcopy(manifest);bad['effective'][key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):d.validate_manifest(bad,101)
        with self.assertRaises(ValueError):d.validate_manifest(manifest,2)
        cli=parser().parse_args(['--server-file','missing_capacity_source.pt','--output-dir','unused'])
        self.assertEqual(cli.client_ids,'0,10,18');self.assertEqual(cli.fc1_ranks,'120,200,720');self.assertEqual(cli.source_round,101)
        with self.assertRaises(FileNotFoundError):d.resolve_source(cli)
        for bad in ('','120,120','1.5','a'):
            with self.assertRaises(ValueError):d.integers(bad,'ranks')

    def test_complete_command_source_rank_ids_and_independent_output(self):
        lines=[x.strip()[1:-1] for x in (ROOT/'system/fc1_capacity_diagnostic_commands.sh').read_text().splitlines() if x.strip().startswith('"')]
        self.assertEqual(len(lines),1);words=shlex.split(lines[0]);get=lambda key:words[words.index(key)+1]
        self.assertEqual(words[3],'-did');self.assertEqual(get('--source-round'),'101');self.assertEqual(get('--client-ids'),'0,10,18')
        self.assertEqual(get('--fc1-ranks'),'120,200,720');self.assertIn('/B_fc1',get('--source-run'))
        self.assertTrue(get('--output-dir').startswith('fc1_capacity_diagnostic_runs/'));self.assertNotIn('$',lines[0])


if __name__=='__main__':unittest.main()

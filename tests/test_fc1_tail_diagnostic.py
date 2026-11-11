"""Offline-only factor gauges, CE signals, read-only observations and cleanup."""
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
from fc1_tail_diagnostic import parser
from utils import fc1_tail_diagnostic as d
from utils import multi_group_svd as svd
from flcore.trainmodel.models import Hyper_CNN_512,FactorizedLinear
from common.data_loading import make_data_loader
from utils.single_avg_proto import loader_seed


class Mini(torch.nn.Module):
    def __init__(self):
        super().__init__();self.fc1=FactorizedLinear(6,8,5/6).double()
    def forward(self,x):return self.fc1(x)


def example():
    model=Mini();p=torch.eye(8,dtype=torch.float64)[:,:6];qh=torch.eye(6,dtype=torch.float64)
    singular=torch.tensor([4.,1.,1e-8,0.,0.,0.],dtype=torch.float64);r=model.fc1.rank
    with torch.no_grad():
        model.fc1.weight_u.copy_(p[:,:r]@torch.diag(singular[:r].sqrt()))
        model.fc1.weight_v.copy_(torch.diag(singular[:r].sqrt())@qh[:r,:]);model.fc1.bias.zero_()
    return model,dict(P=p,S=singular,Qh=qh,actual_rank=r)


class TailTests(unittest.TestCase):
    def setUp(self):torch.set_num_threads(2);torch.random.default_generator.manual_seed(29)
    def test_gauge_prefix_other_state_zero_singular_and_actual_product(self):
        a,evidence=example();before=copy.deepcopy(a.state_dict());rng=d.rng_fingerprint(d.rng_snapshot(a))
        b,info=d.rescale_tail(a,evidence,2)
        self.assertEqual(info['floor_factor_scale'],1.)
        torch.testing.assert_close(d.full_matrix(a.fc1),d.full_matrix(b.fc1),atol=1e-20,rtol=1e-15)
        self.assertTrue(torch.equal(a.fc1.weight_u[:,:2],b.fc1.weight_u[:,:2]))
        self.assertTrue(torch.equal(a.fc1.weight_v[:2],b.fc1.weight_v[:2]))
        self.assertTrue(torch.equal(a.fc1.bias,b.fc1.bias))
        self.assertEqual(float(b.fc1.weight_u[:,3].norm()),1.)
        self.assertEqual(float(b.fc1.weight_v[3].norm()),0.)
        for n,v in before.items():self.assertTrue(torch.equal(v,a.state_dict()[n]))
        self.assertEqual(rng,d.rng_fingerprint(d.rng_snapshot(a)))
        evidence['S'].zero_();zero,_=d.rescale_tail(a,evidence,2)
        self.assertTrue(torch.isfinite(zero.fc1.weight_u).all());self.assertTrue(torch.isfinite(zero.fc1.weight_v).all())
        for m in (0,5):
            with self.assertRaises(ValueError):d.rescale_tail(a,evidence,m)

    def test_one_actual_download_svd_matches_legacy_and_integer_ranks(self):
        mother=Hyper_CNN_512(n_kernels=2,num_classes=100,ratio_LR=1.)
        initial=d.rng_snapshot(mother)
        with redirect_stdout(io.StringIO()):a,e=d.download_once(mother,.35)
        after=d.rng_fingerprint(d.rng_snapshot(a));d.restore_rng(initial)
        b=copy.deepcopy(mother)
        with redirect_stdout(io.StringIO()):b.decom_larger_model(.35)
        b.ratio_LR=.35
        for n,v in a.state_dict().items():self.assertTrue(torch.equal(v,b.state_dict()[n]),n)
        self.assertEqual(after,d.rng_fingerprint(d.rng_snapshot(b)))
        self.assertEqual(e['svd_calls_for_fc1'],1)
        wide=Hyper_CNN_512(n_kernels=16,ratio_LR=1.)
        before=d.rng_fingerprint(d.rng_snapshot(wide));self.assertEqual(d.factor_rank(wide,.15),120)
        self.assertEqual(d.factor_rank(wide,.35),280);self.assertEqual(d.factor_rank(wide,.90),720)
        self.assertEqual(before,d.rng_fingerprint(d.rng_snapshot(wide)))

    def test_CE_signal_zero_balanced_tail_and_unbalanced_V_gradient_no_F_proto(self):
        a,evidence=example();b,_=d.rescale_tail(a,evidence,2)
        x=torch.tensor([[0.,0.,0.,1.,0.,0.]],dtype=torch.float64);y=torch.tensor([0])
        for model in (a,b):
            model.zero_grad();F.cross_entropy(model(x),y).backward()
        self.assertEqual(float(a.fc1.weight_u.grad[:,3:].norm()),0.)
        self.assertEqual(float(a.fc1.weight_v.grad[3:].norm()),0.)
        self.assertGreater(float(b.fc1.weight_v.grad[3:].norm()),0.)
        client=SimpleNamespace(model_set=[b],optimizer_set=[torch.optim.SGD(b.parameters(),lr=.005)],loss=torch.nn.CrossEntropyLoss(),
            args=SimpleNamespace(local_grad_clip=10.,local_learning_rate=.005),local_epochs=1)
        b.zero_grad();initial=d.full_matrix(b.fc1).clone();observer=d.Observer(client,evidence,2)
        with observer:
            client.optimizer_set[0].zero_grad();loss=client.loss(b(x),y)
            self.assertTrue(all(p.grad is None for p in b.parameters()))
            loss.backward();torch.nn.utils.clip_grad_norm_(b.parameters(),10);client.optimizer_set[0].step()
        self.assertGreater(observer.signal[0]['CE_G_Q_download_tail_frobenius'],0.)
        self.assertGreater(observer.rows[0]['fixed_slot_tail_product_change_frobenius'],0.)
        self.assertFalse(torch.equal(initial,d.full_matrix(b.fc1)))

    def test_gradient_probe_matches_direct_restored_weight_gradient_and_hook_cleanup(self):
        a,evidence=example();x=torch.randn(3,6,dtype=torch.float64);y=torch.tensor([0,2,5])
        w=d.full_matrix(a.fc1).requires_grad_();loss=F.cross_entropy(F.linear(x,w,a.fc1.bias.detach()),y)
        G=torch.autograd.grad(loss,w)[0]
        client=SimpleNamespace(model_set=[a],optimizer_set=[torch.optim.SGD(a.parameters(),lr=.005)],loss=torch.nn.CrossEntropyLoss(),
            args=SimpleNamespace(local_grad_clip=.1,local_learning_rate=.005),local_epochs=1)
        optimizer=client.optimizer_set[0];original=optimizer.step
        observer=d.Observer(client,evidence,2)
        with observer:
            optimizer.zero_grad();ce=client.loss(a(x),y);ce.backward()
            torch.nn.utils.clip_grad_norm_(a.parameters(),.1);optimizer.step()
        self.assertAlmostEqual(observer.signal[0]['CE_restored_weight_gradient_frobenius'],float(G.norm()),places=13)
        self.assertAlmostEqual(observer.signal[0]['CE_G_Q_download_tail_frobenius'],float((G@evidence['Qh'][2:5].T).norm()),places=13)
        row=observer.gradients[0]
        self.assertLess(row['clip_coefficient'],1.)
        self.assertAlmostEqual(row['V_tail_grad_norm_post_clip'],row['V_tail_grad_norm_pre_clip']*row['clip_coefficient'],places=12)
        self.assertEqual(optimizer.step,original)
        self.assertNotIn('step',optimizer.__dict__)
        try:
            with d.Observer(client,evidence,2):raise RuntimeError('injected')
        except RuntimeError:pass
        self.assertEqual(optimizer.step,original);self.assertIsInstance(client.loss,torch.nn.CrossEntropyLoss)
        self.assertFalse(a.fc1._forward_hooks);self.assertFalse(a.fc1._forward_pre_hooks)
        for p in a.parameters():self.assertFalse(p._backward_hooks)

    def test_indexed_epoch_loader_identical_original_batches_and_rng(self):
        args=SimpleNamespace(seed=37,batch_size=3);data=list(zip(torch.arange(20).float().reshape(10,2),torch.arange(10)))
        first=d.EpochLoader(data,args,7,101)
        old=make_data_loader(data,3,shuffle=True,drop_last=False,num_workers=0,
            generator=torch.Generator().manual_seed(loader_seed(args,7,101)))
        before=torch.get_rng_state()
        for _ in range(5):
            got=list(first);expected=list(old)
            for (x,y),(xx,yy) in zip(got,expected):self.assertTrue(torch.equal(x,xx));self.assertTrue(torch.equal(y,yy))
        self.assertTrue(torch.equal(before,torch.get_rng_state()));self.assertEqual(len(first.batches),20)
        for epoch in range(1,6):self.assertEqual(sorted(i for row in first.batches if row['epoch']==epoch for i in row['indices']),list(range(10)))

    def test_endpoint_truncates_only_fc1_and_is_read_only(self):
        mother=Hyper_CNN_512(n_kernels=2,num_classes=100,ratio_LR=1.)
        with redirect_stdout(io.StringIO()):model,_=d.download_once(mother,.35)
        model.train();model.fc3.eval();model.conv1.bias.grad=torch.ones_like(model.conv1.bias)
        data=[(torch.randn(3,32,32),torch.tensor(i)) for i in range(2)]
        before=d.rng_fingerprint(d.rng_snapshot(model));state=copy.deepcopy(model.state_dict());modes=[m.training for m in model.modules()]
        metrics,logits=d.evaluate(model,data,1,'cpu');truncated=d.truncated_fc1(model,15)
        self.assertEqual(metrics['samples'],2);self.assertEqual(logits.shape,(2,100))
        self.assertIsInstance(truncated.fc1,torch.nn.Linear)
        for n,v in model.named_parameters():
            if not n.startswith('fc1.'):self.assertTrue(torch.equal(v,dict(truncated.named_parameters())[n]))
        for n,v in state.items():self.assertTrue(torch.equal(v,model.state_dict()[n]))
        self.assertEqual(modes,[m.training for m in model.modules()]);self.assertTrue(torch.equal(model.conv1.bias.grad,torch.ones_like(model.conv1.bias)))
        self.assertEqual(before,d.rng_fingerprint(d.rng_snapshot(model)))
        energy=torch.linalg.svdvals(truncated.fc1.weight.double()).square()
        self.assertLess(float(energy[15:].sum()/energy.sum()),1e-12)

    def test_missing_input_and_parser_defaults_and_weighted_summary(self):
        cli=parser().parse_args(['--server-file','missing_fc1_tail_input.pt','--output-dir','unused'])
        self.assertEqual(cli.source_round,101);self.assertEqual(cli.client_rates,'0.35,0.90');self.assertFalse(cli.save_models)
        with self.assertRaises(FileNotFoundError):d.resolve_source(cli)
        pairs=[]
        for cid,n,a,b in [(1,1,1,0),(2,9,0,9)]:
            pairs.append(dict(client_id=cid,rank_rate=.35,test_samples=n,A_test_correct=a,B_test_correct=b,A_test_CE=1.,B_test_CE=2.,
                A_fc1_truncated_test_correct=a,B_fc1_truncated_test_correct=b,A_epoch_final_tail_product_change=1.,B_epoch_final_tail_product_change=2.))
        row=d.capacity_summary(pairs)[0];self.assertEqual(row['A_test_accuracy'],.1);self.assertEqual(row['B_test_accuracy'],.9)

    def test_one_complete_command_defaults_and_no_auto_execution(self):
        text=(ROOT/'system/fc1_tail_diagnostic_commands.sh').read_text()
        lines=[line.strip()[1:-1] for line in text.splitlines() if line.strip().startswith('"')]
        self.assertEqual(len(lines),1);words=shlex.split(lines[0]);self.assertEqual(words[3:5],['-did','0'])
        args=parser().parse_args(words[3:])
        self.assertEqual(args.source_round,101);self.assertEqual(args.epochs,5)
        self.assertEqual(args.batch_size,16);self.assertEqual(args.lr,.005);self.assertEqual(args.f_lambda,.001)
        self.assertEqual(args.grad_clip,10.);self.assertEqual(args.proto_lambda,1.)
        self.assertFalse(args.save_models);self.assertIn('FA02',args.source_run)
        self.assertNotIn('${',text);self.assertNotIn('for ',text)

    def test_run_root_unique_final_export_resolution_no_other_run_fallback(self):
        with tempfile.TemporaryDirectory(dir='E:/' if Path('E:/').is_dir() else None,prefix='fc1_tail_resolve_') as folder:
            root=Path(folder);export=root/'final_models'/'one'/'runs'/'r101';export.mkdir(parents=True)
            for name in ('Server_model_0.pt','manifest.json','Global_class_sample_latest_generated.packet'):(export/name).touch()
            args=parser().parse_args(['--source-run',str(root),'--output-dir',str(root/'unused')])
            self.assertEqual(d.resolve_source(args),export/'Server_model_0.pt')
            (root/'final_models'/'duplicate').mkdir();(root/'final_models'/'duplicate'/'Server_model_0.pt').touch()
            with self.assertRaisesRegex(FileNotFoundError,'exactly ONE'):d.resolve_source(args)


if __name__=='__main__':unittest.main()

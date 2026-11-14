"""Offline four-way construction, known attenuation, native factors and training."""
import copy
from contextlib import redirect_stdout
import io
from pathlib import Path
import shlex
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'system'))
from whole_network_spectrum_diagnostic import parser
from utils import whole_network_spectrum_diagnostic as d,fc1_tail_diagnostic as old,single_avg_proto as train
from utils.multi_rank_partition import actual_matrix
from flcore.trainmodel.models import Hyper_CNN_512


class FourWayTests(unittest.TestCase):
    def setUp(self):torch.set_num_threads(2);torch.random.default_generator.manual_seed(144)
    def cli(self):return parser().parse_args(['--device','cpu','--output-dir','unused_offline_four'])

    def test_gamma_actual_integer_bins_duplicates_and_invalid(self):
        g=d.attenuation([120,200,280,720],800)
        self.assertTrue(torch.equal(g[:120],torch.ones(120,dtype=torch.float64)))
        for lo,hi,value in [(120,200,.75),(200,280,.5),(280,720,.25),(720,800,0.)]:
            self.assertTrue(torch.equal(g[lo:hi],torch.full((hi-lo,),value,dtype=torch.float64)))
        self.assertEqual(d.attenuation([2,2,4,9],10).tolist(),[1.,1.,.5,.5,.25,.25,.25,.25,.25,0.])
        for ranks in ([],[0,2],[3,11],[1.5,3]):
            with self.assertRaises(ValueError):d.attenuation(ranks,10)

    def test_build_uses_six_original_SVDs_equal_full_Avg_and_no_training(self):
        native=torch.linalg.svd;aggregate=d.equal_full_avg;seen={}
        def capture(mother,uploads):
            result,receipt=aggregate(mother,uploads)
            for name,p in result.named_parameters():
                expected=torch.zeros_like(p)
                for upload in uploads:expected.add_(upload[name]*.25)
                self.assertTrue(torch.equal(p,expected),name)
            seen['uploads']=len(uploads);return result,receipt
        before=old.rng_fingerprint(old.rng_snapshot(Hyper_CNN_512(n_kernels=2,ratio_LR=1.)))
        with patch.object(torch.linalg,'svd',wraps=native) as count,patch.object(d,'equal_full_avg',side_effect=capture),\
            patch.object(torch.optim.SGD,'step',side_effect=AssertionError('No SGD before grouping')),redirect_stdout(io.StringIO()):
            models,mother,ledger,tables=d.build_experiments(0,[.15,.25,.35,.9],n_kernels=2)
        self.assertEqual(count.call_count,6);self.assertEqual(seen['uploads'],4)
        self.assertTrue(all(row['passed'] for row in ledger['construction_checks']))
        self.assertEqual(ledger['actual_layer_ranks']['fc1'],[15,25,35,90])
        self.assertEqual(ledger['actual_layer_ranks']['conv2'],[2,2,4,9])
        self.assertEqual(ledger['actual_layer_ranks']['fc2'],[77,128,179,461])
        self.assertEqual(before,old.rng_fingerprint(old.rng_snapshot(mother)))
        self.assertEqual({r['aggregation_weight'] for r in ledger['simulation_aggregation']['upload_weights']},{.25})
        self.assertEqual({r['ideal_compensation_gain'] for r in tables['spectral_interval_energy'] if r['contributing_tiers']>0},{1.,4/3,2.,4.})
        self.assertEqual(len({sum(p.numel() for p in models[c].parameters()) for c in ('E1','E2','E4')}),1)
        self.assertEqual({r['layer'] for r in tables['E1_E2_spectral_gap']},set(d.LAYERS))
        self.assertTrue(all(r['E2_over_E1'] is None for r in tables['E1_E2_spectral_gap'] if not r['retained_in_large']))
        for name in d.LAYERS:
            # The paired download directions stay the same; only factor scales differ.
            a,b=d.layer_factors(getattr(models['E1'],name));u,v=d.layer_factors(getattr(models['E2'],name))
            sa=a.detach().norm(dim=0);sb=u.detach().norm(dim=0)
            self.assertTrue(torch.allclose(a/sa,u/sb,atol=1e-6,rtol=1e-5))
            self.assertTrue(torch.allclose(b/b.detach().norm(dim=1)[:,None],v/v.detach().norm(dim=1)[:,None],atol=1e-6,rtol=1e-5))

    def test_native_large_retains_original_factors_small_native_factors_common_parameters(self):
        with redirect_stdout(io.StringIO()):models,mother,ledger,_=d.build_experiments(7,[.15,.25,.35,.9],n_kernels=2)
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(7);large=Hyper_CNN_512(n_kernels=2,num_classes=100,ratio_LR=.9)
            torch.random.default_generator.manual_seed(7);small=Hyper_CNN_512(n_kernels=2,num_classes=100,ratio_LR=.15)
        for name,p in large.state_dict().items():self.assertTrue(torch.equal(p,models['E4'].state_dict()[name]),name)
        for name in d.LAYERS:
            for a,b in zip(d.layer_factors(getattr(small,name)),d.layer_factors(getattr(models['E3'],name))):self.assertTrue(torch.equal(a,b))
            for case in ('E1','E2','E4'):
                self.assertEqual(getattr(models[case],name).rank,getattr(large,name).rank)
            check=d.error_metrics(actual_matrix(getattr(models['E2'],name)),actual_matrix(getattr(models['E4'],name)),1e-5,1e-5,name)
            self.assertTrue(check['passed'])
        for case,model in models.items():
            for name,p in large.named_parameters():
                if id(p) not in {id(p) for n in d.LAYERS for p in d.layer_factors(getattr(large,n))}:
                    self.assertTrue(torch.equal(p,dict(model.named_parameters())[name]),(case,name))
        self.assertFalse(torch.equal(models['E2'].fc1.weight_u,models['E4'].fc1.weight_u))
        self.assertLess(sum(p.numel() for p in models['E3'].parameters()),sum(p.numel() for p in models['E4'].parameters()))
        self.assertTrue(torch.equal(models['E1'].fc1.bias,models['E4'].fc1.bias))

    def test_nondefault_tiers_seed_and_rounding_are_layer_specific(self):
        with redirect_stdout(io.StringIO()):
            a,_,ledger,_=d.build_experiments(3,[.2,.4,.8],n_kernels=2,diagnostics=False)
            b,_,other,_=d.build_experiments(4,[.2,.4,.8],n_kernels=2,diagnostics=False)
        self.assertEqual(ledger['actual_layer_ranks']['conv2'],[2,4,8])
        self.assertEqual(ledger['actual_layer_ranks']['fc1'],[20,40,80])
        self.assertEqual(ledger['actual_layer_ranks']['fc2'],[102,205,410])
        self.assertNotEqual(ledger['native_large_state_sha256'],other['native_large_state_sha256'])
        for name in d.LAYERS:
            for model in a.values():self.assertTrue(torch.isfinite(actual_matrix(getattr(model,name))).all())

    def test_training_original_trainer_and_endpoint_isolation_all_four_batch_RNG(self):
        cli=self.cli();cli.batch_size=2;args=d.configuration(cli)
        with redirect_stdout(io.StringIO()):models,mother,ledger,_=d.build_experiments(0,[.15,.25,.35,.9],n_kernels=2,diagnostics=False)
        args.initial_sha256=ledger['recovered_mother_sha256'];args.data_fingerprint='b'*64
        data=[(torch.randn(3,32,32),torch.tensor(i)) for i in range(3)]
        start=old.rng_snapshot(mother);records=[];initial=copy.deepcopy(models['E4'])
        with tempfile.TemporaryDirectory(dir='E:/' if Path('E:/').exists() else None) as folder,redirect_stdout(io.StringIO()):
            for case,model in models.items():
                level=args.multi_svd_effective_ratios.index(model.ratio_LR);a=copy.deepcopy(args);a.multi_svd_effective_assignment=[level]*20
                epoch0=[]
                for split in ('train','test'):
                    metric,_=old.evaluate(model,data,2,'cpu');epoch0.append(dict(epoch=0,split=split,**metric))
                old.restore_rng(start);r=d.train_case(model,mother,a,0,data,data,epoch0,folder)
                self.assertEqual(r['training']['train_batches'],10);self.assertEqual(r['training']['processed_train_samples'],15)
                self.assertEqual({p['epoch'] for p in r['endpoints']},{0,1,5});self.assertEqual(r['training']['raw_mse'],0.)
                records.append(r)
            old.restore_rng(start);a=copy.deepcopy(args);a.multi_svd_effective_assignment=[3]*20
            c=old.local_client(initial,mother,a,0,None,None,data,folder);loader=old.EpochLoader(data,a,0,0);c.load_train_data=lambda:loader
            train.train(c)
            self.assertEqual(records[-1]['final_model_sha256'],d.tensor_fingerprint(initial.state_dict()))
            self.assertEqual(records[-1]['final_rng_sha256'],old.rng_fingerprint(old.rng_snapshot(initial)))
            self.assertEqual(records[-1]['batches'],loader.batches)
            for key in ('ce_loss','regularization_loss','total_loss','clip_trigger_fraction'):
                self.assertEqual(records[-1]['training'][key],c.train_stats[0][key])
        self.assertEqual(len({r['batch_order_sha256'] for r in records}),1)
        self.assertEqual(len({r['initial_rng_sha256'] for r in records}),1)
        self.assertEqual(len({r['final_rng_sha256'] for r in records}),1)

    def test_comparison_pair_sign_fixed_epochs_and_missing_rejected(self):
        points=[]
        for index,case in enumerate(d.CASES):
            for epoch in (0,1,5):
                for split in ('train','test'):points.append(dict(case_id=case,epoch=epoch,split=split,samples=10,correct=index,
                    accuracy=index/10,ce=4-index))
        rows=d.comparisons(points,5);self.assertEqual(len(rows),9)
        primary=[r for r in rows if r['contrast']=='E2_minus_E1']
        self.assertTrue(all(r['test_accuracy_difference']==.1 for r in primary));self.assertTrue(all(r['train_CE_difference']==-1 for r in primary))
        with self.assertRaises(ValueError):d.comparisons(points[:-1],5)
        with self.assertRaises(ValueError):d.comparisons(points+points[:1],5)

    def test_configuration_and_complete_command_no_source_checkpoint(self):
        cli=self.cli();a=d.configuration(cli);self.assertEqual(a.multi_svd_effective_ratios,[.15,.25,.35,.9]);self.assertEqual(a.offline_loop_round,0)
        for key,value in [('capacity_ratios','.15,.15'),('capacity_ratios','0,.9'),('capacity_ratios','.15,nan'),
            ('seed',-1),('client_id',20),('lr',0.),('grad_clip',0.),('epochs',0),('atol',float('inf'))]:
            bad=copy.deepcopy(cli);setattr(bad,key,value)
            with self.subTest(key=key),self.assertRaises(ValueError):d.configuration(bad)
        with self.assertRaises(ValueError):d.error_metrics(torch.ones(3)*2,torch.ones(3),1e-5,1e-5,'bad')
        lines=[line.strip()[1:-1] for line in (ROOT/'system/whole_network_spectrum_diagnostic_commands.sh').read_text().splitlines() if line.strip().startswith('"')]
        self.assertEqual(len(lines),1);words=shlex.split(lines[0]);value=lambda k:words[words.index(k)+1]
        self.assertEqual(words[3],'-did');self.assertEqual(value('--seed'),'0');self.assertEqual(value('--capacity-ratios'),'0.15,0.25,0.35,0.90')
        self.assertEqual(value('--epochs'),'5');self.assertEqual(value('--client-id'),'0');self.assertNotIn('--server-file',words)
        self.assertNotIn('$',lines[0]);self.assertIn('whole_network_spectrum_runs/',value('--output-dir'))


if __name__=='__main__':unittest.main()

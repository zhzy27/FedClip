"""Same-rank spectrum intervention: one SVD, pairing and independent slots."""
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
from fc1_capacity_diagnostic import parser
from utils import fc1_capacity_diagnostic as d,fc1_tail_diagnostic as old,multi_svd_spectral as spectral
from flcore.trainmodel.models import Hyper_CNN_512
from test_fc1_capacity_diagnostic import options


class SpectrumTests(unittest.TestCase):
    def setUp(self):torch.set_num_threads(2);torch.random.default_generator.manual_seed(141)

    def test_one_svd_balanced_raw_corrected_same_prefix_parameter_count_other_state(self):
        mother=Hyper_CNN_512(n_kernels=2,num_classes=100,ratio_LR=1.);mother.register_buffer('test_buffer',torch.ones(2))
        args=options();before=old.rng_fingerprint(old.rng_snapshot(mother));native=torch.linalg.svd
        with patch.object(torch.linalg,'svd',wraps=native) as calls:evidence,receipt=d.common_fc1(mother,args,[25],include_raw=True)
        self.assertEqual(calls.call_count,1);self.assertEqual(receipt['k'],15);self.assertEqual(receipt['b'],25)
        for rate in (.90,.25):
            template=d.client_template(mother,rate)
            raw=d.spectrum_branch(template,evidence,25,'raw');corrected=d.spectrum_branch(template,evidence,25,'corrected')
            d.verify_prefix_and_other_state(raw,corrected,prefix_rank=15)
            self.assertEqual(sum(p.numel() for p in raw.parameters()),sum(p.numel() for p in corrected.parameters()))
            self.assertEqual(raw.ratio_LR,rate);self.assertEqual(corrected.fc1.rank,25)
            for variant,model,s in [('raw',raw,evidence['S']),('corrected',corrected,evidence['corrected_S'])]:
                root=s[:25].sqrt()
                self.assertTrue(torch.equal(model.fc1.weight_u,evidence['P'][:,:25]@torch.diag(root)),variant)
                self.assertTrue(torch.equal(model.fc1.weight_v,torch.diag(root)@evidence['Qh'][:25,:]),variant)
                self.assertTrue(torch.equal(model.fc1.bias,mother.fc1.bias))
                self.assertEqual((model.conv2.rank,model.fc2.rank),(round(10*rate),round(512*rate)))
        self.assertEqual(before,old.rng_fingerprint(old.rng_snapshot(mother)))

    def test_intervention_changes_only_tail_and_zero_intervention_is_allowed(self):
        mother=Hyper_CNN_512(n_kernels=2,num_classes=100,ratio_LR=1.);args=options()
        with torch.no_grad():
            mother.fc1.weight.zero_();s=torch.exp(-torch.arange(100)*.01);s[15:]*=.01
            mother.fc1.weight[:100,:100].copy_(torch.diag(s))
        evidence,receipt=d.common_fc1(mother,args,[25],include_raw=True);template=d.client_template(mother,.9)
        a=d.spectrum_branch(template,evidence,25,'raw');b=d.spectrum_branch(template,evidence,25,'corrected')
        self.assertTrue(receipt['changed']);d.verify_prefix_and_other_state(a,b,prefix_rank=15)
        self.assertFalse(torch.equal(a.fc1.weight_u[:,15:],b.fc1.weight_u[:,15:]))
        torch.testing.assert_close(evidence['corrected_S'][15:25],evidence['S'][15:25]*1.5,rtol=1e-6,atol=1e-8)
        with torch.no_grad():mother.fc1.weight.zero_()
        zero,_=d.common_fc1(mother,args,[25],include_raw=True)
        self.assertTrue(torch.equal(zero['raw_weight_u'],zero['weight_u']))
        self.assertTrue(torch.equal(zero['raw_weight_v'],zero['weight_v']))
        with self.assertRaises(ValueError):d.spectrum_branch(template,evidence,25,'unknown')

    def test_each_branch_uses_own_initial_product_and_paired_batches_RNG(self):
        args=options();args.batch_size=2
        mother=Hyper_CNN_512(n_kernels=2,num_classes=100,ratio_LR=1.)
        evidence,_=d.common_fc1(mother,args,[25],include_raw=True);template=d.client_template(mother,.9)
        data=[(torch.randn(3,32,32),torch.tensor(i)) for i in range(3)]
        initial=old.rng_snapshot(template);results=[]
        with tempfile.TemporaryDirectory(dir='E:/' if Path('E:/').exists() else None) as folder,redirect_stdout(io.StringIO()):
            for variant in ('raw','corrected'):
                model=d.spectrum_branch(template,evidence,25,variant);old.restore_rng(initial)
                result,_=d.train_branch(model,mother,args,0,data,data,folder,evidence,[(0,15),(15,25)])
                zero=[r for r in result['updates'] if r['stage']=='epoch_0']
                self.assertTrue(all(r['absolute_product_change_frobenius']==0 for r in zero))
                tail=next(r for r in zero if r['slot_start']==16)
                u,v=result['initial_factors']['weight_u'],result['initial_factors']['weight_v']
                self.assertEqual(tail['download_block_frobenius_norm'],float((u[:,15:].double()@v[15:].double()).norm()))
                results.append(result)
        self.assertEqual(results[0]['batches'],results[1]['batches'])
        self.assertEqual(results[0]['initial_training_rng_sha256'],results[1]['initial_training_rng_sha256'])
        self.assertEqual(results[0]['final_rng_sha256'],results[1]['final_rng_sha256'])

    def test_summaries_pair_variants_at_same_rank_and_CE_epoch1_to5(self):
        points=[]
        for cid,n in [(0,2),(10,8)]:
            for variant in ('raw','corrected'):
                for split in ('train','test'):
                    for epoch in (0,1,5):
                        delta={'raw':0,'corrected':1}[variant]
                        points.append(dict(client_id=cid,rank_rate=.9,fc1_rank=200,spectrum_variant=variant,split=split,epoch=epoch,
                            correct=delta,samples=n,accuracy=delta/n,ce=10-epoch+(delta*epoch)))
        pairs=d.spectrum_comparisons(points,[0,10]);self.assertEqual(len(pairs),2)
        self.assertEqual(pairs[0]['epoch_0_corrected_minus_raw_test_accuracy'],.5)
        self.assertEqual(pairs[0]['epoch_5_minus_epoch_0_test_accuracy_gap_change'],0)
        self.assertEqual(pairs[0]['raw_train_CE_epoch5_minus_epoch1'],-4)
        self.assertEqual(pairs[0]['corrected_test_CE_epoch5_minus_epoch1'],0)
        summary=d.spectrum_capacity_summary(points);self.assertEqual(len(summary),12)
        corrected=next(r for r in summary if r['spectrum_variant']=='corrected' and r['epoch']==0 and r['split']=='test')
        raw=next(r for r in summary if r['spectrum_variant']=='raw' and r['epoch']==0 and r['split']=='test')
        self.assertEqual(corrected['accuracy'],.2);self.assertEqual(raw['accuracy'],0.)
        self.assertIn('测试CE',d.spectrum_summary_text(pairs,'S101 -> offline R102'))
        with self.assertRaises(ValueError):d.spectrum_comparisons(points+points[:1],[0,10])
        with self.assertRaises(ValueError):d.spectrum_comparisons(points[1:],[0,10])

    def test_cli_defaults_and_fixed_rank_explicit_validation_and_command(self):
        args=parser().parse_args(['--server-file','unused','--output-dir','unused'])
        self.assertEqual(args.comparison,'rank');self.assertEqual(args.fc1_ranks,'120,200,720');self.assertEqual(d.comparison_mode(args),'rank')
        args.comparison='spectrum'
        with self.assertRaisesRegex(ValueError,'--fc1-ranks 200'):d.comparison_mode(args)
        args.fc1_ranks='200';self.assertEqual(d.comparison_mode(args),'spectrum')
        lines=[s.strip()[1:-1] for s in (ROOT/'system/fc1_spectrum_diagnostic_commands.sh').read_text().splitlines() if s.strip().startswith('"')]
        self.assertEqual(len(lines),1);words=shlex.split(lines[0]);value=lambda k:words[words.index(k)+1]
        self.assertEqual(words[3],'-did');self.assertEqual(value('--comparison'),'spectrum');self.assertEqual(value('--fc1-ranks'),'200')
        self.assertEqual(value('--client-ids'),'0,10,18');self.assertEqual(value('--source-round'),'101');self.assertIn('/B_fc1',value('--source-run'))
        self.assertNotIn('$',lines[0]);self.assertIn('raw_vs_corrected',value('--output-dir'))


if __name__=='__main__':unittest.main()

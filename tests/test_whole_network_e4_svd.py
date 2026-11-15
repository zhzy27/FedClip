"""Optional direct-SVD control reuses balanced; four-case default unchanged."""
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
from utils import whole_network_spectrum_diagnostic as d,fc1_tail_diagnostic as old


class E4SVDTests(unittest.TestCase):
    def setUp(self):torch.set_num_threads(2);torch.random.default_generator.manual_seed(151)

    def test_reuses_balanced_without_more_original_SVDs_and_preserves_four_initial_states(self):
        from flcore.trainmodel.models import Hyper_CNN_512
        model=Hyper_CNN_512(n_kernels=2,ratio_LR=1.);before=old.rng_fingerprint(old.rng_snapshot(model))
        native=torch.linalg.svd
        with redirect_stdout(io.StringIO()),patch.object(torch.linalg,'svd',wraps=native) as calls:
            five,mother,ledger,tables=d.build_experiments(0,[.15,.25,.35,.9],n_kernels=2,include_e4_svd=True)
        self.assertEqual(calls.call_count,6);self.assertEqual(list(five),['E1','E2','E3','E4','E4_SVD'])
        self.assertEqual(ledger['E4_SVD_source']['additional_original_SVD_calls'],0)
        self.assertEqual(ledger['E4_SVD_source']['pre_aggregation_balanced_state_sha256'],d.tensor_fingerprint(five['E4_SVD'].state_dict()))
        self.assertEqual(ledger['E4_SVD_source']['actual_layer_ranks'],dict(conv2=9,fc1=90,fc2=461))
        with redirect_stdout(io.StringIO()):
            four,_,_,_=d.build_experiments(0,[.15,.25,.35,.9],n_kernels=2)
            direct,_=d.decomposed_once(mother,.9)
        for case in d.CASES:
            self.assertEqual(d.tensor_fingerprint(five[case].state_dict()),d.tensor_fingerprint(four[case].state_dict()),case)
        self.assertEqual(d.tensor_fingerprint(direct.state_dict()),d.tensor_fingerprint(five['E4_SVD'].state_dict()))
        self.assertEqual(before,old.rng_fingerprint(old.rng_snapshot(model)))
        self.assertEqual(len([r for r in ledger['construction_checks'] if 'E4_SVD' in r['label']]),6)
        self.assertTrue(all(r['passed'] for r in ledger['construction_checks']))
        self.assertEqual(len({sum(p.numel() for p in five[c].parameters()) for c in ('E1','E2','E4','E4_SVD')}),1)

    def test_same_data_new_optimizer_five_case_training_and_function_checks(self):
        cli=parser().parse_args(['--device','cpu','--output-dir','unused','--include-e4-svd']);cli.batch_size=2
        args=d.configuration(cli)
        with redirect_stdout(io.StringIO()):models,mother,ledger,_=d.build_experiments(1,[.15,.25,.35,.9],n_kernels=2,diagnostics=False,include_e4_svd=True)
        args.initial_sha256=ledger['recovered_mother_sha256'];args.data_fingerprint='b'*64
        data=[(torch.randn(3,32,32)*.2,torch.tensor(i)) for i in range(3)]
        start=old.rng_snapshot(mother);cache={};records=[]
        for case,model in models.items():
            cache[case]=[]
            for split in ('train','test'):
                metrics,logits=old.evaluate(model,data,2,'cpu');cache[case].append(dict(epoch=0,split=split,**metrics))
            if case=='E4_SVD':svd_logits=logits
            if case=='E2':e2_logits=logits
            if case=='E4':native_logits=logits
        d.error_metrics(e2_logits,svd_logits,1e-5,1e-5,'E2/E4_SVD initial logits')
        d.error_metrics(svd_logits,native_logits,1e-5,1e-5,'E4_SVD/E4 initial logits')
        with tempfile.TemporaryDirectory(dir='E:/' if Path('E:/').exists() else None) as temp,redirect_stdout(io.StringIO()):
            for case,model in models.items():
                a=copy.deepcopy(args);a.multi_svd_effective_assignment=[args.multi_svd_effective_ratios.index(model.ratio_LR)]*20
                old.restore_rng(start);result=d.train_case(model,mother,a,0,data,data,cache[case],temp)
                self.assertEqual(result['training']['train_batches'],10)
                self.assertEqual({r['epoch'] for r in result['endpoints']},{0,1,5});records.append(result)
        self.assertEqual(len({r['batch_order_sha256'] for r in records}),1)
        self.assertEqual(len({r['initial_rng_sha256'] for r in records}),1)
        self.assertEqual(len({r['final_rng_sha256'] for r in records}),1)

    def test_contrasts_completeness_and_optional_command_default_four(self):
        cli=parser().parse_args(['--device','cpu','--output-dir','unused']);self.assertFalse(cli.include_e4_svd)
        points=[]
        for index,case in enumerate(d.CASES+('E4_SVD',)):
            for epoch in (0,1,5):
                for split in ('train','test'):points.append(dict(case_id=case,epoch=epoch,split=split,samples=10,correct=index,accuracy=index/10,ce=5-index))
        rows=d.comparisons(points,5);self.assertEqual(len(rows),15)
        self.assertIn('E2_minus_E4_SVD',{r['contrast'] for r in rows});self.assertIn('E4_SVD_minus_E4',{r['contrast'] for r in rows})
        self.assertEqual(next(r for r in rows if r['contrast']=='E2_minus_E4_SVD')['test_CE_difference'],3.)
        self.assertIn('五组',d.summary_text(rows,'test'))
        with self.assertRaises(ValueError):d.comparisons(points[:-1],5)
        text=(ROOT/'system/whole_network_e4_svd_commands.sh').read_text()
        command=next(s.strip()[1:-1] for s in text.splitlines() if s.strip().startswith('"'));words=shlex.split(command)
        self.assertIn('--include-e4-svd',words);self.assertEqual(words[3],'-did');self.assertNotIn('$',command)
        self.assertNotIn('--include-e4-svd',(ROOT/'system/whole_network_spectrum_diagnostic_commands.sh').read_text())


if __name__=='__main__':unittest.main()

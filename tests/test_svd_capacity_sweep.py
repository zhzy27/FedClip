"""Real RGB32/100-class counts, width compatibility and offline prefix sweep."""
import ast
import copy
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'system'))
from flcore.trainmodel import models
from utils import multi_group_svd as svd,svd_capacity_sweep as sweep,fc1_tail_diagnostic as old
from utils.whole_network_spectrum_diagnostic import truncated_model
from test_single_avg_proto import options

RATES=[.9,.6,.45,.35,.25,.15]
COUNTS=[3246372,2181764,1649460,1295428,941396,587364]


class SweepTests(unittest.TestCase):
    def setUp(self):torch.set_num_threads(2);torch.manual_seed(25)

    def test_actual_counts_and_equal_ranks_at_both_widths(self):
        ranks={}
        for width in (2000,1536):
            for rate in RATES:
                model=models.Hyper_CNN_512(num_classes=100,ratio_LR=rate,hidden_width=width)
                actual={n:getattr(model,n).rank for n in sweep.LAYERS}
                if width==2000:
                    self.assertEqual(sum(p.numel() for p in model.parameters()),COUNTS[RATES.index(rate)]);ranks[rate]=actual
                else:self.assertEqual(actual,ranks[rate])
                self.assertEqual(model.fc1.out_features,width);self.assertEqual(model.fc2.in_features,width)
                if rate==.9:self.assertEqual(sum(p.numel() for p in model.parameters()),3246372 if width==2000 else 2697924)
                self.assertIs(model.base[7],model.fc1);self.assertIs(model.head[1],model.fc3)

    def test_default_model_state_forward_and_RNG_exact_against_751adbd(self):
        source=subprocess.check_output(['git','show','751adbd:system/flcore/trainmodel/models.py'],cwd=ROOT,text=True,encoding='utf-8')
        node=next(n for n in ast.parse(source).body if isinstance(n,ast.ClassDef) and n.name=='Hyper_CNN_512')
        namespace=dict(vars(models));exec(compile(ast.Module(body=[node],type_ignores=[]),'frozen_Hyper.py','exec'),namespace)
        oldcls=namespace['Hyper_CNN_512'];x=torch.randn(2,3,32,32)
        for rate in (.15,.9,1.):
            torch.manual_seed(61);ref=oldcls(num_classes=100,ratio_LR=rate);rng=torch.get_rng_state().clone()
            torch.manual_seed(61);current=models.Hyper_CNN_512(num_classes=100,ratio_LR=rate)
            self.assertTrue(torch.equal(rng,torch.get_rng_state()))
            self.assertEqual(list(ref.state_dict()),list(current.state_dict()))
            self.assertTrue(all(torch.equal(v,current.state_dict()[n]) for n,v in ref.state_dict().items()))
            with torch.no_grad():self.assertTrue(torch.equal(ref(x),current(x)))

    def test_width_factory_payload_reload_and_old_missing_width(self):
        for width in (2000,1536):
            a=options(cnn_hidden_width=width,multi_svd_init_rank_rate=.9)
            mother=svd.initial_model(a)
            self.assertEqual(mother.fc1.out_features,width)
            spec=svd.model_spec(a,.9);model=models.Hyper_CNN_512(**spec['model_kwargs'])
            if width==2000:self.assertNotIn('hidden_width',spec['model_kwargs'])
            else:self.assertEqual(spec['model_kwargs']['hidden_width'],1536)
            with tempfile.TemporaryDirectory(dir='E:/' if Path('E:/').exists() else None) as folder:
                file=Path(folder)/'model.pt';torch.save(dict(spec,state_dict=model.state_dict()),file)
                loaded,p=svd.load_checkpoint(file)
                self.assertEqual(loaded.fc1.out_features,width)
                self.assertTrue(all(torch.equal(v,loaded.state_dict()[n]) for n,v in model.state_dict().items()))
        for width in (0,-1,1.5):
            with self.assertRaises(ValueError):models.Hyper_CNN_512(hidden_width=width)

    def test_shared_three_original_SVDs_and_roundtrip_logits_sample_order(self):
        for width in (2000,1536):
            native=models.Hyper_CNN_512(num_classes=100,ratio_LR=.9,hidden_width=width)
            data=[(torch.randn(3,32,32)*.1,torch.tensor(i)) for i in range(3)]
            native.eval();digest=old.tensor_fingerprint(native.state_dict());rng=old.rng_fingerprint(old.rng_snapshot(native))
            original=torch.linalg.svd
            with patch.object(torch.linalg,'svd',wraps=original) as count,redirect_stdout(io.StringIO()):
                balanced,receipts=sweep.shared_download(native,.9)
                self.assertEqual(count.call_count,3)
                self.assertTrue(all('driver' not in call.kwargs for call in count.call_args_list))
                self.assertTrue(all(r['svd_driver'] is None and r['svd_device']=='cpu' for r in receipts))
                reference,logits=sweep.evaluate(native,data,2,'cpu')
                for rate in RATES:
                    model=truncated_model(balanced,rate);row,out=sweep.evaluate(model,data,2,'cpu')
                    self.assertEqual(count.call_count,3);self.assertEqual(row['sample_order_sha256'],reference['sample_order_sha256'])
                    self.assertTrue(torch.equal(model.conv1.weight,native.conv1.weight))
                    self.assertTrue(torch.equal(model.fc3.weight,native.fc3.weight))
                    for name in sweep.LAYERS:
                        self.assertTrue(torch.equal(getattr(model,name).bias,getattr(native,name).bias))
                    if rate==.9:
                        torch.testing.assert_close(out,logits,atol=1e-5,rtol=1e-5)
                        expected=svd.recover_parameters(native);reconstructed=svd.recover_parameters(model)
                        for name in expected:torch.testing.assert_close(reconstructed[name],expected[name],atol=1e-5,rtol=1e-5)
            self.assertEqual(len(receipts),3);self.assertEqual(digest,old.tensor_fingerprint(native.state_dict()))
            self.assertEqual(rng,old.rng_fingerprint(old.rng_snapshot(native)))

    def test_training_default_decomposition_SGD_upload_RNG_exact_vs_1d28412(self):
        source=subprocess.check_output(['git','show','1d28412:system/flcore/trainmodel/models.py'],cwd=ROOT,text=True,encoding='utf-8')
        nodes=[n for n in ast.parse(source).body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name in ('Decom_COV','Decom_LINEAR','Hyper_CNN_512')]
        namespace=dict(vars(models));exec(compile(ast.Module(body=nodes,type_ignores=[]),'frozen_1d28412.py','exec'),namespace)
        frozen=namespace['Hyper_CNN_512'](num_classes=100,n_kernels=2,ratio_LR=1.)
        current=models.Hyper_CNN_512(num_classes=100,n_kernels=2,ratio_LR=1.);current.load_state_dict(frozen.state_dict())
        start=torch.get_rng_state().clone()
        with redirect_stdout(io.StringIO()):frozen.decom_larger_model(.37)
        expected_rng=torch.get_rng_state().clone();torch.set_rng_state(start)
        original=torch.linalg.svd
        with patch.object(torch.linalg,'svd',wraps=original) as observed,redirect_stdout(io.StringIO()):current.decom_larger_model(.37)
        self.assertEqual(observed.call_count,3)
        self.assertTrue(all('driver' not in call.kwargs for call in observed.call_args_list))
        self.assertTrue(torch.equal(expected_rng,torch.get_rng_state()))
        self.assertTrue(all(torch.equal(v,current.state_dict()[n]) for n,v in frozen.state_dict().items()))
        x=torch.randn(2,3,32,32);y=torch.tensor([1,4])
        training_rng=torch.get_rng_state().clone()
        for model in (frozen,current):
            torch.set_rng_state(training_rng)
            model.ratio_LR=.37;optimizer=torch.optim.SGD(model.parameters(),lr=.005)
            objective=torch.nn.functional.cross_entropy(model(x),y)+.001*model.frobenius_decay()
            optimizer.zero_grad();objective.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),10.);optimizer.step()
            self.assertTrue(torch.equal(training_rng,torch.get_rng_state()))
        self.assertTrue(all(torch.equal(v,current.state_dict()[n]) for n,v in frozen.state_dict().items()))
        left=svd.recover_parameters(frozen);right=svd.recover_parameters(current)
        self.assertTrue(all(torch.equal(value,right[name]) for name,value in left.items()))
        self.assertTrue(torch.equal(training_rng,torch.get_rng_state()))

    def test_weighted_paired_summary_nonmonotone_threshold_and_missing_round(self):
        rows=[]
        for cid,n,correct in [(0,2,2),(5,8,4)]:
            base=dict(source='L',client_id=cid,completed_round=100,hidden_width=2000,samples=n,sample_order_sha256=str(cid))
            rows.append(dict(base,variant='native',rank_rate=.9,correct=correct))
            for rate,p,delta in [(.9,100,0),(.6,60,1 if cid==5 else 0),(.15,15,0)]:
                rows.append(dict(base,variant='resvd',rank_rate=rate,correct=correct-delta,parameter_count=p,drop_pp=100*delta/n))
        result=sweep.summarize(rows,[0,5],.5)
        self.assertEqual(result['cohort'][1]['accuracy'],.5);self.assertEqual(result['cohort'][1]['drop_pp'],10.)
        self.assertEqual(result['cohort_threshold']['minimum_tested_feasible']['parameter_count'],15)
        self.assertEqual(len(result['cohort_threshold']['adjacent_crossing_intervals']),2)
        with tempfile.TemporaryDirectory(dir='E:/' if Path('E:/').exists() else None) as folder:
            torch.save(dict(client_id=0,model_id=0,completed_train_count=99,role='ordinary_post_local_low_rank'),Path(folder)/'Client_0_model_0.pt')
            cli=SimpleNamespace(checkpoint=None,source_run=folder,model_id=0,completed_round=100,manifest=None)
            with self.assertRaisesRegex(FileNotFoundError,'C0.*R100'):sweep.resolve_inputs(cli,[0,5])
        for values in ('nan,.9','.9,.9','0','.95'):
            with self.assertRaises(ValueError):sweep.rates(values)

    def test_wrong_checkpoint_identity_and_provided_binding_rejected(self):
        model=models.Hyper_CNN_512(num_classes=100,ratio_LR=.9)
        cli=SimpleNamespace(model_id=0,completed_round=100)
        manifest=dict(effective=dict(model_family='Decom_CNN-5-512'))
        payload=dict(role='ordinary_post_local_low_rank',prediction_checkpoint=True,client_id=5,model_id=0,
            completed_train_count=100,loop_round=99,model_family='Decom_CNN-5-512',state_dict=model.state_dict())
        receipt=sweep.validate_checkpoint(model,payload,5,cli,manifest,'actual')
        self.assertEqual(receipt['data_binding'],'independent_same_run_manifest_audit')
        for fields in [dict(role='server'),dict(client_id=0),dict(completed_train_count=99),dict(loop_round=100),
            dict(data_fingerprint='wrong'),dict(single_avg_prototype_usage=dict(downloaded_model=dict(data_fingerprint='wrong'))),
            dict(initial_parameter_sha256='other-run')]:
            with self.subTest(fields=fields),self.assertRaises(ValueError):
                sweep.validate_checkpoint(model,dict(payload,**fields),5,cli,manifest,'actual')
        good=dict(payload,data_fingerprint='actual',single_avg_prototype_usage=dict(downloaded_model=dict(data_fingerprint='actual')))
        self.assertEqual(len(sweep.validate_checkpoint(model,good,5,cli,manifest,'actual')['direct_binding_locations']),2)
        good['single_avg_prototype_usage']['downloaded_model']['data_fingerprint']='wrong'
        with self.assertRaises(ValueError):sweep.validate_checkpoint(model,good,5,cli,manifest,'actual')

    def test_paired_exports_and_cohort_round_data_guards(self):
        with tempfile.TemporaryDirectory() as folder,patch.object(sweep,'render'):
            root=Path(folder);sources=[]
            for label,correct in [('L',6),('H',5)]:
                source=root/label;source.mkdir();sources.append(str(source))
                manifest=dict(status='complete',role='offline_post_local_whole_network_capacity_sweep',client_ids=[0,5],data_fingerprint='same',
                    requested=dict(source_name=label,completed_round=100,max_drop_pp=.5))
                cohort=dict(rank_rate=.9,source=label,completed_round=100,client_ids=[0,5],parameter_count=3246372,samples=10,
                    correct=correct,accuracy=correct/10,native_accuracy=correct/10,drop_pp=0.)
                (source/'manifest.json').write_text(json.dumps(manifest));(source/'summary.json').write_text(json.dumps(dict(cohort=[cohort])))
            sweep.compare_results(SimpleNamespace(output_dir=str(root/'comparison'),compare_results=sources))
            contrast=json.loads((root/'comparison/paired_comparisons.json').read_text())[0]
            self.assertAlmostEqual(contrast['accuracy_difference_pp'],-10)
            for field,value in [('client_ids',[0]),('data_fingerprint','wrong')]:
                wrong=dict(manifest,**{field:value});(root/'H/manifest.json').write_text(json.dumps(wrong))
                with self.assertRaises(ValueError):sweep.compare_results(SimpleNamespace(output_dir=str(root/'reject'),compare_results=sources))
            manifest['requested']['completed_round']=99;(root/'H/manifest.json').write_text(json.dumps(manifest))
            with self.assertRaises(ValueError):sweep.compare_results(SimpleNamespace(output_dir=str(root/'reject'),compare_results=sources))


if __name__=='__main__':unittest.main()

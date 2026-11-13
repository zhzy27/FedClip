"""Full-spectrum D01 corrections: formulas, zero/ordering, timing and same SVD."""
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
from test_single_avg_proto import options as base_options
from test_control_population import assert_rng
from common.experiment import rng_state,set_rng
from utils import multi_svd_spectral as sp,multi_group_svd as svd
from utils.cnn512_reporting import model_digest
from flcore.clients.clientMultiGroupSVD import clientMultiGroupSVD
from flcore.trainmodel.models import Decom_LINEAR,Hyper_CNN_512


def options(**kw):
    values=dict(device='cpu',multi_proto_mode='none',multi_proto_lambda=0.,multi_svd_init_rank_rate=.90,
        multi_svd_spectral_mode='single_point',multi_svd_spectral_layers='conv2,fc1,fc2',multi_svd_spectral_max_gain=1.5)
    values.update(kw);return base_options(**values)


class SpectralTests(unittest.TestCase):
    def setUp(self):torch.set_num_threads(2);torch.random.default_generator.manual_seed(39)

    def test_hand_formulas_single_gain_vs_next_tier_actual_boundary_gain(self):
        s=torch.tensor([8.,4.,2.,1.,.8,.2,.1,0.],dtype=torch.float64)
        a,am=sp.correct_spectrum(s,4,6,'single_point',1.5,4)
        b,bm=sp.correct_spectrum(s,4,6,'next_tier',1.5,4)
        torch.testing.assert_close(a,s,rtol=0,atol=0)
        torch.testing.assert_close(b,torch.tensor([8.,4.,2.,1.,.8,.25,.125,0.],dtype=torch.float64),rtol=0,atol=1e-14)
        self.assertAlmostEqual(bm['slope'],-torch.log(torch.tensor(2.)).item(),places=7)
        self.assertAlmostEqual(bm['gain_b'],1.25,places=14)
        self.assertFalse(bm['counterfactual_same_full_corrected_spectrum'])

    def test_both_rules_full_cap_are_tagged_identical_not_claimed_different(self):
        s=torch.tensor([8.,4.,2.,1.,.2,.1,.05,.02,0.],dtype=torch.float64)
        outputs=[]
        for mode in ('single_point','next_tier'):
            out,row=sp.correct_spectrum(s,4,6,mode,1.5,3);outputs.append(out)
            torch.testing.assert_close(out[4:],s[4:]*1.5,rtol=0,atol=1e-15)
            self.assertTrue(row['counterfactual_both_entire_interval_at_cap']);self.assertTrue(row['counterfactual_same_full_corrected_spectrum'])
            self.assertEqual(row['next_tier_cap_hits'],2);self.assertIn('identical',row['equivalence_note'])
        self.assertTrue(torch.equal(*outputs))

    def test_zero_pivot_zero_b_missing_positive_fit_and_short_windows(self):
        s=torch.tensor([8.,4.,2.,1.,0.,0.,0.])
        out,row=sp.correct_spectrum(s,4,6,'single_point',1.5,20)
        self.assertIs(out,s);self.assertEqual(row['skipped_reason'],'zero_sigma_k_plus_1');self.assertEqual(row['fit_window_actual'],4)
        s=torch.tensor([8.,4.,2.,1.,.1,0.,0.])
        out,row=sp.correct_spectrum(s,4,6,'next_tier',1.5,4)
        self.assertTrue(row['zero_sigma_b']);self.assertTrue(torch.equal(out[5:],s[5:]));self.assertGreater(float(out[4]),float(s[4]))
        self.assertFalse(row['counterfactual_both_entire_interval_at_cap'])
        for s,k,b in [(torch.zeros(6),4,6),(torch.tensor([2.,0.,0.]),1,2)]:
            out,row=sp.correct_spectrum(s,k,b,'next_tier',1.5,20)
            self.assertIs(out,s);self.assertEqual(row['skipped_reason'],'insufficient_positive_fit_values')

    def test_invariants_dtype_shared_full_curve_and_no_input_or_rng_change(self):
        generator=torch.Generator().manual_seed(91);state=rng_state()
        for dtype in (torch.float32,torch.float64):
            for _ in range(25):
                s=torch.exp(-torch.cumsum(torch.rand(40,generator=generator),0)).to(dtype);s[7:]*=.03;s[32:]=0
                before=s.clone()
                for mode in ('single_point','next_tier'):
                    a,row=sp.correct_spectrum(s,7,15,mode,1.5,5,retain_rank=15)
                    b,_=sp.correct_spectrum(s,7,15,mode,1.5,5,retain_rank=40)
                    self.assertTrue(torch.equal(a,b));self.assertTrue(torch.equal(a[:7],s[:7]))
                    self.assertTrue(torch.isfinite(a).all());self.assertTrue((a[:-1]>=a[1:]).all())
                    self.assertTrue((a>=s).all());self.assertTrue(torch.equal(a[s==0],s[s==0]))
                    self.assertTrue(((a[s>0].double()/s[s>0])<=1.5+1e-6).all())
                self.assertTrue(torch.equal(s,before))
        assert_rng(self,state)
        for bad in (torch.tensor([2.,3.,1.]),torch.tensor([2.,float('nan'),0.]),torch.tensor([2.,1.,-1.])):
            with self.assertRaises(ValueError):sp.correct_spectrum(bad,1,2,'single_point',1.5,2)

    def test_scope_layers_gain_and_actual_tier_rule(self):
        a=options();model=svd.initial_model(a);bound=sp.boundary_ranks(model,a)
        self.assertEqual([(bound[n]['k'],bound[n]['b'],bound[n]['fit_window']) for n in sp.LAYERS],[(12,20,8),(120,200,20),(77,128,20)])
        self.assertEqual(sp.layer_names(options(multi_svd_spectral_layers='fc2, conv2')),['conv2','fc2'])
        for change in [dict(multi_svd_spectral_layers=''),dict(multi_svd_spectral_layers='fc1,fc1'),dict(multi_svd_spectral_layers='head'),
            dict(multi_svd_spectral_max_gain=.9),dict(multi_svd_spectral_max_gain=float('nan')),dict(multi_svd_spectral_max_gain=float('inf')),
            dict(multi_fc1_tail_scale_strength=1.),dict(multi_svd_aggregation='rank_partition'),dict(num_server_models=3),
            dict(multi_proto_mode='global_last_epoch',multi_proto_lambda=1.),dict(multi_svd=False,is_regular=0)]:
            with self.subTest(change=change),self.assertRaises(ValueError):options(**change)
        self.assertFalse(sp.enabled(options(multi_svd_spectral_mode='none',multi_fc1_tail_scale_strength=1.)))
        same=options(multi_svd_assignment=','.join(['0']*20))
        with self.assertRaisesRegex(ValueError,'two distinct'):sp.boundary_ranks(model,same)

    def test_same_svd_before_sqrt_balanced_construction_and_old_observer_contract(self):
        linear=torch.nn.Linear(8,8,bias=True);seen={};counter=[0];original=torch.linalg.svd
        with torch.no_grad():linear.weight.copy_(torch.diag(torch.tensor([8.,4.,2.,1.,.1,.05,.02,0.])))
        def record(*a,**kw):counter[0]+=1;return original(*a,**kw)
        def transform(s,rank):seen['raw']=s.clone();seen['modified']=sp.correct_spectrum(s,4,6,'next_tier',1.5,4)[0];return seen['modified']
        def observer(layer,p,s,q):seen['observer_raw']=s.clone();seen['p']=p;seen['q']=q
        with patch.object(torch.linalg,'svd',side_effect=record):layer=Decom_LINEAR(linear,.9,svd_observer=observer,singular_value_transform=transform)
        self.assertEqual(counter[0],1);self.assertTrue(torch.equal(seen['raw'],seen['observer_raw']))
        root=seen['modified'][:layer.rank].sqrt()
        self.assertTrue(torch.equal(layer.weight_u,seen['p'][:,:layer.rank]@torch.diag(root)))
        self.assertTrue(torch.equal(layer.weight_v,torch.diag(root)@seen['q'][:layer.rank]))
        self.assertTrue(torch.equal(layer.bias,linear.bias))

    def test_first_download_minimum_and_unselected_layers_exact_rng_and_optimizer(self):
        a=options(multi_svd_spectral_layers='fc1');mother=svd.initial_model(a)
        with tempfile.TemporaryDirectory(dir='E:/' if Path('E:/').is_dir() else None) as folder,redirect_stdout(io.StringIO()):
            a.save_folder_name_full=folder
            for cid,loop in [(0,0),(4,1),(0,1)]:
                args=copy.deepcopy(a);c=clientMultiGroupSVD(args,cid,3,3,initial_model=mother,train_slow=False,send_slow=False)
                before=rng_state();count=[0];original=torch.linalg.svd
                def SVD(*p,**kw):count[0]+=1;return original(*p,**kw)
                with patch.object(torch.linalg,'svd',side_effect=SVD):c.set_parameters([mother],loop)
                after=rng_state();set_rng(before);old=copy.deepcopy(mother);old.decom_larger_model(c.rank_rate);old.ratio_LR=c.rank_rate
                assert_rng(self,after);self.assertEqual(count[0],3)
                current=c.model_set[0]
                if loop==0 or cid==4:self.assertEqual(model_digest(current),model_digest(old))
                else:
                    for name,p in current.named_parameters():
                        if not name.startswith('fc1.weight_'):self.assertTrue(torch.equal(p,dict(old.named_parameters())[name]),name)
                    self.assertTrue(torch.equal(current.fc1.weight_u[:,:120],old.fc1.weight_u[:,:120]))
                    self.assertTrue(torch.equal(current.fc1.weight_v[:120],old.fc1.weight_v[:120]))
                self.assertEqual([id(p) for p in current.parameters()],[id(p) for g in c.optimizer_set[0].param_groups for p in g['params']])
                self.assertEqual([r['completed_aggregations_before_download'] for r in c.svd_spectral_latest],[loop]*3)
                self.assertEqual([r['upcoming_completed_train_count'] for r in c.svd_spectral_latest],[loop+1]*3)

    def test_eight_commands_and_optional_diagnostic_D01(self):
        text=(ROOT/'system/svd_spectral_commands.sh').read_text()
        lines=[line.strip()[1:-1] for line in text.splitlines() if line.strip().startswith('"')]
        self.assertEqual(len(lines),9);combinations=set();directories=set()
        for index,line in enumerate(lines):
            words=shlex.split(line);get=lambda name:words[words.index(name)+1]
            self.assertEqual(words[3],'-did');self.assertEqual(get('-gr'),'100');self.assertEqual(get('--multi_svd_init_rank_rate'),'0.90')
            self.assertEqual(get('--multi_fc1_tail_scale_strength'),'0.0');self.assertEqual(get('--multi_svd_aggregation'),'full_avg')
            self.assertEqual(get('--multi_proto_mode'),'none');self.assertEqual(float(get('--multi_proto_lambda')),0.)
            self.assertEqual(get('--multi_svd_spectral_max_gain'),'1.5')
            self.assertEqual(get('--multi_rank_spectrum_rounds'),'1,2,3,4,5,6,7,8,9,10,20,50,101')
            if index<8:combinations.add((get('--multi_svd_spectral_mode'),get('--multi_svd_spectral_layers')))
            else:self.assertEqual(get('--multi_svd_spectral_mode'),'none')
            directories.add(get('--final-model-root'));self.assertNotIn('$',line)
        self.assertEqual(len(combinations),8);self.assertEqual(len(directories),9)


if __name__=='__main__':unittest.main()

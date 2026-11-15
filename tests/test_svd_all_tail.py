"""All-tail Avg-mass formula and side-effect-free five-round observations."""
import copy
import io
from contextlib import redirect_stdout
from pathlib import Path
import sys
import tempfile
import json
import shlex
import unittest
from unittest.mock import patch

import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'system'))
from test_multi_svd_spectral import options
from test_control_population import assert_rng
from common.experiment import rng_state,set_rng
from flcore.clients.clientMultiGroupSVD import clientMultiGroupSVD
from flcore.trainmodel.models import Hyper_CNN_512
from utils import multi_svd_spectral as sp,multi_svd_five_round as five
from utils.cnn512_reporting import model_digest


class AllTailTests(unittest.TestCase):
    def setUp(self):torch.set_num_threads(2)

    def test_mass_unequal_samples_forward_limit_zero_and_no_cap(self):
        s=torch.tensor([20.,10.,3.,2.,.1,.05,0.,0.],dtype=torch.float64);before=s.clone();state=rng_state()
        out,row=sp.all_tail_spectrum(s,[2,4,6],[1,2,7])
        expected=torch.tensor([20.,10.,3/.9,2/.9,.1/.7,.05/.7,0.,0.],dtype=torch.float64)
        torch.testing.assert_close(out,expected,rtol=0,atol=1e-15)
        self.assertTrue(torch.equal(s,before));assert_rng(self,state)
        self.assertEqual([r['contributor_train_samples'] for r in row['rank_corrections']],[9,9,7,7])
        s=torch.tensor([10.,9.,8.,7.,6.,0.]);out,row=sp.all_tail_spectrum(s,[2,5],[9,1])
        self.assertTrue(torch.equal(out,torch.tensor([10.,9.,9.,9.,9.,0.])))
        self.assertEqual(row['nonincreasing_limited_ranks'],3)
        self.assertEqual(row['rank_corrections'][0]['theoretical_gain'],10.)
        self.assertGreater(row['rank_corrections'][2]['actual_gain'],1.)
        # No 1.5 ceiling, and beyond maximum client rank untouched.
        out,_=sp.all_tail_spectrum(torch.tensor([10.,9.,1.,.5,.1]),[2,4],[4,1])
        self.assertTrue(torch.equal(out,torch.tensor([10.,9.,5.,2.5,.1])))
        same,_=sp.all_tail_spectrum(before,[2,2],[1,2]);self.assertIs(same,before)
        for ranks,ns in [([0],[1]),([2],[0]),([2,3],[1]),([9],[1])]:
            with self.assertRaises(ValueError):sp.all_tail_spectrum(before,ranks,ns)

    def test_actual_layer_boundaries_and_theoretical_fc1_gains(self):
        from utils.multi_group_svd import initial_model
        a=options(multi_svd_spectral_mode='all_tail');m=initial_model(a);b=sp.boundary_ranks(m,a)
        self.assertEqual(b['fc1']['actual_ranks'],[120,200,280,296,720])
        self.assertEqual(b['conv2']['actual_ranks'],[12,20,28,30,72])
        self.assertEqual(b['fc2']['actual_ranks'],[77,128,179,189,461])
        s=torch.linspace(100.,.01,800);_,meta=sp.all_tail_spectrum(s,b['fc1']['client_ranks'],[10]*20)
        by={r['rank']:r for r in meta['rank_corrections']}
        for rank,gain in [(121,1.25),(201,5/3),(281,2.5),(297,5),(720,5)]:self.assertAlmostEqual(by[rank]['theoretical_gain'],gain)
        self.assertIsNone(sp.policy(a)['max_gain'])

    def test_same_svd_raw_twin_and_disabled_observation_rng(self):
        a=options(multi_svd_spectral_mode='all_tail',multi_svd_five_diagnostics=1)
        a.multi_svd_avg_train_samples=[2+i%3 for i in range(20)]
        torch.manual_seed(8);m=Hyper_CNN_512(n_kernels=2,ratio_LR=.9);m.recover_larger_model();m.ratio_LR=1.
        with tempfile.TemporaryDirectory(dir='E:/' if Path('E:/').is_dir() else None) as folder,redirect_stdout(io.StringIO()):
            a.save_folder_name_full=folder
            for cid,loop in [(0,0),(4,1),(0,1),(2,1)]:
                c=clientMultiGroupSVD(a,cid,3,2,initial_model=m,train_slow=False,send_slow=False)
                state=rng_state();calls=[];native=torch.linalg.svd
                def SVD(*p,**kw):calls.append(1);return native(*p,**kw)
                captured=[]
                with patch.object(five,'after_download',side_effect=lambda c,rs,l:captured.append(rs)),patch.object(torch.linalg,'svd',side_effect=SVD):
                    c.set_parameters([m],loop)
                after=rng_state();self.assertEqual(len(calls),3)
                set_rng(state);raw=copy.deepcopy(m);raw.decom_larger_model(c.rank_rate);raw.ratio_LR=c.rank_rate
                assert_rng(self,after)
                from utils.multi_rank_partition import factors
                for name in sp.LAYERS:
                    f=factors(getattr(raw,name));u,v=captured[0][name]['raw_factors']
                    self.assertTrue(torch.equal(f['U'],u));self.assertTrue(torch.equal(f['V'],v))
                if loop==0 or cid==4:self.assertEqual(model_digest(raw),model_digest(c.model_set[0]))
                disabled=copy.deepcopy(a);disabled.multi_svd_five_diagnostics=0
                d=clientMultiGroupSVD(disabled,cid,3,2,initial_model=m,train_slow=False,send_slow=False)
                set_rng(state);d.set_parameters([m],loop);assert_rng(self,after)
                self.assertEqual(model_digest(d.model_set[0]),model_digest(c.model_set[0]))

    def test_readonly_eval_exception_and_slots_weighted_fixed_cohorts(self):
        from types import SimpleNamespace
        from utils.multi_rank_partition import factors
        from common.data_loading import make_data_loader
        a=options(multi_svd_spectral_mode='none',multi_svd_five_diagnostics=1)
        model=Hyper_CNN_512(n_kernels=2,ratio_LR=.15);model.train();model.fc1.eval()
        state=rng_state();before=model_digest(model);modes=[m.training for m in model.modules()]
        data=[(torch.randn(3,32,32),torch.tensor(i)) for i in range(3)];state=rng_state()
        c=SimpleNamespace(args=a,dataset='Cifar100',id=2,download_round=1,device='cpu',rank_rate=.15,
            train_samples=3,test_samples=3,five_endpoint_rows=[],five_cost_rows=[])
        with patch('utils.data_utils.read_client_data',return_value=data):five.evaluate(c,model,'download')
        assert_rng(self,state);self.assertEqual(before,model_digest(model));self.assertEqual(modes,[m.training for m in model.modules()])
        with patch('utils.data_utils.read_client_data',side_effect=RuntimeError('test')):
            with self.assertRaises(RuntimeError):five.evaluate(c,model,'download')
        assert_rng(self,state);self.assertEqual(modes,[m.training for m in model.modules()])
        source=dict(U=torch.eye(4),V=torch.eye(4));dest=dict(U=torch.eye(4)*2,V=torch.eye(4))
        slots=five.fixed_slot_rows(source,dest,[1,2,4]);self.assertEqual([r['slot_end'] for r in slots],[1,2,4])
        self.assertAlmostEqual(slots[-1]['absolute_product_change'],2**.5)
        rows=[dict(client_id=i,loop_round=0,completed_train_count=1,stage='download',split=s,correct=i,samples=i+1,CE=i+.5) for i in range(3) for s in ('train','test')]
        points=five.weighted_points(rows,{0:0,1:1,2:0});allrow=next(r for r in points if r['evaluation_scope']=='all' and r['split']=='test')
        self.assertEqual(allrow['accuracy'],3/6)
        self.assertEqual(next(r for r in points if r['evaluation_scope']=='fixed_tier_0')['member_ids'],[0,2])

    def test_four_commands_actual_five_round_budget_and_comparison_decomposition(self):
        from compare_svd_five_round import compare
        text=(ROOT/'system/svd_five_round_commands.sh').read_text()
        lines=[shlex.split(line.strip()[1:-1]) for line in text.splitlines() if line.strip().startswith('"')]
        self.assertEqual(len(lines),4);names=[];common=None
        for words in lines:
            get=lambda key:words[words.index(key)+1]
            self.assertEqual(words[3],'-did');self.assertEqual(get('-gr'),'4');self.assertEqual(get('-ls'),'5')
            self.assertEqual(get('--multi_svd_init_rank_rate'),'0.90');self.assertEqual(get('--multi_svd_five_diagnostics'),'1')
            self.assertEqual(get('--multi_proto_mode'),'none');self.assertEqual(get('--multi_proto_lambda'),'0')
            names.append(get('--final-model-root'));self.assertNotIn('$',' '.join(words))
            self.assertEqual(get('--multi_svd_spectral_mode'),'all_tail' if 'Hplus/' in get('-sfn') else 'none')
        self.assertEqual(len(set(names)),4)
        with tempfile.TemporaryDirectory(dir='E:/' if Path('E:/').is_dir() else None) as directory:
            paths={}
            for key,pre,post in [('S',.1,.2),('L',.2,.4),('H',.15,.3),('Hplus',.18,.38)]:
                folder=Path(directory)/key;folder.mkdir();paths[key]=folder
                m=dict(effective={k:0 for k in ('seed','num_clients','num_server_models','num_classes','model_family','local_epochs','batch_size',
                    'local_learning_rate','global_rounds','eval_gap','local_grad_clip','test_batch_size','is_regular','regular_lamda',
                    'multi_svd_init','multi_svd_init_rank_rate','multi_proto_mode','multi_proto_lambda')},five_round_diagnostics=dict(
                    shared_initialization_sha256='init',data_fingerprint='data',fixed_cohort_ratios=[.9,.15]))
                rows=[dict(completed_train_count=2,evaluation_scope='all',split='test',stage=stage,member_ids=[0,1],
                    accuracy=acc,CE=1-acc) for stage,acc in [('download',pre),('post_local',post)]]
                if key=='Hplus':rows.append(dict(rows[0],stage='same_mother_raw_counterfactual',accuracy=.17))
                (folder/'manifest.json').write_text(json.dumps(m));(folder/'five_cohort_metrics.json').write_text(json.dumps(rows))
            rows=compare(paths);row=next(r for r in rows if r['contrast']=='Hplus_minus_H')
            self.assertAlmostEqual(row['pre_accuracy_difference'],.03);self.assertAlmostEqual(row['post_accuracy_difference'],.08)
            self.assertAlmostEqual(row['same_round_local_gain_difference'],.05)
            self.assertAlmostEqual(rows[-1]['pre_accuracy_difference'],.01)


if __name__=='__main__':unittest.main()

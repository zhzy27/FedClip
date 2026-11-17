"""First-band isolation, frozen old spectra and readonly epoch observations."""
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
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'system'))
from test_multi_svd_spectral import options
from common.experiment import rng_state,set_rng
from test_control_population import assert_rng
from utils import multi_svd_spectral as sp,multi_svd_five_round as five
from utils.multi_rank_partition import factors
from flcore.trainmodel.models import Hyper_CNN_512


class FirstBandTests(unittest.TestCase):
    def setUp(self):torch.set_num_threads(2);torch.manual_seed(317)

    def test_unequal_mass_formula_no_cap_outside_and_true_zero(self):
        s=torch.tensor([10.,8.,1.,.3,.2,.1,0.],dtype=torch.float64)
        ranks=[2,4,6];counts=[80,10,10];snapshot=s.clone();state=rng_state()
        out,row=sp.first_band_spectrum(s,ranks,counts)
        expected=s.clone();expected[2]=min(s[2]/.2,expected[1]);expected[3]=min(s[3]/.2,expected[2])
        self.assertTrue(torch.equal(out,expected));self.assertGreater(row['rank_corrections'][0]['actual_gain'],1.5)
        self.assertEqual((row['k'],row['b']),(2,4));self.assertEqual(row['rank_corrections'][0]['contributor_ids'],[1,2])
        self.assertEqual(row['rank_corrections'][0]['contributor_mass'],.2);self.assertEqual(row['rank_corrections'][0]['contributor_count'],2)
        self.assertTrue(torch.equal(out[:2],s[:2]));self.assertTrue(torch.equal(out[4:],s[4:]));self.assertTrue(torch.equal(s,snapshot));assert_rng(self,state)
        z=torch.tensor([3.,2.,0.,0.,0.,0.]);out,_=sp.first_band_spectrum(z,ranks,counts);self.assertIs(out,z)
        with self.assertRaises(ValueError):sp.first_band_spectrum(s,[2,2],[1,1])
        with self.assertRaises(ValueError):sp.first_band_spectrum(s,[2,4],[0,1])

    def test_monotonic_clamp_and_dtype_rounding(self):
        for dtype in (torch.float32,torch.float64):
            s=torch.tensor([10.,8.,7.,6.,5.,1.],dtype=dtype)
            out,row=sp.first_band_spectrum(s,[2,4,6],[1,1,1])
            self.assertTrue(torch.equal(out,torch.tensor([10.,8.,8.,8.,5.,1.],dtype=dtype)))
            self.assertEqual(row['nonincreasing_limited_ranks'],2)
            self.assertTrue(all(r['nonincreasing_limited'] for r in row['rank_corrections']))
            self.assertTrue((out[:-1]>=out[1:]).all())

    def test_original_twenty_client_first_band_mass_is_sixteen_twentieths(self):
        singular=torch.exp(-torch.arange(800,dtype=torch.float64)/100)
        ranks=[720,296,280,200,120]*4;counts=[2250]*20
        corrected,row=sp.first_band_spectrum(singular,ranks,counts)
        self.assertEqual((row['k'],row['b']),(120,200));self.assertEqual(len(row['rank_corrections']),80)
        for item in row['rank_corrections']:
            self.assertEqual(item['contributor_count'],16);self.assertEqual(item['contributor_mass'],.8);self.assertEqual(item['theoretical_gain'],1.25)
        self.assertTrue(torch.equal(corrected[:120],singular[:120]));self.assertTrue(torch.equal(corrected[200:],singular[200:]))

    def test_actual_bounds_same_three_svds_R1_minimum_and_factor_outside(self):
        a=options(multi_svd_spectral_mode='first_band',multi_svd_five_diagnostics=1)
        a.multi_svd_avg_train_samples=[2+i%3 for i in range(20)]
        mother=Hyper_CNN_512(n_kernels=2,ratio_LR=1.);bounds=sp.boundary_ranks(mother,a)
        native=torch.linalg.svd
        for cid,loop in [(0,0),(4,1),(0,1),(3,1)]:
            rate=a.multi_svd_effective_ratios[cid%5];actual=copy.deepcopy(mother);ordinary=copy.deepcopy(mother);state=rng_state();observation={}
            with redirect_stdout(io.StringIO()),patch.object(torch.linalg,'svd',wraps=native) as called:
                records=sp.decompose(actual,rate,a,cid,0,loop,observation=observation)
            self.assertEqual(called.call_count,3);ending=rng_state();set_rng(state)
            with redirect_stdout(io.StringIO()):ordinary.decom_larger_model(rate)
            assert_rng(self,ending)
            for name in sp.LAYERS:
                left=factors(getattr(actual,name));right=factors(getattr(ordinary,name));k,b=bounds[name]['k'],bounds[name]['b']
                if loop==0 or left['rank']==k:
                    self.assertTrue(torch.equal(left['U'],right['U']));self.assertTrue(torch.equal(left['V'],right['V']))
                else:
                    self.assertTrue(torch.equal(left['U'][:,:k],right['U'][:,:k]));self.assertTrue(torch.equal(left['V'][:k],right['V'][:k]))
                    self.assertTrue(torch.equal(left['U'][:,b:],right['U'][:,b:]));self.assertTrue(torch.equal(left['V'][b:],right['V'][b:]))
                    self.assertTrue(torch.equal(observation[name]['used'][b:],observation[name]['raw'][b:]))
                self.assertTrue(torch.equal(getattr(actual,name).bias,getattr(ordinary,name).bias))
            for name in ('conv1','fc3'):
                self.assertTrue(torch.equal(getattr(actual,name).weight,getattr(ordinary,name).weight))
            self.assertTrue(all(not r['changed'] for r in records) if loop==0 else True)
        full=Hyper_CNN_512(ratio_LR=1.);b=sp.boundary_ranks(full,a)
        self.assertEqual([(b[n]['k'],b[n]['b']) for n in sp.LAYERS],[(12,20),(120,200),(77,128)])

    def test_old_all_tail_and_none_policy_frozen_b49c54e(self):
        source=subprocess.check_output(['git','show','b49c54e:system/utils/multi_svd_spectral.py'],cwd=ROOT,text=True,encoding='utf-8')
        frozen={'__name__':'frozen_b49c54e_spectral'};exec(compile(source,'frozen_spectral.py','exec'),frozen)
        gen=torch.Generator().manual_seed(19)
        for dtype in (torch.float32,torch.float64):
            s=torch.sort(torch.rand(30,generator=gen,dtype=dtype),descending=True).values;s[24:]=0
            out,meta=sp.all_tail_spectrum(s,[4,10,24],[3,5,7]);old,oldmeta=frozen['all_tail_spectrum'](s,[4,10,24],[3,5,7])
            self.assertTrue(torch.equal(out,old));self.assertEqual(meta,oldmeta)
        for mode in ('none','all_tail'):
            a=options(multi_svd_spectral_mode=mode);self.assertEqual(sp.policy(a),frozen['policy'](a))

    def test_epoch_evaluation_state_grad_RNG_exception_and_budget(self):
        a=options(multi_svd_spectral_mode='none',multi_svd_five_diagnostics=1,
            multi_svd_local_epoch_diagnostics=1,multi_svd_local_epoch_client_ids='0')
        model=Hyper_CNN_512(n_kernels=2,ratio_LR=.25);model.register_buffer('probe',torch.tensor([3.]));model.train();model.head.eval()
        for p in model.parameters():p.grad=torch.ones_like(p)
        c=SimpleNamespace(id=0,K=1,args=a,model=model,device='cpu',dataset='Cifar100',download_round=0,local_epochs=5,
            rank_rate=.25,train_samples=3,test_samples=2,five_endpoint_rows=[],five_epoch_endpoint_rows=[],five_cost_rows=[])
        data=[(torch.randn(3,32,32),torch.tensor(i)) for i in range(3)]
        state=rng_state();weights=copy.deepcopy(model.state_dict());grads=[p.grad.clone() for p in model.parameters()];modes=[m.training for m in model.modules()]
        def read(dataset,cid,args,is_train=True,**kw):return data if is_train else data[:2]
        with patch('utils.data_utils.read_client_data',side_effect=read):
            self.assertGreater(five.after_training_epoch(c,model,0,1),0)
            with patch.object(model,'forward',side_effect=RuntimeError('injected')):
                with self.assertRaisesRegex(RuntimeError,'injected'):five.after_training_epoch(c,model,0,2)
            with patch.object(model,'forward') as forward:
                self.assertEqual(five.after_training_epoch(c,model,0,5),0);forward.assert_not_called()
        self.assertEqual(len(c.five_epoch_endpoint_rows),2);self.assertEqual(c.five_cost_rows[0]['forward_samples'],5)
        self.assertEqual(c.five_endpoint_rows,[]);self.assertEqual(modes,[m.training for m in model.modules()]);assert_rng(self,state)
        self.assertTrue(all(torch.equal(p,weights[n]) for n,p in model.state_dict().items()))
        self.assertTrue(all(torch.equal(p.grad,g) for p,g in zip(model.parameters(),grads)))
        for change in [dict(multi_svd_five_diagnostics=0),dict(multi_svd_local_epoch_client_ids='0,0'),dict(multi_svd_local_epoch_client_ids='20'),dict(multi_svd_local_epoch_client_ids='')]:
            values=dict(multi_svd_spectral_mode='none',multi_svd_five_diagnostics=1,multi_svd_local_epoch_diagnostics=1,multi_svd_local_epoch_client_ids='0');values.update(change)
            with self.assertRaises(ValueError):options(**values)

    def test_four_commands_actual_R10_and_shared_protocol(self):
        lines=[l.strip()[1:-1] for l in (ROOT/'system/svd_first_band_commands.sh').read_text().splitlines() if l.strip().startswith('"')]
        self.assertEqual(len(lines),4)
        for gpu,line in enumerate(lines):
            words=shlex.split(line);get=lambda key:words[words.index(key)+1]
            self.assertEqual(get('-did'),str(gpu));self.assertEqual(get('-gr'),'9');self.assertEqual(get('-ls'),'5')
            self.assertEqual(get('--multi_svd_local_epoch_client_ids'),'3,8,13,18,0,5,10,15')
            self.assertEqual(get('--multi_proto_mode'),'none');self.assertEqual(get('--multi_svd_init_rank_rate'),'0.90')
            self.assertEqual(get('--multi_svd_spectral_mode'),'first_band' if gpu==3 else 'none')


if __name__=='__main__':unittest.main()

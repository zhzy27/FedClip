"""Subgroup residual directions, independent stages and native layouts."""
import copy
from pathlib import Path
import sys
import unittest

import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'system'))
from utils.matrix_tail_compensation import compensate_matrix,compensate_model,matrix_weight,weight_matrix
from utils.multi_rank_partition import actual_matrix
from utils import multi_group_svd as svd
from utils import fc1_tail_diagnostic as old
from flcore.trainmodel.models import Hyper_CNN_512


class MatrixCompensationTests(unittest.TestCase):
    def setUp(self):torch.set_num_threads(2);torch.manual_seed(53)

    def test_aligned_directions_exactly_restore_Avg_dilution_unequal_samples(self):
        s=torch.tensor([40.,16.,8.,3.,1.],dtype=torch.float64)
        ranks={0:1,1:2,2:3,3:4,4:5};samples={0:1,1:2,2:3,3:4,4:5};order=[4,1,3,0,2]
        matrices={i:torch.diag(torch.cat([s[:r],torch.zeros(5-r,dtype=s.dtype)])) for i,r in ranks.items()}
        average=sum(matrices[i]*(samples[i]/15) for i in order);before={i:v.clone() for i,v in matrices.items()}
        probe=torch.nn.Linear(1,1);rng=old.rng_fingerprint(old.rng_snapshot(probe))
        out,report=compensate_matrix(average,matrices.__getitem__,ranks,samples,order)
        torch.testing.assert_close(out,torch.diag(s),rtol=0,atol=1e-13)
        self.assertEqual([r['actual_taken_rank'] for r in report['segments']],[1]*4)
        self.assertEqual(report['segments'][0]['contributor_ids'],[4,1,3,2])
        self.assertAlmostEqual(report['segments'][-1]['q'],5/15)
        self.assertTrue(all(torch.equal(before[i],matrices[i]) for i in matrices));self.assertTrue(torch.equal(average,sum(matrices[i]*(samples[i]/15) for i in order)))
        self.assertEqual(rng,old.rng_fingerprint(old.rng_snapshot(probe)))

    def test_rank_shortfall_not_filled_and_later_width_unchanged(self):
        matrices={0:torch.diag(torch.tensor([9.,0.,0.,0.],dtype=torch.float64)),
            1:torch.diag(torch.tensor([9.,6.,0.,-4.],dtype=torch.float64)),
            2:torch.diag(torch.tensor([9.,6.,0.,4.],dtype=torch.float64))}
        avg=sum(matrices.values())/3
        out,r=compensate_matrix(avg,matrices.__getitem__,{0:1,1:3,2:4},{0:1,1:1,2:1},[0,1,2])
        self.assertEqual([x['planned_rank'] for x in r['segments']],[2,1])
        self.assertEqual([x['actual_taken_rank'] for x in r['segments']],[1,1])
        self.assertEqual([x['prefix_rank_after'] for x in r['segments']],[2,3])
        torch.testing.assert_close(out,torch.diag(torch.tensor([9.,6.,0.,8/3],dtype=torch.float64)),atol=1e-13,rtol=0)

    def test_strong_tail_reorders_across_minimum_boundary(self):
        matrices={0:torch.diag(torch.tensor([5.,0.,0.],dtype=torch.float64)),1:torch.diag(torch.tensor([5.,10.,0.],dtype=torch.float64))}
        avg=.9*matrices[0]+.1*matrices[1]
        out,r=compensate_matrix(avg,matrices.__getitem__,{0:1,1:2},{0:9,1:1},[0,1])
        torch.testing.assert_close(out,torch.diag(torch.tensor([5.,10.,0.],dtype=torch.float64)),rtol=0,atol=1e-13)
        b=r['boundaries'][0];self.assertEqual(b['before_sigma_at'],5.);self.assertAlmostEqual(b['after_sigma_at'],10.)
        self.assertAlmostEqual(b['prefix_left_subspace_overlap'],0.)
        self.assertAlmostEqual(b['prefix_right_subspace_overlap'],0.)

    def test_float32_rank_threshold_zero_uniform_and_nonfinite(self):
        avg=torch.diag(torch.tensor([4.,1e-8,0.]));matrices={0:avg.clone(),1:avg.clone()}
        out,r=compensate_matrix(avg,matrices.__getitem__,{0:1,1:3},{0:2,1:1},[0,1])
        self.assertEqual(r['segments'][0]['actual_taken_rank'],0);self.assertTrue(torch.equal(out,avg))
        self.assertAlmostEqual(r['segments'][0]['rank_threshold'],3*torch.finfo(torch.float32).eps*4)
        out,r=compensate_matrix(avg,matrices.__getitem__,{0:2,1:2},{0:2,1:1},[0,1]);self.assertTrue(torch.equal(out,avg));self.assertEqual(r['segments'],[])
        bad=avg.clone();bad[0,0]=float('nan')
        with self.assertRaises(FloatingPointError):compensate_matrix(avg,lambda i:bad,{0:1,1:3},{0:1,1:1},[0,1])
        with self.assertRaises(ValueError):compensate_matrix(avg,matrices.__getitem__,{0:0,1:3},{0:1,1:1},[0,1])

    def test_model_original_conv_layout_bias_aliases_and_readonly(self):
        from utils.cnn512_reporting import model_digest
        model=Hyper_CNN_512(n_kernels=2,ratio_LR=1.);params={n:p.detach().clone() for n,p in model.named_parameters()}
        files={0:copy.deepcopy(params),1:copy.deepcopy(params)};samples={0:1,1:3}
        ranks={name:{0:1,1:2} for name in ('conv2','fc1','fc2')}
        state=old.rng_fingerprint(old.rng_snapshot(model));digest=model_digest(model)
        out,records=compensate_model(model,files,ranks,samples,[1,0])
        self.assertEqual(len(records),3);self.assertEqual(digest,model_digest(model));self.assertEqual(state,old.rng_fingerprint(old.rng_snapshot(model)))
        for name,p in model.named_parameters():
            if name not in ('conv2.weight','fc1.weight','fc2.weight'):self.assertTrue(torch.equal(p,dict(out.named_parameters())[name]),name)
        matrix=weight_matrix(model.conv2.weight,model.conv2)
        self.assertTrue(torch.equal(matrix,actual_matrix(model.conv2)))
        self.assertTrue(torch.equal(matrix_weight(matrix,model.conv2),model.conv2.weight))
        self.assertIs(out.base[3],out.conv2);self.assertIs(out.head[1],out.fc3)

    def test_unified_native_download_prefixes_match_original_per_client_SVD(self):
        from test_svd_r6_replay import args
        from utils import svd_r6_replay as replay
        from utils.whole_network_spectrum_diagnostic import truncated_model
        from contextlib import redirect_stdout
        import io
        a=args();mother=Hyper_CNN_512(n_kernels=2,ratio_LR=1.)
        with redirect_stdout(io.StringIO()):
            template=replay.paired_download(mother,.9,a,0,False)[0]
            for rate in (.9,.37,.35,.25,.15):
                actual=truncated_model(template,rate);native=replay.paired_download(mother,rate,a,0,False)[0]
                for name,p in actual.state_dict().items():self.assertTrue(torch.equal(p,native.state_dict()[name]),(rate,name))

    def test_R7_counterfactual_uses_own_pre_compensation_mother_and_rank_rows(self):
        from types import SimpleNamespace
        from unittest.mock import patch
        from test_svd_r6_replay import args
        from utils import matrix_comp_replay as d
        a=args();avg={'H':'avgH','Hplus':'avgP','matrix':'avgM'};pool={name:object() for name in avg}
        calls=[]
        def download(mother,rate,config,cid,repair):
            calls.append((mother,repair))
            receipt=[dict(client_id=0,layer='fc1',rank_corrections=[dict(rank=121,actual_gain=2.)])] if repair else []
            return 'raw_'+mother,'repaired_'+mother if repair else None,receipt
        costs=[];segments=[];boundaries=[];spectra=[];downloads=[]
        with patch.object(d.replay,'paired_download',side_effect=download),patch.object(d.comp,'compensate_model',return_value=('compM',[])) as kernel:
            templates,mothers,ordinary=d.prepare_round(avg,pool,{}, {},list(range(20)),a,
                SimpleNamespace(matrix_rank_rtol=None,matrix_rank_atol=0),7,costs,segments,boundaries,spectra,downloads)
        self.assertEqual(templates,dict(H='raw_avgH',Hplus='repaired_avgP',matrix='raw_compM'))
        self.assertEqual(ordinary,dict(Hplus='raw_avgP',matrix='raw_avgM'))
        self.assertEqual(mothers['matrix'],'compM');self.assertIs(kernel.call_args.args[1],pool['matrix'])
        self.assertEqual(calls,[('avgH',False),('avgP',True),('compM',False),('avgM',False)])
        self.assertEqual(downloads,[dict(round=7,branch='Hplus',client_id=0,layer='fc1',rank=121,actual_gain=2.)])


if __name__=='__main__':unittest.main()

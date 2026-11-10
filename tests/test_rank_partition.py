"""Original-factor partitions, original convolution layout and read-only spectra."""
import copy
from contextlib import redirect_stdout
import io
import math
from pathlib import Path
import random
import shlex
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'system'))
from test_single_avg_proto import options as base_options
from test_single_avg_class_sample import assert_rng
from common.experiment import rng_state
from utils import multi_rank_partition as rp,multi_group_svd as svd
from utils.group_projection import aggregate_group_updates
from utils.head_snapshot import tensor_fingerprint
from flcore.trainmodel.models import FactorizedConv,FactorizedLinear,Hyper_CNN_512


def options(**kwargs):
    config=dict(multi_svd_aggregation='rank_partition',multi_proto_mode='none',multi_proto_lambda=0)
    config.update(kwargs);return base_options(**config)


class RankPartitionTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        if torch.cuda.is_available():torch.cuda.get_rng_state_all()  # Existing legacy kernel's protection may query devices.

    def test_unequal_samples_multiple_blocks_single_contributor_not_mean_factors(self):
        entries=[]
        for cid,diagonal in enumerate(([1.],[3.,5.],[7.,11.,13.])):
            rank=len(diagonal);u=torch.eye(3,dtype=torch.float64)[:,:rank].clone().requires_grad_()
            v=torch.zeros(rank,3,dtype=torch.float64)
            for j,d in enumerate(diagonal):v[j,j]=d
            entries.append(dict(client_id=cid,U=u,V=v.requires_grad_(),rank=rank))
        before=[(x['U'].clone(),x['V'].clone()) for x in entries];random=rng_state()
        with patch.object(torch.linalg,'svd',side_effect=AssertionError('No re-SVD')):
            value,rows=rp.partition_product(entries,{0:1,1:3,2:6})
        torch.testing.assert_close(value,torch.diag(torch.tensor([5.2,9.,13.],dtype=torch.float64)),rtol=0,atol=1e-15)
        self.assertEqual([(r['partition_start'],r['partition_end']) for r in rows],[(0,1),(1,2),(2,3)])
        self.assertEqual([r['contributor_ids'] for r in rows],[[0,1,2],[1,2],[2]])
        self.assertEqual([r['contributor_train_samples_total'] for r in rows],[10,9,6])
        self.assertTrue(all(abs(r['weight_sum']-1.)<1e-15 for r in rows));self.assertFalse(value.requires_grad)
        for pair,item in zip(before,entries):
            self.assertTrue(torch.equal(pair[0],item['U']));self.assertTrue(torch.equal(pair[1],item['V']))
            self.assertIsNone(item['U'].grad);self.assertIsNone(item['V'].grad)
        assert_rng(self,random,rng_state())
        with self.assertRaises(ValueError):rp.partition_product(entries,{0:1,1:0,2:6})
        bad=copy.copy(entries[0]);bad['rank']=2
        with self.assertRaises(ValueError):rp.partition_product([bad],{0:1})

    def test_original_convolution_permutation_and_actual_rounding_per_layer(self):
        layer=FactorizedConv(2,3,.5,kernel_size=2,bias=True).double()
        with torch.no_grad():layer.conv_u.copy_(torch.arange(12).reshape(6,2));layer.conv_v.copy_(torch.arange(8).reshape(2,4))
        matrix=layer.conv_u.detach()@layer.conv_v.detach()
        self.assertTrue(torch.equal(rp.matrix_to_weight(matrix,layer),layer.reconstruct_full_weight()))
        dense=torch.nn.Conv2d(2,3,2).double()
        with torch.no_grad():dense.weight.copy_(layer.reconstruct_full_weight())
        self.assertTrue(torch.equal(rp.actual_matrix(dense),matrix))
        profiles=[]
        for ratio in (.9,.37,.35,.25,.15):
            m=Hyper_CNN_512(ratio_LR=ratio);profiles.append({name:rp.factors(getattr(m,name))['rank'] for name in rp.LAYERS})
        self.assertEqual([p['conv2'] for p in profiles],[72,30,28,20,12])
        self.assertEqual([p['fc1'] for p in profiles],[720,296,280,200,120])
        self.assertEqual([p['fc2'] for p in profiles],[461,189,179,128,77])

    def _server(self,n=2,ratios=None):
        args=options(num_clients=n,multi_rank_spectrum=0)
        mother=Hyper_CNN_512(n_kernels=2,ratio_LR=1.);clients=[]
        for cid in range(n):
            post=Hyper_CNN_512(n_kernels=2,ratio_LR=(ratios or [.25]*n)[cid])
            full=svd.recover_parameters(post)
            c=SimpleNamespace(id=cid,train_samples=cid+1,model_set=[post],upload_round=0)
            c.full_upload_parameters=lambda column,device=None,full=full:{k:v.to(device or 'cpu') for k,v in full.items()}
            clients.append(c)
        s=SimpleNamespace(K=1,args=args,groups=[list(range(n))],num_clients=n,selected_clients=clients,clients=clients,
            loop_round=0,global_model_set=[mother],device='cpu',aggregation_records=[],weight_records=[])
        rp.initialize(s);return s

    def test_uniform_and_single_client_exact_legacy_bias_alias_and_no_recovery_or_svd(self):
        for n in (1,3):
            s=self._server(n);base=dict(s.global_model_set[0].named_parameters());mass=sum(c.train_samples for c in s.clients)
            samples={c.id:c.train_samples/mass for c in s.clients}
            factory=lambda:iter((c.id,samples[c.id],c.full_upload_parameters(0)) for c in s.clients)
            expected=aggregate_group_updates(base,factory,samples,list(range(n)),'group_avg_only')[0]
            before=[(tensor_fingerprint(c.model_set[0].state_dict()),[m.training for m in c.model_set[0].modules()]) for c in s.clients]
            random=rng_state()
            with patch.object(torch.linalg,'svd',side_effect=AssertionError('No SVD')),\
                 patch.object(Hyper_CNN_512,'recover_larger_model',side_effect=AssertionError('Do not recover/re-decompose factor inputs')),redirect_stdout(io.StringIO()):
                rp.aggregate(s)
            for name,p in s.global_model_set[0].named_parameters():self.assertTrue(torch.equal(p,expected[name]),name)
            if n==1:
                for name,p in s.global_model_set[0].named_parameters():self.assertTrue(torch.equal(p,s.clients[0].full_upload_parameters(0)[name]))
            for c,(sha,modes) in zip(s.clients,before):
                self.assertEqual(tensor_fingerprint(c.model_set[0].state_dict()),sha);self.assertEqual([m.training for m in c.model_set[0].modules()],modes)
            self.assertEqual(len(s.rank_partition_records),3);self.assertTrue(all(r['uniform_rank_legacy_fallback'] for r in s.rank_partition_records))
            self.assertEqual([r['canonical_weight_name'] for r in s.rank_partition_records],['conv2.weight','fc1.weight','fc2.weight'])
            assert_rng(self,random,rng_state())

    def test_heterogeneous_bias_head_ordinary_Avg_and_original_factor_snapshot(self):
        s=self._server(2,[.25,.9]);mass=3
        ordinary={name:sum(c.full_upload_parameters(0)[name]*(c.train_samples/mass) for c in s.clients) for name,_ in s.global_model_set[0].named_parameters()}
        expected={}
        for layer in rp.LAYERS:
            entries=[dict(client_id=c.id,**rp.factors(getattr(c.model_set[0],layer))) for c in s.clients]
            product,_=rp.partition_product(entries,{c.id:c.train_samples for c in s.clients})
            expected[layer+'.weight']=rp.matrix_to_weight(product,getattr(s.clients[0].model_set[0],layer))
        with redirect_stdout(io.StringIO()):rp.aggregate(s)
        for name,p in s.global_model_set[0].named_parameters():
            target=expected.get(name,ordinary[name]);torch.testing.assert_close(p,target,rtol=1e-6,atol=1e-7)
        self.assertFalse(torch.equal(dict(s.global_model_set[0].named_parameters())['fc2.weight'],ordinary['fc2.weight']))

    def test_spectrum_actual_singular_energy_and_rng_readonly(self):
        s=self._server(2);s.args.multi_rank_spectrum=1;s.args.multi_rank_effective_spectrum_rounds=[1]
        s.args.multi_rank_spectrum_client_id=0
        for layer in rp.LAYERS:
            linear=torch.nn.Linear(3,3,bias=False).double()
            with torch.no_grad():linear.weight.copy_(torch.diag(torch.tensor([4.,3.,2.],dtype=torch.float64)))
            setattr(s.global_model_set[0],layer,linear)
            for c in s.clients:setattr(c.model_set[0],layer,FactorizedLinear(3,3,1/3))
        before=tensor_fingerprint(s.global_model_set[0].state_dict());modes=[m.training for m in s.global_model_set[0].modules()];random_before=rng_state()
        native=torch.linalg.svdvals
        def spectrum(matrix):random.random();np.random.rand();torch.rand(1);return native(matrix)
        with patch.object(torch.linalg,'svdvals',spectrum):rp.observe(s,'server_post_aggregation')
        self.assertEqual(len(s.rank_spectrum_records),3)
        for row in s.rank_spectrum_records:
            self.assertEqual(row['minimum_actual_client_rank'],1);self.assertEqual(row['spectral_energy_within_min_rank'],16.)
            self.assertEqual(row['spectral_energy_outside_min_rank'],13.);self.assertAlmostEqual(row['actual_matrix_frobenius_norm'],math.sqrt(29))
        self.assertEqual(before,tensor_fingerprint(s.global_model_set[0].state_dict()));self.assertEqual(modes,[m.training for m in s.global_model_set[0].modules()])
        assert_rng(self,random_before,rng_state())

    def test_options_bounds_sampling_final_and_unsupported_combinations(self):
        a=options(global_rounds=100);self.assertEqual(a.multi_rank_effective_spectrum_rounds,[1,10,50,101])
        a=options(global_rounds=2,multi_rank_spectrum_rounds='2');self.assertEqual(a.multi_rank_effective_spectrum_rounds,[2,3])
        options(multi_proto_mode='global_last_epoch',multi_proto_lambda=1,multi_proto_aggregation='class_sample')
        for values in [dict(num_server_models=3),dict(multi_avg_proto_ablation=False),dict(multi_rank_spectrum_rounds='0'),
            dict(multi_rank_spectrum_rounds='1,1'),dict(multi_rank_spectrum_client_id=20),dict(multi_rank_spectrum=0,multi_rank_spectrum_rounds='1'),
            dict(multi_proto_mode='global_pre_train',multi_proto_lambda=1,multi_proto_aggregation='class_sample'),
            dict(multi_proto_mode='global_last_epoch',multi_proto_lambda=1),dict(multi_svd_ratios='1.0')]:
            with self.subTest(values=values),self.assertRaises(ValueError):options(**values)

    def test_sum_can_exceed_max_client_algebraic_rank_without_server_truncation(self):
        eye=torch.eye(3,dtype=torch.float64)
        entries=[dict(client_id=0,U=eye[:,:1],V=eye[:1],rank=1),
                 dict(client_id=1,U=eye[:,1:],V=eye[1:],rank=2)]
        value,_=rp.partition_product(entries,{0:1,1:1})
        self.assertTrue(torch.equal(value,torch.diag(torch.tensor([.5,.5,1.],dtype=torch.float64))))
        self.assertEqual(int(torch.linalg.matrix_rank(value)),3)

    def test_two_new_and_two_optional_commands_exact_public_configuration(self):
        text=(ROOT/'system/rank_partition_commands.sh').read_text()
        self.assertIn('declare -a COMMANDS=(',text);self.assertIn('declare -a OPTIONAL_CONTROL_COMMANDS=(',text)
        commands=[shlex.split(line.strip()[1:-1]) for line in text.splitlines() if line.strip().startswith('"')]
        self.assertEqual(len(commands),4);roots=[]
        for i,command in enumerate(commands):
            self.assertEqual(command[3:5],['-did',str(i)]);self.assertNotIn('$',' '.join(command))
            for flag,value in [('-gr','100'),('-ls','5'),('-nsm','1'),('-nc','20'),('--local_grad_clip','10'),
                ('--multi_svd_aggregation','rank_partition' if i<2 else 'full_avg'),
                ('--multi_proto_mode','global_last_epoch' if i%2 else 'none'),
                ('--multi_proto_lambda','1.0' if i%2 else '0.0'),('--multi_proto_aggregation','class_sample')]:
                self.assertEqual(command[command.index(flag)+1],value)
            self.assertNotIn('global_pre_train',command);self.assertNotIn('--multi_group_file',command)
            roots.append(command[command.index('-sfn')+1])
        self.assertEqual(len(set(roots)),4)


if __name__=='__main__':unittest.main()

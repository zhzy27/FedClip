"""Opt-in fc1 download gauges; zero path, same-SVD offline parity and next round."""
import ast
from contextlib import redirect_stdout
import copy
import io
import math
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'system'))
from test_single_avg_proto import options as base_options
from test_control_population import assert_rng
from common.experiment import rng_state,set_rng
from utils import multi_fc1_tail_scale as tail
from utils import multi_group_svd as svd
from utils import fc1_tail_diagnostic as offline
from utils.cnn512_reporting import model_digest
from utils.head_snapshot import tensor_fingerprint
from flcore.clients.clientMultiGroupSVD import clientMultiGroupSVD
from flcore.trainmodel.models import Hyper_CNN_512,factorized_linear_rank


def options(**changes):
    values=dict(device='cpu',multi_proto_mode='none',multi_proto_lambda=0.,multi_fc1_tail_scale_strength=1.,local_epochs=2,batch_size=2)
    values.update(changes);return base_options(**values)


def frozen_modules():
    code=subprocess.check_output(['git','show','17e5ebb:system/flcore/trainmodel/models.py'],cwd=ROOT,text=True,encoding='utf-8')
    names={'FactorizedConv','Decom_COV','Recover_COV','FactorizedLinear','Decom_LINEAR','Recover_LINEAR','Hyper_CNN_512'}
    tree=ast.parse(code);tree.body=[node for node in tree.body if isinstance(node,(ast.ClassDef,ast.FunctionDef)) and node.name in names]
    ns=dict(torch=torch,nn=nn,F=F,math=math)
    exec(compile(ast.fix_missing_locations(tree),'17e5ebb/models.py','exec'),ns)
    return ns


def frozen_B():
    code=subprocess.check_output(['git','show','17e5ebb:system/utils/fc1_tail_diagnostic.py'],cwd=ROOT,text=True,encoding='utf-8')
    tree=ast.parse(code);tree.body=[node for node in tree.body if isinstance(node,ast.FunctionDef) and node.name=='rescale_tail']
    ns=dict(copy=copy,torch=torch,finite=offline.finite)
    exec(compile(ast.fix_missing_locations(tree),'17e5ebb/rescale_tail.py','exec'),ns)
    return ns['rescale_tail']


class TailScaleTests(unittest.TestCase):
    def setUp(self):torch.set_num_threads(2);torch.random.default_generator.manual_seed(19)

    def test_options_explicit_scope_and_finite_nonnegative_strength(self):
        self.assertEqual(options().multi_fc1_tail_scale_strength,1.)
        self.assertEqual(options(multi_fc1_tail_scale_strength=2.).multi_fc1_tail_scale_strength,2.)
        for changes in [dict(multi_fc1_tail_scale_strength=-1),dict(multi_fc1_tail_scale_strength=float('nan')),
            dict(multi_fc1_tail_scale_strength=float('inf')),dict(multi_svd=False,is_regular=0),dict(multi_svd_aggregation='rank_partition'),
            dict(multi_proto_mode='global_last_epoch',multi_proto_lambda=1.,multi_proto_aggregation='class_sample'),
            dict(multi_proto_mode='global_last_epoch',multi_proto_lambda=0),dict(target_curriculum_mode='random_rank'),
            dict(num_server_models=3),dict(multi_svd_ratios='1.,.15'),dict(multi_group_mode='group_projection_softmax')]:
            with self.subTest(changes=changes),self.assertRaises(ValueError):options(**changes)
        # Off never expands restrictions on existing combinations.
        self.assertFalse(tail.enabled(options(multi_fc1_tail_scale_strength=0,multi_proto_mode='global_last_epoch',multi_proto_lambda=1.)))
        with self.assertRaises(ValueError):tail.validate(type('Args',(),dict(algorithm='FedTargetProj',multi_fc1_tail_scale_strength=1))())

    def test_configurable_floor_strength_zero_singular_and_zero_floor(self):
        from utils.fc1_tail_factors import scales
        singular=torch.tensor([4.,1.,.25,0.,0.],dtype=torch.float64)
        for strength,expected in [(.5,[.5,.5,.5]),(1.,[1.,1.,1.]),(2.,[2.,2.,2.])]:
            a,b,floor=scales(singular,2,5,strength)
            torch.testing.assert_close(a,torch.tensor(expected,dtype=torch.float64),rtol=0,atol=0)
            torch.testing.assert_close(a*b,singular[2:],rtol=0,atol=0)
            self.assertEqual(float(floor),strength)
        a,b,floor=scales(torch.zeros_like(singular),2,5,1.)
        self.assertEqual(float(floor),0.);self.assertTrue(torch.equal(a,torch.zeros(3,dtype=torch.float64)))
        self.assertTrue(torch.equal(b,torch.zeros(3,dtype=torch.float64)))

    def test_default_model_functions_initialization_and_rng_match_frozen_source(self):
        old=frozen_modules();before=rng_state()
        current=Hyper_CNN_512(n_kernels=2,num_classes=100,ratio_LR=.15);current.recover_larger_model();current.ratio_LR=1.
        after=rng_state();set_rng(before)
        frozen=old['Hyper_CNN_512'](n_kernels=2,num_classes=100,ratio_LR=.15);frozen.recover_larger_model();frozen.ratio_LR=1.
        self.assertEqual(model_digest(current),model_digest(frozen));assert_rng(self,after)
        before=rng_state()
        with redirect_stdout(io.StringIO()):current.decom_larger_model(.35)
        after=rng_state();set_rng(before)
        with redirect_stdout(io.StringIO()):frozen.decom_larger_model(.35)
        self.assertEqual(model_digest(current),model_digest(frozen));assert_rng(self,after)
        self.assertEqual([factorized_linear_rank(800,2000,r) for r in (.9,.37,.35,.25,.15)],[720,296,280,200,120])

    def test_all_five_downloads_same_svd_offline_B_products_logits_counts_and_optimizer(self):
        args=options();mother=svd.initial_model(args);data=torch.randn(3,3,32,32);old_B=frozen_B()
        with tempfile.TemporaryDirectory(dir='E:/' if Path('E:/').is_dir() else None) as folder,redirect_stdout(io.StringIO()):
            for cid in range(5):
                a=copy.deepcopy(args);a.save_folder_name_full=folder
                c=clientMultiGroupSVD(a,cid,3,3,initial_model=mother,train_slow=False,send_slow=False)
                state=rng_state();observed=[];original=torch.linalg.svd
                def count(matrix,*p,**kw):observed.append(tuple(matrix.shape));return original(matrix,*p,**kw)
                with patch.object(torch.linalg,'svd',side_effect=count):c.set_parameters([mother],0)
                self.assertEqual(len(observed),3);self.assertEqual(observed.count((2000,800)),1)
                after=rng_state();set_rng(state);download,evidence=offline.download_once(mother,c.rank_rate)
                expected=old_B(download,evidence,120)[0] if cid!=4 else download
                self.assertEqual(model_digest(c.model_set[0]),model_digest(expected));assert_rng(self,after)
                native=download.eval();actual=c.model_set[0].eval()
                self.assertTrue(torch.allclose(offline.full_matrix(native.fc1),offline.full_matrix(actual.fc1),atol=1e-6,rtol=1e-5))
                with torch.no_grad():self.assertTrue(torch.allclose(native(data),actual(data),atol=1e-6,rtol=1e-5))
                self.assertEqual(sum(p.numel() for p in native.parameters()),sum(p.numel() for p in actual.parameters()))
                self.assertTrue(torch.equal(native.fc1.weight_u[:,:120],actual.fc1.weight_u[:,:120]))
                self.assertTrue(torch.equal(native.fc1.weight_v[:120],actual.fc1.weight_v[:120]))
                for name,p in native.named_parameters():
                    if name not in ('fc1.weight_u','fc1.weight_v'):self.assertTrue(torch.equal(p,dict(actual.named_parameters())[name]))
                self.assertEqual([id(p) for p in actual.parameters()],[id(p) for g in c.optimizer_set[0].param_groups for p in g['params']])
                receipt=c.fc1_tail_scale_latest[0]
                self.assertEqual(receipt['minimum_actual_fc1_rank'],120)
                self.assertEqual(receipt['tail_slot_count'],max(actual.fc1.rank-120,0))
                self.assertEqual(receipt['effective_strength'],1. if cid!=4 else 0.)

    def test_zero_path_does_not_enter_gauge_or_create_receipts(self):
        args=options(multi_fc1_tail_scale_strength=0);mother=svd.initial_model(args)
        source=subprocess.check_output(['git','show','17e5ebb:system/flcore/clients/clientMultiGroupSVD.py'],cwd=ROOT,text=True,encoding='utf-8')
        ns={'__name__':'frozen_17e5ebb_client'};exec(compile(source,'17e5ebb/clientMultiGroupSVD.py','exec'),ns)
        with tempfile.TemporaryDirectory(dir='E:/' if Path('E:/').is_dir() else None) as folder,redirect_stdout(io.StringIO()):
            args.save_folder_name_full=folder;results=[];base=rng_state()
            for cls in (ns['clientMultiGroupSVD'],clientMultiGroupSVD):
                set_rng(base);client=cls(copy.deepcopy(args),2,3,3,initial_model=mother,train_slow=False,send_slow=False)
                with patch.object(tail,'decompose',side_effect=AssertionError('Zero must bypass gauge')):client.set_parameters([mother],0)
                self.assertFalse(hasattr(client,'fc1_tail_scale_records'))
                results.append((model_digest(client.model_set[0]),rng_state()))
            self.assertEqual(results[0][0],results[1][0]);assert_rng(self,results[0][1])

    def test_every_round_rebuild_current_optimizer_and_current_svd_floor(self):
        args=options();mother=svd.initial_model(args);dataset=[(torch.randn(3,32,32),torch.tensor(i)) for i in range(3)]
        with tempfile.TemporaryDirectory(dir='E:/' if Path('E:/').is_dir() else None) as folder,redirect_stdout(io.StringIO()):
            args.save_folder_name_full=folder;c=clientMultiGroupSVD(args,2,3,3,initial_model=mother,train_slow=False,send_slow=False)
            c.set_parameters([mother],0);first=c.fc1_tail_scale_latest[0];parameters=list(c.model_set[0].parameters())
            before=tensor_fingerprint(c.model_set[0].state_dict())
            with patch('flcore.clients.clientMultiGroupProj.read_client_data',return_value=dataset):c.train()
            self.assertNotEqual(before,tensor_fingerprint(c.model_set[0].state_dict()))
            self.assertEqual(c.train_stats[0]['train_batches'],4);self.assertEqual(c.train_stats[0]['processed_train_samples'],6)
            with torch.no_grad():mother.fc1.weight.add_(torch.eye(2000,800)*.3)
            c.set_parameters([mother],1);second=c.fc1_tail_scale_latest[0]
            self.assertNotEqual(first['source_server_sha256'],second['source_server_sha256'])
            self.assertNotEqual(first['floor_singular_value'],second['floor_singular_value'])
            self.assertEqual(second['loop_round'],1);self.assertEqual(len(c.fc1_tail_scale_records),2)
            self.assertTrue(all(p is not q for p,q in zip(parameters,c.model_set[0].parameters())))
            self.assertEqual([id(p) for p in c.model_set[0].parameters()],[id(p) for g in c.optimizer_set[0].param_groups for p in g['params']])

    def test_two_complete_command_definitions_B_first_full_avg_and_no_proto(self):
        path=ROOT/'system/fc1_tail_scale_commands.sh';text=path.read_text()
        lines=[line.strip()[1:-1] for line in text.splitlines() if line.strip().startswith('"')]
        self.assertEqual(len(lines),2)
        for index,line in enumerate(lines):
            words=shlex.split(line);value=lambda key:words[words.index(key)+1]
            self.assertEqual(words[3],'-did');self.assertEqual(value('-gr'),'100')
            self.assertEqual(value('--multi_fc1_tail_scale_strength'),'1.0' if index==0 else '0.0')
            self.assertEqual(value('--multi_svd_aggregation'),'full_avg');self.assertEqual(value('--multi_proto_mode'),'none')
            self.assertEqual(value('--multi_proto_lambda'),'0');self.assertEqual(value('--local_grad_clip'),'10')
            self.assertEqual(value('-ls'),'5');self.assertEqual(value('-lr'),'0.005');self.assertEqual(value('-lbs'),'16')
            self.assertEqual(value('--multi_svd_ratios'),'0.90,0.37,0.35,0.25,0.15');self.assertNotIn('$',line)


if __name__=='__main__':unittest.main()

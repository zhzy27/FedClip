"""Explicit population protocol, unequal counts and read-only full CNN evaluation."""
import copy
import hashlib
import io
from contextlib import redirect_stdout
from pathlib import Path
import random
import sys
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'system'))
from common.models import CNN_512
from common.metrics import summarize_evaluations
from common.experiment import rng_state
from flcore.clients.clientTargetProj import clientTargetProj
from flcore.clients.clientbase import load_item, save_item
from flcore.servers.serverTargetGroup import downloaded_group_accuracy
from utils.cnn512_control import validate_control
from utils.cnn512_reporting import primary_point
from utils.group_projection import group_accuracy_summary

IDS=[0,1,2,3,4,6,7]


def options(**kwargs):
    values=dict(algorithm='FedTargetProj',model_family='CNN-512',dataset='Cifar100',num_classes=100,
        num_clients=20,target_client_id=0,seed=0,cnn512_control=True,target_proj_mode='avg',
        projection_self_weight=None,local_grad_clip=None,join_ratio=1.,random_join_ratio=False,
        client_drop_rate=0.,train_slow_rate=0.,send_slow_rate=0.,auto_break=False,learning_rate_decay=False,
        resume=False,few_shot=0,is_regular=0,target_prox_mu=0.,meta_c0_split='',meta_collect_snapshots=False,
        target_head_mode='shared',target_local_schedule='joint',head_diag_collect=False,time_select=False,
        local_learning_rate=.005,batch_size=16,local_epochs=5,eval_gap=1,control_test_batch_size=16,
        data_root='existing',data_subdir='pat_20',data_formats='npz_only',control_eval_scope='all',
        target_group_ids=None,target_group_weight=None,num_server_models=3,smooth_mu=.01,
        fedfews_test_batch_size=None)
    values.update(kwargs)
    return SimpleNamespace(**values)


def assert_rng(test, before):
    after=rng_state()
    test.assertEqual(before['python'],after['python'])
    test.assertEqual(before['numpy'][0],after['numpy'][0])
    np.testing.assert_array_equal(before['numpy'][1],after['numpy'][1])
    test.assertEqual(before['numpy'][2:],after['numpy'][2:])
    test.assertTrue(torch.equal(before['torch'],after['torch']))
    test.assertEqual(len(before['cuda']),len(after['cuda']))
    for x,y in zip(before['cuda'],after['cuda']): test.assertTrue(torch.equal(x,y))


class PopulationTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        # Protected evaluation may lazily query CUDA states even on CPU.
        if torch.cuda.is_available(): torch.cuda.get_rng_state_all()

    def test_scope_validation_new_protocol_only_fixed_group_and_mass(self):
        validate_control(options())
        validate_control(options(algorithm='FedFewS'))
        validate_control(options(target_proj_mode='group_projection_softmax',control_eval_scope='group',
                                 target_group_ids=IDS,target_group_weight=.5))
        for changed in [dict(cnn512_control=False),dict(control_eval_scope='invalid'),dict(control_eval_scope='group'),
            dict(target_proj_mode='group_projection_softmax',target_group_ids=IDS,target_group_weight=.5),
            dict(target_proj_mode='group_projection_softmax',control_eval_scope='group',target_group_ids=IDS,target_group_weight=1.35),
            dict(target_proj_mode='group_projection_softmax',control_eval_scope='group',target_group_ids=[0,20],target_group_weight=.5)]:
            with self.subTest(changed=changed),self.assertRaises(ValueError): validate_control(options(**changed))
        # Previous single-target command still uses exactly the default scope.
        a=options(control_eval_scope='c0',target_proj_mode='projection_softmax',projection_self_weight=.125)
        validate_control(a)
        self.assertEqual(a.control_eval_scope,'c0')

    def test_group_projection_kernel_is_unchanged_from_7f72099(self):
        import ast
        import subprocess
        path='system/utils/group_projection.py'
        before=subprocess.check_output(['git','show','7f72099:'+path],cwd=ROOT).decode('utf-8')
        current=(ROOT/path).read_text(encoding='utf-8')
        def kernel(source):
            return next(n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef) and n.name=='aggregate_group_updates')
        import utils.group_projection as module
        namespace=dict(vars(module))
        exec(compile(ast.Module(body=[kernel(before)],type_ignores=[]),'frozen_7f','exec'),namespace)
        initial={'head.weight':torch.tensor([2.,3.],dtype=torch.float64)}
        posts=[{'head.weight':torch.tensor([2.+i,4.-i],dtype=torch.float64)} for i in range(4)]
        factory=lambda:iter((i,.25,posts[i]) for i in [2,0,3,1])
        args=(initial,factory,dict(enumerate([.25]*4)),[0,1],'group_projection_softmax',.5)
        old=namespace['aggregate_group_updates'](*args)
        new=module.aggregate_group_updates(*args)
        self.assertTrue(torch.equal(old[0]['head.weight'],new[0]['head.weight']))
        self.assertEqual(old[1:],new[1:])

    def test_formal_commands_have_exactly_three_populations_and_old_file_unchanged(self):
        import re
        import shlex
        import subprocess
        text=(ROOT/'system/cnn512_population_commands.sh').read_text(encoding='utf-8')
        commands=re.findall(r'^"([^\n]+)"$',text,re.MULTILINE)
        self.assertEqual(len(commands),3)
        for index,command in enumerate(commands):
            args=shlex.split(command)
            for flag,value in [('-m','CNN-512'),('-gr','100'),('-ls','5'),('-nc','20'),
                ('-ncl','100'),('-lr','0.005'),('-lbs','16'),('-eg','1'),('-is_regular','0'),
                ('--local_grad_clip','10'),('--data-formats','npz_only'),('--data-subdir','pat_20')]:
                self.assertEqual(args[args.index(flag)+1],value)
            self.assertNotIn('--projection_self_weight',args)
            self.assertEqual(args[args.index('--control_eval_scope')+1],'group' if index==1 else 'all')
            if index==1:
                self.assertEqual(args[args.index('--target_group_weight')+1],'0.50')
                self.assertEqual(args[args.index('--target_group_ids')+1],','.join(map(str,IDS)))
            else:
                self.assertNotIn('--target_group_ids',args)
                self.assertNotIn('--target_group_weight',args)
        old=subprocess.check_output(['git','show','7f72099:system/cnn512_control_commands.sh'],cwd=ROOT).decode('utf-8')
        self.assertEqual(old,(ROOT/'system/cnn512_control_commands.sh').read_text(encoding='utf-8'))

    def test_unequal_counts_reduction_all_and_group_and_strict_membership(self):
        for scope,ids in [('all',list(range(20))),('group',IDS)]:
            args=options(control_eval_scope=scope,target_group_ids=IDS)
            counts=[dict(client_id=i,correct=i%3,test_samples=i+3) for i in ids]
            point=primary_point(args,5,counts)
            expected=sum(c['correct'] for c in counts)/sum(c['test_samples'] for c in counts)
            macro=sum(c['correct']/c['test_samples'] for c in counts)/len(counts)
            self.assertEqual(point['test_acc_percent'],100*expected)
            self.assertNotAlmostEqual(expected,macro)
            self.assertEqual((point['loop_round'],point['completed_train_count']),(5,6))
            with self.assertRaises(ValueError): primary_point(args,5,counts[:-1])
            with self.assertRaises(ValueError): primary_point(args,5,counts+counts[:1])

    def test_best_is_same_round_weighted_and_earliest_tie_not_member_bests(self):
        for scope,ids in [('all',list(range(20))),('group',IDS)]:
            args=options(control_eval_scope=scope,target_group_ids=IDS)
            points=[]; history=[]
            for loop in range(3):
                counts=[dict(client_id=i,correct=(1 if i==0 else 0) if loop==0 else (0 if i==0 else 9),
                             test_samples=1 if i==0 else 9) for i in ids]
                points.append(primary_point(args,loop,counts))
                members=[dict(client_id=r['client_id'],post_local_acc=r['correct']/r['test_samples'],
                              test_samples=r['test_samples'],post_local_correct=r['correct']) for r in counts]
                history.append(dict(mode='group_projection_softmax',round=loop+1,
                    primary_average='test_sample_weighted_average',group_post_local_acc=sum(m['post_local_acc'] for m in members)/len(ids),
                    group_post_local_sample_weighted_acc=points[-1]['test_acc_percent']/100,members=members))
            s=summarize_evaluations(points,value_key='test_acc_percent',unit='percent',evaluation_object=scope+'_ordinary_post_local')
            self.assertEqual(s['best_loop_round'],1)
            self.assertEqual(s['best_completed_train_count'],2)
            self.assertLess(s['best'],100.)
            g=group_accuracy_summary(history)['group_post_local']
            self.assertEqual(g['best_round'],2)
            self.assertEqual(g['best'],points[1]['test_acc_percent']/100)
            self.assertEqual(g['last20_count'],3)
        # Old protocol remains macro primary, and keeps its old summary fields.
        for row in history: row.pop('primary_average')
        old=group_accuracy_summary(history)['group_post_local']
        self.assertNotIn('last20',old)

    def test_full_group_and_post_local_evaluations_preserve_model_checkpoint_rng(self):
        with tempfile.TemporaryDirectory() as temp,redirect_stdout(io.StringIO()):
            client=clientTargetProj.__new__(clientTargetProj)
            client.args=options();client.device='cpu';client.role='Client_0';client.save_folder_name=temp
            model=CNN_512(out_dim=100)
            model.register_buffer('buffer_probe',torch.tensor([7.]))
            model.train();model.head.eval()
            server=copy.deepcopy(model)
            with torch.no_grad(): server.head[1].weight.add_(.1)
            save_item(model,client.role,'model',temp);save_item(server,'Server','model',temp)
            checkpoint=Path(temp)/'Client_0_model.pt'
            checksum=hashlib.sha256(checkpoint.read_bytes()).hexdigest()
            baseline=copy.deepcopy(model.state_dict())
            def batches():
                random.random();np.random.rand()
                return [(torch.randn(3,3,32,32),torch.tensor([0,1,2]))]
            client.load_test_data=batches
            for evaluate in [client.test_post_local,lambda:downloaded_group_accuracy(client)]:
                before=rng_state()
                evaluate()
                assert_rng(self,before)
                self.assertEqual(hashlib.sha256(checkpoint.read_bytes()).hexdigest(),checksum)
                current=load_item(client.role,'model',temp)
                for name,value in baseline.items(): self.assertTrue(torch.equal(value,current.state_dict()[name]),name)
                self.assertTrue(current.training);self.assertFalse(current.head.training)


if __name__=='__main__': unittest.main()

"""Frozen downloaded helper heads, CE gradient path, and frozen 6fe2a52 train reference."""
import copy
from contextlib import ExitStack,redirect_stdout
import io
from pathlib import Path
import random
import shlex
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'system'))
from test_multi_last_epoch_proto import options
from test_multi_shared_proto import fake
from test_control_population import assert_rng
from common.experiment import rng_state,set_rng
from utils import multi_group_head as heads,multi_last_epoch_proto as post,single_avg_proto as shared
from utils.multi_group_svd import load_checkpoint
from utils.head_snapshot import tensor_fingerprint
from flcore.clients.clientMultiGroupSVD import clientMultiGroupSVD
from flcore.servers.serverMultiGroupProj import FedMultiGroupProj
from flcore.trainmodel.models import Hyper_CNN_512


class HelperHeadTests(unittest.TestCase):
    def setUp(self):torch.set_num_threads(2)

    def test_validation(self):
        options(multi_head_mode='group_only_reset',multi_projection_scope='backbone',multi_helper_head_update='frozen')
        for kw in [dict(multi_helper_head_update='bad'),dict(multi_helper_head_update='frozen'),
            dict(multi_helper_head_update='frozen',multi_head_mode='group_only_local',multi_projection_scope='backbone'),
            dict(multi_helper_head_update='frozen',multi_head_mode='group_only_reset',multi_projection_scope='full')]:
            with self.subTest(kw=kw),self.assertRaises(ValueError):options(**kw)

    def test_one_formal_command_keeps_paired_protocol(self):
        lines=(ROOT/'system/multi_helper_head_commands.sh').read_text().splitlines()
        commands=[shlex.split(s.strip().strip('"')) for s in lines if s.strip().startswith('"')]
        self.assertEqual(len(commands),1);cmd=commands[0]
        self.assertEqual(cmd[0],'/data/zh/miniconda3/envs/htfllib/bin/python');self.assertEqual(cmd[3],'-did')
        for key,value in [('--multi_head_mode','group_only_reset'),('--multi_projection_scope','backbone'),
            ('--multi_helper_head_update','frozen'),('--multi_proto_scope','group_priority'),('--multi_proto_bank_mode','per_model'),
            ('--multi_proto_loss','mse'),('--multi_proto_target_type','raw_mean'),('--multi_proto_feature','base_output'),
            ('--multi_proto_lambda','1.0'),('--local_grad_clip','10'),('-gr','100'),('-nsm','3'),('-nc','20'),('-ls','5')]:
            self.assertEqual(cmd[cmd.index(key)+1],value)

    def trajectory(self,folder,update='train',legacy=False,diagnostics=1,observe=True,ce_only=False,omitted=False):
        a=options(num_clients=3,num_server_models=3,local_epochs=1,batch_size=2,local_grad_clip=10,
            multi_head_mode='group_only_reset',multi_projection_scope='backbone',multi_helper_head_update=update,
            multi_proto_diagnostics=diagnostics,multi_svd_diagnostics=diagnostics,
            **(dict(is_regular=0,multi_proto_mode='none',multi_proto_lambda=0.,multi_proto_target_type='fixed_norm') if ce_only else {}))
        if omitted:delattr(a,'multi_helper_head_update')
        a.device='cpu';a.save_folder_name_full=str(folder)
        random.seed(31);np.random.seed(31);torch.manual_seed(31)
        mother=Hyper_CNN_512(n_kernels=2,num_classes=100,ratio_LR=.15);mother.recover_larger_model();mother.ratio_LR=1.
        x=torch.randn(4,3,32,32)*.2;data=[list(zip(x[:3+i%2],torch.tensor([i,i+1,i+2,i+3][:3+i%2]))) for i in range(3)]
        client_cls=clientMultiGroupSVD;aggregate=FedMultiGroupProj.aggregate_parameters;runtime=post;old_post=None
        if legacy:
            scopes={}
            for name,path in [('common','utils/single_avg_proto.py'),('post','utils/multi_last_epoch_proto.py'),
                ('client','flcore/clients/clientMultiGroupSVD.py'),('server','flcore/servers/serverMultiGroupProj.py'),('heads','utils/multi_group_head.py')]:
                d={};source=subprocess.check_output(['git','show',f'6fe2a52:system/{path}'],cwd=ROOT,text=True)
                exec(compile(source,'6fe2a52/'+path,'exec'),d);scopes[name]=d
            scopes['post']['shared']=SimpleNamespace(**scopes['common']);old_post=scopes['post']
            client_cls=scopes['client']['clientMultiGroupSVD'];aggregate=scopes['server']['FedMultiGroupProj'].aggregate_parameters
            scopes['server']['heads']=SimpleNamespace(**scopes['heads']);runtime=SimpleNamespace(**old_post)
        clients=[client_cls(a,i,len(data[i]),len(data[i]),initial_model=mother,train_slow=False,send_slow=False) for i in range(3)]
        server=fake(a,clients,mother);server.device='cpu';server.aggregation_records=[];server.weight_records=[];server.selected_clients=clients
        heads.initialize(server,mother);events=[];batches=[];packets=[];all_rows=[]
        class Trace(DataLoader):
            def __iter__(self):
                for xx,yy in super().__iter__():batches.append(yy.tolist());yield xx,yy
        for loop in range(2):
            server.loop_round=loop
            for c in clients:
                previous=[heads.fingerprint(heads.clone_head(m)) for m in c.model_set]
                heads.bind_roles(c,server);c.set_parameters(server.global_model_set,loop)
                for k,m in enumerate(c.model_set):
                    inside=c.multi_head_roles[k];_,head=heads.split(m)
                    self.assertEqual(heads.fingerprint(heads.clone_head(m)),heads.fingerprint(heads.clone_head(server.global_model_set[k])))
                    if update=='frozen':
                        self.assertTrue(all(p.requires_grad==inside for p in head.values()))
                        ids={id(p) for g in c.optimizer_set[k].param_groups for p in g['params']}
                        self.assertEqual(sum(id(p) in ids for p in head.values()),2 if inside else 0)
                        if loop and not inside:self.assertNotEqual(previous[k],heads.fingerprint(heads.clone_head(m)))
            runtime.prepare_round(server)
            for c in clients:
                with ExitStack() as stack:
                    stack.enter_context(patch('flcore.clients.clientMultiGroupProj.read_client_data',return_value=data[c.id]))
                    stack.enter_context(patch('flcore.clients.clientMultiGroupProj.DataLoader',Trace))
                    if legacy:stack.enter_context(patch.object(post,'train',old_post['train']))
                    c.train()
                for k,row in enumerate(c.train_stats):
                    self.assertEqual(('fc3.weight' in c.full_uploads[k]),c.multi_head_roles[k])
                    self.assertEqual(row['train_batches'],2);self.assertEqual(row['processed_train_samples'],c.train_samples)
                    if update=='frozen':
                        helper=not c.multi_head_roles[k]
                        self.assertTrue(all(p.requires_grad for p in c.model_set[k].parameters()))  # temporary flags unwound
                        self.assertEqual(row['head_optimizer_parameter_count'],0 if helper else 2)
                        self.assertEqual(row['head_gradient_produced'],not helper)
                        self.assertEqual(row['effective_head_update'],'frozen' if helper else 'train')
                        if helper:
                            self.assertEqual(row['head_update_norm'],0.);self.assertTrue(row['head_unchanged_bitwise'])
                            self.assertTrue(all(p.grad is None for p in c.model_set[k].fc3.parameters()))
                            self.assertTrue(any(p.grad is not None and p.grad.abs().sum()>0 for n,p in c.model_set[k].named_parameters() if not n.startswith('fc3.')))
                            if diagnostics:self.assertEqual(row['backbone_pre_clip_grad_norm_mean_per_step'],row['total_pre_clip_grad_norm_mean_per_step'])
                        else:self.assertGreater(row['head_update_norm'],0.)
                        if ce_only:self.assertEqual(row['regularization_loss'],0.);self.assertEqual(row['raw_mse'],0.)
                        restored,payload=load_checkpoint(Path(c.save_folder_name)/f'Client_{c.id}_model_{k}.pt')
                        self.assertEqual(tensor_fingerprint(restored.state_dict()),tensor_fingerprint(c.model_set[k].state_dict()))
                        self.assertEqual(payload['multi_head_policy']['effective_head_update'],row['effective_head_update'])
                if observe:
                    state=[tensor_fingerprint(m.state_dict()) for m in c.model_set];before=rng_state()
                    with patch('flcore.clients.clientMultiGroupProj.read_client_data',return_value=data[c.id]):c.observe_models()
                    assert_rng(self,before);self.assertEqual(state,[tensor_fingerprint(m.state_dict()) for m in c.model_set])
                server.training_records.extend(c.train_stats);all_rows.extend(c.train_stats)
            aggregate(server);runtime.finish_round(server);packets.append(server.post_proto_packet)
            events.append(dict(local=[tensor_fingerprint(m.state_dict()) for c in clients for m in c.model_set],
                uploads=[tensor_fingerprint(p) for c in clients for p in c.full_uploads],server=[tensor_fingerprint(m.state_dict()) for m in server.global_model_set]))
        return dict(events=events,batches=batches,packets=packets,rng=rng_state(),rows=all_rows)

    def test_train_default_matches_frozen_6fe2a52_and_explicit_train(self):
        with tempfile.TemporaryDirectory() as d,redirect_stdout(io.StringIO()):
            results=[self.trajectory(Path(d),legacy=True),self.trajectory(Path(d),omitted=True),self.trajectory(Path(d))]
        for result in results[1:]:
            for key in ('events','batches','packets'):self.assertEqual(result[key],results[0][key])
            set_rng(results[0]['rng']);assert_rng(self,result['rng'])

    def test_roles_next_round_prototypes_reload_and_diagnostics_exact(self):
        with tempfile.TemporaryDirectory() as d,redirect_stdout(io.StringIO()):
            on=self.trajectory(Path(d),update='frozen');off=self.trajectory(Path(d),update='frozen',diagnostics=0,observe=False)
        for key in ('events','batches','packets'):self.assertEqual(on[key],off[key])
        set_rng(on['rng']);assert_rng(self,off['rng'])
        self.assertTrue(all(r['raw_mse']==0 for r in on['rows'][:9]))
        self.assertTrue(all(r['used_proto_source_completed_train_count']==1 for r in on['rows'][9:]))

    def test_CE_alone_reaches_backbone_through_fixed_head(self):
        with tempfile.TemporaryDirectory() as d,redirect_stdout(io.StringIO()):self.trajectory(Path(d),update='frozen',ce_only=True)

    def test_exception_restores_flags_modes_and_requires_fresh_download(self):
        a=options(num_clients=3,num_server_models=3,local_epochs=1,multi_head_mode='group_only_reset',multi_projection_scope='backbone',multi_helper_head_update='frozen')
        with tempfile.TemporaryDirectory() as d,redirect_stdout(io.StringIO()):
            a.device='cpu';a.save_folder_name_full=d
            model=Hyper_CNN_512(n_kernels=2,num_classes=100,ratio_LR=1.)
            c=clientMultiGroupSVD(a,0,3,3,initial_model=model,train_slow=False,send_slow=False);s=fake(a,[c],model)
            s.groups=[[0],[],[]];s.group_payload['client_to_model']={'0':0};heads.initialize(s,model)
            heads.bind_roles(c,s);c.set_parameters(s.global_model_set,0);c.post_proto_banks=None;c.post_proto_receipt=None
            c.model_set[1].eval();modes=[[m.training for m in model.modules()] for model in c.model_set]
            with patch.object(c,'load_train_data',side_effect=RuntimeError('controlled failure')):
                with self.assertRaisesRegex(RuntimeError,'controlled failure'):c.train()
            self.assertTrue(all(p.requires_grad for m in c.model_set for p in m.parameters()))
            self.assertEqual(modes,[[m.training for m in model.modules()] for model in c.model_set])
            with self.assertRaisesRegex(RuntimeError,'fresh role-bound'):c.train()
            heads.bind_roles(c,s);c.set_parameters(s.global_model_set,0)
            self.assertFalse(c._multi_helper_training_failed)
            self.assertFalse(c.model_set[1].fc3.weight.requires_grad)


if __name__=='__main__':unittest.main()

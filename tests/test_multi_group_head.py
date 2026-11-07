"""Alias-aware head roles, independent backbone kernel, frozen full regression."""
import copy
from contextlib import ExitStack,redirect_stdout
import io
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from torch.utils.data import DataLoader

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'system'))
from test_multi_last_epoch_proto import options
from test_multi_shared_proto import fake
from test_control_population import assert_rng
from common.experiment import rng_state,set_rng
from utils import multi_group_head as heads,multi_last_epoch_proto as post,single_avg_proto as common
from utils.group_projection import aggregate_group_updates
from utils.head_snapshot import tensor_fingerprint
from flcore.clients.clientMultiGroupSVD import clientMultiGroupSVD
from flcore.servers.serverMultiGroupProj import FedMultiGroupProj
from flcore.trainmodel.models import Hyper_CNN_512


class MultiHeadTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        if torch.cuda.is_available():torch.cuda.get_rng_state_all()

    def test_validation_defaults_and_scope(self):
        for mode in ('shared','group_only_reset','group_only_local'):
            options(multi_head_mode=mode,multi_projection_scope='backbone')
        options(multi_head_mode='shared',multi_projection_scope='full')
        for kw in [dict(multi_head_mode='bad'),dict(multi_projection_scope='bad'),
            dict(multi_head_mode='group_only_local',multi_projection_scope='full'),
            dict(multi_head_mode='shared',multi_projection_scope='backbone',multi_last_epoch_protocol=False),
            dict(multi_head_mode='shared',multi_projection_scope='backbone',multi_proto_scope='global'),
            dict(multi_head_mode='shared',multi_projection_scope='backbone',multi_group_mode='group_avg_only'),
            dict(multi_head_mode='group_only_local',multi_projection_scope='backbone',multi_group_file=''),
            dict(multi_head_mode='group_only_reset',multi_projection_scope='backbone',target_prox_mu=.1),
            dict(multi_head_mode='shared',multi_projection_scope='backbone',multi_warmup_rounds=1)]:
            with self.subTest(kw=kw),self.assertRaises(ValueError):options(**kw)

    def test_formal_commands_are_independent_and_keep_training_protocol(self):
        lines=(ROOT/'system/multi_group_head_commands.sh').read_text().splitlines()
        jobs=[shlex.split(s.strip().strip('"')) for s in lines if s.strip().startswith('"python')]
        self.assertEqual(len(jobs),3)
        def value(job,key):return job[job.index(key)+1]
        self.assertEqual([value(j,'--multi_head_mode') for j in jobs],['shared','group_only_reset','group_only_local'])
        self.assertEqual([value(j,'-did') for j in jobs],['0','1','2'])
        for key in ('-sfn','--run_log_dir','--h5_result_root','--final-model-root','-exp_name'):
            self.assertEqual(len({value(j,key) for j in jobs}),3)
        for j in jobs:
            self.assertEqual(j[3],'-did')
            for key,v in [('--multi_projection_scope','backbone'),('-gr','100'),('-ls','5'),('-nc','20'),('-nsm','3'),
                ('--multi_proto_bank_mode','per_model'),('--multi_proto_scope','group_priority'),('--multi_proto_loss','mse'),
                ('--multi_proto_feature','base_output'),('--multi_proto_target_type','raw_mean'),('--multi_proto_lambda','1.0'),
                ('--local_grad_clip','10'),('--multi_group_reuse_policy','members_only'),('--data-expected-alpha','0.5'),
                ('--multi_group_file','/data/zh/AllRepo/FedClip/system/dir05_proto_runs/dir0p5_seed0_gr100_20261105_v1/shared/K3_full_weights.json')]:
                self.assertEqual(value(j,key),v)
            self.assertNotIn('--multi_group_publish_file',j)
        args=options(multi_head_mode='group_only_local',multi_projection_scope='backbone')
        self.assertEqual(heads.capacity_metadata(args)['projection_parameter_scope'],'backbone_excluding_fc3')

    def test_scope_weights_head_mass_and_no_private_placeholder(self):
        torch.manual_seed(8);model=Hyper_CNN_512(n_kernels=2,num_classes=100,ratio_LR=1.)
        base={n:p.detach().clone() for n,p in model.named_parameters()};bb,head=heads.split(model)
        self.assertEqual(tuple(head),('fc3.weight','fc3.bias'));self.assertIn('fc2.weight',bb);self.assertIn('fc1.weight',bb)
        samples={0:.1,1:.3,2:.6};order=[2,0,1];group=[0,1]
        uploads={i:{n:p.clone() for n,p in base.items()} for i in range(3)}
        for i in uploads:
            uploads[i]['conv1.weight'].view(-1)[0]+=(1. if i!=2 else -2.)
            uploads[i]['fc3.weight'].add_(i+1);uploads[i]['fc3.bias'].add_(10*(i+1))
        expected=aggregate_group_updates(bb,lambda:iter((i,samples[i],{n:uploads[i][n] for n in bb}) for i in order),samples,group,'group_projection_softmax',.5)
        results={}
        for mode in ('shared','group_only_reset','group_only_local'):
            view={i:({n:p for n,p in u.items() if n in bb} if i==2 and mode!='shared' else u) for i,u in uploads.items()}
            clients=[SimpleNamespace(train_samples=[1,3,6][i],full_upload_parameters=lambda k,device=None,i=i:view[i]) for i in range(3)]
            args=options(multi_head_mode=mode,multi_projection_scope='backbone')
            result,metrics,rows,_=heads.aggregate_column(model,clients,order,samples,group,0,args)
            self.assertTrue(all(torch.equal(result[n],expected[0][n]) for n in bb))
            self.assertEqual([r['aggregation_weight'] for r in rows],[r['aggregation_weight'] for r in expected[2]])
            self.assertEqual(metrics['head_aggregation_mass'],1.)
            weights={r['client_id']:r['head_aggregation_weight'] for r in rows}
            target={n:torch.zeros_like(p) for n,p in head.items()}
            for i in order:
                if mode!='shared' and i==2:continue
                for n in head:target[n].add_(uploads[i][n]*weights[i])
            self.assertTrue(all(torch.equal(result[n],target[n]) for n in head))
            if mode!='shared':self.assertEqual(weights,{2:0.,0:.25,1:.75})
            self.assertGreater(next(r for r in rows if r['client_id']==2)['removed_component_norm'],0.)
            self.assertTrue(all(not p.requires_grad for p in result.values()))
            results[mode]=result
        self.assertTrue(all(torch.equal(results['group_only_reset'][n],results['group_only_local'][n]) for n in base))
        changed=copy.deepcopy(uploads);changed[2]['fc3.weight'].fill_(999.)
        clients=[SimpleNamespace(train_samples=[1,3,6][i],full_upload_parameters=lambda k,device=None,i=i:changed[i]) for i in range(3)]
        same=heads.aggregate_column(model,clients,order,samples,group,0,options(multi_head_mode='shared',multi_projection_scope='backbone'))[0]
        self.assertTrue(all(torch.equal(same[n],expected[0][n]) for n in bb))
        with self.assertRaises(ValueError):heads.aggregate_column(model,clients,order,samples,group,0,options(multi_head_mode='group_only_local',multi_projection_scope='backbone'))

    def trajectory(self,folder,mode,scope='backbone',legacy=False,observe=True,diagnostics=1):
        args=options(num_clients=3,num_server_models=2,local_epochs=1,batch_size=2,local_grad_clip=10,
            multi_head_mode=mode,multi_projection_scope=scope,multi_proto_diagnostics=diagnostics,multi_svd_diagnostics=diagnostics)
        args.device='cpu';args.save_folder_name_full=str(folder)
        torch.manual_seed(12);mother=Hyper_CNN_512(n_kernels=2,num_classes=100,ratio_LR=.15);mother.recover_larger_model();mother.ratio_LR=1.
        images=torch.randn(4,3,32,32)*.1;data=[list(zip(images[:3+i%2],torch.tensor([0,1,0,2][:3+i%2]))) for i in range(3)]
        client_cls=clientMultiGroupSVD;aggregate=FedMultiGroupProj.aggregate_parameters;runtime=post
        old_common={};old_post={};old_server={};old_client={}
        if legacy:
            for path,d in [('utils/single_avg_proto.py',old_common),('utils/multi_last_epoch_proto.py',old_post),
                ('flcore/clients/clientMultiGroupSVD.py',old_client),('flcore/servers/serverMultiGroupProj.py',old_server)]:
                source=subprocess.check_output(['git','show',f'dc30a19:system/{path}'],cwd=ROOT,text=True);exec(compile(source,'dc30a19/'+path,'exec'),d)
            old_post['shared']=SimpleNamespace(**old_common);runtime=SimpleNamespace(**old_post)
            client_cls=old_client['clientMultiGroupSVD'];aggregate=old_server['FedMultiGroupProj'].aggregate_parameters
        clients=[client_cls(args,i,len(data[i]),len(data[i]),initial_model=mother,train_slow=False,send_slow=False) for i in range(3)]
        server=fake(args,clients,mother);server.device='cpu';server.aggregation_records=[];server.weight_records=[]
        server.selected_clients=clients
        heads.initialize(server,mother)
        events=[];batches=[];packets=[];head_downloads=[]
        class TraceLoader(DataLoader):
            def __iter__(self):
                for x,y in super().__iter__():batches.append(y.tolist());yield x,y
        for loop in range(2):
            server.loop_round=loop
            for c in clients:
                old_heads=[heads.clone_head(m) for m in c.model_set]
                if heads.enabled(args):heads.bind_roles(c,server)
                c.set_parameters(server.global_model_set,loop)
                if heads.enabled(args):
                    head_downloads.extend(copy.deepcopy(c.multi_head_download))
                    for k,m in enumerate(c.model_set):
                        private=mode=='group_only_local' and not c.multi_head_roles[k]
                        expected=old_heads[k] if private and loop else heads.clone_head(mother) if private else heads.clone_head(server.global_model_set[k])
                        self.assertEqual(heads.fingerprint(heads.clone_head(m)),heads.fingerprint(expected))
            runtime.prepare_round(server)
            for c in clients:
                with ExitStack() as stack:
                    stack.enter_context(patch('flcore.clients.clientMultiGroupProj.read_client_data',return_value=data[c.id]))
                    stack.enter_context(patch('flcore.clients.clientMultiGroupProj.DataLoader',TraceLoader))
                    stack.enter_context(patch.object(torch,'save'))
                    if legacy:
                        stack.enter_context(patch.object(post,'train',old_post['train']))
                    c.train()
                server.training_records.extend(c.train_stats)
                if heads.enabled(args):
                    for k,row in enumerate(c.train_stats):
                        self.assertNotEqual(row['downloaded_head_sha256'],row['post_local_head_sha256'])
                        self.assertEqual(('fc3.weight' in c.full_uploads[k]),bool(row['head_upload_enabled']))
                        self.assertEqual(row['diagnostic_parameter_scope'],'backbone_excluding_fc3')
                        if loop==1 and mode=='group_only_local' and row['role']=='helper':self.assertEqual(row['private_head_source_completed_train_count'],1)
                if observe:
                    before=rng_state();state=[tensor_fingerprint(m.state_dict()) for m in c.model_set]
                    modes=[[module.training for module in m.modules()] for m in c.model_set]
                    with patch('flcore.clients.clientMultiGroupProj.read_client_data',return_value=data[c.id]):c.observe_models()
                    assert_rng(self,before);self.assertEqual(state,[tensor_fingerprint(m.state_dict()) for m in c.model_set])
                    self.assertEqual(modes,[[module.training for module in m.modules()] for m in c.model_set])
            aggregate(server);runtime.finish_round(server);packets.append(server.post_proto_packet)
            events.append(dict(local=[tensor_fingerprint(m.state_dict()) for c in clients for m in c.model_set],
                uploads=[tensor_fingerprint(p) for c in clients for p in c.full_uploads],server=[tensor_fingerprint(m.state_dict()) for m in server.global_model_set]))
        return dict(events=events,batches=batches,packets=packets,rng=rng_state(),downloads=head_downloads)

    def test_two_round_roles_private_columns_observation_and_frozen_default(self):
        with tempfile.TemporaryDirectory() as directory,redirect_stdout(io.StringIO()):
            results={}
            for name,kwargs in [('reset',dict(mode='group_only_reset')),('local',dict(mode='group_only_local')),
                ('local_no_observe',dict(mode='group_only_local',observe=False,diagnostics=0)),
                ('default',dict(mode='shared',scope='full')),('legacy',dict(mode='shared',scope='full',legacy=True))]:
                results[name]=self.trajectory(Path(directory),**kwargs)
        self.assertEqual(results['reset']['events'][0],results['local']['events'][0])
        self.assertEqual(results['reset']['packets'][0],results['local']['packets'][0])
        self.assertNotEqual(results['reset']['events'][1],results['local']['events'][1])
        for a,b in [('local','local_no_observe'),('default','legacy')]:
            for key in ('events','batches','packets'):self.assertEqual(results[a][key],results[b][key],(a,b,key))
            set_rng(results[b]['rng']);assert_rng(self,results[a]['rng'])


if __name__=='__main__':unittest.main()

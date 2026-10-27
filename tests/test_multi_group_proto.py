"""Group-priority sums/counts, exact feature location and collective barrier."""
import copy
from contextlib import redirect_stdout
import io
from pathlib import Path
import subprocess
import shlex
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'system'))
from common.experiment import rng_state,set_rng,tensor_digest
from common.data_loading import make_data_loader
from utils.multi_group_proto import (Features,prototype_loss,merge_statistics,collect_client,prepare_round,Session,
                                     encode_packet,decode_packet,packet_sha)
from utils.head_snapshot import tensor_fingerprint,backend_settings
from utils.multi_group_proto_options import validate_proto,active_round
from utils.multi_group_clustering import grouping_context
from flcore.clients.clientMultiGroupSVD import clientMultiGroupSVD
from flcore.trainmodel.models import Hyper_CNN_512
from test_multi_group_svd import options
from test_control_population import assert_rng


def fixture(folder,coefficient=.1,diagnostics=1):
    args=options(num_clients=3,num_server_models=3,multi_group_sizes='1,1,1',local_epochs=1,batch_size=2,
        multi_proto_mode='group_priority',multi_proto_lambda=coefficient,multi_proto_diagnostics=diagnostics)
    args.device='cpu';args.few_shot=0;args.save_folder_name_full=str(folder)
    model=Hyper_CNN_512(n_kernels=2,num_classes=100,ratio_LR=1.)
    servers=[copy.deepcopy(model) for _ in range(3)]
    generator=torch.Generator().manual_seed(88);clients=[];files=[]
    for cid,labels in enumerate(([0,0,1],[0,2,2,2],[3,3])):
        data=[(torch.randn(3,32,32,generator=generator),torch.tensor(y)) for y in labels]
        c=clientMultiGroupSVD(args,cid,len(data),len(data),initial_model=model,train_slow=False,send_slow=False)
        c.load_train_data=lambda data=data:make_data_loader(data,2,shuffle=True,drop_last=False,num_workers=0)
        c.set_parameters(servers,1);clients.append(c)
        y=torch.stack([b for a,b in data]);files.append(dict(client_id=cid,split='train',sha256=f'file{cid}',
            decoded_x_sha256=tensor_digest(torch.stack([a for a,b in data])),decoded_y_sha256=tensor_digest(y),
            labels={str(int(label)):int(n) for label,n in zip(*y.unique(return_counts=True))}))
    server=SimpleNamespace(args=args,loop_round=1,K=3,num_classes=100,clients=clients,groups=[[0],[1],[2]],
        group_payload=dict(client_to_model={str(i):i for i in range(3)},group_fingerprint='fixed'),
        data_manifest=dict(files=files),report_dir=lambda:str(folder),proto_latest_packet=None)
    for name in ('training','source','direction','capacity','collection','upload','round'):setattr(server,f'proto_{name}_records',[])
    return server


def stats_example():
    result={}
    for cid,n in enumerate(([1,0,0],[3,0,0],[7,2,0])):
        s=torch.zeros(2,3,2,dtype=torch.float64)
        s[0,0]=torch.tensor([1.,0.]) if cid==0 else torch.tensor([0.,3.]) if cid==1 else torch.tensor([7.,0.])
        s[1,0]=s[0,0].flip(0)
        if cid==2:s[0,1]=torch.tensor([0.,2.]);s[1,1]=torch.tensor([2.,0.])
        result[cid]=dict(sums=s,counts=torch.tensor(n),capacity_level=cid,rank_rate=.9-.2*cid)
    return result


class PrototypeTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        if torch.cuda.is_available():torch.cuda.get_rng_state_all()

    def test_unequal_counts_group_priority_outside_fallback_and_no_column_mix(self):
        st=stats_example();p,valid,rows,_,_=merge_statistics(st,[[0,1],[2]],1e-12,1,'g')
        torch.testing.assert_close(p[0,0],torch.tensor([1.,3.],dtype=torch.float64)/(10**.5))
        self.assertFalse(torch.allclose(p[0,0],torch.tensor([.5,.5],dtype=torch.float64).div(2**-.5)))
        torch.testing.assert_close(p[0,1],torch.tensor([0.,1.],dtype=torch.float64))
        torch.testing.assert_close(p[1,1],torch.tensor([1.,0.],dtype=torch.float64))
        self.assertEqual(rows[0]['source'],'group');self.assertEqual(rows[0]['class_samples'],4)
        self.assertEqual(rows[1]['source'],'outside');self.assertEqual(rows[1]['source_client_ids'],[2])
        self.assertFalse(valid[:,2].any());self.assertEqual(rows[2]['invalid_reason'],'no_training_samples')
        altered=copy.deepcopy(st);altered[2]['sums'][0,0]=torch.tensor([0.,7000.])
        p2,*_=merge_statistics(altered,[[0,1],[2]],1e-12,1,'g')
        self.assertTrue(torch.equal(p2[0,0],p[0,0]))

    def test_zero_group_vector_invalid_never_falls_back_and_nonfinite_is_error(self):
        st=stats_example();st[0]['sums'][0,0].zero_();st[1]['sums'][0,0].zero_()
        p,valid,rows,_,_=merge_statistics(st,[[0,1],[2]],1e-12,1,'g')
        self.assertFalse(valid[0,0]);self.assertEqual(rows[0]['candidate_source'],'group')
        self.assertEqual(rows[0]['source'],'invalid');self.assertEqual(rows[0]['invalid_reason'],'near_zero_merged_mean')
        self.assertTrue(torch.equal(p[0,0],torch.zeros(2,dtype=torch.float64)))
        st[1]['sums'][0,0,0]=float('nan')
        with self.assertRaises(FloatingPointError):merge_statistics(st,[[0,1],[2]],1e-12,1,'g')

    def test_classifier_actual_input_includes_relu_and_original_forward_unchanged(self):
        model=Hyper_CNN_512(n_kernels=2,num_classes=100,ratio_LR=.15).eval();x=torch.randn(2,3,32,32)
        schema=list(model.state_dict());before=tensor_fingerprint(model.state_dict())
        base=model.base(x);expected=torch.relu(base)
        self.assertTrue((base<0).any())
        with Features(model) as capture:logits=model(x);features=capture.get()
        self.assertTrue(torch.equal(features,expected));self.assertTrue(torch.equal(logits,model.fc3(expected)))
        self.assertEqual(schema,list(model.state_dict()));self.assertEqual(before,tensor_fingerprint(model.state_dict()))
        self.assertEqual(len(model.fc3._forward_pre_hooks),0)

    def test_proto_detach_backbone_gradient_no_classifier_gradient_and_invalid_batch(self):
        model=Hyper_CNN_512(n_kernels=2,num_classes=100,ratio_LR=.15)
        x=torch.randn(3,3,32,32);y=torch.tensor([0,1,2]);p=torch.randn(100,512,requires_grad=True)
        with Features(model) as capture:model(x);h=capture.get()
        raw,n=prototype_loss(h,y,p,torch.ones(100,dtype=torch.bool),1e-12);raw.backward()
        self.assertEqual(n,3);self.assertIsNone(p.grad);self.assertIsNone(model.fc3.weight.grad)
        self.assertTrue(any(v.grad is not None and float(v.grad.abs().sum())>0 for v in model.base.parameters()))
        h=torch.zeros(3,512,requires_grad=True)
        loss,n=prototype_loss(h,y,p,torch.zeros(100,dtype=torch.bool),1e-12)
        self.assertEqual(n,0);self.assertEqual(float(loss),0.);loss.backward();self.assertTrue(torch.equal(h.grad,torch.zeros_like(h)))
        huge=torch.full((3,512),3e38,requires_grad=True)
        empty,_=prototype_loss(huge,y,p,torch.zeros(100,dtype=torch.bool),1e-12)
        self.assertEqual(float(empty),0.);empty.backward();self.assertTrue(torch.isfinite(huge.grad).all())
        bad=h.detach().clone();bad[0,0]=float('inf')
        with self.assertRaises(FloatingPointError):prototype_loss(bad,y,p,torch.ones(100,dtype=torch.bool),1e-12)

    def test_packets_actual_lengths_roundtrip_and_truncation(self):
        packet=encode_packet('library',dict(prototypes=torch.randn(2,3,4,dtype=torch.float64),valid=torch.ones(2,3,dtype=torch.bool)),dict(loop_round=1))
        values,meta=decode_packet(packet,'library');again=encode_packet('library',values,meta)
        self.assertEqual(packet,again);self.assertEqual(len(packet),len(again));self.assertEqual(meta['loop_round'],1)
        with self.assertRaises(ValueError):decode_packet(packet[:-1],'library')
        with self.assertRaises(ValueError):decode_packet(packet,'statistics')

    def test_collective_barrier_all_columns_rng_state_and_readonly_collection(self):
        with tempfile.TemporaryDirectory() as folder,redirect_stdout(io.StringIO()):
            server=fixture(folder);before=[[tensor_fingerprint(m.state_dict()) for m in c.model_set] for c in server.clients]
            modes=[[[m.training for m in model.modules()] for model in c.model_set] for c in server.clients]
            state=rng_state();backend=backend_settings();prepare_round(server);assert_rng(self,state)
            self.assertEqual(backend,backend_settings());self.assertEqual(before,[[tensor_fingerprint(m.state_dict()) for m in c.model_set] for c in server.clients])
            self.assertEqual(modes,[[[m.training for m in model.modules()] for model in c.model_set] for c in server.clients])
            self.assertEqual(len(server.proto_collection_records),9)
            self.assertTrue(all(c.upload_round is None for c in server.clients))
            budget=server.proto_round_records[0];self.assertEqual(budget['collected_sample_forwards'],27)
            self.assertEqual(budget['prototype_download_bytes'],3*len(server.proto_latest_packet))
            for c in server.clients:
                self.assertEqual(c.proto_receipt['loop_round'],1);self.assertEqual(len(c.proto_banks),3)
                self.assertTrue(all(not b['prototypes'].requires_grad for b in c.proto_banks))
            server.clients[0].upload_round=1
            with self.assertRaisesRegex(RuntimeError,'All clients'):prepare_round(server)

    def test_diagnostics_toggle_exact_packets_training_uploads_and_rng(self):
        with tempfile.TemporaryDirectory() as folder,redirect_stdout(io.StringIO()):
            server=fixture(folder);other=copy.deepcopy(server);other.args.multi_proto_diagnostics=0
            state=rng_state();prepare_round(server);set_rng(state);prepare_round(other)
            self.assertEqual(server.proto_latest_packet,other.proto_latest_packet)
            self.assertTrue(server.proto_direction_records);self.assertEqual(other.proto_direction_records,[])
            start=rng_state();server.clients[0].train();expected=rng_state();set_rng(start);other.clients[0].train();assert_rng(self,expected)
            for left,right in zip(server.clients[0].model_set,other.clients[0].model_set):self.assertEqual(tensor_fingerprint(left.state_dict()),tensor_fingerprint(right.state_dict()))
            self.assertTrue(all(r['proto_enabled'] for r in server.clients[0].train_stats))
            stale=server.clients[1];stale.proto_receipt['loop_round']=0
            with self.assertRaisesRegex(RuntimeError,'stale'):Session(stale)

    def test_off_zero_first_round_match_fixed_old_train_and_skip_collection(self):
        source=subprocess.run(['git','show','5d6103a:system/flcore/clients/clientMultiGroupSVD.py'],cwd=ROOT,capture_output=True,text=True,check=True).stdout
        scope={'__name__':'fixed_proto_reference'};exec(compile(source,'5d6103a/clientMultiGroupSVD.py','exec'),scope)
        with tempfile.TemporaryDirectory() as folder,redirect_stdout(io.StringIO()):
            server=fixture(folder);c=server.clients[0];c.args.multi_proto_mode='none'
            reference=copy.deepcopy(c);start=rng_state();scope['clientMultiGroupSVD'].train(reference);expected=rng_state()
            for mode,coefficient,loop in [('none',0,1),('group_priority',0,1),('group_priority',.1,0)]:
                trial=copy.deepcopy(c);trial.args.multi_proto_mode=mode;trial.args.multi_proto_lambda=coefficient;trial.download_round=loop
                trial.proto_binding=dict(loop_round=loop,assigned_model_id=None if loop==0 else 0,group_fingerprint=None if loop==0 else 'fixed')
                set_rng(start)
                with patch('utils.multi_group_proto.Features',side_effect=AssertionError('No extra forward path')):trial.train()
                assert_rng(self,expected)
                for k,m in enumerate(trial.model_set):
                    self.assertEqual(tensor_fingerprint(m.state_dict()),tensor_fingerprint(reference.model_set[k].state_dict()))
                    self.assertEqual(tensor_fingerprint(trial.full_uploads[k]),tensor_fingerprint(reference.full_uploads[k]))

    def test_validation_and_prototype_config_not_in_group_compatibility(self):
        base=options();data=dict(config_sha256='c',files=[]);context=grouping_context(base,data,'init')
        active=options(multi_proto_mode='group_priority',multi_proto_lambda=.01)
        self.assertEqual(context,grouping_context(active,data,'init'));self.assertFalse(active_round(active,0));self.assertTrue(active_round(active,1))
        for change in [dict(multi_proto_lambda=-1),dict(multi_proto_lambda=float('nan')),dict(multi_proto_eps=0),
                       dict(multi_proto_eps=float('inf')),dict(multi_proto_start_round=1),dict(multi_proto_batch_size=0),
                       dict(multi_proto_start_round=4),dict(multi_kd_mode='full'),dict(target_head_mode='local'),
                       dict(multi_svd=False,model_family='CNN-512')]:
            params=dict(multi_proto_mode='group_priority',multi_proto_lambda=.1);params.update(change)
            with self.subTest(change=change),self.assertRaises(ValueError):options(**params)
        unsupported=copy.deepcopy(base);unsupported.algorithm='FedFewS';unsupported.multi_proto_mode='group_priority'
        with self.assertRaises(ValueError):validate_proto(unsupported)

    def test_collector_preserves_nontrivial_buffers_modes_and_rng(self):
        class Buffered(torch.nn.Module):
            def __init__(self):
                super().__init__();self.base=torch.nn.Sequential(torch.nn.AdaptiveAvgPool2d(1),torch.nn.Flatten(),
                    torch.nn.BatchNorm1d(3),torch.nn.Linear(3,4));self.fc3=torch.nn.Linear(4,2)
            def forward(self,x):return self.fc3(torch.relu(self.base(x)))
        model=Buffered().train();model.base[2].running_mean.copy_(torch.tensor([1.,2.,3.]));model.base[2].num_batches_tracked.fill_(7)
        data=[(torch.ones(3,32,32)*i,torch.tensor(i%2)) for i in range(3)]
        binding=dict(loop_round=1,assigned_model_id=0,group_fingerprint='g',train_data_sha256='file',
            decoded_x_sha256=tensor_digest(torch.stack([x for x,y in data])),decoded_y_sha256=tensor_digest(torch.stack([y for x,y in data])))
        c=SimpleNamespace(args=SimpleNamespace(multi_proto_batch_size=2,multi_proto_eps=1e-12),id=0,K=1,device='cpu',
            train_samples=3,capacity_level=0,rank_rate=.3,download_round=1,upload_round=None,model_set=[model],
            download_hashes=['mother'],proto_download_local_states=[tensor_fingerprint(model.state_dict())],
            load_train_data=lambda:make_data_loader(data,2,shuffle=True,drop_last=False,num_workers=0))
        before=tensor_fingerprint(model.state_dict());state=rng_state();packet,rows,upload=collect_client(c,binding);assert_rng(self,state)
        values,_=decode_packet(packet,'statistics');self.assertEqual(values['sums'].shape,(1,2,4))
        self.assertEqual(values['counts'].tolist(),[2,1]);self.assertEqual(rows[0]['forward_batches'],2)
        self.assertEqual(upload['statistics_upload_bytes'],len(packet));self.assertEqual(before,tensor_fingerprint(model.state_dict()))
        self.assertTrue(all(m.training for m in model.modules()));self.assertTrue(all(p.requires_grad and p.grad is None for p in model.parameters()))

    def test_formal_array_four_31_round_jobs_share_one_group_artifact(self):
        lines=(ROOT/'system/multi_group_proto_commands.sh').read_text().splitlines()
        commands=[shlex.split(row[1:-1]) for row in lines if row.startswith('"python ')]
        self.assertEqual(len(commands),4)
        for command,coefficient in zip(commands,(0,.01,.1,1)):
            for flag,value in [('-gr','30'),('-ls','5'),('-nsm','3'),('-nc','20'),('--multi_proto_start_round','2'),
                               ('--multi_svd_init','low_rank_recovered'),('--multi_svd_init_rank_rate','0.15'),
                               ('--group_softmax_tau','0.20'),('--multi_group_weight','0.50')]:
                self.assertEqual(command[command.index(flag)+1],value)
            self.assertEqual(float(command[command.index('--multi_proto_lambda')+1]),coefficient)
            self.assertEqual(command[command.index('--multi_kd_mode')+1],'none')
            self.assertEqual(command[command.index('--multi_proto_mode')+1],'none' if coefficient==0 else 'group_priority')
        group=commands[0][commands[0].index('--multi_group_publish')+1]
        for command in commands[1:]:self.assertEqual(command[command.index('--multi_group_file')+1],group)


if __name__=='__main__':unittest.main()

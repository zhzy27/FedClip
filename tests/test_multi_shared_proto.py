"""Two-level equal means, single-bank transport, strict legacy SGD regression."""
import copy
from contextlib import redirect_stdout
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
from test_control_population import assert_rng
from common.experiment import rng_state,set_rng
from utils import single_avg_proto as common,multi_last_epoch_proto as post,multi_shared_proto as bank
from utils.multi_group_proto import encode_packet,decode_packet,packet_sha
from utils.head_snapshot import tensor_fingerprint
from utils.group_projection import aggregate_group_updates
from flcore.clients.clientMultiGroupSVD import clientMultiGroupSVD
from flcore.trainmodel.models import Hyper_CNN_512


def statistics(K=2):
    # Class 0 has unequal owner sample counts; class 1 cancels to zero;
    # class 2 has one owner, class 3 has no observations.
    counts=[[1,2,0,0],[9,1,0,0],[2,0,3,0]]
    columns=[]
    for k in range(K):
        packets=[]
        for i,n in enumerate(counts):
            mean=torch.tensor([[2*i*i+4*k,3*i-k],[(-1 if i else 1),0],[7+k,-2*k],[0,0]],dtype=torch.float64)
            packets.append(dict(client_id=i,model_id=k,source_completed_train_count=1,
                sums=mean*torch.tensor(n)[:,None],counts=torch.tensor(n)))
        columns.append(packets)
    return columns


def fake(args,clients,mother):
    return SimpleNamespace(args=args,clients=clients,K=args.num_server_models,num_clients=len(clients),
        num_classes=100,loop_round=0,global_model_set=[copy.deepcopy(mother) for _ in range(args.num_server_models)],
        groups=[list(range(i,len(clients),args.num_server_models)) for i in range(args.num_server_models)],
        group_source_sha256='source',group_payload=dict(group_fingerprint='group',
            client_to_model={str(i):i%args.num_server_models for i in range(len(clients))}),
        initial_hash='initial',context=dict(data_fingerprint='data'),post_proto_packet=None,
        post_proto_classes=[],post_proto_rounds=[],training_records=[])


class SharedBankTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        if torch.cuda.is_available():torch.cuda.get_rng_state_all()

    def test_two_level_formula_nonuniform_samples_nondefault_K_zero_missing(self):
        for K in (1,2,4):
            columns=statistics(K);before=copy.deepcopy(columns)
            actual,ids=common.equal_client_column_means(columns,4,2,1)
            expected=torch.zeros(4,2,dtype=torch.float64)
            for label in range(4):
                owners=[i for i in range(3) if columns[0][i]['counts'][label]>0]
                for i in owners:
                    client_mean=torch.zeros(2,dtype=torch.float64)
                    for k in range(K):client_mean+=columns[k][i]['sums'][label]/int(columns[k][i]['counts'][label])
                    expected[label]+=client_mean/K
                if owners:expected[label]/=len(owners)
            self.assertTrue(torch.equal(actual['prototypes'],expected))
            torch.testing.assert_close(actual['prototypes'][0],torch.tensor([(10+6*(K-1))/3,3-(K-1)/2],dtype=torch.float64),rtol=0,atol=0)
            self.assertEqual(ids,[[0,1,2],[0,1],[2],[]])
            self.assertEqual(actual['valid'].tolist(),[True,True,True,False])
            self.assertEqual(actual['samples'].tolist(),[12,3,3,0])
            self.assertEqual(actual['model_contributors'].tolist(),[3*K,2*K,K,0])
            self.assertEqual(actual['statistics_samples'].tolist(),[12*K,3*K,3*K,0])
            self.assertTrue(torch.equal(actual['prototypes'][1],torch.zeros(2,dtype=torch.float64)))
            for a,b in zip(columns,before):
                for x,y in zip(a,b):
                    self.assertTrue(torch.equal(x['sums'],y['sums']));self.assertTrue(torch.equal(x['counts'],y['counts']))
            # Class-sample weighting or grouping clients before averaging is a different rule.
            self.assertNotEqual(float(actual['prototypes'][0,0]),sum(float(columns[k][i]['sums'][0,0]) for k in range(K) for i in range(3))/(12*K))

    def test_statistics_reject_incomplete_stale_mismatched_and_nonfinite(self):
        for kind in ('missing_column','missing_client','duplicate','wrong_column','stale','count_mismatch','float_count','nonfinite','requires_grad','sum_without_samples'):
            c=statistics()
            if kind=='missing_column':c[1]=[]
            elif kind=='missing_client':c[1].pop()
            elif kind=='duplicate':c[0].append(c[0][0])
            elif kind=='wrong_column':c[1][0]['model_id']=0
            elif kind=='stale':c[1][0]['source_completed_train_count']=2
            elif kind=='count_mismatch':c[1][0]['counts'][0]+=1
            elif kind=='float_count':c[1][0]['counts']=c[1][0]['counts'].float()
            elif kind=='nonfinite':c[1][0]['sums'][0,0]=float('nan')
            elif kind=='requires_grad':c[1][0]['sums'].requires_grad_()
            elif kind=='sum_without_samples':c[1][0]['sums'][3,0]=1.
            with self.subTest(kind=kind),self.assertRaises((ValueError,FloatingPointError)):
                common.equal_client_column_means(c,4,2,1)

    def test_transport_single_bank_scopes_counts_rounds_frozen_loss_and_reload(self):
        args=options(multi_proto_bank_mode='shared',multi_proto_scope='global',num_clients=3,num_server_models=2)
        clients=[SimpleNamespace(id=i,train_samples=sum([1,2,0,0] if i==0 else [9,1,0,0] if i==1 else [2,0,3,0]),
            upload_round=0,post_proto_local_packets=[],train_stats=[dict(prototype_statistics_encode_seconds=0.)]*2) for i in range(3)]
        model=SimpleNamespace(fc3=SimpleNamespace(in_features=2,out_features=4))
        server=fake(args,clients,model);server.num_classes=4;columns=statistics(2)
        for k in range(2):
            for i,c in enumerate(clients):
                p=columns[k][i];c.post_proto_local_packets.append(encode_packet('last_epoch_column_local',
                    dict(sums=p['sums'],counts=p['counts']),dict(client_id=i,model_id=k,source_completed_train_count=1,feature_position='base_output')))
        post.prepare_round(server);before=rng_state();post.finish_round(server);assert_rng(self,before)
        values,meta=decode_packet(server.post_proto_packet,bank.KIND)
        self.assertEqual(values['prototypes'].shape,(4,2));self.assertEqual(meta['actual_prototype_bank_count'],1)
        self.assertAlmostEqual(float(values['prototypes'][0,0]),16/3)
        self.assertNotEqual(float(values['prototypes'][0,0]),5.)  # Equal group means for sizes 2/1 would be wrong.
        self.assertEqual(server.post_proto_rounds[0]['deduplicated_training_samples'],18)
        self.assertEqual(server.post_proto_rounds[0]['collected_training_samples'],36)
        self.assertEqual(server.post_proto_rounds[0]['actual_download_bytes'],0)
        # Different group sizes/members do not change the all-client equal rule.
        self.assertEqual(server.post_proto_classes[0]['contributor_ids'],[0,1,2])
        server.loop_round=1
        for c in clients:c.model_set=[model]*2;c.K=2;c.args=args;c.download_round=1
        post.prepare_round(server)
        for c in clients:
            self.assertIsNone(c.post_proto_banks)
            self.assertIs(bank.training_bank(c,0),bank.training_bank(c,1))
            self.assertEqual(c.post_proto_receipt['sha256'],packet_sha(server.post_proto_packet))
            self.assertTrue(torch.equal(c.post_proto_shared_bank['prototypes'],values['prototypes']))
            self.assertFalse(c.post_proto_shared_bank['prototypes'].requires_grad)
        z=torch.randn(2,2,requires_grad=True);loss,_=common.global_mse(z,torch.tensor([0,1]),values['prototypes'],values['valid']);loss.backward()
        self.assertTrue(torch.isfinite(z.grad).all());self.assertIsNone(values['prototypes'].grad)
        with tempfile.TemporaryDirectory() as directory:
            server.save_folder_name=directory;server.final_model_dir=lambda:directory
            post.save_latest(server)
            raw=(Path(directory)/'Last_epoch_shared_prototypes.packet').read_bytes()
            self.assertEqual(raw,server.post_proto_packet)
            self.assertEqual((Path(directory)/'Last_epoch_shared_training_prototypes.packet').read_bytes(),raw)
        for change in ('round','shape','kind','mode'):
            broken=copy.deepcopy(meta);tensors=copy.deepcopy(values)
            if change=='round':broken['source_completed_train_count']=2
            if change=='shape':tensors['prototypes']=tensors['prototypes'][None]
            if change=='mode':broken['bank_mode']='per_model'
            server.post_proto_packet=encode_packet('last_epoch_columns_library' if change=='kind' else bank.KIND,tensors,broken)
            with self.subTest(change=change),self.assertRaises(ValueError):post.prepare_round(server)

    def test_validation_and_formal_command(self):
        options(multi_proto_bank_mode='shared',multi_proto_scope='global')
        options(multi_proto_bank_mode='shared',multi_proto_scope='group_priority',multi_proto_lambda=0.)
        for change in [dict(multi_proto_bank_mode='invalid'),dict(multi_proto_bank_mode='shared'),
            dict(multi_last_epoch_protocol=False,multi_proto_bank_mode='shared',multi_proto_mode='none',multi_proto_lambda=0.)]:
            with self.subTest(change=change),self.assertRaises(ValueError):options(**change)
        path=ROOT/'system/multi_last_epoch_shared_commands.sh'
        self.assertTrue(path.exists())
        if path.exists():
            rows=[shlex.split(l.strip().strip('"')) for l in path.read_text().splitlines() if l.strip().startswith('"python')]
            self.assertEqual(len(rows),1);c=rows[0];self.assertEqual(c[3],'-did');self.assertTrue(c[4].isdecimal())
            for key,value in [('--multi_proto_bank_mode','shared'),('--multi_proto_scope','global'),('--local_grad_clip','10'),
                ('-nsm','3'),('-nc','20'),('-ls','5'),('-gr','100'),('--multi_proto_feature','base_output'),('--multi_group_reuse_policy','members_only')]:
                self.assertEqual(c[c.index(key)+1],value)

    def trajectory(self,directory,mode='per_model',proto=1.,diagnostics=1,legacy=False,omitted=False,no_proto=False):
        a=options(num_clients=2,num_server_models=2,local_epochs=2,batch_size=2,local_grad_clip=10,
            multi_proto_scope='global',multi_proto_bank_mode=mode,multi_proto_lambda=proto,multi_proto_diagnostics=diagnostics)
        a.device='cpu';a.save_folder_name_full=directory
        if omitted:delattr(a,'multi_proto_bank_mode')
        if no_proto:a.multi_proto_mode='none';a.multi_proto_target_type='fixed_norm'
        torch.manual_seed(119);mother=Hyper_CNN_512(n_kernels=2,num_classes=100,ratio_LR=.15)
        mother.recover_larger_model();mother.ratio_LR=1.
        images=torch.randn(4,3,32,32)*.1;data=[list(zip(images[:3+i],torch.tensor([0,1,0,2][:3+i]))) for i in range(2)]
        clients=[clientMultiGroupSVD(a,i,len(data[i]),len(data[i]),initial_model=mother,train_slow=False,send_slow=False) for i in range(2)]
        server=fake(a,clients,mother);events=[];batches=[];packets=[]
        runtime=post
        if legacy:
            old_common={'__name__':'frozen_shared_8080557'};old_post={'__name__':'frozen_post_8080557'}
            for module,scope in [('single_avg_proto',old_common),('multi_last_epoch_proto',old_post)]:
                source=subprocess.check_output(['git','show',f'8080557:system/utils/{module}.py'],cwd=ROOT,text=True)
                exec(compile(source,'8080557/'+module,'exec'),scope)
            old_post['shared']=SimpleNamespace(**old_common);runtime=SimpleNamespace(**old_post)
        class TraceLoader(DataLoader):
            def __iter__(self):
                for x,y in super().__iter__():batches.append(y.tolist());yield x,y
        for loop in range(2):
            server.loop_round=loop
            for c in clients:c.set_parameters(server.global_model_set,loop)
            runtime.prepare_round(server)
            for c in clients:
                with patch('flcore.clients.clientMultiGroupProj.read_client_data',return_value=data[c.id]),\
                     patch('flcore.clients.clientMultiGroupProj.DataLoader',TraceLoader),patch.object(torch,'save'):
                    runtime.train(c)
                server.training_records.extend(c.train_stats)
                if mode=='shared':
                    self.assertTrue(all(r['actual_prototype_bank_count']==int(proto>0 and loop>0) for r in c.train_stats))
            with torch.no_grad():
                weights={c.id:c.train_samples/7 for c in clients}
                for k,m in enumerate(server.global_model_set):
                    factory=lambda:iter((c.id,weights[c.id],c.full_upload_parameters(k,'cpu')) for c in clients)
                    result=aggregate_group_updates(dict(m.named_parameters()),factory,weights,server.groups[k],a.multi_group_mode,.5)[0]
                    for n,p in m.named_parameters():p.copy_(result[n])
            runtime.finish_round(server);packets.append(server.post_proto_packet)
            events.append(dict(local=[tensor_fingerprint(m.state_dict()) for c in clients for m in c.model_set],
                uploads=[tensor_fingerprint(p) for c in clients for p in c.full_uploads],
                server=[tensor_fingerprint(m.state_dict()) for m in server.global_model_set]))
        return dict(events=events,batches=batches,rng=rng_state(),packets=packets)

    def test_frozen_per_model_and_zero_first_round_diagnostics_trajectories(self):
        results={}
        with tempfile.TemporaryDirectory() as directory,redirect_stdout(io.StringIO()):
            for label,settings in dict(old=dict(legacy=True),omitted=dict(omitted=True),explicit={},shared=dict(mode='shared'),
                shared_off=dict(mode='shared',diagnostics=0),zero=dict(mode='shared',proto=0.),none=dict(proto=0.,no_proto=True)).items():
                results[label]=self.trajectory(directory,**settings)
        for a,b in [('old','omitted'),('old','explicit'),('shared','shared_off'),('zero','none')]:
            for key in ('events','batches','packets'):
                # Diagnostics does not enter packet metadata, so library bytes also match.
                self.assertEqual(results[a][key],results[b][key],(a,b,key))
            set_rng(results[b]['rng']);assert_rng(self,results[a]['rng'])
        self.assertEqual(results['old']['events'][0],results['shared']['events'][0])
        self.assertEqual(results['shared']['batches'],results['old']['batches'])


if __name__=='__main__':unittest.main()

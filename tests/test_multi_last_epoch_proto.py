"""K-column equal-client means, controlled membership and paired K1-compatible SGD."""
import copy
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'system'))
from test_multi_group_svd import options as svd_options
from test_control_population import assert_rng
from common.experiment import rng_state,set_rng,sha256
from utils import single_avg_proto as shared,multi_last_epoch_proto as post
from utils.multi_group_proto import encode_packet,decode_packet
from utils.multi_group_reuse import validate_members,load_members
from utils.multi_group_clustering import grouping_context,digest_json,validate_group_file
from utils.multi_group_protocol import grouping_protocol
from flcore.clients.clientMultiGroupSVD import clientMultiGroupSVD
from flcore.trainmodel.models import Hyper_CNN_512


def options(**changes):
    values=dict(multi_last_epoch_protocol=True,multi_proto_scope='group_priority',multi_group_reuse_policy='members_only',
        multi_group_file='fixed.json',multi_proto_mode='last_epoch',multi_proto_loss='mse',multi_proto_target_type='raw_mean',
        multi_proto_feature='base_output',multi_proto_lambda=1.,local_grad_clip=0,
        multi_svd_init='low_rank_recovered',multi_svd_init_rank_rate=.15)
    values.update(changes);return svd_options(**values)


def source_artifact(context):
    n,k=context['num_clients'],context['K'];proto=context['grouping_protocol']
    columns=[np.eye(n).tolist() for _ in range(k)];matrix=np.mean(np.array(columns),axis=0).tolist()
    row=dict(loop_round=0,completed_train_count=1,matrix=matrix,matrix_sha256=digest_json(matrix),diagnostics=dict(per_model_cosine_matrices=columns))
    evidence=dict(protocol=proto,rounds=[row],final_similarity_sha256=digest_json(matrix));evidence['evidence_sha256']=digest_json(evidence)
    groups=[];offset=0
    for size in context['capacities']:groups.append(list(range(offset,offset+size)));offset+=size
    return dict(schema_version=2,context=context,context_sha256=digest_json(context),groups=groups,
        group_fingerprint=digest_json(dict(context=context,groups=groups)),client_to_model={str(i):c for c,g in enumerate(groups) for i in g},
        grouping_protocol=proto,evidence=evidence,similarity_matrix=matrix,completed_train_count=1,diagnostics={})


class LastEpochColumnTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        if torch.cuda.is_available():torch.cuda.get_rng_state_all()

    def test_group_priority_outside_global_equal_means_and_zero_invalid(self):
        packets=[dict(client_id=i,source_completed_train_count=1,
            sums=torch.tensor(s,dtype=torch.float64),counts=torch.tensor(n)) for i,s,n in [
                (0,[[2.,0.],[0.,0.],[0.,0.],[0.,0.]],[1,0,1,0]),
                (1,[[0.,18.],[0.,0.],[0.,0.],[0.,0.]],[9,0,0,0]),
                (2,[[100.,100.],[0.,12.],[8.,8.],[0.,0.]],[1,3,1,0])]]
        selected,sources=post.contributors(packets,[0,1],'group_priority',4)
        self.assertEqual(selected,[[0,1],[2],[0],[]]);self.assertEqual(sources,['group','outside','group','invalid'])
        value,ids=shared.equal_client_means(packets,4,2,1,selected)
        self.assertTrue(torch.equal(value['prototypes'],torch.tensor([[1.,1.],[0.,4.],[0.,0.],[0.,0.]],dtype=torch.float64)))
        self.assertEqual(value['valid'].tolist(),[True,True,True,False]);self.assertEqual(value['samples'].tolist(),[10,3,1,0])
        global_ids,_=post.contributors(packets,[0,1],'global',4)
        global_value,_=shared.equal_client_means(packets,4,2,1,global_ids)
        torch.testing.assert_close(global_value['prototypes'][0],torch.tensor([34.,34.],dtype=torch.float64))
        changed=copy.deepcopy(packets);changed[-1]['sums'][0]*=99
        fixed,_=shared.equal_client_means(changed,4,2,1,selected)
        self.assertTrue(torch.equal(fixed['prototypes'],value['prototypes']))

    def test_members_only_validates_source_and_immutable_context_without_writing(self):
        old=svd_options(num_clients=4,num_server_models=2,multi_svd_init='low_rank_recovered',local_grad_clip=10)
        new=options(num_clients=4,num_server_models=2)
        data=dict(files=[],config_sha256='config');source=source_artifact(grouping_context(old,data,'initial'))
        current=grouping_context(new,data,'initial')
        with self.assertRaises(ValueError):validate_group_file(source,current)
        groups,diffs=validate_members(source,current)
        self.assertEqual(groups,[[0,1],[2,3]])
        self.assertIn('training.grad_clip',[d['field'] for d in diffs])
        for key,value in [('initial_parameter_sha256','bad'),('data_fingerprint','bad'),('K',3),('capacities',[1,3]),('model_family','other')]:
            changed=copy.deepcopy(current);changed[key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):validate_members(source,changed)
        changed=copy.deepcopy(current);changed['svd_capacity_protocol']['rank_rates'][0]=.8
        with self.assertRaises(ValueError):validate_members(source,changed)
        corrupted=copy.deepcopy(source);corrupted['group_fingerprint']='bad'
        with self.assertRaises(ValueError):validate_members(corrupted,current)
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'source.json';path.write_text(json.dumps(source));before=path.read_bytes()
            new.multi_group_file=str(path);server=SimpleNamespace(args=new,context=current,report_dir=lambda:directory)
            receipt=load_members(server)
            self.assertEqual(before,path.read_bytes());self.assertEqual(receipt['source_file_sha256'],sha256(path))
            new.multi_group_reuse_policy='strict'
            with self.assertRaises(ValueError):load_members(server)
            new.multi_group_file=str(path.with_name('missing.json'))
            with self.assertRaises(FileNotFoundError):load_members(server)

    def test_column0_seed_matches_K1_others_independent_without_rng(self):
        a=options(seed=19);before=rng_state()
        self.assertEqual(shared.loader_seed(a,3,4),shared.loader_seed(a,3,4,0))
        self.assertEqual(len({shared.loader_seed(a,3,4,k) for k in range(5)}),5)
        assert_rng(self,before)

    def test_K_columns_train_last_epoch_next_round_and_no_extra_steps(self):
        torch.manual_seed(4);mother=Hyper_CNN_512(n_kernels=2,num_classes=100,ratio_LR=.15)
        mother.recover_larger_model();mother.ratio_LR=1.
        images=torch.randn(3,3,32,32)*.1;data=list(zip(images,torch.tensor([0,1,0])))
        with tempfile.TemporaryDirectory() as directory,redirect_stdout(io.StringIO()):
            args=options(local_epochs=2,batch_size=2);args.device='cpu';args.save_folder_name_full=directory
            c=clientMultiGroupSVD(args,0,3,3,initial_model=mother,train_slow=False,send_slow=False)
            fake=SimpleNamespace(args=args,post_proto_packet=None,clients=[c],loop_round=0,K=3,num_classes=100,
                global_model_set=[mother]*3,group_source_sha256='source',group_payload=dict(group_fingerprint='group',client_to_model={'0':0}))
            with patch('flcore.clients.clientMultiGroupProj.read_client_data',return_value=data),\
                 patch.object(torch.nn.utils,'clip_grad_norm_',side_effect=AssertionError('clip must be disabled')):
                c.set_parameters([mother]*3,0);post.prepare_round(fake);c.train()
                self.assertEqual(len(c.train_stats),3)
                for k,packet in enumerate(c.post_proto_local_packets):
                    values,meta=decode_packet(packet,'last_epoch_column_local');self.assertEqual(meta['model_id'],k)
                    self.assertEqual(int(values['counts'].sum()),3);self.assertEqual(c.train_stats[k]['train_batches'],4)
                    self.assertEqual(c.train_stats[k]['raw_mse'],0.)
                library=torch.stack([torch.full((100,512),float(k+1),dtype=torch.float64) for k in range(3)])
                meta=dict(source_completed_train_count=1,K=3,feature_position='base_output',contributor_scope='group_priority',group_fingerprint='group')
                fake.post_proto_packet=encode_packet('last_epoch_columns_library',dict(prototypes=library,valid=torch.ones(3,100,dtype=torch.bool)),meta)
                fake.loop_round=1;c.set_parameters([mother]*3,1);post.prepare_round(fake)
                observed=[];native=shared.global_mse
                def checked(z,y,prototypes,valid):
                    observed.append(float(prototypes[0,0]));return native(z,y,prototypes,valid)
                with patch.object(shared,'global_mse',side_effect=checked):c.train()
                self.assertEqual(observed,[1.]*4+[2.]*4+[3.]*4)
                self.assertTrue(all(r['used_proto_source_completed_train_count']==1 for r in c.train_stats))
                payload=torch.load(Path(directory)/'Client_0_model_1.pt',map_location='cpu',weights_only=False)
                self.assertEqual(payload['prototype_training_source_completed_train_count'],1)
                self.assertEqual(payload['prototype_statistics_source_completed_train_count'],2)
                self.assertEqual(payload['prototype_training_library_sha256'],c.post_proto_receipt['sha256'])
                c.post_proto_banks[0],c.post_proto_banks[1]=c.post_proto_banks[1],c.post_proto_banks[0]
                c.upload_round=None
                with self.assertRaisesRegex(RuntimeError,'wrong model column'):c.train()

    def test_options_and_five_complete_commands(self):
        for change in [dict(multi_avg_proto_ablation=True),dict(multi_group_file=''),dict(multi_warmup_rounds=1),
            dict(multi_proto_loss='z2'),dict(multi_proto_feature='classifier_input'),dict(multi_proto_lambda=float('nan')),
            dict(multi_proto_target_norm=2.),dict(multi_proto_scope='bad')]:
            with self.subTest(change=change),self.assertRaises(ValueError):options(**change)
        text=(ROOT/'system/multi_last_epoch_commands.sh').read_text()
        rows=[shlex.split(l.strip().strip('"')) for l in text.splitlines() if l.strip().startswith('"python')]
        self.assertEqual(len(rows),5)
        for i,c in enumerate(rows):
            self.assertEqual(c[3],'-did');self.assertTrue(c[4].isdecimal());self.assertNotIn('$',' '.join(c))
            for key,value in [('-nsm','3'),('-nc','20'),('-ls','5'),('-gr','100'),('--local_grad_clip','0'),
                ('--multi_group_reuse_policy','members_only'),('--multi_proto_feature','base_output')]:
                self.assertEqual(c[c.index(key)+1],value)
            self.assertEqual(c[c.index('--multi_proto_lambda')+1],'1.0' if i>=2 else '0.0')
            self.assertEqual(c[c.index('--multi_proto_scope')+1],'global' if i==4 else 'group_priority')

    def test_frozen_a4ea487_K1_eight_training_paths_and_packets_exact(self):
        from test_single_avg_proto import options as k1_options
        from flcore.clients.clientMultiGroupProj import clientMultiGroupProj
        source=subprocess.check_output(['git','show','a4ea487:system/utils/single_avg_proto.py'],cwd=ROOT,text=True)
        old={'__name__':'fixed_a4ea487_shared'};exec(compile(source,'a4ea487/single_avg_proto.py','exec'),old)
        torch.manual_seed(11);mother=Hyper_CNN_512(n_kernels=2,num_classes=100,ratio_LR=.15)
        mother.recover_larger_model();mother.ratio_LR=1.
        data=list(zip(torch.randn(3,3,32,32)*.1,torch.tensor([0,1,0])));start=rng_state()
        factors=[(1,1,10,0),(1,1,10,1),(1,0,10,0),(1,0,10,1),(0,0,10,0),(0,0,10,1),(1,1,0,0),(1,1,0,1)]
        with tempfile.TemporaryDirectory() as directory,redirect_stdout(io.StringIO()):
            for svd,f,clip,proto in factors:
                results=[]
                for legacy in (True,False):
                    a=k1_options(multi_svd=bool(svd),is_regular=f,local_grad_clip=clip,multi_proto_lambda=proto,
                        local_epochs=2,batch_size=2);a.device='cpu';a.save_folder_name_full=directory
                    cls=clientMultiGroupSVD if svd else clientMultiGroupProj
                    c=cls(a,0,3,3,initial_model=mother,train_slow=False,send_slow=False)
                    set_rng(start);c.set_parameters([mother],1)
                    c.avg_proto_bank=dict(prototypes=torch.full((100,512),.02,dtype=torch.float64),valid=torch.ones(100,dtype=torch.bool)) if proto else None
                    c.avg_proto_receipt=dict(source_completed_train_count=1) if proto else None
                    with patch('flcore.clients.clientMultiGroupProj.read_client_data',return_value=data),patch.object(torch,'save'):
                        (old['train'] if legacy else shared.train)(c)
                    results.append((copy.deepcopy(c.model_set[0].state_dict()),getattr(c,'full_uploads',[]),c.avg_proto_local,rng_state(),c.train_stats[0]))
                for name,value in results[0][0].items():self.assertTrue(torch.equal(value,results[1][0][name]),(factors,name))
                if svd:
                    for name,value in results[0][1][0].items():self.assertTrue(torch.equal(value,results[1][1][0][name]),name)
                self.assertEqual(results[0][2],results[1][2]);set_rng(results[1][3]);assert_rng(self,results[0][3])
                for key in ('ce_loss','regularization_loss','raw_mse','weighted_mse','total_loss','head_weight_l2_norm',
                            'pre_clip_grad_norm_mean_per_step','clip_coefficient_mean_per_step','prototype_collected_samples'):
                    self.assertEqual(results[0][4][key],results[1][4][key],key)


if __name__=='__main__':unittest.main()

"""K1 class-count weighting and actual-download collection/source isolation."""
import copy
from contextlib import redirect_stdout
import io
from pathlib import Path
import random
import shlex
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'system'))
from test_single_avg_proto import options
from utils import single_avg_proto as p
from utils.multi_group_proto import decode_packet,encode_packet
from common.experiment import rng_state
from utils.head_snapshot import tensor_fingerprint


def assert_rng(test,a,b):
    test.assertEqual(a['python'],b['python']);test.assertEqual(a['numpy'][0],b['numpy'][0])
    np.testing.assert_array_equal(a['numpy'][1],b['numpy'][1]);test.assertEqual(a['numpy'][2:],b['numpy'][2:])
    test.assertTrue(torch.equal(a['torch'],b['torch']));test.assertEqual(len(a['cuda']),len(b['cuda']))
    for x,y in zip(a['cuda'],b['cuda']):test.assertTrue(torch.equal(x,y))


class Encoder(torch.nn.Module):
    def __init__(self):
        super().__init__();self.base=torch.nn.Linear(2,2);self.fc3=torch.nn.Linear(2,3)
        self.head=torch.nn.Sequential(torch.nn.ReLU(inplace=True),self.fc3)
        self.register_buffer('probe',torch.zeros(1));self.fail=False;self.read_random=False;self.calls=0
    def forward(self,x):
        self.calls+=1;self.probe.add_(1)
        if self.read_random:random.random();np.random.rand();torch.rand(2)
        if self.fail:raise RuntimeError('collection error')
        return self.head(self.base(x))


def client(model,data,loop=1,cid=0):
    args=options(multi_proto_mode='global_pre_train',multi_proto_aggregation='class_sample',multi_proto_batch_size=2)
    args.num_classes=3
    args.num_clients=1
    c=SimpleNamespace(id=cid,args=args,device='cpu',model_set=[model],train_samples=len(data),download_round=loop,
        avg_proto_download_context=dict(server_sha256='server',download_sha256=tensor_fingerprint(model.state_dict()),
            data_fingerprint='data',initial_sha256='init'))
    c.load_train_data=lambda:DataLoader(data,batch_size=2,shuffle=True,generator=torch.Generator().manual_seed(47))
    return c


class ClassSampleTests(unittest.TestCase):
    def setUp(self):torch.set_num_threads(2);torch.manual_seed(19)

    def test_count_weighting_unequal_zero_missing_and_invalid_statistics(self):
        a=dict(client_id=0,source_completed_train_count=1,sums=torch.tensor([[20.,-10.],[0.,0.],[0.,0.]],dtype=torch.float64),counts=torch.tensor([10,2,0]))
        b=dict(client_id=3,source_completed_train_count=1,sums=torch.tensor([[6.,12.],[0.,0.],[0.,0.]],dtype=torch.float64),counts=torch.tensor([1,0,0]))
        result,ids,counts=p.class_sample_means([b,a],3,2,1)
        torch.testing.assert_close(result['prototypes'][0],torch.tensor([26/11,2/11],dtype=torch.float64),rtol=0,atol=0)
        self.assertEqual(result['valid'].tolist(),[True,True,False]);self.assertEqual(counts,[[10,1],[2],[]])
        self.assertEqual(ids,[[0,3],[0],[]]);self.assertEqual(result['samples'].tolist(),[11,2,0])
        for packets in ([],[a,a],[{k:v for k,v in a.items() if k!='counts'}]):
            with self.assertRaises(ValueError):p.class_sample_means(packets,3,2,1)
        for change in ('nan','negative','shape','dtype','unobserved_sum','wrong_source'):
            bad=copy.deepcopy(a)
            if change=='nan':bad['sums'][0,0]=float('nan')
            if change=='negative':bad['counts'][0]=-1
            if change=='shape':bad['sums']=torch.zeros(3,4,dtype=torch.float64)
            if change=='dtype':bad['counts']=bad['counts'].float()
            if change=='unobserved_sum':bad['sums'][2,0]=1
            if change=='wrong_source':bad['source_completed_train_count']=2
            with self.subTest(change=change),self.assertRaises(ValueError):p.class_sample_means([bad,b],3,2,1)

    def test_distributed_matches_same_encoder_concatenation_raw_signed_features(self):
        model=Encoder().double().eval();data=[(torch.tensor([float(i),-float(i+2)],dtype=torch.float64),torch.tensor(i%2)) for i in range(7)]
        with torch.no_grad():z=model.base(torch.stack([x for x,y in data]));labels=torch.stack([y for x,y in data])
        packets=[]
        for cid,indices in enumerate(([0,1],[2,3,4,5,6])):
            stats=p.LastEpochStats(3,2);stats.add(z[list(indices)],labels[list(indices)])
            packets.append(dict(client_id=cid,source_completed_train_count=1,sums=stats.sums,counts=stats.counts))
        values,_,_=p.class_sample_means(packets,3,2,1)
        for label in (0,1):torch.testing.assert_close(values['prototypes'][label],z[labels==label].mean(0),rtol=1e-14,atol=1e-14)
        self.assertFalse(bool(values['valid'][2]));self.assertTrue((z<0).any())

    def test_collection_restores_model_modes_buffers_rng_and_never_iterates_train_sampler(self):
        model=Encoder();model.train();model.fc3.eval();model.read_random=True
        data=list(zip(torch.randn(5,2),torch.tensor([0,0,1,1,1])));c=client(model,data)
        before=copy.deepcopy(model.state_dict());modes=[m.training for m in model.modules()];random_before=rng_state()
        with patch.object(torch.cuda,'get_rng_state_all',side_effect=AssertionError('Do not initialize all GPUs')):
            packet,cost=p.collect_pre_train(c)
        values,meta=decode_packet(packet,'single_avg_local_v2')
        self.assertEqual(values['counts'].tolist(),[2,3,0]);self.assertEqual(cost['samples'],5);self.assertEqual(cost['batches'],3)
        for n,v in before.items():self.assertTrue(torch.equal(v,model.state_dict()[n]),n)
        self.assertEqual(modes,[m.training for m in model.modules()]);assert_rng(self,random_before,rng_state())
        self.assertEqual(meta['collection_loop_round'],1);self.assertEqual(meta['source_completed_train_count'],1)
        self.assertEqual(meta['intended_use_completed_train_count'],2);self.assertEqual(meta['collection_phase'],'pre_local_download')
        model.fail=True
        with self.assertRaisesRegex(RuntimeError,'collection error'):p.collect_pre_train(c)
        for n,v in before.items():self.assertTrue(torch.equal(v,model.state_dict()[n]),n)
        self.assertEqual(modes,[m.training for m in model.modules()]);assert_rng(self,random_before,rng_state())
        self.assertFalse(model.base._forward_hooks)

    def test_receipts_do_not_confuse_same_number_different_collection_phases(self):
        model=Encoder();c=client(model,[(torch.zeros(2),torch.tensor(0))]);meta=p.local_metadata(c,'pre_local_download')
        meta.pop('client_id');meta.pop('source_download_model_sha256')
        meta.update(source_download_model_sha256_by_client={'0':c.avg_proto_download_context['download_sha256']},sha256='bank')
        values=dict(prototypes=torch.zeros(3,2,dtype=torch.float64),valid=torch.tensor([True,False,False]),
            contributors=torch.tensor([1,0,0]),samples=torch.tensor([1,0,0]))
        p.validate_training_receipt(c,values,meta)
        for key,value in [('collection_phase','last_local_epoch'),('intended_use_loop_round',2),('source_server_completed_aggregations',0),
            ('aggregation','client_equal'),('data_fingerprint','wrong')]:
            with self.subTest(key=key),self.assertRaises(ValueError):p.validate_training_receipt(c,values,dict(meta,**{key:value}))
        with self.assertRaises(ValueError):p.validate_training_receipt(c,values,{k:v for k,v in meta.items() if k!='collection_phase'})
        wrong=dict(meta,source_download_model_sha256_by_client={'0':'another low-rank download'})
        with self.assertRaises(ValueError):p.validate_training_receipt(c,values,wrong)
        c.download_round=2
        with self.assertRaises(ValueError):p.validate_training_receipt(c,values,meta)

    @unittest.skipUnless(torch.cuda.is_available(),'CUDA is not available')
    def test_CUDA_collection_preserves_only_actual_device_rng_and_buffers(self):
        device=torch.cuda.current_device();model=Encoder().to(device);model.read_random=True
        data=list(zip(torch.randn(3,2),torch.tensor([0,0,1])));c=client(model,data);c.device=f'cuda:{device}'
        def state():
            return dict(python=random.getstate(),numpy=np.random.get_state(),torch=torch.get_rng_state(),cuda=[torch.cuda.get_rng_state(device)])
        before=copy.deepcopy(model.state_dict());random_before=state();native=Encoder.forward
        def forward(m,x):torch.rand(2,device=x.device);return native(m,x)
        with patch.object(Encoder,'forward',forward),patch.object(torch.cuda,'get_rng_state_all',side_effect=AssertionError('All-device query forbidden')):
            packet,cost=p.collect_pre_train(c)
        assert_rng(self,random_before,state())
        for name,value in before.items():self.assertTrue(torch.equal(value,model.state_dict()[name]),name)
        self.assertEqual(cost['samples'],3);self.assertEqual(cost['batches'],2)
        self.assertFalse(model.base._forward_hooks)

    def test_options_only_K1_no_ignored_aggregation_or_timing(self):
        options(multi_proto_aggregation='class_sample')
        options(multi_proto_mode='global_pre_train',multi_proto_aggregation='class_sample')
        for kwargs in (dict(multi_proto_aggregation='wrong'),dict(multi_proto_mode='global_pre_train'),
            dict(multi_proto_mode='global_pre_train',multi_proto_aggregation='class_sample',multi_svd=False,is_regular=0),
            dict(multi_avg_proto_ablation=False,multi_proto_aggregation='class_sample'),
            dict(multi_proto_aggregation='class_sample',num_server_models=3),
            dict(multi_proto_aggregation='class_sample',multi_proto_batch_size=3)):
            with self.subTest(kwargs=kwargs),self.assertRaises(ValueError):options(**kwargs)
        for mode in ('none','global_pre_train','global_last_epoch'):
            args=options(multi_proto_mode=mode,multi_proto_lambda=0,multi_proto_aggregation='class_sample')
            self.assertFalse(p.uses_bank(args))

    def test_two_complete_101_round_commands_share_all_public_parameters(self):
        path=ROOT/'system/single_avg_class_sample_commands.sh'
        commands=[shlex.split(line.strip()[1:-1]) for line in path.read_text().splitlines() if line.strip().startswith('"')]
        self.assertEqual(len(commands),2)
        different={'-did','-exp_name','-sfn','--run_log_dir','--h5_result_root','--final-model-root','--multi_proto_mode'}
        def mapping(command):
            result={};i=3
            while i<len(command):
                flag=command[i]
                if i+1<len(command) and not command[i+1].startswith('-'):result[flag]=command[i+1];i+=2
                else:result[flag]=True;i+=1
            return result
        maps=[mapping(command) for command in commands]
        for gpu,(command,m) in enumerate(zip(commands,maps)):
            self.assertNotIn('$',' '.join(command));self.assertEqual(command[3],'-did');self.assertEqual(command[4],str(gpu))
            for flag,value in [('-gr','100'),('-ls','5'),('-nc','20'),('-nsm','1'),('--multi_proto_aggregation','class_sample'),
                ('--multi_proto_lambda','1.0'),('--multi_proto_start_round','2'),('--local_grad_clip','10')]:
                self.assertEqual(command[command.index(flag)+1],value)
            self.assertNotIn('--multi_group_file',command)
        self.assertEqual({k:v for k,v in maps[0].items() if k not in different},{k:v for k,v in maps[1].items() if k not in different})
        self.assertEqual(commands[0][commands[0].index('--multi_proto_mode')+1],'global_last_epoch')
        self.assertEqual(commands[1][commands[1].index('--multi_proto_mode')+1],'global_pre_train')


if __name__=='__main__':unittest.main()

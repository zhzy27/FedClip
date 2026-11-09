"""Frozen 63b91d5 K1 client-equal low-rank uploads, kernel, packets and RNG."""
import copy
from contextlib import ExitStack,redirect_stdout
import io
from pathlib import Path
import subprocess
import sys
import tempfile
from types import FunctionType,SimpleNamespace
import unittest
from unittest.mock import patch

import torch

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'system'))
from test_single_avg_proto import options
from common.experiment import rng_state,set_rng
from fedfews_compat.observe import rng_digest
from utils import single_avg_proto as proto
from utils.head_snapshot import tensor_fingerprint
from utils.multi_group_proto import packet_sha
from utils.group_projection import aggregate_group_updates
from flcore.clients.clientMultiGroupSVD import clientMultiGroupSVD
from flcore.trainmodel.models import Hyper_CNN_512


class LegacyDefaultRuntime(unittest.TestCase):
    def test_omitted_and_explicit_equal_match_frozen_parameters_uploads_packets_batches_rng(self):
        torch.set_num_threads(2)
        if torch.cuda.is_available():torch.cuda.get_rng_state_all()
        namespace={};source=subprocess.check_output(['git','show','63b91d5:system/utils/single_avg_proto.py'],cwd=ROOT,text=True)
        exec(compile(source,'63b91d5/single_avg_proto.py','exec'),namespace)
        frozen={k:v for k,v in namespace.items() if isinstance(v,FunctionType) or k=='LastEpochStats'}
        generator=torch.Generator().manual_seed(37);images=torch.randn(5,3,32,32,generator=generator)
        torch.manual_seed(53);mother=Hyper_CNN_512(n_kernels=2,num_classes=100,ratio_LR=.15)
        mother.recover_larger_model();mother.ratio_LR=1.;initial_rng=rng_state();results=[]
        with tempfile.TemporaryDirectory() as tmp,redirect_stdout(io.StringIO()):
            for kind in ('frozen','omitted','explicit'):
                set_rng(initial_rng);args=options(num_clients=5,local_epochs=2,batch_size=2,global_rounds=1,local_grad_clip=10)
                args.device='cpu';args.save_folder_name_full=str(Path(tmp)/kind);Path(args.save_folder_name_full).mkdir()
                if kind=='omitted':del args.multi_proto_aggregation
                clients=[clientMultiGroupSVD(args,cid,3+cid%3,2,initial_model=mother,train_slow=False,send_slow=False) for cid in range(5)]
                server=SimpleNamespace(args=args,num_clients=5,num_classes=100,clients=clients,global_model_set=[copy.deepcopy(mother)],
                    initial_hash='same_mother',context={'data_fingerprint':'same_training_data'})
                events=[];batches=[]
                def reader(dataset,cid,args,is_train=True,**kwargs):
                    return [(images[i],torch.tensor((i//2+cid)%4)) for i in range(3+cid%3)]
                native_forward=Hyper_CNN_512.forward
                def forward(model,x):
                    if model.training and torch.is_grad_enabled():batches.append(x[:,0,0,0].tolist())
                    return native_forward(model,x)
                with ExitStack() as stack:
                    if kind=='frozen':
                        for name,value in frozen.items():stack.enter_context(patch.object(proto,name,value))
                    stack.enter_context(patch('flcore.clients.clientMultiGroupProj.read_client_data',side_effect=reader))
                    stack.enter_context(patch.object(Hyper_CNN_512,'forward',forward))
                    proto.initialize(server)
                    for loop in range(2):
                        server.loop_round=loop
                        for c in clients:c.set_parameters(server.global_model_set,loop)
                        proto.prepare_round(server)
                        for c in clients:
                            c.train()
                            events.append(dict(kind='local',loop=loop,cid=c.id,weights=tensor_fingerprint(c.model_set[0].state_dict()),
                                uploads=tensor_fingerprint(c.full_uploads[0]),packet_sha256=packet_sha(c.avg_proto_local),rng=rng_digest()))
                        total=sum(c.train_samples for c in clients);samples={c.id:c.train_samples/total for c in clients}
                        factory=lambda:iter((c.id,samples[c.id],c.full_uploads[0]) for c in clients)
                        updates=aggregate_group_updates(dict(server.global_model_set[0].named_parameters()),factory,samples,list(range(5)),'group_avg_only')[0]
                        with torch.no_grad():
                            for name,p in server.global_model_set[0].named_parameters():p.copy_(updates[name])
                        proto.finish_round(server)
                        events.append(dict(kind='aggregate',loop=loop,weights=tensor_fingerprint(server.global_model_set[0].state_dict()),
                            library_sha256=packet_sha(server.avg_proto_packet),rng=rng_digest()))
                results.append(dict(events=events,batches=batches,rng=rng_digest()))
        self.assertEqual(results[0],results[1]);self.assertEqual(results[0],results[2])


if __name__=='__main__':unittest.main()

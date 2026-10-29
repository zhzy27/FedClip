"""Frozen 89d442c omitted/zero server trajectories for all three old modes."""
from contextlib import redirect_stdout
import io
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from torch.utils.data import DataLoader

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'system'))
from flcore.clients.clientbase import load_item
from flcore.servers.serverTargetProj import FedTargetProj
from fedfews_compat.observe import rng_digest
from utils.head_snapshot import tensor_fingerprint
from utils.target_head import preserved_head_rng


class ZeroTrajectoryTests(unittest.TestCase):
    def test_three_modes_omitted_and_zero_equal_frozen_local_uploads_aggregation_batches_rng(self):
        torch.set_num_threads(2)
        if torch.cuda.is_available():torch.cuda.get_rng_state_all()
        old_client,old_server={},{}
        for path,scope in [('clients/clientTargetProj.py',old_client),('servers/serverTargetProj.py',old_server)]:
            source=subprocess.check_output(['git','show','89d442c:system/flcore/'+path],cwd=ROOT,text=True)
            exec(compile(source,'89d442c/'+path,'exec'),scope)
        old_server['clientTargetProj']=old_client['clientTargetProj']
        factory='Hyper_CNN_512(in_features=3,num_classes=100,n_kernels=2,ratio_LR={rank},input_size=32)'
        generator=torch.Generator().manual_seed(145);images=torch.randn(4,3,32,32,generator=generator)
        context=dict(training=False,batches=[]);histories={};previous=os.getcwd()
        class TraceLoader(DataLoader):
            def __iter__(self):
                for x,y in super().__iter__():
                    if context['training']:context['batches'].append((context['round'],context['cid'],y.tolist()))
                    yield x,y
        with tempfile.TemporaryDirectory() as directory,redirect_stdout(io.StringIO()):
            root=Path(directory)
            try:
                os.chdir(root)
                def reader(dataset,cid,args,is_train=True,**kwargs):
                    n=3+cid%2 if is_train else 2+cid%2
                    return [(images[i],torch.tensor((cid*7+i)%100)) for i in range(n)]
                for mode in ('avg','softmax_only','projection_softmax'):
                    for label in ('legacy','omitted','zero'):
                        name=mode+'_'+label
                        args=SimpleNamespace(device='cpu',dataset='Synthetic',num_classes=100,global_rounds=1,local_epochs=1,
                            batch_size=2,local_learning_rate=.005,num_clients=5,join_ratio=1.,random_join_ratio=False,
                            few_shot=0,algorithm='FedTargetProj',time_select=False,goal='smoke',time_threthold=1e9,top_cnt=10,
                            auto_break=False,eval_gap=1,client_drop_rate=0.,train_slow_rate=0.,send_slow_rate=0.,resume=False,
                            model_family='Decom_CNN-5-512',models_folder_name='',niid=1,partition='pat',class_per_client=20,
                            exp_name=name,save_file_paths=[],models=[factory.format(rank=r) for r in (.9,.37,.35,.25,.15)],
                            global_model=factory.format(rank=.15),target_client_id=0,seed=0,is_regular=1,regular_lamda=.001,
                            target_proj_mode=mode,projection_self_weight=None if mode=='avg' else .125,target_prox_mu=0.,
                            target_head_mode='shared',target_local_schedule='joint',save_folder_name=str(root/name/'c'),
                            final_model_root=str(root/name/'f'),h5_result_root=str(root/name/'h'),training_log_root=str(root/name/'l'))
                        if label=='zero':args.target_z2_lambda=0.
                        cls=old_server['FedTargetProj'] if label=='legacy' else FedTargetProj
                        context.update(training=False,batches=[]);events=[]
                        with patch('flcore.clients.clientbase.read_client_data',side_effect=reader),\
                             patch('flcore.servers.serverbase.read_client_data',side_effect=reader),\
                             patch('flcore.clients.clientbase.DataLoader',TraceLoader):
                            server=cls(args,0)
                            for client in server.clients:
                                native=client.train
                                def train(current_round,client=client,native=native):
                                    context.update(training=True,round=current_round,cid=client.id)
                                    try:return native(current_round)
                                    finally:context['training']=False
                                client.train=train
                            native_aggregate=server.aggregate_parameters_avg
                            def aggregate():
                                with preserved_head_rng():
                                    local=[tensor_fingerprint(load_item(c.role,'model',c.save_folder_name).state_dict()) for c in server.clients]
                                    uploads=[tensor_fingerprint(server._load_full_parameters(c.id)) for c in server.clients]
                                native_aggregate()
                                events.append(dict(local=local,uploads=uploads,ids=list(server.uploaded_ids),
                                    server=tensor_fingerprint(load_item('Server','model',server.save_folder_name).state_dict()),rng=rng_digest()))
                            server.aggregate_parameters_avg=aggregate;server.train()
                            histories[mode,label]=dict(events=events,batches=list(context['batches']),rng=rng_digest(),summary=server._local_accuracy_summary())
                    self.assertEqual(histories[mode,'legacy'],histories[mode,'omitted'])
                    self.assertEqual(histories[mode,'legacy'],histories[mode,'zero'])
            finally:os.chdir(previous)


if __name__=='__main__':unittest.main()

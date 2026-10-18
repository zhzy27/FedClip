"""Independent column kernels, clustering, endpoint state and K-task budgets."""
import copy
import json
from pathlib import Path
import random
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'system'))
from common.experiment import rng_state
from utils.multi_group_options import group_sizes, validate_multi_group
from utils.multi_group_clustering import balanced_cluster, canonical_groups, update_similarity, digest_json, validate_group_file
from utils.group_projection import aggregate_group_updates
from utils.multi_group_reporting import record_evaluation
from flcore.clients.clientMultiGroupProj import clientMultiGroupProj
from flcore.servers.serverMultiGroupProj import FedMultiGroupProj
from test_control_population import assert_rng, options


def args(**extra):
    values = dict(algorithm='FedMultiGroupProj', partition='pat', class_per_client=20, global_rounds=1, prev=0, times=1,
        seed=0, multi_cluster_seed=4, multi_cluster_restarts=3, multi_cluster_iterations=20,
        group_softmax_tau=.2, projection_softmax_tau=.2, multi_group_sizes=None, multi_group_weight=.5,
        multi_group_file='', multi_group_mode='group_projection_softmax', multi_no_pre_eval=False, fedfews_preview=False,
        data_expected_partition=None, data_expected_alpha=None)
    values.update(extra)
    value = options(**values)
    validate_multi_group(value)
    return value


class MultiGroupTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        if torch.cuda.is_available(): torch.cuda.get_rng_state_all()

    def test_configuration_and_capacity_boundaries(self):
        self.assertEqual(group_sizes(None, 20, 3), [7, 7, 6])
        self.assertEqual(args(num_clients=5,num_server_models=2,multi_group_sizes='2,3').multi_capacities,[3,2])
        for changed in [dict(num_server_models=21), dict(multi_group_sizes='7,7,7'), dict(local_grad_clip=-1),
                        dict(multi_group_weight=float('nan')),dict(multi_group_weight=1.1),dict(target_prox_mu=.1),
                        dict(join_ratio=.5),dict(is_regular=1),dict(num_server_models=1),dict(target_group_ids=[1]),
                        dict(target_head_mode='local'),dict(model_family='Decom_CNN-5-512'),dict(seed=-1)]:
            with self.subTest(changed=changed),self.assertRaises(ValueError):args(**changed)
        self.assertEqual(args(num_server_models=1,multi_group_weight=1.).multi_capacities,[20])

    def test_balanced_deterministic_zero_tied_and_nontrivial_similarity(self):
        state = rng_state()
        for similarity in [np.zeros((20,20)), np.ones((20,20)), np.eye(20)]:
            first, info=balanced_cluster(similarity,[7,7,6],seed=8,restarts=4)
            again,_=balanced_cluster(similarity,[7,7,6],seed=8,restarts=4)
            self.assertEqual(first,again)
            self.assertEqual(first,canonical_groups(first))
            self.assertEqual([len(g) for g in first],[7,7,6])
            self.assertEqual(sorted(i for g in first for i in g),list(range(20)))
            self.assertTrue(info['degenerate_similarity'])
        vectors=np.array([[1,0],[.9,.1],[-1,0],[-.9,-.1]])
        vectors/=np.sqrt((vectors*vectors).sum(axis=1,keepdims=True))
        similarity=(vectors[:,None,:]*vectors[None,:,:]).sum(axis=2)
        self.assertEqual(balanced_cluster(similarity,[2,2],seed=1,restarts=6)[0],[[0,1],[2,3]])
        assert_rng(self,state)

    def test_similarity_is_mean_within_column_delta_cosine_not_weight_or_cross_base(self):
        def model(values):
            m=nn.Linear(2,1,bias=False)
            with torch.no_grad():m.weight.copy_(torch.tensor([values],dtype=torch.float32))
            return m
        servers=[model([100,200]),model([-50,70])]
        clients=[SimpleNamespace(model_set=[model([101,200]),model([-50,71])]),
                 SimpleNamespace(model_set=[model([101,200]),model([-50,69])]),
                 SimpleNamespace(model_set=[model([100,200]),model([-50,70])])]
        state=rng_state()
        sim,info=update_similarity(servers,clients,chunk_size=1)
        np.testing.assert_array_equal(sim,np.array([[1,0,0],[0,1,0],[0,0,0]]))
        self.assertEqual(info['zero_update_clients_by_model'],[[2],[2]])
        assert_rng(self,state)

    def test_group_file_strict_mapping_context_and_fingerprint(self):
        context=dict(num_clients=4,K=2,capacities=[2,2],initial='test')
        groups=[[0,1],[2,3]]
        payload=dict(schema_version=1,context=context,context_sha256=digest_json(context),groups=groups,
            client_to_model={'0':0,'1':0,'2':1,'3':1},group_fingerprint=digest_json(dict(context=context,groups=groups)))
        self.assertEqual(validate_group_file(payload,context),groups)
        for change in [dict(groups=[[0,1],[1,3]]),dict(client_to_model={'0':1}),dict(context_sha256='bad'),dict(group_fingerprint='bad')]:
            with self.assertRaises(ValueError):validate_group_file(dict(payload,**change),context)

    def test_each_column_exactly_reuses_single_group_kernel_and_inputs_immutable(self):
        a=args(num_clients=4,num_server_models=2,multi_group_sizes='2,2')
        server=object.__new__(FedMultiGroupProj)
        server.args=a;server.K=2;server.loop_round=0;server.num_clients=4;server.groups=[[0,1],[2,3]]
        server.device='cpu';server.aggregation_records=[];server.weight_records=[]
        server.global_model_set=[nn.Linear(2,1),nn.Linear(2,1)]
        server.clients=[]
        for cid in range(4):
            models=copy.deepcopy(server.global_model_set)
            for k,m in enumerate(models):
                with torch.no_grad():
                    for p in m.parameters():p.add_((cid-1.4)*(k+1))
            server.clients.append(SimpleNamespace(id=cid,train_samples=cid+1,upload_round=0,model_set=models))
        server.selected_clients=list(reversed(server.clients))
        before=[[p.detach().clone() for p in m.parameters()] for c in server.clients for m in c.model_set]
        state=rng_state()
        for mode in ['group_projection_softmax','group_avg_only']:
            a.multi_group_mode=mode
            expected=[]
            for k,m in enumerate(server.global_model_set):
                weights={c.id:c.train_samples/10 for c in server.clients}
                def uploads():
                    for c in server.selected_clients:yield c.id,weights[c.id],dict(c.model_set[k].named_parameters())
                expected.append(aggregate_group_updates(dict(m.named_parameters()),uploads,weights,server.groups[k],mode,
                    .5 if mode=='group_projection_softmax' else None)[0])
            with patch('builtins.print'):server.aggregate_parameters()
            for model,values in zip(server.global_model_set,expected):
                for name,p in model.named_parameters():self.assertTrue(torch.equal(p,values[name]))
        after=[list(m.parameters()) for c in server.clients for m in c.model_set]
        self.assertTrue(all(torch.equal(x,y) for left,right in zip(before,after) for x,y in zip(left,right)))
        assert_rng(self,state)

    def _client(self):
        a=args(num_clients=4,num_server_models=2,local_epochs=2,batch_size=2,local_grad_clip=10)
        a.device='cpu';a.save_folder_name_full='unused';a.few_shot=0
        model=nn.Sequential(nn.Linear(2,4),nn.ReLU(),nn.Linear(4,100))
        return clientMultiGroupProj(a,0,3,3,initial_model=model,train_slow=False,send_slow=False)

    def test_k_separate_tasks_optimizer_binding_no_duplicate_training_and_download(self):
        client=self._client();data=[(torch.tensor([float(i),1.]),torch.tensor(i)) for i in range(3)]
        initial=copy.deepcopy(client.model_set)
        with patch('flcore.clients.clientMultiGroupProj.read_client_data',return_value=data):
            client.set_parameters(initial,0)
            client.train()
            self.assertEqual(len(client.train_stats),2)
            self.assertEqual([r['processed_train_samples'] for r in client.train_stats],[6,6])
            self.assertEqual([r['train_batches'] for r in client.train_stats],[4,4])
            for model,opt in zip(client.model_set,client.optimizer_set):
                self.assertEqual([id(p) for p in model.parameters()],[id(p) for p in opt.param_groups[0]['params']])
            with self.assertRaises(RuntimeError):client.train()
            client.set_parameters(initial,1);client.train()
            self.assertEqual(client.upload_round,1)

    def test_evaluation_restores_parameters_buffers_modes_rng_and_next_training(self):
        client=self._client()
        client.model_set=[nn.Sequential(nn.Linear(2,4),nn.BatchNorm1d(4),nn.ReLU(),nn.Linear(4,100)) for _ in range(2)]
        # Mixed per-module train/eval flags must also survive observation.
        client.model_set[0][1].eval()
        data=[(torch.tensor([float(i),1.]),torch.tensor(i)) for i in range(3)]
        state=rng_state();before=[copy.deepcopy(m.state_dict()) for m in client.model_set]
        modes=[[m.training for m in model.modules()] for model in client.model_set]
        with patch('flcore.clients.clientMultiGroupProj.read_client_data',return_value=data):
            observations=client.observe_models()
        self.assertEqual(len(observations),2);assert_rng(self,state)
        for model,weights,training in zip(client.model_set,before,modes):
            self.assertEqual([m.training for m in model.modules()],training)
            for name,p in model.state_dict().items():self.assertTrue(torch.equal(p,weights[name]))

    def test_formal_commands_public_parameters_and_reuse_switch(self):
        import shlex
        text=(ROOT/'system/multi_group_commands.sh').read_text()
        commands=[line.strip().strip('"') for line in text.splitlines() if line.startswith('"$MULTI_PYTHON main.py')]
        self.assertEqual(len(commands),2)
        expected_modes=['group_projection_softmax','group_avg_only']
        for command,mode in zip(commands,expected_modes):
            tokens=shlex.split(command)
            for key,value in dict([('-algo','FedMultiGroupProj'),('-m','CNN-512'),('-nc','20'),('-nsm','3'),
                ('--seed','0'),('-lr','0.005'),('-lbs','16'),('-ls','5'),('-gr','100'),('-eg','1'),
                ('--local_grad_clip','10'),('--data-subdir','pat_20'),('--data-formats','npz_only'),
                ('--multi_group_mode',mode),('--multi_group_weight','0.50'),('--group_softmax_tau','0.20')]).items():
                self.assertEqual(tokens[tokens.index(key)+1],value)
            self.assertIn('$MULTI_REUSE_ARG',tokens)

    def test_nonuniform_counts_fixed_membership_weighted_gain_and_summary(self):
        server=SimpleNamespace(loop_round=0,K=2,num_clients=4,groups=[[0,2],[1,3]],
            group_payload={'client_to_model':{'0':0,'2':0,'1':1,'3':1}},args=SimpleNamespace(multi_group_mode='group_avg_only'),
            primary_records=[],endpoint_records=[],group_records=[],rs_test_acc=[])
        after=[dict(client_id=i,model_id=k,correct=(i if k==i%2 else 0),test_samples=i+1,
                    accuracy=(i if k==i%2 else 0)/(i+1)) for i in range(4) for k in range(2)]
        before=[dict(r,correct=0,accuracy=0.) for r in after]
        with patch('builtins.print'):record_evaluation(server,before,after)
        self.assertEqual(server.primary_records[0]['post_local_accuracy'],6/10)
        self.assertEqual([r['post_local_accuracy'] for r in server.group_records],[2/4,4/6])
        self.assertEqual(server.primary_records[0]['same_round_finetune_gain'],.6)
        self.assertEqual(len(server.endpoint_records),16)


if __name__=='__main__':unittest.main()

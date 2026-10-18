"""Warmup stage boundaries, independent similarity formulas and rendezvous IO."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import threading
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch import nn

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'system'))
from common.experiment import rng_state, write_json
from utils.multi_group_clustering import update_similarity, digest_json, validate_group_file
from utils.multi_group_protocol import (GroupEvidence, grouping_protocol, phase_for_loop, read_group_file,
                                       atomic_publish_group)
from utils.multi_group_reporting import record_warmup_evaluation, record_evaluation, primary_summary
from test_multi_group import args
from test_control_population import assert_rng


def model(values):
    m=nn.Linear(len(values),1,bias=False)
    with torch.no_grad():m.weight.copy_(torch.tensor([values],dtype=torch.float32))
    return m


def oracle(servers,clients,full=False):
    matrices=[]
    for k,s in enumerate(servers):
        base=torch.cat([p.reshape(-1) for p in s.parameters()])
        vectors=[]
        for c in clients:
            post=torch.cat([p.reshape(-1) for p in c.model_set[k].parameters()])
            vectors.append((post if full else post-base).double())
        matrix=np.zeros((len(clients),len(clients)))
        for i,x in enumerate(vectors):
            for j,y in enumerate(vectors):
                denom=float(x.norm()*y.norm())
                matrix[i,j]=float(x.dot(y))/denom if denom else 0.
        matrices.append(matrix)
    return np.mean(matrices,axis=0)


class WarmupTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        if torch.cuda.is_available():torch.cuda.get_rng_state_all()

    def test_configuration_budget_and_no_ignored_window(self):
        a=args(global_rounds=100,multi_warmup_rounds=10,multi_group_basis='delta_history',multi_history_window=10)
        p=grouping_protocol(a)
        self.assertEqual(p['formation_completed_train_count'],10)
        self.assertEqual(p['grouped_start_loop_round'],10)
        self.assertEqual(p['evidence_loop_rounds'],list(range(10)))
        self.assertEqual(sum(phase_for_loop(a,r)=='warmup' for r in range(101)),10)
        self.assertEqual(sum(phase_for_loop(a,r)=='grouped' for r in range(101)),91)
        for change in [dict(multi_warmup_rounds=-1),dict(multi_warmup_rounds=True),
            dict(multi_warmup_rounds=2,global_rounds=1),dict(multi_group_basis='delta_history'),
            dict(multi_warmup_rounds=2,multi_group_basis='delta_history',multi_history_window=3),
            dict(multi_warmup_rounds=2,multi_group_basis='delta_history',multi_history_window=0),
            dict(multi_history_window=1),dict(multi_group_basis='full_weights',multi_history_window=1),
            dict(multi_group_wait_seconds=1),dict(multi_group_poll_seconds=.1),
            dict(multi_group_wait_seconds=float('nan'))]:
            with self.subTest(change=change),self.assertRaises(ValueError):args(global_rounds=10,**change) if 'global_rounds' not in change else args(**change)
        self.assertEqual(grouping_protocol(args(global_rounds=10,multi_warmup_rounds=10,
            multi_group_basis='delta_history'))['history_window'],10)
        from utils.cnn512_control import validate_control
        from test_control_population import options
        with self.assertRaises(ValueError):validate_control(options(algorithm='FedFewS',multi_warmup_rounds=10))

    def test_full_weight_cosine_is_not_delta_and_includes_all_parameters(self):
        servers=[model([100,20]),model([-60,100])]
        clients=[SimpleNamespace(model_set=[model([101,20]),model([-60,101])]),
                 SimpleNamespace(model_set=[model([99,20]),model([-60,99])])]
        state=rng_state()
        full,info=update_similarity(servers,clients,chunk_size=1,basis='full_weights')
        delta,_=update_similarity(servers,clients,chunk_size=1)
        np.testing.assert_allclose(full,oracle(servers,clients,True),rtol=0,atol=3e-16)
        self.assertGreater(full[0,1],.99);self.assertEqual(delta[0,1],-1.)
        self.assertIn('upload_full_weights_sha256',info)
        assert_rng(self,state)

    def test_history_tail_is_mean_of_per_round_per_column_cosines(self):
        a=args(global_rounds=4,multi_warmup_rounds=4,multi_group_basis='delta_history',multi_history_window=2)
        history=GroupEvidence(a);expected=[]
        for loop in range(4):
            servers=[model([100*loop,25]),model([17,-50*loop])]
            clients=[]
            for cid in range(3):
                posts=[]
                for k,s in enumerate(servers):
                    delta=torch.tensor([[float((cid+1)*(loop+1)),float((-1)**(cid+loop+k)*(k+1))]])
                    m=copy.deepcopy(s)
                    with torch.no_grad():m.weight.add_(delta)
                    posts.append(m)
                clients.append(SimpleNamespace(model_set=posts))
            if loop>=2:expected.append(oracle(servers,clients))
            history.collect(loop,servers,clients)
            # Later server mutation must never alter retained history evidence.
            with torch.no_grad():
                for s in servers:s.weight.add_(900)
        actual,evidence=history.finalize()
        np.testing.assert_allclose(actual,np.mean(expected,axis=0),rtol=0,atol=3e-16)
        self.assertEqual([r['loop_round'] for r in evidence['rounds']],[2,3])
        self.assertTrue(all(not isinstance(v,torch.Tensor) for r in history.records for v in r.values()))

    def test_atomic_publication_wait_timeout_invalid_and_rng(self):
        state=rng_state()
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'shared.json';payload=dict(message='complete',values=list(range(100)))
            with self.assertRaises(FileNotFoundError):read_group_file(path)
            with self.assertRaises(TimeoutError):read_group_file(path,wait_seconds=.025,poll_seconds=.005)
            errors=[]
            def publish():
                try:
                    time.sleep(.03);atomic_publish_group(path,payload)
                except Exception as e:errors.append(e)
            thread=threading.Thread(target=publish);thread.start()
            loaded,sha,waited=read_group_file(path,wait_seconds=2,poll_seconds=.005)
            thread.join();self.assertFalse(errors)
            self.assertEqual(loaded,payload);self.assertGreater(waited,.01)
            original=path.read_bytes()
            with self.assertRaises(FileExistsError):atomic_publish_group(path,dict(other='no overwrite'))
            self.assertEqual(path.read_bytes(),original)
            self.assertFalse(list(path.parent.glob('.*.tmp.*')))
            path.write_text('{half')
            with self.assertRaises(ValueError):read_group_file(path,wait_seconds=1,poll_seconds=.01)
        assert_rng(self,state)

    def test_evidence_compatibility_cannot_reuse_different_stage_or_window(self):
        a=args(global_rounds=2,multi_warmup_rounds=2,multi_group_basis='delta_history',multi_history_window=1)
        history=GroupEvidence(a)
        history.collect(1,[model([1,2]),model([3,4])],
            [SimpleNamespace(model_set=[model([2,2]),model([4,4])]),
             SimpleNamespace(model_set=[model([1,3]),model([3,5])])])
        matrix,evidence=history.finalize();p=grouping_protocol(a)
        context=dict(num_clients=2,K=2,capacities=[1,1],grouping_protocol=p)
        groups=[[0],[1]]
        payload=dict(schema_version=2,context=context,context_sha256=digest_json(context),groups=groups,
            client_to_model={'0':0,'1':1},group_fingerprint=digest_json(dict(context=context,groups=groups)),
            grouping_protocol=p,completed_train_count=2,evidence=evidence,similarity_matrix=matrix.tolist())
        self.assertEqual(validate_group_file(payload,context),groups)
        for key,value in [('completed_train_count',1),('grouping_protocol',dict(p,basis='full_weights')),
                          ('similarity_matrix',[[0,0],[0,0]])]:
            with self.assertRaises(ValueError):validate_group_file(dict(payload,**{key:value}),context)
        changed=dict(context,grouping_protocol=dict(p,history_window=2))
        with self.assertRaises(ValueError):validate_group_file(payload,changed)

    def test_warmup_endpoints_are_unassigned_and_excluded_from_primary(self):
        s=SimpleNamespace(loop_round=0,K=2,num_clients=2,warmup_records=[],primary_records=[],endpoint_records=[],
            group_records=[],rs_test_acc=[],args=SimpleNamespace(multi_warmup_rounds=2,multi_group_mode='group_avg_only'))
        before=[dict(client_id=i,model_id=k,correct=0,test_samples=i+1,accuracy=0.) for i in range(2) for k in range(2)]
        after=[dict(r,correct=r['test_samples'],accuracy=1.) for r in before]
        with patch('builtins.print'):record_warmup_evaluation(s,before,after)
        self.assertEqual(s.primary_records,[])
        self.assertTrue(all(r['assigned_model_id'] is None for r in s.endpoint_records))
        s.loop_round=2;s.groups=[[0],[1]];s.group_payload={'client_to_model':{'0':0,'1':1}}
        with patch('builtins.print'):record_evaluation(s,before,after)
        summary=primary_summary(s)
        self.assertEqual(summary['evaluation_loop_rounds'],[2]);self.assertEqual(summary['best_completed_train_count'],3)
        self.assertEqual(summary['warmup_train_count'],2);self.assertEqual(summary['grouped_train_count'],1)

    def test_last_warmup_uses_plain_avg_even_after_groups_formed(self):
        from flcore.servers.serverMultiGroupProj import FedMultiGroupProj
        s=object.__new__(FedMultiGroupProj)
        s.args=args(num_clients=3,num_server_models=2,global_rounds=2,multi_warmup_rounds=2)
        s.num_clients=3;s.K=2;s.loop_round=1;s.groups=[[0,1],[2]];s.device='cpu'
        s.global_model_set=[model([1,7]),model([9,-2])]
        s.clients=[SimpleNamespace(id=i,train_samples=i+1,upload_round=1,
            model_set=[model([i+2,9-i]),model([-i,3*i])]) for i in range(3)]
        s.selected_clients=list(reversed(s.clients));s.aggregation_records=[];s.weight_records=[]
        expected=[]
        for k in range(2):
            values=torch.zeros_like(s.global_model_set[k].weight)
            for c in s.selected_clients:values.add_(c.model_set[k].weight.detach()*(c.train_samples/6))
            expected.append(values)
        with patch('builtins.print'):s.aggregate_parameters()
        for k in range(2):self.assertTrue(torch.equal(s.global_model_set[k].weight,expected[k]))
        self.assertTrue(all(r['phase']=='warmup' and r['aggregation_mode']=='group_avg_all' for r in s.aggregation_records))

    def test_six_formal_commands_share_protocol_and_exact_artifact_per_pair(self):
        import shlex
        text=(ROOT/'system/multi_group_warmup_commands.sh').read_text()
        commands=[shlex.split(line.strip().strip('"')) for line in text.splitlines() if line.startswith('"$WARM_PYTHON main.py')]
        self.assertEqual(len(commands),6)
        get=lambda t,key:t[t.index(key)+1]
        for pair,(basis,warmup) in enumerate([('full_weights','0'),('full_weights','10'),('delta_history','10')]):
            producer,consumer=commands[2*pair:2*pair+2]
            self.assertEqual(get(producer,'--multi_group_publish'),get(consumer,'--multi_group_file'))
            self.assertEqual(get(consumer,'--multi_group_wait_seconds'),'$WARM_WAIT_SECONDS')
            for tokens in (producer,consumer):
                for key,value in [('-algo','FedMultiGroupProj'),('-nc','20'),('-nsm','3'),('--seed','0'),('-gr','100'),
                    ('-eg','1'),('-lr','0.005'),('-lbs','16'),('-ls','5'),('--control_test_batch_size','16'),
                    ('--multi_warmup_rounds',warmup),('--multi_group_basis',basis),('--data-subdir','pat_20')]:
                    self.assertEqual(get(tokens,key),value)
                if pair==2:self.assertEqual(get(tokens,'--multi_history_window'),'10')
                else:self.assertNotIn('--multi_history_window',tokens)
        self.assertEqual(len({get(tokens,'-sfn') for tokens in commands}),6)
        self.assertEqual(len({get(tokens,'--run_log_dir') for tokens in commands}),6)


if __name__=='__main__':unittest.main()

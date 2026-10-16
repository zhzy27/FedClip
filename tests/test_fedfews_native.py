"""Native FedFewS policy, public data routing and independent numerical checks."""
from contextlib import redirect_stdout
import argparse
import copy
import io
import json
import os
from pathlib import Path
import random
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'system'))
from common.client_data import audit_image_clients, read_image_client
from common.experiment import rng_state, set_rng
from common.models import FedAvgCNN
from flcore.clients.clientfedfews import clientFedFewS
from flcore.servers.serverfedfews import FedFewS
from utils.data_utils import read_client_data
from utils.fedfews_options import add_fedfews_arguments, validate_fedfews, prepare_fedfews_random
from utils.fedfews_reporting import collected_accuracy


def options(**extra):
    defaults = dict(algorithm='FedFewS', dataset='Cifar100/noniid_dir_20_a0p5', device='cpu', num_classes=100,
                    model_family='FedAvgCNN-plain', num_server_models=3, smooth_mu=.01, global_rounds=2,
                    local_epochs=1, batch_size=4, local_learning_rate=.005, eval_gap=10, num_clients=3,
                    join_ratio=1., target_head_mode='shared', target_local_schedule='joint', is_regular=0,
                    data_root='explicit_root', data_subdir=None, data_formats=None, data_expected_alpha=None,
                    data_expected_partition=None, fedfews_torch_threads=2, fedfews_test_batch_size=None,
                    fedfews_no_cache=False, prev=0, times=1, few_shot=0, save_folder_name_full='unused',
                    fedfews_rng='controlled', fedfews_control_seed=7, learning_rate_decay_gamma=.99)
    defaults.update(extra)
    args = SimpleNamespace(**defaults)
    validate_fedfews(args)
    return args


class NativeTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)

    def test_protocol_rejections_and_old_options_are_inactive(self):
        self.assertEqual((options().test_batch_size, options().use_cache), (256, True))
        for changed in [dict(smooth_mu=-1), dict(smooth_mu=float('nan')), dict(local_learning_rate=float('inf')),
                        dict(num_server_models=0), dict(join_ratio=.5), dict(learning_rate_decay=True),
                        dict(is_regular=1), dict(target_prox_mu=.1), dict(target_head_mode='local'),
                        dict(model_family='FedavgCNN'), dict(resume=True), dict(data_root=None),
                        dict(fedfews_test_batch_size=0), dict(data_subdir='another_suffix')]:
            with self.subTest(changed=changed), self.assertRaises(ValueError):
                options(**changed)
        # Legacy algorithm mu is deliberately unrelated to smooth_mu.
        self.assertEqual(options(mu=999).smooth_mu, .01)

    def test_base_hooks_are_opt_in_and_native_classes_do_not_use_vendor(self):
        from flcore.clients.clientbase import Client
        from flcore.servers.serverbase import Server
        self.assertIs(clientFedFewS.__bases__[0], Client)
        self.assertIs(FedFewS.__bases__[0], Server)
        self.assertEqual(Server.client_init_kwargs(object(), 0), {})
        parser=argparse.ArgumentParser()
        add_fedfews_arguments(parser)
        args=parser.parse_args([])
        self.assertIsNone(args.data_root)
        # Do not accidentally change legacy Prox/head final-directory naming.
        self.assertFalse(hasattr(args,'final_model_root'))
        for path in ['flcore/clients/clientfedfews.py','flcore/servers/serverfedfews.py',
                     'utils/fedfews_reporting.py','utils/fedfews_options.py']:
            self.assertNotIn('fedfews_compat', (ROOT/'system'/path).read_text(encoding='utf-8'))

    def test_public_explicit_reader_audit_formats_alpha_no_rng_or_rewrite(self):
        import pickle
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            directory = root/'Cifar100/noniid_dir_20_a0p5'
            for split in ['train','test']:
                (directory/split).mkdir(parents=True)
                for cid in range(3):
                    x = np.arange((cid+1)*3*32*32, dtype=np.float32).reshape(cid+1,3,32,32)
                    data = dict(x=list(x) if cid==1 else x, y=np.arange(cid+1))
                    with (directory/split/f'{cid}.pkl').open('wb') as stream:
                        pickle.dump(data,stream)
                    np.savez(directory/split/f'{cid}.npz',data=dict(x=x+100, y=np.arange(cid+1)))
            config = directory/'config.json'
            config.write_text(json.dumps(dict(num_clients=3, alpha=.1, partition='dir')))
            args = options(data_root=str(root), data_expected_alpha=.1, data_expected_partition='dir')
            torch_before = torch.get_rng_state().clone()
            original = config.read_bytes()
            audit = audit_image_clients(args)
            self.assertEqual(audit['actual_alpha'], .1)  # Ignore misleading a0p5 directory name.
            self.assertTrue(torch.equal(torch_before, torch.get_rng_state()))
            self.assertEqual(original, config.read_bytes())
            self.assertEqual([r['count'] for r in audit['files'] if r['split']=='train'], [1,2,3])
            for cid in range(3):
                a, b = read_client_data(args.dataset,cid,args), read_image_client(args,cid)
                self.assertTrue(all(torch.equal(x,y) and torch.equal(l,m) for (x,l),(y,m) in zip(a,b)))
            args.data_formats='npz_only'
            self.assertEqual(read_image_client(args,0)[0][0].flatten()[0], 100)
            with self.assertRaisesRegex(ValueError,'must agree'):
                read_client_data('Cifar100/another_partition',0,args)
            args.data_expected_alpha=.5
            with self.assertRaisesRegex(ValueError,'directory name is not evidence'):
                audit_image_clients(args)
            args=options(data_root=str(root),dataset='Cifar100',data_subdir='noniid_dir_20_a0p5')
            self.assertEqual(args.test_batch_size,256)
            self.assertEqual(len(read_image_client(args,1)),2)

    def test_native_client_ce_only_last_epoch_losses_and_optimizer_identity(self):
        args=options(num_server_models=2, local_epochs=2, batch_size=2, local_learning_rate=.005)
        model=torch.nn.Linear(4,3).double()
        data=[(torch.tensor([100.,-20.,5.,2.],dtype=torch.float64),torch.tensor(i%3)) for i in range(5)]
        with patch('flcore.clients.clientfedfews.read_client_data', return_value=data):
            client=clientFedFewS(args,0,5,5,initial_model=model,train_slow=False,send_slow=False)
        references=copy.deepcopy(client.model_set)
        identities=[[id(p) for p in m.parameters()] for m in client.model_set]
        client.set_parameters([model,model])
        self.assertEqual(identities,[[id(p) for p in o.param_groups[0]['params']] for o in client.optimizer_set])
        state=rng_state()
        losses=torch.zeros(2)
        loader=client.load_train_data()
        for k,m in enumerate(references):
            opt=torch.optim.SGD(m.parameters(),lr=.005)
            for ep in range(2):
                losses[k]=0
                for x,y in loader:
                    opt.zero_grad()
                    loss=torch.nn.functional.cross_entropy(m(x),y)
                    loss.backward(); opt.step()
                    losses[k]+=loss.item()*len(y)
        set_rng(state)
        client.train()
        self.assertTrue(np.array_equal(client.train_model_losses,(losses/5).numpy()))
        for a,b in zip(client.model_set,references):
            self.assertTrue(all(torch.equal(p,q) for p,q in zip(a.parameters(),b.parameters())))
        self.assertEqual(client.train_time_cost['num_rounds'],1)
        index, endpoint=client.select_best_model()
        self.assertEqual(index,int(endpoint.argmin()))
        self.assertFalse(np.array_equal(endpoint,client.train_model_losses))
        self.assertTrue(all(o.defaults['momentum']==0 and o.defaults['weight_decay']==0 for o in client.optimizer_set))

    def test_dual_weight_increment_unbalanced_receive_order_and_input_immutability(self):
        s=FedFewS.__new__(FedFewS)
        s.K,s.mu=3,.01
        def model(value):
            result=torch.nn.Linear(2,1,bias=False).double()
            result.weight.data.fill_(value)
            return result
        old=[1.,2.,-3.]
        s.global_model_set=[model(v) for v in old]
        losses=np.array([[1.,2.,3.],[3.,1.,2.],[2.,3.,1.]],dtype=np.float32)
        s.clients=[SimpleNamespace(id=i,train_model_losses=losses[i]) for i in range(3)]
        s.uploaded_ids=[2,0,1]
        counts=np.array([2,5,11])
        s.uploaded_weights=[counts[i]/counts.sum() for i in s.uploaded_ids]
        s.uploaded_models=[[model(old[k]+(i+1)*(k+1)*.2) for k in range(3)] for i in s.uploaded_ids]
        saved=copy.deepcopy(s.uploaded_models)
        weighted=losses[s.uploaded_ids].astype(float)*np.array(s.uploaded_weights)[:,None]
        exp=np.exp(-weighted/s.mu)
        sums=exp.sum(axis=1)
        alpha=(1/sums)/(1/sums).sum()
        w=exp/sums[:,None]
        s.aggregate_parameters()
        for k in range(3):
            expected=old[k]+sum(alpha[r]*w[r,k]*(cid+1)*(k+1)*.2 for r,cid in enumerate(s.uploaded_ids))
            self.assertAlmostEqual(s.global_model_set[k].weight[0,0].item(),expected,delta=2e-14)
        for a,b in zip(saved,s.uploaded_models):
            self.assertTrue(all(torch.equal(p,q) for m,n in zip(a,b) for p,q in zip(m.parameters(),n.parameters())))
        self.assertFalse(np.allclose((s.last_alpha*s.last_w).sum(axis=0),1))

    def test_original_and_controlled_rng_protocol(self):
        args=options(fedfews_rng='original')
        random.seed(9); np.random.seed(11); torch.manual_seed(99)
        before=rng_state()
        prepare_fedfews_random(args)
        after=rng_state()
        self.assertEqual(before['python'],after['python'])
        self.assertTrue(np.array_equal(before['numpy'][1],after['numpy'][1]))
        self.assertFalse(torch.equal(before['torch'],after['torch']))
        args.fedfews_rng='controlled'
        prepare_fedfews_random(args)
        a=(random.random(),np.random.rand(),torch.rand(1))
        prepare_fedfews_random(args)
        b=(random.random(),np.random.rand(),torch.rand(1))
        self.assertEqual(a[:2],b[:2]); self.assertTrue(torch.equal(a[2],b[2]))

    def test_subset_extraction_is_sample_weighted_and_read_only(self):
        e=dict(clients=[dict(client_id=0,correct=2,test_samples=4),dict(client_id=7,correct=9,test_samples=10)])
        original=copy.deepcopy(e)
        self.assertEqual(collected_accuracy(e,[0]),.5)
        self.assertEqual(collected_accuracy(e,[0,7]),11/14)
        self.assertEqual(e,original)


if __name__=='__main__':
    unittest.main()

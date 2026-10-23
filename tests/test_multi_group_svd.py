"""Capacity, real legacy SVD/F, upload isolation and heterogeneous group options."""
from contextlib import redirect_stdout
import copy
import io
from pathlib import Path
import re
import shlex
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'system'))
from test_multi_group import args as multi_args
from test_control_population import assert_rng
from common.experiment import rng_state, set_rng, sha256
from utils.cnn512_reporting import model_digest
from utils.multi_group_svd import (enabled, capacity_protocol, initial_model, recover_parameters,
    distance_squared, load_checkpoint, DEFAULT_RATIOS)
from utils.multi_group_clustering import update_similarity, grouping_context, validate_group_file
from utils.multi_group_clustering import digest_json
from utils.multi_group_reporting import record_evaluation
from flcore.clients.clientMultiGroupSVD import clientMultiGroupSVD
from flcore.trainmodel.models import Hyper_CNN_512


def options(**extra):
    value = dict(multi_svd=True, model_family='Decom_CNN-5-512', multi_group_basis='full_weights',
                 is_regular=1, regular_lamda=.001)
    value.update(extra)
    return multi_args(**value)


def client(folder, cid=0, diagnostics=1, device='cpu'):
    a = options(local_epochs=1, batch_size=2, multi_svd_diagnostics=diagnostics)
    a.device, a.save_folder_name_full, a.few_shot = device, str(folder), 0
    model = Hyper_CNN_512(num_classes=100, n_kernels=2, ratio_LR=1.).to(device)
    c = clientMultiGroupSVD(a, cid, 3, 3, initial_model=model, train_slow=False, send_slow=False)
    return c, [copy.deepcopy(model) for _ in range(3)]


class MultiSVDTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        if torch.cuda.is_available(): torch.cuda.get_rng_state_all()

    def test_options_capacity_assignment_and_rejections(self):
        a = options()
        self.assertEqual(a.multi_svd_effective_ratios, list(DEFAULT_RATIOS))
        self.assertEqual(a.multi_svd_effective_assignment, [i % 5 for i in range(20)])
        for i in range(5): self.assertEqual(a.multi_svd_effective_assignment.count(i), 4)
        custom = options(num_clients=5, num_server_models=1, multi_group_mode='group_avg_only',
            multi_svd_ratios='.5,.2', multi_svd_assignment='0,1,0,0,1')
        self.assertEqual(capacity_protocol(custom)['client_rank_rates'], [.5,.2,.5,.5,.2])
        for changes in [dict(model_family='CNN-512'), dict(multi_warmup_rounds=1), dict(multi_group_basis='single_delta'),
            dict(multi_svd_ratios='nan,.2'), dict(multi_svd_ratios='0'), dict(multi_svd_ratios='1.1'),
            dict(multi_svd_ratios='.2,.2'), dict(multi_svd_assignment='0,1'), dict(multi_svd_assignment=','.join(['5']*20)),
            dict(target_prox_mu=.1), dict(target_head_l2_lambda=.01), dict(regular_lamda=float('inf')),
            dict(target_head_mode='local'), dict(target_local_schedule='fedrep')]:
            with self.subTest(changes=changes), self.assertRaises(ValueError): options(**changes)
        with self.assertRaises(ValueError): multi_args(multi_svd_ratios='.5')
        self.assertFalse(enabled(multi_args()))

    def test_public_full_initialization_three_copies_and_rng(self):
        a = options(); before = rng_state()
        full = initial_model(a)
        assert_rng(self, before)
        self.assertEqual(full.ratio_LR, 1.)
        self.assertIsInstance(full.fc1, torch.nn.Linear)
        self.assertEqual(len({model_digest(full), model_digest(copy.deepcopy(full)), model_digest(copy.deepcopy(full))}), 1)
        self.assertIn('fc3.weight', dict(full.named_parameters()))

    def test_group_file_rejects_homogeneous_and_different_svd_capacity_protocols(self):
        data = dict(config_sha256='config', files=[dict(client_id=i, split='train', sha256='file',
            decoded_x_sha256='x', decoded_y_sha256='y', count=2) for i in range(20)])
        context = grouping_context(options(), data, 'same_initial_hash')
        old = grouping_context(multi_args(multi_group_basis='full_weights'), data, 'same_initial_hash')
        changed = grouping_context(options(multi_svd_assignment=','.join(['0']*20)), data, 'same_initial_hash')
        for incompatible in (old, changed):
            payload = dict(schema_version=2, context=incompatible, context_sha256=digest_json(incompatible))
            with self.assertRaisesRegex(ValueError, 'data/initialization/N/K/capacities/training protocol'):
                validate_group_file(payload, context)

    def test_five_real_capacities_start_truncation_is_not_zero_global_delta(self):
        with tempfile.TemporaryDirectory() as folder, redirect_stdout(io.StringIO()):
            for cid, ratio in enumerate(DEFAULT_RATIOS):
                c, mothers = client(folder, cid)
                c.set_parameters(mothers, 0)
                for k, model in enumerate(c.model_set):
                    self.assertEqual(model.ratio_LR, ratio)
                    for name in ('conv2','fc1','fc2'):
                        layer = getattr(model, name)
                        self.assertEqual(layer.rank, max(1, round(ratio*layer.max_rank)))
                    self.assertTrue(torch.equal(model.conv1.weight, mothers[k].conv1.weight))
                    self.assertTrue(torch.equal(model.fc3.weight, mothers[k].fc3.weight))
                    self.assertGreater(distance_squared(c.download_starts[k], dict(mothers[k].named_parameters())), 0.)
                    self.assertGreater(float(model.frobenius_decay()), 0.)
                    self.assertEqual(c.capacity_metadata[k]['trainable_parameter_count'], sum(p.numel() for p in model.parameters()))

    def test_optimizer_F_upload_checkpoint_reload_and_next_round(self):
        with tempfile.TemporaryDirectory() as folder, redirect_stdout(io.StringIO()):
            c, mothers = client(folder, cid=3)
            data = [(torch.randn(3,32,32), torch.tensor(i)) for i in range(3)]
            with patch('flcore.clients.clientMultiGroupProj.read_client_data', return_value=data):
                c.set_parameters(mothers, 0)
                starts = [model_digest(m) for m in c.model_set]
                factor_starts = [m.fc1.weight_u.detach().clone() for m in c.model_set]
                c.train()
                for k, (m, opt) in enumerate(zip(c.model_set,c.optimizer_set)):
                    self.assertNotEqual(model_digest(m), starts[k])
                    self.assertFalse(torch.equal(m.fc1.weight_u, factor_starts[k]))
                    self.assertEqual([id(p) for p in m.parameters()], [id(p) for p in opt.param_groups[0]['params']])
                    row = c.train_stats[k]
                    self.assertGreater(row['regularization_loss'], 0.)
                    self.assertGreater(row['local_training_recovered_norm'], 0.)
                    self.assertEqual(row['processed_train_samples'], 3)
                    path = Path(folder)/f'Client_3_model_{k}.pt'
                    fingerprint = sha256(path); before = rng_state()
                    independently_recovered = recover_parameters(m)
                    self.assertEqual(independently_recovered.keys(), c.full_upload_parameters(k).keys())
                    for name, value in independently_recovered.items():
                        self.assertTrue(torch.equal(value, c.full_upload_parameters(k)[name]))
                    assert_rng(self, before)
                    self.assertEqual(sha256(path), fingerprint)
                    restored, payload = load_checkpoint(path)
                    self.assertEqual(payload['role'], 'ordinary_post_local_low_rank')
                    self.assertEqual(model_digest(restored), model_digest(m))
                    self.assertEqual(restored.ratio_LR, c.rank_rate)
                    probe = torch.zeros(1,3,32,32)
                    self.assertTrue(torch.equal(restored(probe), copy.deepcopy(m).eval()(probe)))
                with self.assertRaises(RuntimeError): c.train()
                c.set_parameters(mothers, 1)
                self.assertEqual([m.ratio_LR for m in c.model_set], [.25]*3)
                with self.assertRaises(RuntimeError): c.full_upload_parameters(0)
                c.optimizer_set.pop()
                with self.assertRaises(RuntimeError): c.train()

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA unavailable')
    def test_cuda_new_modules_optimizer_and_recovered_CPU_upload(self):
        with tempfile.TemporaryDirectory() as folder, redirect_stdout(io.StringIO()):
            c, mothers = client(folder, cid=4, device='cuda')
            data = [(torch.randn(3,32,32),torch.tensor(i)) for i in range(3)]
            with patch('flcore.clients.clientMultiGroupProj.read_client_data',return_value=data):
                c.set_parameters(mothers,0)
                self.assertTrue(all(p.is_cuda for m in c.model_set for p in m.parameters()))
                c.train()
                self.assertTrue(all(not p.is_cuda for post in c.full_uploads for p in post.values()))
                self.assertTrue(all(r['regularization_loss'] > 0 for r in c.train_stats))

    def test_diagnostics_and_endpoint_observation_do_not_change_SGD_uploads_or_rng(self):
        data = [(torch.randn(3,32,32), torch.tensor(i)) for i in range(3)]
        initial_rng = rng_state(); outcomes = []
        with tempfile.TemporaryDirectory() as folder, redirect_stdout(io.StringIO()):
            for observe in (False, True):
                set_rng(initial_rng)
                c, mothers = client(folder, diagnostics=int(observe))
                with patch('flcore.clients.clientMultiGroupProj.read_client_data', return_value=data):
                    c.set_parameters(mothers, 0)
                    if observe:
                        before = rng_state(); c.observe_models(); assert_rng(self, before)
                    c.train()
                    if observe:
                        before = rng_state(); c.observe_models(); assert_rng(self, before)
                outcomes.append(([model_digest(m) for m in c.model_set],
                    [[v.clone() for v in post.values()] for post in c.full_uploads], rng_state()))
        self.assertEqual(outcomes[0][0], outcomes[1][0])
        for old, new in zip(outcomes[0][1], outcomes[1][1]):
            for x, y in zip(old, new): self.assertTrue(torch.equal(x,y))
        set_rng(outcomes[0][2]); assert_rng(self, outcomes[1][2])

    def test_recovered_full_weight_cosines_average_columns_and_exclude_factor_aliases(self):
        mothers = [torch.nn.Linear(2,1), torch.nn.Linear(2,1)]
        posts = [[{'weight':torch.tensor([[float(i+k),1.]]), 'bias':torch.tensor([.2*i])} for k in range(2)] for i in range(4)]
        from types import SimpleNamespace
        clients = [SimpleNamespace(full_upload_parameters=lambda k, i=i: posts[i][k]) for i in range(4)]
        matrix, diagnostics = update_similarity(mothers, clients, basis='full_weights')
        expected = []
        for k in range(2):
            v = torch.stack([torch.cat([p.flatten() for p in row[k].values()]).double() for row in posts])
            expected.append((v@v.T / (v.norm(dim=1)[:,None]*v.norm(dim=1)[None,:])).numpy())
        np.testing.assert_allclose(matrix, np.mean(expected,axis=0), rtol=0, atol=1e-15)
        np.testing.assert_allclose(diagnostics['per_model_cosine_matrices'], expected, rtol=0, atol=1e-15)

    def test_three_formal_commands_budgets_and_paired_group_artifact(self):
        text = (ROOT/'system/multi_group_svd_commands.sh').read_text()
        commands = [shlex.split(line.strip('"')) for line in text.splitlines() if line.startswith('"python -u main.py')]
        self.assertEqual(len(commands), 3)
        for tokens, k, mode, sizes in zip(commands, ['1','3','3'], ['group_avg_only','group_avg_only','group_projection_softmax'], ['20','7,7,6','7,7,6']):
            self.assertIn('--multi_svd', tokens)
            for flag, value in [('-nsm',k),('--multi_group_mode',mode),('--multi_group_sizes',sizes),('-gr','100'),('-ls','5'),
                ('-is_regular','1'),('-regular_lamda','1e-3'),('-nc','20'),('--multi_group_basis','full_weights'),
                ('--multi_warmup_rounds','0'),('-m','Decom_CNN-5-512'),('--local_grad_clip','10')]:
                self.assertEqual(tokens[tokens.index(flag)+1], value)
        self.assertEqual(commands[1][commands[1].index('--multi_group_file')+1], commands[2][commands[2].index('--multi_group_publish')+1])


if __name__ == '__main__': unittest.main()

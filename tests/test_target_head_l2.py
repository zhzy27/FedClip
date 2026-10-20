"""Explicit head-only gradients, joint CE/F preservation, options and read-only endpoints."""

from contextlib import redirect_stdout
import copy
import io
import json
from pathlib import Path
import re
import shlex
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'system'))
from flcore.clients.clientTargetProj import clientTargetProj
from flcore.clients.clientbase import save_item
from utils.target_head_l2 import (validate_head_l2, ensure_head_l2_names, l2_weight,
    head_l2_penalty, head_weight_norm, observe_target, head_l2_tag)
from utils.head_snapshot import tensor_fingerprint, file_hash
from test_target_local_head import AliasModel
from test_meta_projection import rng_state, assert_rng
import run_target_proj as launcher


def options(**updates):
    data = dict(algorithm='FedTargetProj', target_proj_mode='avg', target_head_mode='local',
        target_local_schedule='joint', model_family='Decom_CNN-5-512', target_prox_mu=0.,
        target_head_l2_lambda=.001, target_head_l2_diagnostics=1, num_clients=20,
        exp_name='avg_local_head', save_folder_name='temp', final_model_root='final',
        h5_result_root='h5', training_log_root='logs', run_log_dir='', is_regular=1, regular_lamda=.001)
    data.update(updates)
    return SimpleNamespace(**data)


class JointModel(AliasModel):
    def __init__(self):
        super().__init__()
        self.register_buffer('probe', torch.tensor([3.]))
        self.dropout = torch.nn.Dropout(.2)
    def forward(self, x):
        if not self.training:
            self.probe.add_(1.)  # Deliberately impure inference; observation must use a private copy.
        return self.head(self.dropout(self.fc2(torch.relu(self.fc1(x)))))
    def frobenius_decay(self):
        return self.fc1.weight.square().sum() + self.fc2.weight.square().sum()


class HeadL2Tests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)

    def test_validation_and_zero_noop_names(self):
        validate_head_l2(options())
        for changes in (dict(target_head_l2_lambda=-1), dict(target_head_l2_lambda=float('nan')),
            dict(target_head_l2_lambda=float('inf')), dict(algorithm='FedFewS'), dict(target_proj_mode='projection_softmax'),
            dict(target_proj_mode='feedback_avg'), dict(target_proj_mode='group_avg_only'), dict(target_proj_mode='dwa_soft'),
            dict(target_proj_mode='meta_projection_fixed'), dict(target_local_schedule='fedrep'),
            dict(target_head_mode='shared'), dict(model_family='CNN-512'), dict(target_prox_mu=.01),
            dict(target_prox_mu=float('nan')), dict(meta_c0_split='split.json')):
            with self.subTest(changes=changes), self.assertRaises(ValueError): validate_head_l2(options(**changes))
        zero = options(target_head_l2_lambda=0., algorithm='anything', target_proj_mode='anything')
        before = copy.deepcopy(vars(zero))
        validate_head_l2(zero); ensure_head_l2_names(zero)
        self.assertEqual(vars(zero), before)
        args = options()
        ensure_head_l2_names(args)
        before = copy.deepcopy(vars(args)); ensure_head_l2_names(args)
        self.assertEqual(vars(args), before)
        for key in ('exp_name', 'save_folder_name', 'final_model_root', 'h5_result_root', 'training_log_root'):
            self.assertIn(head_l2_tag(.001), getattr(args, key))

    def test_penalty_gradient_aliases_and_preserved_CE_F_interface(self):
        model = JointModel().double()
        coefficient = .01
        weight = l2_weight(model)
        self.assertIs(weight, model.head[1].weight)
        penalty = head_l2_penalty(weight, coefficient)
        torch.testing.assert_close(penalty, coefficient / 2 * weight.square().sum(), rtol=0, atol=0)
        penalty.backward()
        torch.testing.assert_close(model.fc3.weight.grad, coefficient * model.fc3.weight, rtol=0, atol=0)
        for name, p in model.named_parameters():
            if name != 'fc3.weight': self.assertIsNone(p.grad)
        client = clientTargetProj.__new__(clientTargetProj)
        client.args, client.loss = options(), torch.nn.CrossEntropyLoss()
        model.zero_grad()
        ce, f = client._objective(model, torch.randn(4, 2, dtype=torch.double), torch.arange(4))
        self.assertGreater(float(f), 0.)
        grads = torch.autograd.grad(ce + f, tuple(model.parameters()), retain_graph=True)
        combined = torch.autograd.grad(ce + f + head_l2_penalty(weight, coefficient), tuple(model.parameters()))
        for (name, p), old, new in zip(model.named_parameters(), grads, combined):
            expected = old + coefficient * p if name == 'fc3.weight' else old
            torch.testing.assert_close(new, expected, atol=1e-14, rtol=1e-14)

    def test_sample_visit_weighted_joint_statistics_and_all_parameter_clipping(self):
        with tempfile.TemporaryDirectory() as directory:
            model = JointModel()
            client = clientTargetProj.__new__(clientTargetProj)
            client.args, client.loss = options(), torch.nn.CrossEntropyLoss()
            client.id, client.role, client.device, client.train_slow = 7, 'Client_7', 'cpu', False
            client.learning_rate, client.local_epochs = .005, 2
            client.save_folder_name = directory
            client.train_time_cost = dict(num_rounds=0, total_cost=0.)
            client._load_model = lambda: model
            data = list(zip(torch.randn(5, 2), torch.arange(5)))
            client.load_train_data = lambda: DataLoader(data, batch_size=3, shuffle=True)
            native_objective, native_clip = client._objective, torch.nn.utils.clip_grad_norm_
            batches, clip_ids = [], []
            def objective(model, x, y):
                ce, f = native_objective(model, x, y)
                l2 = head_l2_penalty(l2_weight(model), .001)
                batches.append((len(y), float(ce), float(f), float(l2), float(ce + f + l2)))
                return ce, f
            def clip(parameters, cap):
                parameters = list(parameters)
                clip_ids.append({id(p) for p in parameters})
                self.assertEqual(cap, 10.)
                return native_clip(parameters, cap)
            client._objective = objective
            with redirect_stdout(io.StringIO()), patch('torch.nn.utils.clip_grad_norm_', side_effect=clip): client.train(3)
            row = client.last_head_l2_train_stats
            self.assertEqual(row['train_sample_visits'], 10)
            self.assertEqual(row['train_batches'], 4)
            for col, name in enumerate(('train_mean_ce', 'train_mean_weighted_frobenius', 'train_mean_weighted_head_l2', 'train_mean_total_loss'), 1):
                self.assertAlmostEqual(row[name], sum(r[0] * r[col] for r in batches) / 10, places=10)
            self.assertEqual(clip_ids, [{id(p) for p in model.parameters()}] * 4)
            self.assertEqual(row['head_weight_norm'], head_weight_norm(model))
            self.assertEqual(client.last_normal_train_stats['ce_loss'], row['train_mean_ce'])

    def test_endpoint_isolation_checkpoint_buffers_modes_and_RNG(self):
        with tempfile.TemporaryDirectory() as directory:
            model = JointModel(); model.fc2.eval()
            client = clientTargetProj.__new__(clientTargetProj)
            client.id, client.device, client.loss = 0, 'cpu', torch.nn.CrossEntropyLoss()
            client._load_model = lambda: model
            data = list(zip(torch.randn(5, 2), torch.arange(5)))
            client.load_train_data = lambda: DataLoader(data, batch_size=3, shuffle=True)
            client.load_test_data = lambda: DataLoader(data, batch_size=3, shuffle=False)
            save_item(model, 'Client_0', 'model', directory)
            path = Path(directory) / 'Client_0_model.pt'
            before_hash, before_state = file_hash(path), tensor_fingerprint(model.state_dict())
            modes, rng = [m.training for m in model.modules()], rng_state()
            with torch.no_grad():
                candidate = copy.deepcopy(model).eval()
                expected = int((candidate(torch.stack([x for x, y in data])).argmax(1) == torch.arange(5)).sum()) / 5
            server = SimpleNamespace(args=options(), clients=[client], target_client_id=0, cur_ground=0,
                target_proj_mode='avg', seed=0, _post_local_acc=expected, head_l2_target=[],
                _append_csv=lambda *a: None)
            with redirect_stdout(io.StringIO()), patch('utils.target_head_l2.save_json'):
                observe_target(server)
            assert_rng(self, rng)
            self.assertEqual(tensor_fingerprint(model.state_dict()), before_state)
            self.assertEqual([m.training for m in model.modules()], modes)
            self.assertEqual(file_hash(path), before_hash)
            self.assertEqual(server.head_l2_target[0]['endpoint_test_accuracy'], expected)
            self.assertEqual(server.head_l2_target[0]['endpoint_train_samples'], 5)

    def test_launcher_forwarding_rejection_and_formal_commands(self):
        with patch.object(sys, 'argv', ['run_target_proj.py', '--modes', 'avg', '--target_head_mode', 'local',
                '--target_head_l2_lambda', '.001', '--target_head_l2_diagnostics', '0', '--rounds', '30', '--dry-run']), redirect_stdout(io.StringIO()) as out:
            launcher.main()
        text = out.getvalue()
        self.assertIn('--target_head_l2_lambda 0.001', text)
        self.assertIn('--target_head_l2_diagnostics 0', text)
        self.assertIn('head_l2_0p001', text)
        with patch.object(sys, 'argv', ['run_target_proj.py', '--modes', 'projection', '--target_head_l2_lambda', '.001', '--dry-run']), redirect_stdout(io.StringIO()), patch('sys.stderr', io.StringIO()):
            with self.assertRaises(SystemExit) as failure: launcher.main()
            self.assertEqual(failure.exception.code, 2)
        commands = re.findall(r'^\s+"([^"]+)"$', (ROOT / 'system/target_head_l2_commands.sh').read_text(), re.MULTILINE)
        self.assertEqual(len(commands), 3)
        roots = []
        for command, value in zip(commands, ('1e-4', '1e-3', '1e-2')):
            parts = shlex.split(command)
            for key, expected in (('--target_head_l2_lambda', value), ('-gr', '30'), ('-ls', '5'), ('-nc', '20'),
                ('-ncl', '100'), ('-m', 'Decom_CNN-5-512'), ('-data', 'Cifar100'), ('-lr', '0.005'), ('-lbs', '16'),
                ('-algo', 'FedTargetProj'), ('--target_proj_mode', 'avg'), ('--target_head_mode', 'local'),
                ('--target_local_schedule', 'joint'), ('--target_prox_mu', '0'), ('-is_regular', '1'),
                ('-regular_lamda', '1e-3'), ('-niid', '1'), ('-pt', 'pat'), ('-cpc', '20'), ('-eg', '1'), ('-jr', '1.0')):
                self.assertEqual(parts[parts.index(key) + 1], expected)
            roots.append([parts[parts.index(k) + 1] for k in ('-sfn', '--h5_result_root', '--final-model-root', '--run_log_dir', '-exp_name')])
        for column in zip(*roots): self.assertEqual(len(set(column)), 3)


if __name__ == '__main__': unittest.main()

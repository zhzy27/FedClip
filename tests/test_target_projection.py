"""Numerical and synthetic lifecycle tests; no CLIP weights or dataset required."""

import copy
import csv
import importlib.util
import io
import json
import math
from pathlib import Path
import random
import sys
import tempfile
import types
import unittest
from contextlib import redirect_stdout
from unittest import mock

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "system"))
from utils.target_projection import EPS, aggregate_target_updates


def params(values, dtype=torch.float64):
    return {"base.weight": torch.tensor(values[:2], dtype=dtype),
            "head.bias": torch.tensor(values[2:], dtype=dtype)}


def vector(parameters):
    return torch.cat([value.reshape(-1) for value in parameters.values()])


class ProjectionMathTests(unittest.TestCase):
    def test_three_modes_match_independent_flattened_reference(self):
        initial = params([5.0, -3.0, 2.0])
        deltas = [params([1.0, 2.0, -1.0]), params([-3.0, 1.0, 4.0]), params([2.0, 0.0, 1.0])]
        weights = [0.2, 0.3, 0.5]
        uploads = [(cid, weights[cid], {n: initial[n] + d[n] for n in initial})
                   for cid, d in enumerate(deltas)]
        target = vector(deltas[0])
        corrected = []
        removed = 0.0
        for cid, delta in enumerate(deltas):
            raw = vector(delta)
            dot = raw @ target
            projected = raw - min(float(dot), 0.0) / (float(target @ target) + EPS) * target if cid else raw
            corrected.append(projected)
            if cid:
                removed += float(torch.linalg.vector_norm(raw - projected))
        expected = {
            "avg": vector(initial) + sum(w * vector(d) for w, d in zip(weights, deltas)),
            "target_only": vector(initial) + target,
            "projection": vector(initial) + sum(w * d for w, d in zip(weights, corrected)),
        }
        snapshot = copy.deepcopy(uploads)
        for mode in expected:
            result, stats, rows = aggregate_target_updates(initial, uploads[0][2], iter(uploads), 0, mode)
            torch.testing.assert_close(vector(result), expected[mode])
            self.assertEqual(stats["conflict_client_ratio"], 0.5)
            denominator = sum(float(torch.linalg.vector_norm(vector(d))) for d in deltas[1:]) + EPS
            self.assertAlmostEqual(stats["removed_update_ratio"], removed / denominator)
            self.assertAlmostEqual(rows[1]["dot_after_projection"], 0.0, places=10)
            self.assertEqual(rows[0]["removed_norm"], 0.0)
            self.assertEqual(rows[2]["removed_norm"], 0.0)
        for original, after in zip(snapshot, uploads):
            torch.testing.assert_close(vector(original[2]), vector(after[2]), rtol=0, atol=0)

    def test_projection_is_global_not_layerwise(self):
        initial = params([0.0, 0.0, 0.0])
        target = params([1.0, 0.0, 1.0])
        other = params([-2.0, 0.0, 3.0])  # first layer conflicts; full model does not
        uploads = [(0, 0.25, target), (1, 0.75, other)]
        result, stats, _ = aggregate_target_updates(initial, target, uploads, 0, "projection")
        torch.testing.assert_close(vector(result), 0.25 * vector(target) + 0.75 * vector(other))
        self.assertEqual(stats["removed_update_ratio"], 0.0)

    def test_negative_full_dot_uses_one_coefficient_for_head_and_base(self):
        initial = params([0.0, 0.0, 0.0])
        target = params([1.0, 0.0, 2.0])
        other = params([1.0, 0.0, -2.0])
        result, _, _ = aggregate_target_updates(initial, target, [(4, 0.8, other), (2, 0.2, target)], 2, "projection")
        projected = vector(other) + 3 / (5 + EPS) * vector(target)
        torch.testing.assert_close(vector(result), 0.8 * projected + 0.2 * vector(target))

    def test_zero_target_and_single_client_are_finite(self):
        initial = params([1.0, 2.0, 3.0])
        for uploads in ([(0, 1.0, initial)], [(0, 0.1, initial), (1, 0.9, params([3.0, -1.0, 2.0]))]):
            for mode in ("avg", "target_only", "projection"):
                _, stats, _ = aggregate_target_updates(initial, initial, uploads, 0, mode)
                self.assertTrue(all(math.isfinite(value) for value in stats.values()))
                self.assertEqual(stats["avg_target_cos"], 0.0)
                self.assertEqual(stats["proj_target_cos"], 0.0)
                self.assertEqual(stats["removed_update_ratio"], 0.0)

    def test_tiny_target_uses_epsilon_without_hidden_threshold(self):
        initial = params([0.0, 0.0, 0.0])
        target = params([1e-8, 0.0, 0.0])
        other = params([-1.0, 2.0, 0.0])
        result, stats, _ = aggregate_target_updates(initial, target, [(0, 0.5, target), (1, 0.5, other)], 0, "projection")
        expected = 0.5 * vector(target) + 0.5 * (vector(other) + 1e-8 / (1e-16 + EPS) * vector(target))
        torch.testing.assert_close(vector(result), expected)
        self.assertEqual(stats["conflict_client_ratio"], 1.0)

    def test_invalid_uploads_fail_clearly(self):
        initial = params([0.0, 0.0, 0.0])
        target = params([1.0, 0.0, 0.0])
        bad_cases = [
            [(1, 1.0, target)],
            [(0, 0.5, target)],
            [(0, 0.5, target), (0, 0.5, target)],
            [(0, 1.1, target), (1, -0.1, target)],
            [(0, 0.5, target), (1, 0.5, {"wrong": torch.ones(3)})],
            [(0, 0.5, target), (1, 0.5, params([float("nan"), 0.0, 0.0]))],
        ]
        for uploads in bad_cases:
            with self.subTest(uploads=uploads), self.assertRaises(ValueError):
                aggregate_target_updates(initial, target, uploads, 0, "projection")


def load_module(name, relative_path, dependencies):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative_path)
    module = importlib.util.module_from_spec(spec)
    with mock.patch.dict(sys.modules, dependencies):
        spec.loader.exec_module(module)
    return module


class TinyModel(torch.nn.Module):
    def __init__(self, low_rank=False, offset=0.0):
        super().__init__()
        self.ratio_LR = 0.5 if low_rank else 1.0
        self.bias = torch.nn.Parameter(torch.tensor([0.1 + offset, 0.2]))
        if low_rank:
            self.weight_u = torch.nn.Parameter(torch.tensor([[1.0 + offset], [2.0]]))
            self.weight_v = torch.nn.Parameter(torch.tensor([[0.3, -0.1]]))
        else:
            self.weight = torch.nn.Parameter(torch.tensor([[0.3, -0.1], [0.6, -0.2]]))
        self.register_buffer("running_stat", torch.tensor([17.0 + offset]))

    def recover_larger_model(self):
        if hasattr(self, "weight_v"):
            full = self.weight_u @ self.weight_v
            del self.weight_u, self.weight_v
            self.weight = torch.nn.Parameter(full.detach())

    def decom_larger_model(self, ratio):
        u, s, vh = torch.linalg.svd(self.weight.detach(), full_matrices=False)
        self.weight_u = torch.nn.Parameter(u[:, :1] * s[:1])
        self.weight_v = torch.nn.Parameter(vh[:1])
        del self.weight
        self.ratio_LR = ratio

    def forward(self, x):
        weight = self.weight_u @ self.weight_v if hasattr(self, "weight_v") else self.weight
        return torch.nn.functional.linear(x, weight, self.bias)

    def frobenius_decay(self):
        return (self.weight_u @ self.weight_v).square().sum()


class FakeServer:
    def _to_json_serializable(self, value):
        return value

    def select_clients(self):
        return self.clients

    def send_parameters(self):
        for client in self.clients:
            client.set_parameters()

    def receive_ids(self):
        clients = random.sample(self.selected_clients, len(self.selected_clients))
        self.uploaded_ids = [client.id for client in clients]
        total = sum(client.train_samples for client in clients)
        self.uploaded_weights = [client.train_samples / total for client in clients]

    def evaluate(self, epoch=0):
        self.rs_test_acc.append(0.5)

    def save_results(self):
        self.saved_results = True

    def save_json_file(self):
        self.saved_json = True


class FakeClientBase:
    pass


class ProjectionIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.store = {}

        def load_item(role, name, folder):
            return copy.deepcopy(self.store.get((folder, role, name)))

        def save_item(value, role, name, folder):
            self.store[folder, role, name] = copy.deepcopy(value)

        self.load_item, self.save_item = load_item, save_item
        clientbase = types.ModuleType("flcore.clients.clientbase")
        clientbase.load_item, clientbase.save_item = load_item, save_item
        clientbase.Client = FakeClientBase
        serverbase = types.ModuleType("flcore.servers.serverbase")
        serverbase.Server = FakeServer
        models = types.ModuleType("flcore.trainmodel.models")
        models.Model_Distribe = mock.Mock()
        deps = {
            "flcore.clients.clientbase": clientbase,
            "flcore.servers.serverbase": serverbase,
            "flcore.trainmodel.models": models,
        }
        self.client_module = load_module("target_test_client", "system/flcore/clients/clientTargetProj.py", deps)
        deps["flcore.clients.clientTargetProj"] = self.client_module
        self.server_module = load_module("target_test_server", "system/flcore/servers/serverTargetProj.py", deps)

    def make_server(self, folder, mode):
        server = self.server_module.FedTargetProj.__new__(self.server_module.FedTargetProj)
        server.target_client_id = 1
        server.target_proj_mode = mode
        server.seed = 0
        server.role, server.save_folder_name, server.device = "Server", folder, "cpu"
        server.num_clients = 3
        server.clients = [types.SimpleNamespace(id=i, role=f"Client_{i}", save_folder_name=folder,
                                               train_samples=i + 1) for i in range(3)]
        server.uploaded_ids = [2, 0, 1]
        server.uploaded_weights = [0.5, 1 / 6, 1 / 3]
        server.cur_ground, server.global_rounds, server.eval_gap = 0, 1, 1
        server.target_proj_history = []
        server.args = types.SimpleNamespace(target_proj_mode=mode, target_client_id=1, seed=0)
        server.clients[1].test_downloaded_global = mock.Mock(return_value=(3, 4, 0))
        self.save_item(TinyModel(), "Server", "model", folder)
        for client in server.clients:
            self.save_item(TinyModel(low_rank=True, offset=client.id - 1.5), client.role, "model", folder)
        return server

    def test_avg_matches_sample_weighted_recovered_models_and_preserves_buffers(self):
        with tempfile.TemporaryDirectory() as folder, redirect_stdout(io.StringIO()):
            server = self.make_server(folder, "avg")
            originals = copy.deepcopy(self.store)
            expected = self.load_item("Server", "model", folder)
            with torch.no_grad():
                for p in expected.parameters():
                    p.zero_()
                for cid, weight in zip(server.uploaded_ids, server.uploaded_weights):
                    source = self.load_item(f"Client_{cid}", "model", folder)
                    source.recover_larger_model()
                    for name, p in expected.named_parameters():
                        p.add_(dict(source.named_parameters())[name] * weight)
            server.aggregate_parameters_avg()
            actual = self.load_item("Server", "model", folder)
            for name, value in actual.state_dict().items():
                torch.testing.assert_close(value, expected.state_dict()[name], rtol=0, atol=0)
            self.assertEqual(actual.running_stat.item(), 17.0)
            for client in server.clients:
                key = (folder, client.role, "model")
                for name, value in self.store[key].state_dict().items():
                    torch.testing.assert_close(value, originals[key].state_dict()[name], rtol=0, atol=0)
            self.assertEqual(server.target_proj_history[0]["target_client_test_acc"], 0.75)
            with open(Path(folder) / "target_proj_metrics.json", encoding="utf-8") as stream:
                saved = json.load(stream)
            self.assertEqual(saved["history"], server.target_proj_history)
            with open(Path(folder) / "target_proj_clients.csv", newline="", encoding="utf-8") as stream:
                self.assertEqual(len(list(csv.DictReader(stream))), 3)

    def make_client(self, folder, client_id):
        client = self.client_module.clientTargetProj.__new__(self.client_module.clientTargetProj)
        client.id, client.role = client_id, f"Client_{client_id}"
        client.save_folder_name, client.device = folder, "cpu"
        client.learning_rate, client.local_epochs, client.train_slow = 0.05, 1, False
        client.train_samples = 2
        client.args = types.SimpleNamespace(is_regular=1, regular_lamda=0.7)
        client.loss = torch.nn.CrossEntropyLoss()
        client.train_time_cost = {"num_rounds": 0, "total_cost": 0.0}
        client.send_time_cost = {"num_rounds": 0, "total_cost": 0.0}
        batches = [(torch.tensor([[1.0, -0.5], [-0.2, 0.3]]), torch.tensor([0, 1]))]
        client.load_train_data = lambda: batches
        client.load_test_data = lambda: batches
        return client

    def test_standalone_training_loop_with_real_local_updates_for_all_modes(self):
        for mode in ("avg", "target_only", "projection"):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as folder, redirect_stdout(io.StringIO()):
                server = self.make_server(folder, mode)
                server.Budget, server.rs_test_acc = [], []
                server.auto_break = False
                server.client_drop_rate, server.current_num_join_clients = 0.0, 3
                server.clients = [self.make_client(folder, cid) for cid in range(3)]
                for client in server.clients:
                    client.train = mock.Mock(side_effect=client.train)
                server.train()
                self.assertEqual(len(server.target_proj_history), 2)
                self.assertEqual([r["round"] for r in server.target_proj_history], [1, 2])
                self.assertTrue(server.saved_results and server.saved_json)
                for client in server.clients:
                    self.assertEqual(client.train.call_count, 2)
                if mode == "target_only":
                    target = self.load_item("Client_1", "model", folder)
                    target.recover_larger_model()
                    global_model = self.load_item("Server", "model", folder)
                    torch.testing.assert_close(global_model.weight, target.weight, rtol=0, atol=0)

    def test_observation_restores_local_model_and_rng_even_on_failure(self):
        client = self.client_module.clientTargetProj.__new__(self.client_module.clientTargetProj)
        client.role, client.save_folder_name = "Client_0", "run"
        original = TinyModel(low_rank=True)
        self.save_item(original, client.role, "model", "run")
        self.assertEqual(type(client).__bases__, (FakeClientBase,))

        def download():
            torch.rand(7)
            random.random()
            np.random.rand(5)
            self.save_item(TinyModel(offset=7), client.role, "model", "run")

        client.set_parameters = download
        for fails in (False, True):
            client.test_metrics = mock.Mock(side_effect=RuntimeError("test failed") if fails else None,
                                            return_value=(2, 3, 0))
            torch_state, python_state, numpy_state = torch.get_rng_state(), random.getstate(), np.random.get_state()
            if fails:
                with self.assertRaisesRegex(RuntimeError, "test failed"):
                    client.test_downloaded_global()
            else:
                self.assertEqual(client.test_downloaded_global(), (2, 3, 0))
            torch.testing.assert_close(torch.get_rng_state(), torch_state, rtol=0, atol=0)
            self.assertEqual(random.getstate(), python_state)
            np.testing.assert_array_equal(np.random.get_state()[1], numpy_state[1])
            restored = self.load_item(client.role, "model", "run")
            for name, value in restored.state_dict().items():
                torch.testing.assert_close(value, original.state_dict()[name], rtol=0, atol=0)

    def test_missing_client_upload_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            server = self.make_server(folder, "projection")
            server.uploaded_ids = [0, 2]
            with self.assertRaisesRegex(RuntimeError, "every client"):
                server.aggregate_parameters_avg()

    def test_local_step_is_ce_plus_frobenius_with_one_learning_rate(self):
        with tempfile.TemporaryDirectory() as folder, redirect_stdout(io.StringIO()):
            client = self.make_client(folder, 0)
            initial = TinyModel(low_rank=True)
            self.save_item(initial, client.role, "model", folder)
            expected = copy.deepcopy(initial)
            x, y = client.load_train_data()[0]
            ce = torch.nn.functional.cross_entropy(expected(x), y)
            reg = 0.7 * expected.frobenius_decay()
            (ce + reg).backward()
            torch.nn.utils.clip_grad_norm_(expected.parameters(), 10.0)
            with torch.no_grad():
                for p in expected.parameters():
                    p.add_(p.grad, alpha=-0.05)
            client.train()
            actual = self.load_item(client.role, "model", folder)
            for name, p in actual.named_parameters():
                torch.testing.assert_close(p, dict(expected.named_parameters())[name], rtol=0, atol=0)
                self.assertFalse(torch.equal(p, dict(initial.named_parameters())[name]))
            losses, samples = client.train_metrics()
            expected_loss = torch.nn.functional.cross_entropy(actual(x), y) + 0.7 * actual.frobenius_decay()
            self.assertAlmostEqual(losses / samples, expected_loss.item())

    def test_download_reconstructs_low_rank_and_preserves_local_buffers(self):
        client = self.make_client("run", 0)
        self.save_item(TinyModel(low_rank=True, offset=7), client.role, "model", "run")
        self.save_item(TinyModel(), "Server", "model", "run")
        client.set_parameters()
        downloaded = self.load_item(client.role, "model", "run")
        self.assertTrue(hasattr(downloaded, "weight_u") and hasattr(downloaded, "weight_v"))
        self.assertEqual(downloaded.running_stat.item(), 24.0)
        torch.testing.assert_close(downloaded.weight_u @ downloaded.weight_v, TinyModel().weight)
        self.assertEqual(self.load_item("Server", "model", "run").running_stat.item(), 17.0)

    def test_eval_gap_final_evaluation_and_h5_export(self):
        import h5py

        with tempfile.TemporaryDirectory() as folder, redirect_stdout(io.StringIO()):
            server = self.make_server(folder, "projection")
            server.global_rounds, server.eval_gap = 2, 2
            for loop_round in range(3):
                server.cur_ground = loop_round
                server.aggregate_parameters_avg()
            self.assertEqual([r["target_client_test_acc"] for r in server.target_proj_history],
                             [None, 0.75, 0.75])
            self.assertEqual(server.clients[1].test_downloaded_global.call_count, 2)
            server.args.save_file_paths = []
            path = str(Path(folder) / "result.h5")

            def save_baseline_results():
                with h5py.File(path, "w") as result:
                    result.create_dataset("rs_test_acc", data=[0.5])
                server.args.save_file_paths.append(path)

            with mock.patch.object(FakeServer, "save_results", side_effect=save_baseline_results):
                server.save_results()
            with h5py.File(path, "r") as result:
                group = result["target_projection"]
                self.assertEqual(group.attrs["mode"], "projection")
                self.assertEqual(group.attrs["target_client_id"], 1)
                np.testing.assert_array_equal(group["round"][:], [1, 2, 3])
                self.assertTrue(np.isnan(group["target_client_test_acc"][0]))
                self.assertEqual(group["target_client_test_acc"][2], 0.75)
                self.assertEqual(result["rs_test_acc"][0], 0.5)
            destination = Path(folder) / "final"
            server.final_model_dir = lambda: str(destination)
            with mock.patch.object(FakeServer, "export_final_models", create=True,
                                   side_effect=lambda: destination.mkdir()):
                server.export_final_models()
            for filename in ("target_proj_metrics.csv", "target_proj_clients.csv", "target_proj_metrics.json"):
                self.assertEqual((destination / filename).read_bytes(), (Path(folder) / filename).read_bytes())

    def test_invalid_experiment_configuration_is_rejected_before_initialization(self):
        valid = dict(target_client_id=0, target_proj_mode="avg", num_clients=20,
                     join_ratio=1.0, random_join_ratio=False, client_drop_rate=0.0,
                     time_select=False, resume=False, global_rounds=100, eval_gap=1)
        for change in (dict(target_client_id=20), dict(join_ratio=0.5), dict(client_drop_rate=0.1),
                       dict(random_join_ratio=True), dict(resume=True), dict(eval_gap=101),
                       dict(use_asymmetric_lr=1), dict(use_loss_specific_u_scaling=1)):
            with self.subTest(change=change), self.assertRaises(ValueError):
                self.server_module.FedTargetProj(types.SimpleNamespace(**(valid | change)), 0)


if __name__ == "__main__":
    unittest.main()

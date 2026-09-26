"""Local-delta masks, logical layers, snapshots and complete diagnostics."""

from contextlib import redirect_stdout
import copy
import csv
import io
import json
from pathlib import Path
import random
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "system"))
from utils.layer_mask import aggregate_layer_mask, logical_layer_groups, print_layer_mask_diagnostics
from utils.target_projection import EPS
import test_target_projection as fixtures


def model(conv, head):
    return {"conv.weight": torch.tensor([conv], dtype=torch.float64),
            "head.weight": torch.tensor([head], dtype=torch.float64)}


class LayerMaskMathTests(unittest.TestCase):
    def test_pure_local_delta_and_hidden_layer_conflict(self):
        global_params = model(0, 0)
        pre = model(100, 100)
        target = model(110, 101)
        helper = model(110, 99)
        result, metrics, clients, layers, matrices = aggregate_layer_mask(
            global_params, target, pre,
            [(1, 0.8, helper, pre), (0, 0.2, target, pre)], 0,
            logical_layer_groups(global_params),
        )
        self.assertGreater(clients[1]["old_global_delta_cos"], 0)
        self.assertGreater(clients[1]["full_local_delta_cos"], 0)
        self.assertEqual(matrices["mask_matrix"], [[1, 1], [1, 0]])
        self.assertEqual(matrices["client_ids"], [0, 1])
        self.assertAlmostEqual(clients[1]["local_update_norm"], np.sqrt(101))
        # Positive layer keeps the COMPLETE 10-unit local update with original 0.8 weight.
        # Negative head helper update is completely absent; target has coefficient one.
        torch.testing.assert_close(result["conv.weight"], torch.tensor([118.0], dtype=torch.float64))
        torch.testing.assert_close(result["head.weight"], target["head.weight"], rtol=0, atol=0)
        self.assertEqual(metrics["total_negative_entries"], 1)
        self.assertEqual(metrics["layer_conflict_ratio"], 0.5)
        self.assertAlmostEqual(metrics["masked_update_ratio"], 0.8 / (11 + EPS))
        self.assertAlmostEqual(matrices["layer_summary"][1]["masked_norm_ratio"], 0.8 / (1 + EPS))
        self.assertEqual(layers[3]["layer"], "head")

    def test_weight_and_bias_share_one_mask(self):
        initial = {"conv.weight": torch.zeros(1), "conv.bias": torch.zeros(1),
                   "head.weight": torch.zeros(1)}
        target = {"conv.weight": torch.tensor([1.0]), "conv.bias": torch.tensor([2.0]),
                  "head.weight": torch.tensor([1.0])}
        helper = {"conv.weight": torch.tensor([-2.0]), "conv.bias": torch.tensor([2.0]),
                  "head.weight": torch.tensor([-1.0])}
        groups = logical_layer_groups(initial)
        self.assertEqual(groups["conv"], ["conv.weight", "conv.bias"])
        result, _, _, _, matrices = aggregate_layer_mask(
            initial, target, initial, [(0, 0.25, target, initial), (1, 0.75, helper, initial)], 0, groups
        )
        self.assertEqual(matrices["mask_matrix"], [[1, 1], [1, 0]])
        self.assertEqual(result["conv.weight"].item(), -0.5)
        self.assertEqual(result["conv.bias"].item(), 3.5)
        self.assertEqual(result["head.weight"].item(), 1.0)

    def test_all_helpers_masked_returns_exact_target_anchor_without_mutating_inputs(self):
        initial, pre, target, helper = model(-9, -7), model(10, 20), model(11, 21), model(8, 19)
        uploads = [(0, 0.01, target, pre), (1, 0.99, helper, pre)]
        originals = copy.deepcopy([initial, pre, target, helper])
        result, _, _, _, matrices = aggregate_layer_mask(
            initial, target, pre, uploads, 0, logical_layer_groups(initial)
        )
        self.assertEqual(matrices["mask_matrix"], [[1, 1], [0, 0]])
        for name in result:
            torch.testing.assert_close(result[name], target[name], rtol=0, atol=0)
        for actual, expected in zip([initial, pre, target, helper], originals):
            for name in actual:
                torch.testing.assert_close(actual[name], expected[name], rtol=0, atol=0)

    def test_zero_near_zero_and_orthogonal_updates_are_finite_and_masked(self):
        for tiny in (0.0, EPS / 10):
            with self.subTest(tiny=tiny):
                initial, target, helper = model(0, 0), model(tiny, 1), model(2, 0)
                result, metrics, _, _, matrices = aggregate_layer_mask(
                    initial, target, initial, [(0, 0.5, target, initial), (1, 0.5, helper, initial)],
                    0, logical_layer_groups(initial),
                )
                self.assertEqual(matrices["mask_matrix"], [[1, 1], [0, 0]])
                self.assertEqual(matrices["cosine_matrix"][0], [1.0, 1.0])
                self.assertEqual(matrices["zero_norm_matrix"][1], [True, True])
                self.assertEqual(metrics["total_negative_entries"], 0)
                self.assertEqual(metrics["total_masked_entries"], 2)
                json.dumps(matrices, allow_nan=False)
                self.assertTrue(all(torch.isfinite(p).all() for p in result.values()))
        initial = {"layer.weight": torch.zeros(2)}
        target, helper = {"layer.weight": torch.tensor([1.0, 0.0])}, {"layer.weight": torch.tensor([0.0, 1.0])}
        _, _, _, rows, _ = aggregate_layer_mask(initial, target, initial,
            [(0, 0.5, target, initial), (1, 0.5, helper, initial)], 0, logical_layer_groups(initial))
        self.assertEqual(rows[1]["mask"], 0)
        self.assertFalse(rows[1]["zero_norm"])

    def test_single_target_and_invalid_layout_or_weights(self):
        initial, target = model(0, 0), model(1, 2)
        groups = logical_layer_groups(initial)
        _, metrics, _, _, _ = aggregate_layer_mask(initial, target, initial, [(0, 1.0, target, initial)], 0, groups)
        self.assertEqual(metrics["total_helper_layer_entries"], 0)
        self.assertEqual(metrics["masked_update_ratio"], 0)
        for uploads in (
            [(1, 1.0, target, initial)],
            [(0, 0.2, target, initial)],
            [(0, 0.5, target, initial), (0, 0.5, target, initial)],
            [(0, 1.0, target, {"conv.weight": torch.zeros(2), "head.weight": torch.zeros(1)})],
            [(0, 1.0, target, {"wrong": torch.zeros(1)})],
        ):
            with self.assertRaises(ValueError):
                aggregate_layer_mask(initial, target, initial, uploads, 0, groups)

    def test_terminal_contains_all_twenty_clients_and_all_layers(self):
        initial, target = model(0, 0), model(1, 2)
        _, metrics, clients, _, matrices = aggregate_layer_mask(
            initial, target, initial, [(cid, 0.05, target, initial) for cid in range(20)],
            0, logical_layer_groups(initial),
        )
        output = io.StringIO()
        with redirect_stdout(output):
            print_layer_mask_diagnostics(12, matrices, clients, metrics)
        blocks = output.getvalue().split("[LayerMask][Round 12]")
        for block in blocks[1:5]:  # cosine, mask, norms, zero-norm matrices
            self.assertIn("conv", block)
            self.assertIn("head", block)
            for cid in range(20):
                self.assertIn(f"Client {cid} ", block)
        self.assertIn("total_negative_entries = 0 / 38", output.getvalue())


class LayerMaskServerTests(unittest.TestCase):
    def setUp(self):
        self.fixture = fixtures.ProjectionIntegrationTests()
        self.fixture.setUp()

    def make_server(self, folder):
        server = self.fixture.make_server(folder, "layer_mask")
        server.target_client_id = 0
        server.clients[0].test_downloaded_global = mock.Mock(return_value=(3, 4, 0))
        server.layer_groups = logical_layer_groups(dict(fixtures.TinyModel().named_parameters()))
        server.layer_mask_history = []
        server._pre_local_round = None
        return server

    def test_snapshots_use_actual_download_preserve_rng_and_uploads_and_reject_stale_round(self):
        with tempfile.TemporaryDirectory() as folder, redirect_stdout(io.StringIO()):
            server = self.make_server(folder)
            server.selected_clients = server.clients
            original_uploads = copy.deepcopy(self.fixture.store)
            original_loader = server._load_full_parameters

            def noisy_recovery(client_id):
                torch.rand(10)
                random.random()
                np.random.rand(4)
                return original_loader(client_id)

            server._load_full_parameters = noisy_recovery
            torch_rng, py_rng, np_rng = torch.get_rng_state(), random.getstate(), np.random.get_state()
            server._capture_pre_local_parameters()
            torch.testing.assert_close(torch.get_rng_state(), torch_rng, rtol=0, atol=0)
            self.assertEqual(py_rng, random.getstate())
            np.testing.assert_array_equal(np_rng[1], np.random.get_state()[1])
            for client in server.clients:
                expected = original_loader(client.id)
                pre = server._load_pre_local_parameters(client.id)
                for name in expected:
                    torch.testing.assert_close(pre[name], expected[name], rtol=0, atol=0)
            server.aggregate_parameters_avg()  # zero training change => exact target anchor
            global_model = self.fixture.load_item("Server", "model", folder)
            target = original_loader(0)
            for name, p in global_model.named_parameters():
                torch.testing.assert_close(p, target[name], rtol=0, atol=0)
            for client in server.clients:
                key = (folder, client.role, "model")
                for name, p in self.fixture.store[key].state_dict().items():
                    torch.testing.assert_close(p, original_uploads[key].state_dict()[name], rtol=0, atol=0)
            server.cur_ground += 1
            with self.assertRaisesRegex(RuntimeError, "pre-local snapshots"):
                server._load_pre_local_parameters(0)

    def test_real_local_training_loop_and_saved_matrices(self):
        with tempfile.TemporaryDirectory() as folder, redirect_stdout(io.StringIO()):
            server = self.make_server(folder)
            server.clients = [self.fixture.make_client(folder, cid) for cid in range(3)]
            server.Budget, server.rs_test_acc, server.auto_break = [], [], False
            for client in server.clients:
                client.train = mock.Mock(side_effect=client.train)
            server.train()
            for client in server.clients:
                self.assertEqual(client.train.call_count, 2)
            with open(Path(folder) / "layer_mask_matrices.json") as stream:
                history = json.load(stream)["history"]
            self.assertEqual([row["round"] for row in history], [1, 2])
            self.assertEqual(history[0]["client_ids"], [0, 1, 2])
            self.assertEqual(len(history[0]["cosine_matrix"]), 3)
            self.assertEqual(history[0]["mask_matrix"][0], [1])
            for filename, row_count in (("layer_mask_metrics.csv", 2), ("layer_mask_clients.csv", 6),
                                        ("layer_mask_cosines.csv", 6)):
                with open(Path(folder) / filename, newline="") as stream:
                    self.assertEqual(len(list(csv.DictReader(stream))), row_count)


if __name__ == "__main__":
    unittest.main()

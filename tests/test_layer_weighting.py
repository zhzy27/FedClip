"""Positive-only mass redistribution, untouched anchors and isolated result files."""

from contextlib import redirect_stdout
import copy
import csv
import io
import json
import math
from pathlib import Path
import random
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "system"))
from utils.layer_mask import aggregate_layer_mask, logical_layer_groups
from utils.layer_weighting import (
    WEIGHTING_MODES, aggregate_layer_weighting, positive_helper_weights, print_layer_weighting_diagnostics,
)
import test_layer_mask as fixtures


def scalar_weights(cosines, samples, mode):
    clients = [{"client_id": cid, "weight": p} for cid, p in enumerate(samples)]
    rows = [{"client_id": cid, "layer": "conv", "layer_local_delta_cos": c}
            for cid, c in enumerate(cosines)]
    return positive_helper_weights(clients, rows, ["conv"], 0, mode)


def vector(values):
    return {"conv.weight": torch.tensor(values, dtype=torch.float64)}


class ContinuousWeightMathTests(unittest.TestCase):
    def test_softmax_rejects_nonpositive_and_matches_fixed_temperature_formula(self):
        cosines, samples = [1, -0.4, 0, 0.2, 0.8], [0.2] * 5
        weights, summaries = scalar_weights(cosines, samples, "layer_softmax")
        self.assertEqual([weights[cid, "conv"] for cid in (0, 1, 2)], [0, 0, 0])
        self.assertGreater(weights[4, "conv"], weights[3, "conv"])
        self.assertAlmostEqual(weights[4, "conv"] / weights[3, "conv"], math.exp(3))
        self.assertAlmostEqual(sum(weights.values()), 0.4, places=15)
        self.assertEqual(summaries[0]["positive_clients"], 2)
        self.assertAlmostEqual(summaries[0]["original_weight_mass"], 0.4)

    def test_relu_rejects_nonpositive_and_grows_linearly_with_cosine(self):
        weights, _ = scalar_weights([1, -1, 0, 0.2, 0.8], [0.2] * 5, "layer_relu")
        self.assertEqual([weights[cid, "conv"] for cid in (0, 1, 2)], [0, 0, 0])
        self.assertAlmostEqual(weights[4, "conv"] / weights[3, "conv"], 4)
        self.assertAlmostEqual(sum(weights.values()), 0.4, places=15)

    def test_relu_weak_positive_tends_to_zero_with_another_positive_helper(self):
        weak_weights = []
        for cosine in (0.1, 1e-5, 1e-12):
            weights, _ = scalar_weights([1, cosine, 0.5], [0.5, 0.25, 0.25], "layer_relu")
            weak_weights.append(weights[1, "conv"])
            self.assertAlmostEqual(sum(weights.values()), 0.5, places=15)
        self.assertGreater(weak_weights[0], weak_weights[1])
        self.assertGreater(weak_weights[1], weak_weights[2])
        self.assertLess(weak_weights[2], 2e-12)

    def test_equal_cosines_preserve_sample_ratios_and_effective_count(self):
        for mode in WEIGHTING_MODES:
            with self.subTest(mode=mode):
                weights, stats = scalar_weights([1, 0.7, 0.7], [0.2, 0.4, 0.4], mode)
                self.assertEqual(weights[1, "conv"], weights[2, "conv"])
                self.assertAlmostEqual(stats[0]["effective_helper_count"], 2)
                weights, _ = scalar_weights([1, 0.7, 0.7], [0.2, 0.2, 0.6], mode)
                self.assertAlmostEqual(weights[2, "conv"] / weights[1, "conv"], 3)
                self.assertAlmostEqual(sum(weights.values()), 0.8)

    def test_empty_zero_mass_single_and_tiny_positive_sets_are_finite(self):
        for mode in WEIGHTING_MODES:
            for cosines, samples, mass in (
                ([1, 0, -1], [0.5, 0.25, 0.25], 0),
                ([1, 1, 1], [1, 0, 0], 0),
                ([1, 1e-300], [0.8, 0.2], 0.2),
                ([1, 1e-300, 2e-300], [0.5, 0.2, 0.3], 0.5),
            ):
                with self.subTest(mode=mode, cosines=cosines):
                    weights, rows = scalar_weights(cosines, samples, mode)
                    self.assertAlmostEqual(sum(weights.values()), mass, places=15)
                    self.assertTrue(all(math.isfinite(value) for value in weights.values()))
                    json.dumps(rows, allow_nan=False)

    def test_full_anchor_kept_and_no_helper_budget_is_applied(self):
        initial, pre, target = vector([0, 0]), vector([99, 20]), vector([100, 20])
        uploads = [(0, 0.5, target, pre), (1, 0.5, vector([100, 0]), initial)]
        originals = copy.deepcopy(uploads)
        for mode in WEIGHTING_MODES:
            result, _, _, _, matrices = aggregate_layer_weighting(
                initial, target, pre, lambda: iter(uploads), 0, logical_layer_groups(initial), mode)
            torch.testing.assert_close(result["conv.weight"], torch.tensor([150., 20.], dtype=torch.float64))
            self.assertEqual(matrices["final_weight_matrix"][0], ["anchor"])
            self.assertEqual(matrices["helper_weight_matrix"][0], [0])
            self.assertNotIn("budget_layers", matrices)
        for actual, before in zip(uploads, originals):
            for index in (2, 3):
                torch.testing.assert_close(actual[index]["conv.weight"], before[index]["conv.weight"], rtol=0, atol=0)

    def test_aggregation_uses_weights_and_retains_all_raw_mask_diagnostics(self):
        initial, pre, target = vector([0, 0]), vector([10, 20]), vector([11, 20])
        helper_a, helper_b = vector([0.2, math.sqrt(0.96)]), vector([0.8, 0.6])
        uploads = [(2, 0.25, helper_b, initial), (0, 0.5, target, pre), (1, 0.25, helper_a, initial)]
        groups = logical_layer_groups(initial)
        old = aggregate_layer_mask(initial, target, pre, iter(uploads), 0, groups)
        for mode in WEIGHTING_MODES:
            result, _, clients, rows, matrices = aggregate_layer_weighting(
                initial, target, pre, lambda: iter(uploads), 0, groups, mode)
            for key, value in old[4].items():
                self.assertEqual(matrices[key], value)
            self.assertEqual(clients, old[2])
            alpha = matrices["helper_weight_matrix"]
            expected = target["conv.weight"] + alpha[1][0] * helper_a["conv.weight"] + alpha[2][0] * helper_b["conv.weight"]
            torch.testing.assert_close(result["conv.weight"], expected)
            self.assertAlmostEqual(sum(row[0] for row in alpha), 0.5)
            self.assertEqual(rows[0]["aggregation_role"], "anchor")
            self.assertEqual(rows[0]["anchor_coefficient"], 1)

    def test_no_positive_helpers_returns_exact_target(self):
        initial, target = vector([0, 0]), vector([1, 0])
        uploads = [(0, 0.5, target, initial), (1, 0.5, vector([-9, 4]), initial)]
        for mode in WEIGHTING_MODES:
            result, _, _, _, matrices = aggregate_layer_weighting(
                initial, target, initial, lambda: iter(uploads), 0, logical_layer_groups(initial), mode)
            torch.testing.assert_close(result["conv.weight"], target["conv.weight"], rtol=0, atol=0)
            self.assertEqual(matrices["helper_weight_matrix"], [[0], [0]])
            self.assertEqual(matrices["effective_helper_count"], [0])

    def test_second_checkpoint_read_does_not_change_rng(self):
        initial, target = vector([0, 0]), vector([1, 0])
        uploads = [(0, 0.5, target, initial), (1, 0.5, target, initial)]
        for mode in WEIGHTING_MODES:
            states, calls = [], []

            def factory():
                calls.append(1)
                for upload in uploads:
                    torch.rand(3)
                    random.random()
                    np.random.rand(3)
                    yield upload
                if len(calls) == 1:
                    states.append((torch.get_rng_state(), random.getstate(), np.random.get_state()))

            aggregate_layer_weighting(initial, target, initial, factory, 0, logical_layer_groups(initial), mode)
            self.assertEqual(len(calls), 2)
            torch.testing.assert_close(torch.get_rng_state(), states[0][0], rtol=0, atol=0)
            self.assertEqual(random.getstate(), states[0][1])
            np.testing.assert_array_equal(np.random.get_state()[1], states[0][2][1])

    def test_weight_matrix_prints_all_twenty_clients_and_anchor(self):
        initial, target = vector([0, 0]), vector([1, 0])
        uploads = [(cid, 0.05, target, initial) for cid in range(20)]
        for mode in WEIGHTING_MODES:
            _, _, _, _, matrices = aggregate_layer_weighting(
                initial, target, initial, lambda: iter(uploads), 0, logical_layer_groups(initial), mode)
            output = io.StringIO()
            with redirect_stdout(output):
                print_layer_weighting_diagnostics(10, matrices)
            for cid in range(20):
                self.assertIn(f"Client {cid} ", output.getvalue())
            self.assertIn("anchor", output.getvalue())
            self.assertIn("effective_helper_count=19", output.getvalue())


class ContinuousWeightServerTests(unittest.TestCase):
    def test_both_modes_train_and_export_isolated_weight_logs(self):
        for mode in WEIGHTING_MODES:
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as folder:
                fixture = fixtures.LayerMaskServerTests()
                fixture.setUp()
                server = fixture.make_server(folder)
                server.target_proj_mode = mode
                server.args.target_proj_mode = mode
                server.clients = [fixture.fixture.make_client(folder, cid) for cid in range(3)]
                server.Budget, server.rs_test_acc, server.auto_break = [], [], False
                sentinel = Path(folder) / "layer_mask_matrices.json"
                sentinel.write_text("old experiment", encoding="utf-8")
                for client in server.clients:
                    client.train = mock.Mock(side_effect=client.train)
                output = io.StringIO()
                with redirect_stdout(output):
                    server.train()
                for client in server.clients:
                    self.assertEqual(client.train.call_count, 2)
                self.assertEqual(sentinel.read_text(encoding="utf-8"), "old experiment")
                self.assertNotIn("[LayerBudget]", output.getvalue())
                self.assertIn("Aggregation weight matrix", output.getvalue())
                with open(Path(folder) / f"{mode}_matrices.json") as stream:
                    history = json.load(stream)["history"]
                self.assertEqual(len(history), 2)
                for entry in history:
                    self.assertIn("target_client_test_acc", entry)
                    self.assertEqual(entry["final_weight_matrix"][0], ["anchor"])
                    self.assertNotIn("budget_beta", entry)
                    for row in entry["weight_summary"]:
                        self.assertAlmostEqual(row["original_weight_mass"], row["final_weight_mass"], places=15)
                for suffix, expected_count in (("metrics", 2), ("clients", 6), ("weights", 6), ("layers", 2)):
                    with open(Path(folder) / f"{mode}_{suffix}.csv", newline="") as stream:
                        self.assertEqual(len(list(csv.DictReader(stream))), expected_count)
                destination = Path(folder) / "export"
                server.final_model_dir = lambda: str(destination)
                with mock.patch.object(fixtures.fixtures.FakeServer, "export_final_models", create=True,
                                       side_effect=lambda: destination.mkdir()):
                    server.export_final_models()
                self.assertTrue((destination / f"{mode}_weights.csv").is_file())
                self.assertFalse((destination / "layer_mask_matrices.json").exists())


if __name__ == "__main__":
    unittest.main()

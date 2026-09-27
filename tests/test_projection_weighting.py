"""Full-model projection followed by pure helper-similarity mass redistribution."""

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
from utils.projection_variants import (
    PROJECTION_WEIGHTING_MODES, aggregate_projection_weighting, projection_similarity_weights,
)
from utils.target_projection import EPS, aggregate_target_updates
import test_target_projection as fixtures


def scalar_weights(cosines, samples, mode):
    rows = [dict(client_id=cid, sample_weight=weight, cosine_before_projection=cosines[cid])
            for cid, weight in enumerate(samples)]
    return projection_similarity_weights(rows, 0, mode)


def parameters(values, dtype=torch.float64):
    # Include weight/bias/head in the same full-model projection direction.
    return {"conv.weight": torch.tensor(values[:2], dtype=dtype),
            "conv.bias": torch.tensor(values[2:3], dtype=dtype),
            "fc.weight": torch.tensor(values[3:], dtype=dtype)}


def flattened(parameters):
    return torch.cat([value.reshape(-1) for value in parameters.values()])


class ProjectionWeightingMathTests(unittest.TestCase):
    def test_softmax_keeps_target_mass_and_uniform_nineteen_helpers(self):
        weights, _, summary = scalar_weights([1.] + [.3] * 19, [.05] * 20, "projection_softmax")
        self.assertEqual(weights[0], .05)
        for cid in range(1, 20):
            self.assertAlmostEqual(weights[cid], .95 / 19, places=15)
        self.assertAlmostEqual(summary["helper_total_weight"], .95, places=15)
        self.assertAlmostEqual(summary["effective_helper_count"], 19)
        self.assertEqual(summary["temperature"], .2)

    def test_softmax_is_monotonic_finite_and_never_rejects_negative_cosine(self):
        weights, scores, summary = scalar_weights([1., -1., -.2, 0., 1.], [.2] * 5, "projection_softmax")
        self.assertTrue(0 < weights[1] < weights[2] < weights[3] < weights[4])
        self.assertAlmostEqual(weights[4] / weights[1], math.exp(10))
        self.assertEqual(scores[4], 1.)
        self.assertEqual(summary["softmax_shift_max"], 1.)
        self.assertAlmostEqual(sum(weights.values()), 1., places=15)
        self.assertTrue(all(math.isfinite(value) for value in weights.values()))

    def test_normal_weights_ignore_unequal_helper_sample_counts(self):
        for mode in PROJECTION_WEIGHTING_MODES:
            weights, _, _ = scalar_weights([1., .5, .5], [.2, .1, .7], mode)
            self.assertEqual(weights[0], .2)
            self.assertEqual(weights[1], weights[2])
            self.assertAlmostEqual(weights[1], .4)
            changed, _, _ = scalar_weights([1., .5, .5], [.2, .7, .1], mode)
            self.assertEqual(weights, changed)

    def test_relu_rejects_nonpositive_and_distributes_mass_by_cosine(self):
        weights, scores, summary = scalar_weights([1., -.7, 0., .2, .8], [.2] * 5, "projection_relu")
        self.assertEqual(weights[0], .2)
        self.assertEqual(weights[1], 0.)
        self.assertEqual(weights[2], 0.)
        self.assertAlmostEqual(weights[4] / weights[3], 4.)
        self.assertAlmostEqual(summary["helper_total_weight"], .8)
        self.assertEqual(summary["relu_fallback_used"], 0)
        self.assertEqual(scores[3], .2)

    def test_relu_all_nonpositive_falls_back_to_unequal_sample_weights_exactly(self):
        for cosines in ([1., -.2, -1.], [1., 0., 0.], [1., 0., -.5]):
            samples = [.2, .1, .7]
            weights, _, summary = scalar_weights(cosines, samples, "projection_relu")
            self.assertEqual(weights, dict(enumerate(samples)))
            self.assertEqual(summary["relu_fallback_used"], 1)
            self.assertAlmostEqual(summary["helper_total_weight"], .8)

    def test_relu_tiny_positive_scores_keep_mass_without_epsilon_shrinkage(self):
        weights, _, summary = scalar_weights([1., 1e-300, 2e-300], [.2, .3, .5], "projection_relu")
        self.assertAlmostEqual(weights[2] / weights[1], 2.)
        self.assertAlmostEqual(weights[1] + weights[2], .8)
        self.assertEqual(summary["relu_fallback_used"], 0)

    def test_single_positive_helper_gets_whole_mass_and_effective_count_one(self):
        weights, _, summary = scalar_weights([1., -.1, 1e-10], [.2, .3, .5], "projection_relu")
        self.assertEqual(weights, {0: .2, 1: 0., 2: .8})
        self.assertEqual(summary["effective_helper_count"], 1.)

    def test_no_helpers_and_zero_helper_mass_are_finite(self):
        for mode in PROJECTION_WEIGHTING_MODES:
            for cosines, samples in (([1.], [1.]), ([1., 1.], [1., 0.])):
                weights, _, summary = scalar_weights(cosines, samples, mode)
                self.assertEqual(weights[0], 1.)
                self.assertEqual(summary["helper_total_weight"], 0.)
                self.assertEqual(summary["effective_helper_count"], 0.)
                json.dumps(summary, allow_nan=False)

    def test_aggregation_matches_independent_project_then_weight_reference(self):
        initial = parameters([5., 10., -4., 2.])
        target_delta = parameters([1., 0., 1., 2.])
        deltas = [target_delta, parameters([-2., 3., -1., -1.]), parameters([1., -2., 1., 1.])]
        posts = [{key: initial[key] + delta[key] for key in initial} for delta in deltas]
        uploads = [(cid, weight, posts[cid]) for cid, weight in enumerate([.2, .1, .7])]
        original = copy.deepcopy(uploads)
        target_vector = flattened(target_delta)
        target_sq = target_vector.dot(target_vector).item()
        for mode in PROJECTION_WEIGHTING_MODES:
            result, _, rows, _, matrices = aggregate_projection_weighting(initial, posts[0], lambda: iter(uploads), 0, mode)
            expected = flattened(initial).clone()
            for row, delta in zip(rows, deltas):
                raw = flattened(delta)
                dot = raw.dot(target_vector).item()
                coefficient = dot / (target_sq + EPS) if row["client_id"] != 0 and dot < 0 else 0.
                projected = raw - coefficient * target_vector
                expected += row["aggregation_weight"] * projected
                self.assertAlmostEqual(row["projection_coefficient"], coefficient)
                self.assertAlmostEqual(row["projected_update_norm"], projected.norm().item())
                self.assertAlmostEqual(row["cosine_before_projection"], dot / (raw.norm().item() * math.sqrt(target_sq)))
            torch.testing.assert_close(flattened(result), expected, rtol=1e-12, atol=1e-12)
            self.assertEqual(rows[0]["aggregation_weight"], .2)
            self.assertLess(rows[1]["cosine_before_projection"], 0)
            self.assertLess(rows[1]["projection_coefficient"], 0)
            self.assertEqual(matrices["projection_scope"], "full_model")
            self.assertNotIn("layer_groups", matrices)
            if mode == "projection_softmax":
                self.assertGreater(rows[1]["aggregation_weight"], 0)
                # Its nonzero orthogonal coordinate is retained after projection.
                self.assertAlmostEqual((result["conv.weight"][1] - initial["conv.weight"][1]).item(),
                    3 * rows[1]["aggregation_weight"] - 2 * rows[2]["aggregation_weight"])
            else:
                self.assertEqual(rows[1]["aggregation_weight"], 0.)
        for actual, before in zip(uploads, original):
            torch.testing.assert_close(flattened(actual[2]), flattened(before[2]), rtol=0, atol=0)

    def test_relu_fallback_aggregates_projected_negative_updates_and_matches_old_bitwise(self):
        initial, target = parameters([2., 3., 4., 5.]), parameters([3., 3., 4., 5.])
        uploads = [(0, .2, target), (1, .1, parameters([0., 7., 4., 5.])),
                   (2, .7, parameters([1., 5., 4., 5.]))]
        result, metrics, rows, _, _ = aggregate_projection_weighting(initial, target, lambda: iter(uploads), 0, "projection_relu")
        old, old_metrics, _ = aggregate_target_updates(initial, target, uploads, 0, "projection")
        torch.testing.assert_close(flattened(result), flattened(old), rtol=0, atol=0)
        self.assertEqual(metrics["relu_fallback_used"], 1)
        self.assertEqual([row["aggregation_weight"] for row in rows], [.2, .1, .7])
        self.assertGreater(rows[1]["projected_update_norm"], 3.99)
        for key, value in old_metrics.items():
            self.assertEqual(metrics[key], value)

    def test_equal_cosines_and_balanced_samples_match_legacy_projection(self):
        initial, target = parameters([1., 2., 3., 4.]), parameters([2., 2., 3., 4.])
        uploads = [(cid, .25, target) for cid in range(4)]
        old = aggregate_target_updates(initial, target, uploads, 0, "projection")[0]
        for mode in PROJECTION_WEIGHTING_MODES:
            result = aggregate_projection_weighting(initial, target, lambda: iter(uploads), 0, mode)[0]
            torch.testing.assert_close(flattened(result), flattened(old), rtol=0, atol=0)

    def test_zero_and_tiny_target_use_original_projection_rules(self):
        for tiny in (0., 1e-8):
            initial, target = parameters([0.] * 4), parameters([tiny, 0., 0., 0.])
            uploads = [(0, .2, target), (1, .8, parameters([-1., 2., 0., 0.]))]
            for mode in PROJECTION_WEIGHTING_MODES:
                result, metrics, rows, _, matrices = aggregate_projection_weighting(initial, target, lambda: iter(uploads), 0, mode)
                old = aggregate_target_updates(initial, target, uploads, 0, "projection")[0]
                torch.testing.assert_close(flattened(result), flattened(old), rtol=0, atol=0)
                json.dumps([metrics, rows, matrices], allow_nan=False)
                self.assertTrue(torch.isfinite(flattened(result)).all())

    def test_second_read_restores_python_numpy_and_torch_rng(self):
        initial, target = parameters([0.] * 4), parameters([1., 0., 0., 0.])
        for mode in PROJECTION_WEIGHTING_MODES:
            calls, states = [], []

            def factory():
                calls.append(1)
                for cid in range(3):
                    random.random()
                    np.random.rand(3)
                    torch.rand(3)
                    if torch.cuda.is_available():
                        torch.rand(3, device="cuda")
                    yield cid, 1 / 3, target
                if len(calls) == 1:
                    states.append((random.getstate(), np.random.get_state(), torch.get_rng_state(),
                                   torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []))

            aggregate_projection_weighting(initial, target, factory, 0, mode)
            self.assertEqual(len(calls), 2)
            self.assertEqual(random.getstate(), states[0][0])
            np.testing.assert_array_equal(np.random.get_state()[1], states[0][1][1])
            self.assertEqual(np.random.get_state()[2:], states[0][1][2:])
            torch.testing.assert_close(torch.get_rng_state(), states[0][2], rtol=0, atol=0)
            if torch.cuda.is_available():
                for before, after in zip(states[0][3], torch.cuda.get_rng_state_all()):
                    torch.testing.assert_close(before, after, rtol=0, atol=0)

    def test_changed_second_pass_uploads_are_rejected_and_rng_restored(self):
        initial, target = parameters([0.] * 4), parameters([1., 0., 0., 0.])
        for changed in ([(0, 1., target)], [(0, .2, target), (1, .7, target)], [(0, .2, target)]):
            calls, states = [], []

            def factory():
                calls.append(1)
                for upload in ([(0, .2, target), (1, .8, target)] if len(calls) == 1 else changed):
                    torch.rand(2)
                    yield upload
                if len(calls) == 1:
                    states.append(torch.get_rng_state())

            with self.assertRaises(ValueError):
                aggregate_projection_weighting(initial, target, factory, 0, "projection_softmax")
            torch.testing.assert_close(torch.get_rng_state(), states[0], rtol=0, atol=0)


class ProjectionWeightingServerTests(unittest.TestCase):
    def test_both_modes_train_all_twenty_clients_without_snapshots_and_save_complete_logs(self):
        torch.set_num_threads(2)
        for mode in PROJECTION_WEIGHTING_MODES:
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as folder, redirect_stdout(io.StringIO()) as output:
                fixture = fixtures.ProjectionIntegrationTests()
                fixture.setUp()
                server = fixture.make_server(folder, mode)
                server.target_client_id, server.num_clients = 0, 20
                server.Budget, server.rs_test_acc, server.auto_break = [], [], False
                server.clients = [fixture.make_client(folder, cid) for cid in range(20)]
                self.assertNotIn(mode, fixture.server_module.PRE_LOCAL_MODES)
                self.assertNotIn(mode, fixture.server_module.LAYER_GROUP_MODES)
                server._capture_pre_local_parameters = mock.Mock(side_effect=AssertionError("Unexpected snapshot"))
                server._load_pre_local_parameters = mock.Mock(side_effect=AssertionError("Unexpected pre read"))
                for client in server.clients:
                    fixture.save_item(fixtures.TinyModel(low_rank=True, offset=client.id * .01), client.role, "model", folder)
                    client.train = mock.Mock(side_effect=client.train)
                sentinel = Path(folder) / "projection_local_metrics.csv"
                sentinel.write_text("prior experiment")
                server.train()
                for client in server.clients:
                    self.assertEqual(client.train.call_count, 2)
                self.assertEqual(sentinel.read_text(), "prior experiment")
                for suffix, count in (("metrics.csv", 2), ("clients.csv", 40)):
                    with open(Path(folder) / f"{mode}_{suffix}") as stream:
                        self.assertEqual(len(list(csv.DictReader(stream))), count)
                with open(Path(folder) / f"{mode}_matrices.json") as stream:
                    saved = json.load(stream)
                self.assertEqual(len(saved["history"]), 2)
                self.assertEqual(saved["summary"], server._local_accuracy_summary())
                for row in saved["history"]:
                    self.assertIsNotNone(row["target_post_local_acc"])
                    self.assertEqual(row["target_weight"], .05)
                    self.assertAlmostEqual(row["helper_total_weight"], .95)
                    self.assertEqual(len(row["clients"]), 20)
                    self.assertNotIn("layer_groups", row)
                    for client in row["clients"]:
                        self.assertTrue({"client_id", "is_target", "sample_weight", "cosine_before_projection",
                            "dot_before_projection", "conflict", "projection_coefficient", "removed_component_norm",
                            "projected_update_norm", "raw_similarity_score", "aggregation_weight"} <= client.keys())
                self.assertIn("client_id=19", output.getvalue())
                self.assertIn("Final Client 0 post-local accuracy:", output.getvalue())
                destination = Path(folder) / "export"
                server.final_model_dir = lambda: str(destination)
                with mock.patch.object(fixtures.FakeServer, "export_final_models", create=True,
                                       side_effect=lambda: destination.mkdir()):
                    server.export_final_models()
                self.assertTrue((destination / f"{mode}_clients.csv").is_file())
                self.assertFalse((destination / "projection_local_metrics.csv").exists())


if __name__ == "__main__":
    unittest.main()

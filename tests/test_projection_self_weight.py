"""ProjectionSoftmax mass-only ablation, unchanged scoring/kernel and frozen settings."""

from contextlib import redirect_stdout
import copy
import csv
import io
import json
import math
from pathlib import Path
import re
import shlex
import sys
import tempfile
import unittest

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "system"))
from utils.projection_variants import aggregate_projection_weighting, projection_similarity_weights
from utils.projection_self_weight import validate_projection_self_weight, self_weight_tag
from utils.target_projection import EPS
import test_target_projection as fixtures


def parameters(values, dtype=torch.float64):
    return {"conv.weight": torch.tensor(values[:2], dtype=dtype), "head.bias": torch.tensor(values[2:], dtype=dtype)}


class ProjectionSelfWeightMathTests(unittest.TestCase):
    def test_none_and_explicit_point_zero_five_match_original_numeric_fields_exactly(self):
        for dtype in (torch.float32, torch.float64):
            generator = torch.Generator().manual_seed(47)
            initial = parameters([1., -2., .3, .4], dtype)
            posts = [parameters(torch.randn(4, generator=generator).tolist(), dtype) for _ in range(20)]
            uploads = [(cid, .05, posts[cid]) for cid in reversed(range(20))]
            before = aggregate_projection_weighting(initial, posts[0], lambda: iter(uploads), 0, "projection_softmax")
            after = aggregate_projection_weighting(initial, posts[0], lambda: iter(uploads), 0, "projection_softmax", .05)
            for name in before[0]:
                torch.testing.assert_close(after[0][name], before[0][name], rtol=0, atol=0)
            for key, value in before[1].items():
                self.assertEqual(after[1][key], value)
            self.assertEqual(before[2], after[2])
            self.assertNotIn("projection_self_weight", before[1])
            self.assertNotIn("effective_all_client_count", before[1])

    def test_masses_and_helper_ratios_scoring_projection_coefficients_unchanged(self):
        initial = parameters([0., 0., 0., 0.])
        posts = [parameters([1., 0., .1, 0.]), parameters([-2., 3., .2, 1.]), parameters([2., 1., .3, -1.])]
        samples = [.05, .2, .75]
        uploads = [(cid, samples[cid], post) for cid, post in enumerate(posts)]
        reference = aggregate_projection_weighting(initial, posts[0], lambda: iter(uploads), 0, "projection_softmax")
        for s in (.10, .20, .50):
            result, metrics, clients, _, _ = aggregate_projection_weighting(initial, posts[0], lambda: iter(uploads), 0, "projection_softmax", s)
            self.assertEqual(metrics["target_weight"], s)
            self.assertEqual(metrics["projection_self_weight"], s)
            self.assertAlmostEqual(metrics["helper_total_weight"], 1 - s)
            self.assertAlmostEqual(sum(c["aggregation_weight"] for c in clients), 1.)
            self.assertAlmostEqual(clients[1]["aggregation_weight"] / clients[2]["aggregation_weight"],
                                   reference[2][1]["aggregation_weight"] / reference[2][2]["aggregation_weight"])
            for client, old in zip(clients, reference[2]):
                for key in ("raw_similarity_score", "cosine_before_projection", "projection_coefficient",
                            "dot_before_projection", "removed_component_norm", "projected_update_norm"):
                    self.assertEqual(client[key], old[key])
            self.assertEqual(metrics["temperature"], .2)
            self.assertTrue(all(torch.isfinite(p).all() for p in result.values()))

    def test_analytic_full_global_projection_with_new_mass(self):
        initial = parameters([1., 0., 0., 0.])
        posts = [parameters([2., 0., 0., 0.]), parameters([-1., 3., 0., 0.]), parameters([2., 1., 0., 0.])]
        uploads = [(0, .05, posts[0]), (1, .45, posts[1]), (2, .5, posts[2])]
        result, _, clients, _, _ = aggregate_projection_weighting(initial, posts[0], lambda: iter(uploads), 0, "projection_softmax", .2)
        scores = [math.exp((-2 / math.sqrt(13)) / .2), math.exp((1 / math.sqrt(2)) / .2)]
        a1, a2 = [.8 * score / sum(scores) for score in scores]
        expected = torch.tensor([1 + .2 + a1 * (-2 + 2 / (1 + EPS)) + a2, 3 * a1 + a2], dtype=torch.float64)
        torch.testing.assert_close(result["conv.weight"], expected, rtol=1e-14, atol=1e-14)
        self.assertAlmostEqual(clients[1]["aggregation_weight"], a1)

    def test_zero_and_one_boundary_finite_conventions(self):
        rows = [dict(client_id=cid, sample_weight=.05, cosine_before_projection=.3) for cid in range(20)]
        for value in (0., 1.):
            weights, scores, metrics = projection_similarity_weights(rows, 0, "projection_softmax", value)
            self.assertEqual(weights[0], value)
            self.assertAlmostEqual(sum(weights.values()), 1.)
            self.assertEqual(metrics["helper_total_weight"], 1 - value)
            self.assertTrue(all(math.isfinite(x) for x in metrics.values()))
            self.assertTrue(all(score > 0 for score in scores.values()))
            if value == 1.:
                self.assertEqual(metrics["effective_helper_count"], 0.)
                self.assertEqual(metrics["effective_all_client_count"], 1.)
                self.assertEqual((metrics["min_helper_weight"], metrics["max_helper_weight"]), (0., 0.))
        initial, target = parameters([0.] * 4), parameters([1., 0., 0., 0.])
        uploads = [(0, .5, target), (1, .5, parameters([-2., 3., 0., 0.]))]
        result = aggregate_projection_weighting(initial, target, lambda: iter(uploads), 0, "projection_softmax", 1.)[0]
        torch.testing.assert_close(result["conv.weight"], target["conv.weight"], rtol=0, atol=0)

    def test_invalid_weights_other_modes_and_meta_holdout_rejected(self):
        for value in (-.1, 1.1, float("nan"), float("inf"), -float("inf")):
            with self.assertRaises(ValueError):
                validate_projection_self_weight(value, "projection_softmax")
        for mode in ("avg", "projection", "softmax_only", "projection_relu", "apa", "dwa_soft", "meta_projection_fixed"):
            with self.assertRaises(ValueError):
                validate_projection_self_weight(.2, mode)
            validate_projection_self_weight(None, mode)
        with self.assertRaises(ValueError):
            validate_projection_self_weight(.2, "projection_softmax", "split.json")
        initial = parameters([0.] * 4)
        for mode in ("softmax_only", "projection_relu"):
            with self.assertRaises(ValueError):
                aggregate_projection_weighting(initial, initial, lambda: iter([]), 0, mode, .2)

    def test_input_models_unchanged(self):
        initial, target = parameters([0.] * 4), parameters([1., 2., 3., 4.])
        uploads = [(0, .05, target), (1, .95, parameters([-1., 3., 4., 5.]))]
        originals = copy.deepcopy((initial, target, uploads))
        aggregate_projection_weighting(initial, target, lambda: iter(uploads), 0, "projection_softmax", .2)
        for actual, old in [(initial, originals[0]), (target, originals[1]), (uploads[1][2], originals[2][1][2])]:
            for name in actual:
                torch.testing.assert_close(actual[name], old[name], rtol=0, atol=0)


class ProjectionSelfWeightServerTests(unittest.TestCase):
    def setUp(self):
        self.fixture = fixtures.ProjectionIntegrationTests()
        self.fixture.setUp()

    def test_three_rounds_local_metrics_and_full_data_protocol(self):
        torch.set_num_threads(2)
        with tempfile.TemporaryDirectory() as folder, redirect_stdout(io.StringIO()):
            server = self.fixture.make_server(folder, "projection_softmax")
            server.args.projection_self_weight = .2
            server.target_client_id, server.global_rounds = 0, 2
            server.Budget, server.rs_test_acc, server.auto_break = [], [], False
            server.clients = [self.fixture.make_client(folder, cid) for cid in range(3)]
            server.train()
            self.assertFalse(hasattr(server, "meta_split_record"))
            self.assertFalse(hasattr(server.clients[0], "_meta_c0_train_data"))
            for client in server.clients:
                self.assertEqual(client.train_time_cost["num_rounds"], 3)
            for row in server.target_proj_history:
                self.assertEqual(row["projection_self_weight"], .2)
                self.assertEqual(row["target_weight"], .2)
                self.assertAlmostEqual(row["helper_total_weight"], .8)
            summary = server._local_accuracy_summary()
            self.assertEqual(summary["last10_target_local_count"], 3)
            self.assertAlmostEqual(summary["last10_target_local_acc"], sum(r["target_post_local_acc"] for r in server.target_proj_history) / 3)
            with open(Path(folder) / "projection_softmax_matrices.json") as stream:
                history = json.load(stream)
            self.assertEqual(history["summary"], summary)
            with open(Path(folder) / "projection_softmax_metrics.csv") as stream:
                self.assertIn("effective_all_client_count", next(csv.DictReader(stream)))


class ProjectionSelfWeightCommandTests(unittest.TestCase):
    def test_array_priority_full_configuration_names_and_directories(self):
        path = Path(__file__).resolve().parents[1] / "system" / "projection_self_weight_commands.sh"
        text = path.read_text(encoding="utf-8")
        self.assertIn("declare -a COMMANDS=(", text)
        commands = re.findall(r'^\s+"([^"]+)"$', text, re.MULTILINE)
        self.assertEqual(len(commands), 4)
        names, folders = [], []
        for command, s in zip(commands, (.1, .2, .05, .5)):
            args = shlex.split(command)
            for key, expected in (("--projection_self_weight", str(s)), ("-gr", "100"), ("-ls", "5"),
                ("-lr", "0.005"), ("-lbs", "16"), ("-nc", "20"), ("-data", "Cifar100"),
                ("-m", "Decom_CNN-5-512"), ("-regular_lamda", "1e-3"), ("-pt", "pat"), ("-cpc", "20"),
                ("--target_client_id", "0"), ("--seed", "0"), ("--target_proj_mode", "projection_softmax")):
                actual = args[args.index(key) + 1]
                self.assertEqual(float(actual), float(expected)) if key == "--projection_self_weight" else self.assertEqual(actual, expected)
            name, folder = args[args.index("-exp_name") + 1], args[args.index("-sfn") + 1]
            self.assertIn(self_weight_tag(s), name)
            self.assertIn(self_weight_tag(s), folder)
            self.assertNotIn("--meta_c0_split", args)
            self.assertNotIn("--apa_self_weight", args)
            names.append(name); folders.append(folder)
        self.assertEqual(len(set(names)), 4)
        self.assertEqual(len(set(folders)), 4)


if __name__ == "__main__":
    unittest.main()

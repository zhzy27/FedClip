"""Adaptive target mass, helper ratios, near-one stability and shared DWA smoke."""

from contextlib import redirect_stdout
import copy
import io
import json
import math
from pathlib import Path
import re
import shlex
import sys
import tempfile
import unittest
from unittest import mock

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "system"))
from utils.dwa_adaptive_aggregation import (
    DWA_ADAPTIVE_MODES, adaptive_guidance_weights, aggregate_dwa_adaptive,
    target_weight_history_summary, target_weight_phase,
)
from utils.dwa_aggregation import guidance_distance_weights
from utils.target_projection import aggregate_target_updates
import test_dwa_aggregation as dwa_tests
import test_target_projection as fixtures


def aggregate(initial, posts, guidance, mode, loop_round=0, **kwargs):
    samples = {cid: 1 / len(posts) for cid in range(len(posts))}
    return aggregate_dwa_adaptive(initial, posts[0],
        lambda: ((cid, samples[cid], post) for cid, post in enumerate(posts)),
        guidance, 0, samples, loop_round, {cid: loop_round for cid in samples}, mode, **kwargs)


class AdaptiveDWAMathTests(unittest.TestCase):
    def test_equal_twenty_distances_give_point_zero_five(self):
        for value in (0., 1., 1e200):
            weights, q, _, summary = adaptive_guidance_weights({cid: value for cid in range(20)})
            self.assertEqual(list(weights.values()), [.05] * 20)
            self.assertAlmostEqual(sum(q.values()), 1.)
            self.assertAlmostEqual(summary["helper_total_weight"], .95)
            self.assertAlmostEqual(summary["effective_helper_count"], 19.)
            self.assertAlmostEqual(summary["effective_all_client_count"], 20.)

    def test_target_mass_increases_when_closer_and_can_be_less_than_point_zero_five(self):
        for target_distance, relation in ((.1, "greater"), (10., "less")):
            distances = {cid: 1. for cid in range(20)}
            distances[0] = target_distance
            weights, _, _, _ = adaptive_guidance_weights(distances)
            expected = (1 / (target_distance + 1e-12)) / (1 / (target_distance + 1e-12) + 19 / (1 + 1e-12))
            self.assertAlmostEqual(weights[0], expected, places=15)
            getattr(self, "assertGreater" if relation == "greater" else "assertLess")(weights[0], .05)

    def test_helper_ratios_match_fixed_dwa_over_extreme_target_masses(self):
        helpers = {cid: cid ** 2 / 7 for cid in range(1, 20)}
        fixed, fixed_q, _, _ = guidance_distance_weights(helpers)
        for distance in (0., .1, 1., 1e100):
            adaptive, helper_q, _, summary = adaptive_guidance_weights({0: distance, **helpers})
            for cid in helpers:
                self.assertAlmostEqual(helper_q[cid], fixed_q[cid], places=14)
                self.assertAlmostEqual(adaptive[cid] / adaptive[1], fixed[cid] / fixed[1], places=14)
            self.assertAlmostEqual(summary["same_label_helper_share"], sum(helper_q[cid] for cid in (1, 2, 3)))
            self.assertAlmostEqual(summary["cross_label_helper_share"], sum(helper_q[cid] for cid in range(4, 20)))

    def test_near_one_target_uses_actual_helper_sum_and_finite_diagnostics(self):
        distances = {0: 0., **{cid: 1e308 for cid in range(1, 20)}}
        weights, q, _, summary = adaptive_guidance_weights(distances)
        self.assertEqual(weights[0], 1.)  # Rounds to one; helpers are still representable.
        self.assertEqual(1. - weights[0], 0.)
        self.assertGreater(summary["helper_total_weight"], 0.)
        self.assertEqual(summary["helper_total_weight"], math.fsum(weights[cid] for cid in range(1, 20)))
        self.assertTrue(all(math.isfinite(value) for value in summary.values()))
        self.assertAlmostEqual(summary["effective_helper_count"], 19.)
        self.assertEqual(summary["effective_all_client_count"], 1.)
        self.assertAlmostEqual(summary["same_label_helper_share"], 3 / 19, places=3)
        self.assertAlmostEqual(sum(q.values()), 1.)

    def test_extreme_distances_and_epsilon_do_not_change_rule(self):
        for distances, epsilon in (({0: 0., 1: 0., 2: 1.}, 1e-12),
                                   ({0: 1e308, 1: 1e308}, 1e308), ({0: 0., 1: 1e-320}, 1e-320)):
            weights, _, _, summary = adaptive_guidance_weights(distances, epsilon=epsilon)
            self.assertTrue(all(math.isfinite(w) and w > 0 for w in weights.values()))
            self.assertAlmostEqual(sum(weights.values()), 1.)
            self.assertTrue(all(math.isfinite(value) for value in summary.values()))

    def test_modes_identical_weights_and_projection_uses_ordinary_global_delta(self):
        params, guide = dwa_tests.params, dwa_tests.guide
        initial = params([4., 7.], head=1)
        posts = [params([5., 7.], head=1), params([2., 10.], head=1), params([7., 9.], head=1)]
        guidance = guide(params([5.01, 7.], head=1))
        plain, _, rows_a = aggregate(initial, posts, guidance, DWA_ADAPTIVE_MODES[0])
        projected, metrics, rows_b = aggregate(initial, posts, guidance, DWA_ADAPTIVE_MODES[1])
        for a, b in zip(rows_a, rows_b):
            for key in ("guidance_squared_distance", "aggregation_weight", "helper_q", "all_client_q"):
                self.assertEqual(a[key], b[key])
        weights = {row["client_id"]: row["aggregation_weight"] for row in rows_a}
        self.assertNotEqual(weights[0], .05)  # First aggregation has no Avg override.
        self.assertGreater(weights[0], .95)  # No upper cap.
        reference = aggregate_target_updates(initial, posts[0],
            [(cid, weights[cid], post) for cid, post in enumerate(posts)], 0, "projection")[0]
        for name in projected:
            torch.testing.assert_close(projected[name], reference[name], rtol=0, atol=0)
            expected_plain = sum(post[name] * weights[cid] for cid, post in enumerate(posts))
            torch.testing.assert_close(plain[name], expected_plain, rtol=0, atol=0)
        self.assertEqual([row["conflict"] for row in rows_b], [0, 1, 0])
        self.assertEqual(metrics["conflict_client_count"], 1)

    def test_zero_target_delta_zero_distance_and_near_one_aggregate_remain_finite(self):
        params, guide = dwa_tests.params, dwa_tests.guide
        initial = params([0., 0.])
        posts = [copy.deepcopy(initial)] + [params([1e150, 1e150]) for _ in range(19)]
        for mode in DWA_ADAPTIVE_MODES:
            result, metrics, _ = aggregate(initial, posts, guide(initial), mode)
            self.assertTrue(all(torch.isfinite(value).all() for value in result.values()))
            self.assertGreater(metrics["helper_total_weight"], 0.)
            self.assertTrue(all(math.isfinite(v) for v in metrics.values()))

    def test_plain_never_calls_projection_and_inputs_rng_remain_unchanged(self):
        initial = dwa_tests.params([0., 0.])
        posts = [dwa_tests.params([1., 0.]), dwa_tests.params([-1., 2.])]
        guidance = dwa_tests.guide(dwa_tests.params([1.1, 0.]))
        originals = copy.deepcopy((initial, posts, guidance))
        before = dwa_tests.rng_snapshot()
        with mock.patch("utils.dwa_adaptive_aggregation.aggregate_target_updates", side_effect=AssertionError("No projection")):
            aggregate(initial, posts, guidance, DWA_ADAPTIVE_MODES[0])
        aggregate(initial, posts, guidance, DWA_ADAPTIVE_MODES[1])
        dwa_tests.assert_rng_equal(self, before)
        for actual, old in [(initial, originals[0]), *zip(posts, originals[1]), (guidance["parameters"], originals[2]["parameters"])]:
            for name in actual:
                torch.testing.assert_close(actual[name], old[name], rtol=0, atol=0)

    def test_stale_guidance_layout_missing_clients_and_invalid_distances_rejected(self):
        initial = dwa_tests.params([0., 0.])
        posts = [dwa_tests.params([1., 0.]), dwa_tests.params([-1., 2.])]
        for field in ("loop_round", "source_post_local_round", "client_id"):
            guidance = dwa_tests.guide(initial)
            guidance[field] = 1
            with self.assertRaises(ValueError):
                aggregate(initial, posts, guidance, DWA_ADAPTIVE_MODES[0])
        for invalid in ({1: 1.}, {0: 0.}, {0: float("nan"), 1: 1.}, {0: -1., 1: 1.}):
            with self.assertRaises(ValueError):
                adaptive_guidance_weights(invalid)
        with self.assertRaisesRegex(ValueError, "uploads.*round"):
            aggregate_dwa_adaptive(initial, posts[0], lambda: iter([]), dwa_tests.guide(initial),
                0, {0: .5, 1: .5}, 0, {0: -1, 1: 0}, DWA_ADAPTIVE_MODES[0])
        with self.assertRaises(ValueError):
            aggregate(initial, posts, dwa_tests.guide({"wrong": torch.zeros(1)}), DWA_ADAPTIVE_MODES[0])

    def test_planned_phase_boundaries_and_threshold_counts(self):
        expected = {0: "early", 33: "early", 34: "mid", 67: "mid", 68: "late", 100: "late"}
        for loop, phase in expected.items():
            self.assertEqual(target_weight_phase(loop, 100), phase)
        rows = [dict(loop_round=loop, target_weight=weight) for loop, weight in
                ((0, .05), (33, .5), (34, .8), (67, .9), (68, .95), (100, 1.))]
        summary = target_weight_history_summary(rows, 100)
        self.assertEqual(summary["target_weight_count"], 6)
        self.assertEqual(summary["early_target_weight_min"], .05)
        self.assertAlmostEqual(summary["mid_target_weight_mean"], .85)
        self.assertEqual(summary["late_target_weight_max"], 1.)
        self.assertEqual(summary["target_weight_gt_0_5_count"], 4)
        self.assertEqual(summary["target_weight_gt_0_8_count"], 3)
        self.assertEqual(summary["target_weight_gt_0_95_count"], 1)
        partial = target_weight_history_summary(rows[:1], 100)
        self.assertIsNone(partial["late_target_weight_mean"])
        self.assertEqual(partial["late_target_weight_count"], 0)


class AdaptiveDWAServerTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        self.fixture = fixtures.ProjectionIntegrationTests()
        self.fixture.setUp()

    def test_twenty_clients_three_rounds_existing_guidance_shared_download_logs_and_export(self):
        for mode in DWA_ADAPTIVE_MODES:
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as folder, redirect_stdout(io.StringIO()):
                server = self.fixture.make_server(folder, mode)
                server.target_client_id, server.num_clients, server.global_rounds = 0, 20, 2
                server.Budget, server.rs_test_acc, server.auto_break = [], [], False
                server.clients = [self.fixture.make_client(folder, cid) for cid in range(20)]
                for client in server.clients:
                    client.local_epochs, client.learning_rate = 5, .005
                    client.args.regular_lamda = .001
                    self.fixture.save_item(fixtures.TinyModel(low_rank=True), client.role, "model", folder)
                    batches = [(torch.tensor([[1., -.5], [-.2, .3]]), torch.tensor([0 if client.id == 0 else 1, 1]))]
                    client.load_train_data = lambda batches=batches: batches
                    client.set_parameters = mock.Mock(wraps=client.set_parameters)
                    client.build_dwa_guidance = mock.Mock(wraps=client.build_dwa_guidance)
                target = server.clients[0]
                post_local_test = target.test_post_local
                def ordinary_test():
                    self.assertEqual(target.build_dwa_guidance.call_count, server.cur_ground)
                    return post_local_test()
                target.test_post_local = ordinary_test
                old_file = Path(folder) / "dwa_soft_history.json"
                old_file.write_text("fixed DWA control")
                server.train()
                self.assertEqual(old_file.read_text(), "fixed DWA control")
                history = json.loads((Path(folder) / f"{mode}_history.json").read_text())
                self.assertEqual(len(history["history"]), 3)
                self.assertEqual(history["summary"]["target_weight_count"], 3)
                self.assertEqual([row["target_weight_phase"] for row in history["history"]], ["early", "mid", "late"])
                self.assertNotEqual(history["history"][0]["target_weight"], .05)
                for client in server.clients:
                    self.assertEqual(client.train_time_cost["num_rounds"], 3)
                    # Normal shared download each round; C0 has three additional read-only diagnostic downloads.
                    self.assertEqual(client.set_parameters.call_count, 6 if client.id == 0 else 3)
                    self.assertEqual(client.build_dwa_guidance.call_count, 3 if client.id == 0 else 0)
                for row in history["history"]:
                    clients = row["clients"]
                    self.assertEqual(clients[0]["guidance_squared_distance"], row["target_squared_distance"])
                    self.assertEqual(clients[0]["aggregation_weight"], row["target_weight"])
                    self.assertAlmostEqual(math.fsum(c["aggregation_weight"] for c in clients[1:]), row["helper_total_weight"])
                    self.assertEqual(row["guidance_source_post_local_round"], row["loop_round"])
                    self.assertIsNotNone(row["target_post_local_acc"])
                summary = server._local_accuracy_summary()
                self.assertEqual(summary["last10_target_local_count"], 3)
                destination = Path(folder) / "export"
                server.final_model_dir = lambda: str(destination)
                with mock.patch.object(fixtures.FakeServer, "export_final_models", create=True,
                                       side_effect=lambda: destination.mkdir()):
                    server.export_final_models()
                self.assertTrue((destination / f"{mode}_history.json").is_file())
                with self.assertRaisesRegex(RuntimeError, "fresh guidance"):
                    server.aggregate_parameters_avg()


class AdaptiveCommandTests(unittest.TestCase):
    def test_commands_array_full_frozen_parameters_without_automatic_execution(self):
        path = Path(__file__).resolve().parents[1] / "system" / "dwa_adaptive_self_commands.sh"
        text = path.read_text(encoding="utf-8")
        self.assertIn("declare -a COMMANDS=(", text)
        commands = re.findall(r'^\s+"([^"]+)"$', text, re.MULTILINE)
        self.assertEqual(len(commands), 2)
        for command, mode in zip(commands, DWA_ADAPTIVE_MODES):
            argv = shlex.split(command)
            self.assertEqual(argv[:2], ["python", "main.py"])
            for key, expected in (("-gr", "100"), ("-ls", "5"), ("-nc", "20"), ("-lbs", "16"),
                ("-lr", "0.005"), ("-regular_lamda", "1e-3"), ("-pt", "pat"), ("-cpc", "20"),
                ("-data", "Cifar100"), ("-m", "Decom_CNN-5-512"), ("--seed", "0"),
                ("--target_client_id", "0"), ("--dwa_distance_eps", "1e-12"), ("--target_proj_mode", mode)):
                self.assertEqual(argv[argv.index(key) + 1], expected)
        self.assertNotIn("nohup", text)
        self.assertNotIn("bash -lc", text)


if __name__ == "__main__":
    unittest.main()

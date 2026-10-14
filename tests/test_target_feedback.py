"""Feedback sample-mass rules, legacy kernel equivalence, boundaries and summaries."""

from contextlib import redirect_stdout
import copy
import io
import json
import math
from pathlib import Path
import re
import shlex
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "system"))
from utils.feedback_options import FEEDBACK_MODES, feedback_tag, ensure_feedback_names, validate_feedback_options
from utils.feedback_aggregation import aggregate_feedback_updates, feedback_weights
from utils.target_projection import aggregate_target_updates, EPS
from flcore.servers.serverTargetFeedback import feedback_accuracy_summary
from test_meta_projection import rng_state, assert_rng
from test_target_proj_launcher import launcher


def parameters(vector, dtype=torch.float64):
    value = torch.tensor(vector, dtype=dtype)
    return {"conv.weight": value[:2], "head.weight": value[2:]}


class FeedbackAggregationTests(unittest.TestCase):
    def setUp(self):
        self.initial = parameters([3., -2., 5., .4])
        deltas = ([1., 0., .5, 0.], [-2., 3., -.5, 1.], [2., 1., .2, -1.], [-1., -3., -2., 0.])
        self.posts = [{n: p + parameters(d)[n] for n, p in self.initial.items()} for d in deltas]
        self.samples = {0: .05, 1: .1, 2: .25, 3: .6}
        self.order = (3, 1, 0, 2)

    def aggregate(self, mode):
        return aggregate_feedback_updates(self.initial, self.posts[0],
            ((cid, self.samples[cid], self.posts[cid]) for cid in self.order), self.samples, 0, mode)

    def test_self_modes_strict_equal_sample_based_weights_and_mass(self):
        avg = self.aggregate("feedback_avg_self")
        proj = self.aggregate("feedback_projection_self")
        self.assertEqual([c["aggregation_weight"] for c in avg[2]], [c["aggregation_weight"] for c in proj[2]])
        self.assertEqual(avg[1]["target_weight"], .125)
        self.assertAlmostEqual(avg[1]["helper_total_weight"], .875)
        for mode in FEEDBACK_MODES:
            result, metrics, rows = self.aggregate(mode)
            self.assertAlmostEqual(math.fsum(r["aggregation_weight"] for r in rows), 1.)
            self.assertTrue(all(math.isfinite(v) for v in metrics.values()))
        # Pure count ratios rather than cosine/softmax.
        self.assertEqual(avg[2][1]["aggregation_weight"] / avg[2][2]["aggregation_weight"],
                         self.samples[1] / self.samples[2])

    def test_avg_is_bitwise_legacy_sample_parameter_arithmetic(self):
        for dtype in (torch.float32, torch.float64):
            self.initial = {n: p.to(dtype) for n, p in self.initial.items()}
            self.posts = [{n: p.to(dtype) for n, p in post.items()} for post in self.posts]
            expected = aggregate_target_updates(self.initial, self.posts[0],
                ((cid, self.samples[cid], self.posts[cid]) for cid in self.order), 0, "avg")
            actual = self.aggregate("feedback_avg")
            for name in self.initial:
                torch.testing.assert_close(actual[0][name], expected[0][name], atol=0, rtol=0)
            self.assertEqual(actual[1]["target_weight"], .05)
            self.assertEqual(actual[1]["projection_enabled"], 0)
            self.assertGreater(actual[1]["conflict_client_count"], 0)
            self.assertEqual(actual[1]["removed_update_ratio"], 0)
            self.assertTrue(all(row["projection_coefficient"] == row["removed_component_norm"] == 0 for row in actual[2]))

    def test_projection_exactly_matches_existing_per_helper_kernel_and_differs_with_conflict(self):
        weights = feedback_weights(self.samples, 0, "feedback_avg_self")
        expected = aggregate_target_updates(self.initial, self.posts[0],
            ((cid, weights[cid], self.posts[cid]) for cid in self.order), 0, "projection")
        actual = self.aggregate("feedback_projection_self")
        for name in self.initial:
            torch.testing.assert_close(actual[0][name], expected[0][name], atol=0, rtol=0)
        for key, value in expected[1].items():
            self.assertEqual(actual[1][key], value)
        self.assertEqual(actual[2][0]["projection_applied"], 0)
        self.assertGreater(actual[2][1]["removed_component_norm"], 0)
        self.assertEqual(actual[2][2]["projection_coefficient"], 0)
        reference = {n: self.posts[0][n] - self.initial[n] for n in self.initial}
        reference_sq = sum(float(p.square().sum()) for p in reference.values())
        expected_coefficient = sum(float((self.posts[1][n] - self.initial[n]).mul(reference[n]).sum())
                                   for n in self.initial) / (reference_sq + EPS)
        self.assertEqual(actual[2][1]["projection_coefficient"], expected_coefficient)
        self.assertTrue(any(not torch.equal(actual[0][n], self.aggregate("feedback_avg_self")[0][n]) for n in self.initial))

    def test_no_conflicts_self_modes_agree_and_manual_weighted_average_includes_head(self):
        self.posts = [{n: p + parameters([cid + 1., .2, .1, .3])[n] for n, p in self.initial.items()}
                      for cid in range(4)]
        avg, proj = self.aggregate("feedback_avg_self"), self.aggregate("feedback_projection_self")
        weights = feedback_weights(self.samples, 0, "feedback_avg_self")
        for name in self.initial:
            expected = sum(self.posts[cid][name] * weights[cid] for cid in self.order)
            torch.testing.assert_close(avg[0][name], expected, atol=0, rtol=0)
            torch.testing.assert_close(proj[0][name], expected, atol=1e-14, rtol=1e-14)
        self.assertEqual(proj[1]["conflict_client_count"], 0)

    def test_zero_reference_and_inputs_rng_unchanged(self):
        self.posts[0] = copy.deepcopy(self.initial)
        old = copy.deepcopy((self.initial, self.posts))
        before = rng_state()
        for mode in FEEDBACK_MODES:
            result, metrics, rows = self.aggregate(mode)
            self.assertEqual(metrics["target_delta_norm"], 0)
            self.assertEqual(metrics["conflict_client_count"], 0)
            self.assertTrue(all(torch.isfinite(p).all() for p in result.values()))
            self.assertTrue(all(row["projection_coefficient"] == 0 for row in rows))
        assert_rng(self, before)
        for actual, saved in [(self.initial, old[0]), *zip(self.posts, old[1])]:
            for name in actual:
                torch.testing.assert_close(actual[name], saved[name], atol=0, rtol=0)

    def test_invalid_mass_upload_names_shapes_and_nonfinite_rejected(self):
        for samples in ({0: 1.}, {0: .2, 1: float("nan")}, {0: .2, 1: -.1}, {0: .2, 1: .5}):
            with self.assertRaises(ValueError):
                feedback_weights(samples, 0, "feedback_avg_self")
        for uploads in ([(0, .05, self.posts[0])], [(0, .05, self.posts[0])] * 2,
                        [(0, .5, self.posts[0])]):
            with self.assertRaises(ValueError):
                aggregate_feedback_updates(self.initial, self.posts[0], iter(uploads), self.samples, 0, "feedback_avg")
        for bad in (parameters([float("nan"), 0., 0., 0.]), {"conv.weight": torch.ones(3)}):
            self.posts[1] = bad
            with self.assertRaises(ValueError):
                self.aggregate("feedback_projection_self")


class FeedbackOptionsSummaryTests(unittest.TestCase):
    def test_new_names_are_mode_scoped_idempotent_and_do_not_change_old_options(self):
        args = SimpleNamespace(target_proj_mode="feedback_avg_self", exp_name="trial", save_folder_name="temp",
            h5_result_root="h5", final_model_root="final", training_log_root="logs", run_log_dir="")
        with patch.dict("os.environ", {"FEDCLIP_TRAIN_LOG_DIR": "external_logs"}):
            ensure_feedback_names(args)
        first = copy.deepcopy(vars(args))
        ensure_feedback_names(args)
        self.assertEqual(vars(args), first)
        for key in ("exp_name", "save_folder_name", "h5_result_root", "final_model_root", "training_log_root", "run_log_dir"):
            self.assertIn("feedback_avg_self_self0p125", getattr(args, key))
        old = SimpleNamespace(target_proj_mode="projection", exp_name="untouched", save_folder_name="temp")
        before = vars(old).copy()
        ensure_feedback_names(old)
        self.assertEqual(vars(old), before)

    def test_only_complete_c0_two_stage_configuration_and_no_prox_meta(self):
        args = SimpleNamespace(algorithm="FedTargetProj", target_client_id=0, num_clients=20,
                               local_epochs=5, eval_gap=1, target_proj_mode="feedback_avg")
        for mode in FEEDBACK_MODES:
            args.target_proj_mode = mode
            validate_feedback_options(args)
            self.assertIn(mode, feedback_tag(mode))
        for key, value in (("target_client_id", 1), ("num_clients", 19), ("local_epochs", 1),
                           ("eval_gap", 2), ("auto_break", True), ("few_shot", 1), ("train_slow_rate", .1),
                           ("target_prox_mu", .01), ("meta_c0_split", "split.json"),
                           ("meta_collect_snapshots", True), ("meta_weight_file", "weights.json"),
                           ("projection_self_weight", .125)):
            changed = copy.copy(args)
            setattr(changed, key, value)
            with self.assertRaises(ValueError):
                validate_feedback_options(changed)
        args.target_proj_mode = "avg"
        args.local_epochs = 1
        validate_feedback_options(args)  # No changes to old options.

    def test_three_accuracy_series_and_primary_best_earliest_last10_last20(self):
        history = [dict(round=index + 1, target_stage1_post_local_acc=.1,
            target_intermediate_download_acc=.2, target_stage2_post_local_acc=.8 if index in (2, 3) else index / 100)
                   for index in range(25)]
        summary = feedback_accuracy_summary(history)
        self.assertEqual(summary["stage2"]["final"], .24)
        self.assertEqual(summary["stage2"]["best_round"], 3)
        self.assertEqual(summary["stage1"]["best_round"], 1)
        self.assertEqual(summary["stage2"]["last10_count"], 10)
        self.assertEqual(summary["stage2"]["last20_count"], 20)
        self.assertAlmostEqual(summary["stage2"]["last20"], sum(r["target_stage2_post_local_acc"] for r in history[-20:]) / 20)

    def test_launcher_all_three_distinct_names_and_rejects_meta_mass_or_prox(self):
        output = io.StringIO()
        with patch.object(sys, "argv", ["run_target_proj.py", "--modes", *FEEDBACK_MODES, "--rounds", "200", "--dry-run"]), \
                redirect_stdout(output):
            launcher.main()
        lines = output.getvalue().splitlines()
        self.assertEqual(len(lines), 3)
        for line, mode in zip(lines, FEEDBACK_MODES):
            for value in (feedback_tag(mode), "-gr 200", "-ls 5", "-eg 1", "-lr 0.005", "--run_log_dir"):
                self.assertIn(value, line)
            self.assertNotIn("--projection_self_weight", line)
        for extra in (["--target_prox_mu", ".01"], ["--meta_c0_split", "split.json"],
                      ["--projection_self_weight", ".125"], ["--meta_weight_file", "weights.json"],
                      ["--target_group_ids", "0,1"]):
            with patch.object(sys, "argv", ["run_target_proj.py", "--modes", "feedback_avg", "--dry-run", *extra]), \
                    patch.object(sys, "stderr", io.StringIO()), self.assertRaises(SystemExit):
                launcher.main()

    def test_array_fixed_public_settings_distinct_gpus_and_model_log_roots(self):
        source = (Path(__file__).resolve().parents[1] / "system/target_feedback_commands.sh").read_text()
        self.assertIn("declare -a COMMANDS=(", source)
        commands = re.findall(r'^\s+"([^"]+)"$', source, re.MULTILINE)
        self.assertEqual(len(commands), 3)
        paths, gpus = [], []
        for command, mode in zip(commands, FEEDBACK_MODES):
            args = shlex.split(command)
            for flag, value in (("--target_proj_mode", mode), ("--seed", "0"), ("--target_client_id", "0"),
                ("-t", "1"), ("-gr", "200"), ("-ls", "5"), ("-eg", "1"), ("-lr", "0.005"), ("-lbs", "16"),
                ("-nc", "20"), ("-ncl", "100"), ("-jr", "1.0"), ("-m", "Decom_CNN-5-512"),
                ("-data", "Cifar100"), ("-pt", "pat"), ("-cpc", "20"), ("-regular_lamda", "1e-3"),
                ("--target_prox_mu", "0")):
                self.assertEqual(args[args.index(flag) + 1], value)
            locations = [args[args.index(key) + 1] for key in ("-sfn", "--h5_result_root", "--final-model-root", "--run_log_dir")]
            self.assertTrue(all(feedback_tag(mode) in location for location in locations))
            self.assertTrue(locations[-1].startswith("training_logs/"))
            for flag in ("--projection_self_weight", "--meta_c0_split", "--apa_self_weight"):
                self.assertNotIn(flag, args)
            paths.append(tuple(locations)); gpus.append(args[args.index("-did") + 1])
        self.assertEqual(len(set(paths)), 3)
        self.assertEqual(len(set(gpus)), 3)


if __name__ == "__main__":
    unittest.main()

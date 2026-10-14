"""Helper-first feedback math, server-anchor semantics and disabled Projection gate."""

from contextlib import redirect_stdout
import copy
import io
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
from utils.feedback_options import (FEEDBACK_MODES, feedback_tag, feedback_anchor_weight,
                                    ensure_feedback_names, validate_feedback_options)
from utils.feedback_aggregation import aggregate_feedback_updates, feedback_weights
from utils.target_projection import aggregate_target_updates
from flcore.servers.serverTargetFeedback import feedback_accuracy_summary
from flcore.servers.serverTargetProj import FedTargetProj
from test_meta_projection import rng_state, assert_rng
from test_target_proj_launcher import launcher


def parameters(vector, dtype=torch.float64):
    value = torch.tensor(vector, dtype=dtype)
    return {"conv.weight": value[:2], "head.weight": value[2:]}


class FeedbackAggregationTests(unittest.TestCase):
    def setUp(self):
        self.initial = parameters([3., -2., 5., .4])
        self.posts = {cid: {n: p + parameters([(-1.) ** cid * cid, .3, -.1, .2])[n]
                           for n, p in self.initial.items()} for cid in range(1, 20)}
        self.samples = {cid: cid * 10 for cid in self.posts}
        self.order = list(reversed(range(1, 20)))

    def aggregate(self, mode, alpha=None):
        return aggregate_feedback_updates(self.initial,
            ((cid, self.samples[cid], self.posts[cid]) for cid in self.order), self.samples, 0, mode, alpha)

    def test_only_helper_train_counts_normalize_and_anchor_is_not_a_C0_upload(self):
        avg = self.aggregate("feedback_avg")
        retained = self.aggregate("feedback_avg_self")
        self.assertEqual(len(retained[2]), 19)
        self.assertNotIn(0, [r["client_id"] for r in retained[2]])
        self.assertEqual(retained[1]["server_anchor_weight"], .125)
        self.assertAlmostEqual(retained[1]["helper_total_weight"], .875)
        self.assertEqual(avg[1]["server_anchor_weight"], 0.)
        for raw, weighted in zip(avg[2], retained[2]):
            self.assertEqual(weighted["helper_sample_weight"], raw["helper_sample_weight"])
            self.assertEqual(weighted["aggregation_weight"], .875 * raw["helper_sample_weight"])
            self.assertEqual(weighted["projection_applied"], 0)
        self.assertNotIn("target_weight", retained[1])

    def test_avg_bitwise_existing_parameter_average_on_helpers_only(self):
        for dtype in (torch.float32, torch.float64):
            self.initial = {n: p.to(dtype) for n, p in self.initial.items()}
            self.posts = {cid: {n: p.to(dtype) for n, p in post.items()} for cid, post in self.posts.items()}
            _, q, _ = feedback_weights(self.samples, 0, "feedback_avg")
            expected = aggregate_target_updates(self.initial, self.posts[1],
                ((cid, q[cid], self.posts[cid]) for cid in self.order), 1, "avg")
            with patch("utils.target_projection.aggregate_target_updates", side_effect=AssertionError("target kernel called")):
                actual = self.aggregate("feedback_avg")
            for name in self.initial:
                torch.testing.assert_close(actual[0][name], expected[0][name], atol=0, rtol=0)

    def test_anchor_zero_is_bitwise_identical_to_pure_avg(self):
        for dtype in (torch.float32, torch.float64):
            self.initial = {n: p.to(dtype) for n, p in self.initial.items()}
            self.posts = {cid: {n: p.to(dtype) for n, p in post.items()} for cid, post in self.posts.items()}
            avg, self_zero = self.aggregate("feedback_avg"), self.aggregate("feedback_avg_self", 0.)
            for name in avg[0]:
                torch.testing.assert_close(avg[0][name], self_zero[0][name], atol=0, rtol=0)
            self.assertEqual(avg[1:], self_zero[1:])

    def test_retention_matches_alpha_server_plus_helper_average_including_head(self):
        avg = self.aggregate("feedback_avg")
        retained = self.aggregate("feedback_avg_self", .125)
        for name in self.initial:
            expected = .125 * self.initial[name] + .875 * avg[0][name]
            torch.testing.assert_close(retained[0][name], expected, atol=0, rtol=0)
            torch.testing.assert_close(retained[0][name] - self.initial[name],
                .875 * (avg[0][name] - self.initial[name]), atol=1e-14, rtol=1e-14)

    def test_zero_helper_updates_keep_start_exactly_and_anchor_one_is_finite(self):
        for dtype in (torch.float32, torch.float64):
            self.initial = {n: p.to(dtype) for n, p in self.initial.items()}
            self.posts = {cid: copy.deepcopy(self.initial) for cid in range(1, 20)}
            for mode, alpha in (("feedback_avg", None), ("feedback_avg_self", 0.),
                                ("feedback_avg_self", .125), ("feedback_avg_self", 1.)):
                result, metrics, rows = self.aggregate(mode, alpha)
                for name in result:
                    torch.testing.assert_close(result[name], self.initial[name], atol=0, rtol=0)
                self.assertEqual(metrics["intermediate_vs_server_start_delta_norm"], 0.)
                self.assertTrue(all(math.isfinite(v) for v in metrics.values()))
                if alpha == 1.:
                    self.assertEqual(metrics["effective_helper_count"], 0)
                    self.assertEqual(metrics["helper_total_weight"], 0)

    def test_inputs_rng_unchanged_and_C0_extra_missing_duplicate_nonfinite_uploads_rejected(self):
        saved = copy.deepcopy((self.initial, self.posts))
        before = rng_state()
        for mode in FEEDBACK_MODES:
            self.aggregate(mode)
        assert_rng(self, before)
        for actual, old in [(self.initial, saved[0]), *[(self.posts[cid], saved[1][cid]) for cid in self.posts]]:
            for name in actual:
                torch.testing.assert_close(actual[name], old[name], atol=0, rtol=0)
        for samples in ({0: 10, 1: 10}, {}, {1: 0}, {1: float("nan")}, {1: -1}):
            with self.assertRaises(ValueError):
                feedback_weights(samples, 0, "feedback_avg")
        uploads = [(cid, self.samples[cid], self.posts[cid]) for cid in self.order]
        for invalid in (uploads[:-1], uploads + uploads[:1], uploads + [(0, 10, self.initial)]):
            with self.assertRaises(ValueError):
                aggregate_feedback_updates(self.initial, iter(invalid), self.samples, 0, "feedback_avg")
        self.posts[1] = parameters([float("inf"), 0., 0., 0.])
        with self.assertRaises(ValueError):
            self.aggregate("feedback_avg_self")


class FeedbackOptionsSummaryTests(unittest.TestCase):
    def test_corrected_names_are_idempotent_distinct_from_old_local10_paths(self):
        args = SimpleNamespace(target_proj_mode="feedback_avg_self", exp_name="trial", save_folder_name="temp",
            h5_result_root="h5", final_model_root="final", training_log_root="logs", run_log_dir="")
        with patch.dict("os.environ", {"FEDCLIP_TRAIN_LOG_DIR": "external_logs"}):
            ensure_feedback_names(args)
        first = copy.deepcopy(vars(args))
        ensure_feedback_names(args)
        self.assertEqual(vars(args), first)
        for key in ("exp_name", "save_folder_name", "h5_result_root", "final_model_root", "training_log_root", "run_log_dir"):
            self.assertIn("feedback_avg_self_anchor0p125_local5", getattr(args, key))
        old = SimpleNamespace(target_proj_mode="projection", exp_name="untouched", save_folder_name="temp")
        before = vars(old).copy()
        ensure_feedback_names(old)
        self.assertEqual(vars(old), before)

    def test_options_anchor_scope_range_and_no_extra_protocols(self):
        args = SimpleNamespace(algorithm="FedTargetProj", target_client_id=0, num_clients=20,
                               local_epochs=5, eval_gap=1, target_proj_mode="feedback_avg_self")
        for mode in FEEDBACK_MODES:
            args.target_proj_mode = mode
            validate_feedback_options(args)
        for key, value in (("target_client_id", 1), ("num_clients", 19), ("local_epochs", 1),
                           ("eval_gap", 2), ("auto_break", True), ("few_shot", 1), ("train_slow_rate", .1),
                           ("target_prox_mu", .01), ("meta_c0_split", "split.json"),
                           ("meta_collect_snapshots", True), ("meta_weight_file", "weights.json"),
                           ("projection_self_weight", .125)):
            changed = copy.copy(args); setattr(changed, key, value)
            with self.assertRaises(ValueError):
                validate_feedback_options(changed)
        for alpha in (0., .125, 1.):
            self.assertEqual(feedback_anchor_weight("feedback_avg_self", alpha), alpha)
        for alpha in (-.1, 1.1, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                feedback_anchor_weight("feedback_avg_self", alpha)
        for mode in ("feedback_avg", "projection"):
            changed = copy.copy(args); changed.target_proj_mode = mode; changed.feedback_anchor_weight = .125
            with self.assertRaises(ValueError):
                validate_feedback_options(changed)

    def test_projection_deferred_at_cli_runtime_and_kernel_before_any_training(self):
        args = SimpleNamespace(target_proj_mode="feedback_projection_self", target_client_id=0)
        with self.assertRaisesRegex(ValueError, "deferred.*reference"):
            validate_feedback_options(args)
        with self.assertRaisesRegex(ValueError, "deferred.*reference"):
            FedTargetProj(args, 0)
        with self.assertRaisesRegex(ValueError, "deferred.*reference"):
            aggregate_feedback_updates({}, iter([]), {1: 10}, 0, "feedback_projection_self")
        output = io.StringIO()
        with patch.object(sys, "argv", ["run_target_proj.py", "--modes", "feedback_projection_self", "--dry-run"]), \
                redirect_stdout(output), patch.object(sys, "stderr", io.StringIO()), self.assertRaises(SystemExit):
            launcher.main()
        self.assertEqual(output.getvalue(), "")

    def test_two_accuracy_series_primary_earliest_best_last10_last20(self):
        history = [dict(round=i + 1, target_intermediate_download_acc=.2,
                       target_post_local_acc=.8 if i in (2, 3) else i / 100) for i in range(25)]
        summary = feedback_accuracy_summary(history)
        self.assertEqual(set(summary), {"intermediate_download", "target_local"})
        self.assertEqual(summary["target_local"]["final"], .24)
        self.assertEqual(summary["target_local"]["best_round"], 3)
        self.assertEqual(summary["target_local"]["last10_count"], 10)
        self.assertEqual(summary["target_local"]["last20_count"], 20)

    def test_launcher_two_modes_no_C0_upload_flag_and_anchor_parameter_forwarded(self):
        output = io.StringIO()
        with patch.object(sys, "argv", ["run_target_proj.py", "--modes", *FEEDBACK_MODES, "--rounds", "200", "--dry-run"]), \
                redirect_stdout(output):
            launcher.main()
        lines = output.getvalue().splitlines()
        self.assertEqual(len(lines), 2)
        for line, mode in zip(lines, FEEDBACK_MODES):
            for value in (feedback_tag(mode), "-gr 200", "-ls 5", "-eg 1", "-lr 0.005", "--run_log_dir"):
                self.assertIn(value, line)
        output = io.StringIO()
        with patch.object(sys, "argv", ["run_target_proj.py", "--modes", "feedback_avg_self",
                    "--feedback_anchor_weight", ".125", "--dry-run"]), redirect_stdout(output):
            launcher.main()
        self.assertIn("--feedback_anchor_weight 0.125", output.getvalue())

    def test_formal_array_two_modes_public_settings_and_independent_roots(self):
        source = (Path(__file__).resolve().parents[1] / "system/target_feedback_commands.sh").read_text()
        commands = re.findall(r'^\s+"([^"]+)"$', source, re.MULTILINE)
        self.assertEqual(len(commands), 2)
        paths, gpus = [], []
        for command, mode in zip(commands, FEEDBACK_MODES):
            args = shlex.split(command)
            for flag, value in (("--target_proj_mode", mode), ("--seed", "0"), ("--target_client_id", "0"),
                ("-t", "1"), ("-gr", "200"), ("-ls", "5"), ("-eg", "1"), ("-lr", "0.005"), ("-lbs", "16"),
                ("-nc", "20"), ("-ncl", "100"), ("-jr", "1.0"), ("-m", "Decom_CNN-5-512"),
                ("-data", "Cifar100"), ("-pt", "pat"), ("-cpc", "20"), ("-regular_lamda", "1e-3"),
                ("--target_prox_mu", "0")):
                self.assertEqual(args[args.index(flag) + 1], value)
            if mode == "feedback_avg_self":
                self.assertEqual(args[args.index("--feedback_anchor_weight") + 1], "0.125")
            else:
                self.assertNotIn("--feedback_anchor_weight", args)
            locations = [args[args.index(key) + 1] for key in ("-sfn", "--h5_result_root", "--final-model-root", "--run_log_dir")]
            self.assertTrue(all(feedback_tag(mode) in location for location in locations))
            self.assertTrue(locations[-1].startswith("training_logs/"))
            self.assertNotIn("--projection_self_weight", args)
            paths.append(tuple(locations)); gpus.append(args[args.index("-did") + 1])
        self.assertEqual(len(set(paths)), 2)
        self.assertEqual(len(set(gpus)), 2)


if __name__ == "__main__":
    unittest.main()

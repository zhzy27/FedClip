"""Mass-only softmax controls: strict paired weights, raw deltas and legacy regressions."""

from contextlib import redirect_stdout
import copy
import io
import math
from pathlib import Path
import re
import shlex
import subprocess
import sys
import unittest
from unittest.mock import patch

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "system"))
from utils import projection_variants as single
from utils.group_projection import aggregate_group_updates, accuracy_statistics, group_accuracy_summary
from utils.target_group_options import DEFAULT_GROUP_IDS, normalize_group_ids, validate_group_options
from utils.projection_self_weight import validate_projection_self_weight
from utils.target_prox_options import validate_target_prox
from test_meta_projection import rng_state, assert_rng
from test_target_proj_launcher import launcher


def parameters(vector, dtype=torch.float64):
    values = torch.tensor(vector, dtype=dtype)
    return {"conv.weight": values[:2], "head.weight": values[2:]}


class NoProjectionTests(unittest.TestCase):
    def setUp(self):
        self.initial = parameters([4., -3., 2., 7.])
        self.samples = dict.fromkeys(range(20), .05)
        self.posts = [{name: value + parameters([(-1.) ** cid * (cid + 1), .3 * cid, .2, -.1])[name]
                       for name, value in self.initial.items()} for cid in range(20)]
        self.order = list(reversed(range(20)))
        self.uploads = lambda: iter((cid, self.samples[cid], self.posts[cid]) for cid in self.order)

    def test_single_point125_strict_weights_raw_sum_and_conflict_not_erased(self):
        projected = single.aggregate_projection_weighting(self.initial, self.posts[0], self.uploads, 0,
                                                           "projection_softmax", .125)
        with patch.object(single, "aggregate_target_updates", side_effect=AssertionError("projection called")):
            plain = single.aggregate_softmax_only(self.initial, self.posts[0], self.uploads, 0, self_weight=.125)
        for old, new in zip(projected[2], plain[2]):
            for key in ("aggregation_weight", "sample_weight", "cosine_before_projection",
                        "raw_similarity_score", "dot_before_projection", "conflict"):
                self.assertEqual(new[key], old[key])
            self.assertEqual(new["projection_coefficient"], 0.)
            self.assertEqual(new["removed_component_norm"], 0.)
        self.assertEqual(plain[1]["target_weight"], .125)
        self.assertAlmostEqual(plain[1]["helper_total_weight"], .875)
        self.assertGreater(plain[1]["conflict_client_count"], 0)
        self.assertEqual(plain[1]["projection_enabled"], 0)
        self.assertGreater(plain[1]["effective_all_client_count"], 0)
        self.check_raw_sum(plain)
        self.assertTrue(any(not torch.equal(plain[0][n], projected[0][n]) for n in self.initial))

    def test_group_half_strict_weights_reference_scores_raw_sum_and_conflict_semantics(self):
        args = (self.initial, self.uploads, self.samples, DEFAULT_GROUP_IDS)
        projected = aggregate_group_updates(*args, "group_projection_softmax", .5)
        plain = aggregate_group_updates(*args, "group_softmax_only", .5)
        for old, new in zip(projected[2], plain[2]):
            for key in ("aggregation_weight", "original_sample_weight", "group_sample_share",
                        "cosine_before_projection", "dot_before_projection", "raw_softmax_score",
                        "outside_q", "raw_conflict_with_reference"):
                self.assertEqual(new[key], old[key])
            self.assertEqual(new["projection_coefficient"], 0.)
            self.assertEqual(new["removed_component_norm"], 0.)
            self.assertEqual(new["conflict"], 0)  # Old group field means projection actually performed.
            if new["in_target_group"]:
                self.assertAlmostEqual(new["aggregation_weight"], .5 / 7)
        for key in ("group_reference_norm", "effective_all_client_count", "effective_group_count",
                    "effective_outside_count", "actual_group_mass", "actual_outside_mass"):
            self.assertEqual(plain[1][key], projected[1][key])
        self.assertGreater(plain[1]["raw_outside_conflict_count"], 0)
        self.assertEqual(plain[1]["outside_conflict_count"], 0)
        self.assertEqual(plain[1]["projection_enabled"], 0)
        self.assertEqual(plain[3]["weighting_scope"], "outside_raw_cosine_softmax")
        self.check_raw_sum(plain)
        self.assertTrue(any(not torch.equal(plain[0][n], projected[0][n]) for n in self.initial))

    def check_raw_sum(self, output):
        weights = {row["client_id"]: row["aggregation_weight"] for row in output[2]}
        self.assertAlmostEqual(math.fsum(weights.values()), 1.)
        for name, initial in self.initial.items():
            expected = initial + sum((self.posts[cid][name] - initial) * weights[cid] for cid in self.order)
            torch.testing.assert_close(output[0][name], expected, atol=1e-14, rtol=1e-14)

    def test_legacy_single_and_three_group_modes_bitwise_identical_to_prechange_commit(self):
        root = Path(__file__).resolve().parents[1]
        previous = {}
        for key, path in (("single", "projection_variants"), ("group", "group_projection")):
            source = subprocess.check_output(["git", "show", f"6449fc5:system/utils/{path}.py"], cwd=root, text=True)
            namespace = {}
            exec(compile(source, f"6449fc5/{path}.py", "exec"), namespace)
            previous[key] = namespace
        for dtype in (torch.float32, torch.float64):
            self.initial = {n: p.to(dtype) for n, p in self.initial.items()}
            self.posts = [{n: p.to(dtype) for n, p in post.items()} for post in self.posts]
            for mode, weight in (("softmax_only", None), ("projection_softmax", None),
                                ("projection_softmax", .125), ("projection_relu", None)):
                args = (self.initial, self.posts[0], self.uploads, 0, mode, weight)
                actual = single.aggregate_projection_weighting(*args)
                expected = previous["single"]["aggregate_projection_weighting"](*args)
                self.check_exact_output(actual, expected)
            for mode in ("group_avg_all", "group_avg_only", "group_projection_softmax"):
                for beta in ((None, .5) if mode == "group_projection_softmax" else (None,)):
                    args = (self.initial, self.uploads, self.samples, DEFAULT_GROUP_IDS, mode, beta)
                    self.check_exact_output(aggregate_group_updates(*args),
                                            previous["group"]["aggregate_group_updates"](*args))

    def check_exact_output(self, actual, expected):
        for name in actual[0]:
            torch.testing.assert_close(actual[0][name], expected[0][name], atol=0, rtol=0)
        self.assertEqual(actual[1:], expected[1:])

    def test_boundaries_defaults_and_single_member_agree_with_single_softmax(self):
        for beta in (None, 0., .125, 1.):
            single_result = single.aggregate_projection_weighting(self.initial, self.posts[0], self.uploads,
                                                                 0, "softmax_only", beta)
            group_result = aggregate_group_updates(self.initial, self.uploads, self.samples, [0],
                                                   "group_softmax_only", beta)
            for name in self.initial:
                torch.testing.assert_close(single_result[0][name], group_result[0][name], atol=1e-14, rtol=1e-14)
            for left, right in zip(single_result[2], group_result[2]):
                self.assertAlmostEqual(left["aggregation_weight"], right["aggregation_weight"], places=14)
        default = aggregate_group_updates(self.initial, self.uploads, self.samples, DEFAULT_GROUP_IDS,
                                          "group_softmax_only")
        self.assertAlmostEqual(default[1]["actual_group_mass"], .35)
        for beta in (0., 1.):
            output = aggregate_group_updates(self.initial, self.uploads, self.samples, DEFAULT_GROUP_IDS,
                                             "group_softmax_only", beta)
            self.check_raw_sum(output)
            self.assertEqual(output[1]["effective_group_count"] if beta == 0 else output[1]["effective_outside_count"], 0)
            self.assertTrue(all(math.isfinite(x) for x in output[1].values()))
        for beta in (None, 1.):
            aggregate_group_updates(self.initial, self.uploads, self.samples, list(range(20)), "group_softmax_only", beta)
        with self.assertRaises(ValueError):
            aggregate_group_updates(self.initial, self.uploads, self.samples, list(range(20)), "group_softmax_only", .5)
        # A sole target can retain mass 1; positive mass with no helpers is invalid.
        one = lambda: iter([(0, 1., self.posts[0])])
        single.aggregate_softmax_only(self.initial, self.posts[0], one, 0, 1.)
        with self.assertRaises(ValueError):
            single.aggregate_softmax_only(self.initial, self.posts[0], one, 0, .5)

    def test_zero_near_zero_reference_and_readonly_inputs_rng(self):
        original = copy.deepcopy((self.initial, self.posts))
        for tiny in (0., 1e-15, 1e-8):
            self.initial = parameters([0.] * 4)
            self.posts = [parameters([tiny, 0., 0., 0.])] + [parameters([-1., 2., 0., 0.])] * 19
            before = rng_state()
            expected = copy.deepcopy((self.initial, self.posts))
            plain = single.aggregate_softmax_only(self.initial, self.posts[0], self.uploads, 0, .125)
            group = aggregate_group_updates(self.initial, self.uploads, self.samples, [0], "group_softmax_only", .5)
            assert_rng(self, before)
            for actual, saved in [(self.initial, expected[0]), *zip(self.posts, expected[1])]:
                for name in actual:
                    torch.testing.assert_close(actual[name], saved[name], atol=0, rtol=0)
            for result in (plain, group):
                self.assertTrue(all(torch.isfinite(p).all() for p in result[0].values()))
                self.assertTrue(all(row["projection_coefficient"] == 0 for row in result[2]))
                if tiny == 0:
                    self.assertTrue(all(row["cosine_before_projection"] == 0 for row in result[2]))
        self.initial, self.posts = original

    def test_invalid_parameters_members_and_nonzero_prox_rejected(self):
        for value in (-.1, 1.1, float("nan"), float("inf")):
            for mode in ("softmax_only", "projection_softmax"):
                with self.assertRaises(ValueError):
                    validate_projection_self_weight(value, mode)
            with self.assertRaises(ValueError):
                validate_group_options(None, value, "group_softmax_only", 20)
        for ids in ([], "", "0,0", "20", "-1", [True]):
            with self.assertRaises(ValueError):
                normalize_group_ids(ids, 20)
        for mode in ("softmax_only", "group_softmax_only"):
            validate_target_prox(0., "helpers", mode=mode)
            with self.assertRaises(ValueError):
                validate_target_prox(.01, "helpers", mode=mode)
        with self.assertRaises(ValueError):
            validate_group_options(None, .5, "group_avg_all", 20)

    def test_weighted_group_summary_counts_tail20_and_earliest_best(self):
        history = []
        for index in range(25):
            correct = (1, 0) if index in (0, 1) else (0, 5)
            members = [dict(client_id=cid, post_local_acc=c / n, post_local_correct=c, test_samples=n)
                       for cid, (c, n) in enumerate(zip(correct, (1, 9)))]
            stats = accuracy_statistics(members, "post_local_acc")
            history.append(dict(round=index + 1, mode="group_softmax_only", members=members,
                group_post_local_acc=stats["macro"], group_post_local_sample_weighted_acc=stats["sample_weighted"]))
        summary = group_accuracy_summary(history)
        self.assertEqual(summary["group_post_local"]["final"], .5)
        self.assertEqual(summary["group_post_local"]["best_round"], 3)
        self.assertEqual(summary["group_post_local_macro"]["best_round"], 1)
        self.assertEqual(summary["group_post_local"]["last20_count"], 20)
        self.assertEqual(summary["group_post_local"]["last20"], .5)
        self.assertEqual(summary["members"]["0"]["best_round"], 1)
        self.assertEqual(summary["members"]["1"]["best_round"], 3)


class NoProjectionLauncherTests(unittest.TestCase):
    def test_launcher_accepts_modes_masses_and_rejects_cross_mode_or_prox(self):
        for mode, flag, value, tag in (("softmax_only", "--projection_self_weight", ".125", "self0p125"),
                                      ("group_softmax_only", "--target_group_weight", ".5", "beta0p5")):
            output = io.StringIO()
            with patch.object(sys, "argv", ["run_target_proj.py", "--dry-run", "--modes", mode,
                                           flag, value, "--rounds", "200"]), redirect_stdout(output):
                launcher.main()
            command = output.getvalue()
            for token in (mode, tag, "-gr 200", "-eg 1", "--run_log_dir", flag):
                self.assertIn(token, command)
        for options in (["--modes", "softmax_only", "--target_prox_mu", ".1"],
                        ["--modes", "group_softmax_only", "--target_prox_mu", ".1"],
                        ["--modes", "group_softmax_only", "--projection_self_weight", ".125"],
                        ["--modes", "softmax_only", "--target_group_weight", ".5"]):
            with patch.object(sys, "argv", ["run_target_proj.py", "--dry-run", *options]), \
                    patch.object(sys, "stderr", io.StringIO()), self.assertRaises(SystemExit):
                launcher.main()

    def test_formal_array_public_settings_masses_independent_names_and_roots(self):
        source = (Path(__file__).resolve().parents[1] / "system/softmax_no_projection_commands.sh").read_text()
        self.assertIn("declare -a COMMANDS=(", source)
        commands = re.findall(r'^\s+"([^"]+)"$', source, re.MULTILINE)
        self.assertEqual(len(commands), 2)
        directories, gpus = [], []
        for command, mode, flag, value in zip(commands, ("softmax_only", "group_softmax_only"),
                ("--projection_self_weight", "--target_group_weight"), ("0.125", "0.50")):
            args = shlex.split(command)
            for key, expected in (("--target_proj_mode", mode), (flag, value), ("-gr", "200"), ("-eg", "1"),
                    ("-ls", "5"), ("-lr", "0.005"), ("-lbs", "16"), ("-jr", "1.0"), ("-nc", "20"),
                    ("-ncl", "100"), ("-data", "Cifar100"), ("-m", "Decom_CNN-5-512"),
                    ("-is_regular", "1"), ("-regular_lamda", "1e-3"), ("-pt", "pat"), ("-cpc", "20"),
                    ("--seed", "0"), ("--target_prox_mu", "0")):
                self.assertEqual(args[args.index(key) + 1], expected)
            for key in ("--meta_c0_split", "--apa_self_weight"):
                self.assertNotIn(key, args)
            paths = [args[args.index(key) + 1] for key in ("-sfn", "--h5_result_root", "--final-model-root", "--run_log_dir")]
            self.assertTrue(all(mode in p for p in paths))
            self.assertTrue(paths[-1].startswith("training_logs/"))
            directories.append(tuple(paths)); gpus.append(args[args.index("-did") + 1])
        self.assertEqual(len(set(directories)), 2)
        self.assertEqual(len(set(gpus)), 2)


if __name__ == "__main__":
    unittest.main()

"""Group formulas, finite boundaries, state protection, metrics and launchers."""

from contextlib import redirect_stdout
import copy
import io
import json
import math
from pathlib import Path
import random
import shlex
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "system"))
from utils.group_projection import (GROUP_MODES, aggregate_group_updates, accuracy_statistics,
                                    group_accuracy_summary, normalize_group_ids, validate_group_options)
from utils.projection_variants import aggregate_projection_weighting
from utils.target_projection import aggregate_target_updates
from test_meta_projection import rng_state, assert_rng
from test_target_proj_launcher import launcher


def params(vector):
    v = torch.tensor(vector, dtype=torch.float64)
    return {"base.weight": v[:2], "head.weight": v[2:]}


class GroupProjectionTests(unittest.TestCase):
    def setUp(self):
        self.initial = params([10., 20., 30., 40.])
        self.deltas = [params([-2., 0., 0., 0.]), params([4., 0., 0., 0.]),
                       params([2., 3., 0., 4.]), params([-4., 5., 6., 0.])]
        self.posts = [{n: self.initial[n] + delta[n] for n in self.initial} for delta in self.deltas]
        self.samples = {0: .1, 1: .1, 2: .3, 3: .5}
        self.order = [3, 1, 2, 0]

    def aggregate(self, mode, group=(0, 1), beta=None, initial=None, posts=None, samples=None):
        initial, posts, samples = initial or self.initial, posts or self.posts, samples or self.samples
        order = self.order if len(samples) == 4 else list(samples)
        return aggregate_group_updates(initial, lambda: iter((cid, samples[cid], posts[cid]) for cid in order),
                                       samples, list(group), mode, beta)

    def test_avg_all_is_bitwise_original_avg_and_avg_only_is_direct_group_average(self):
        result, metrics, rows, _ = self.aggregate("group_avg_all")
        expected = aggregate_target_updates(self.initial, self.posts[0],
            ((cid, self.samples[cid], self.posts[cid]) for cid in self.order), 0, "avg")[0]
        for name in result:
            torch.testing.assert_close(result[name], expected[name], atol=0, rtol=0)
        result, metrics, rows, _ = self.aggregate("group_avg_only")
        for name in result:
            torch.testing.assert_close(result[name], (self.posts[0][name] + self.posts[1][name]) * .5, atol=0, rtol=0)
        self.assertEqual(metrics["actual_group_mass"], 1.)
        self.assertEqual(metrics["actual_outside_mass"], 0.)
        self.assertEqual(metrics["effective_outside_count"], 0.)
        self.assertTrue(all(row["projection_coefficient"] == 0 for row in rows))

    def test_projection_analytic_result_keeps_opposing_group_member_and_uses_raw_cosines(self):
        result, metrics, rows, matrices = self.aggregate("group_projection_softmax", beta=.35)
        reference = {n: (self.deltas[0][n] + self.deltas[1][n]) / 2 for n in self.initial}
        lookup = {r["client_id"]: r for r in rows}
        q2 = math.exp(lookup[2]["cosine_before_projection"] / .2)
        q3 = math.exp(lookup[3]["cosine_before_projection"] / .2)
        total = q2 + q3
        for name in result:
            outside = sum((self.deltas[cid][name] - lookup[cid]["projection_coefficient"] * reference[name])
                          * .65 * q / total for cid, q in ((2, q2), (3, q3)))
            expected = self.initial[name] + .35 * reference[name] + outside
            torch.testing.assert_close(result[name], expected, atol=1e-14, rtol=1e-14)
        self.assertEqual(lookup[0]["conflict"], 0)  # D0 is opposite to the group's axis.
        self.assertEqual(lookup[0]["projection_coefficient"], 0)
        self.assertEqual(lookup[2]["projection_coefficient"], 0)
        self.assertAlmostEqual(lookup[3]["projection_coefficient"], -4 / (1 + 1e-12))
        self.assertEqual(lookup[3]["cosine_before_projection"], -4 / math.sqrt(77))
        self.assertAlmostEqual(metrics["actual_group_mass"], .35)
        self.assertAlmostEqual(metrics["actual_outside_mass"], .65)
        self.assertEqual(matrices["group_pair_conflict_matrix"], [[0, 1], [1, 0]])
        self.assertGreater(result["head.weight"][1], self.initial["head.weight"][1])

    def test_single_member_matches_legacy_projection_softmax_for_each_beta(self):
        for cid in (0, 1):
            for beta in (None, 0., .05, .125, .35, 1.):
                uploads = lambda: iter((i, self.samples[i], self.posts[i]) for i in self.order)
                original = aggregate_projection_weighting(self.initial, self.posts[cid], uploads, cid,
                                                            "projection_softmax", beta)
                actual = self.aggregate("group_projection_softmax", group=(cid,), beta=beta)
                for name in actual[0]:
                    torch.testing.assert_close(actual[0][name], original[0][name], atol=2e-14, rtol=2e-14)
                expected = {r["client_id"]: r["aggregation_weight"] for r in original[2]}
                for row in actual[2]:
                    self.assertAlmostEqual(row["aggregation_weight"], expected[row["client_id"]], places=14)

    def test_default_mass_uses_sample_counts_and_equal_20_clients_default_is_point35(self):
        result = self.aggregate("group_projection_softmax")
        self.assertAlmostEqual(result[1]["actual_group_mass"], .2)
        samples = dict.fromkeys(range(20), .05)
        posts = [self.posts[i % 4] for i in range(20)]
        result = self.aggregate("group_projection_softmax", (0, 1, 2, 3, 4, 6, 7), posts=posts, samples=samples)
        self.assertAlmostEqual(result[1]["actual_group_mass"], .35)
        self.assertAlmostEqual(result[1]["actual_outside_mass"], .65)

    def test_zero_and_near_zero_reference_follow_exact_epsilon_rules(self):
        posts = copy.deepcopy(self.posts)
        posts[1] = {n: self.initial[n] - self.deltas[0][n] for n in self.initial}
        result, metrics, rows, _ = self.aggregate("group_projection_softmax", posts=posts)
        self.assertEqual(metrics["reference_is_zero"], 1)
        self.assertTrue(all(r["cosine_before_projection"] == 0 and r["projection_coefficient"] == 0 for r in rows))
        self.assertEqual(rows[2]["outside_q"], .5)
        self.assertEqual(rows[3]["outside_q"], .5)
        tiny = params([1e-15, 0., 0., 0.])
        initial = params([0., 0., 0., 0.])
        result, metrics, rows, _ = self.aggregate("group_projection_softmax", (0,), initial=initial,
            posts=[tiny, params([-1., 0., 0., 0.]), tiny, tiny])
        self.assertEqual(metrics["reference_is_zero"], 0)
        self.assertEqual(metrics["reference_epsilon_dominated"], 1)
        self.assertAlmostEqual(rows[1]["projection_coefficient"], -1e-15 / (1e-30 + 1e-12))
        self.assertTrue(all(torch.isfinite(p).all() for p in result.values()))

    def test_zero_one_mass_and_no_outside_conventions(self):
        for beta in (0., 1.):
            result, metrics, rows, _ = self.aggregate("group_projection_softmax", beta=beta)
            self.assertAlmostEqual(sum(r["aggregation_weight"] for r in rows), 1.)
            self.assertEqual(metrics["effective_group_count"] if beta == 0 else metrics["effective_outside_count"], 0.)
            self.assertTrue(all(math.isfinite(v) for v in metrics.values()))
        self.aggregate("group_projection_softmax", (0, 1, 2, 3), beta=1.)
        self.aggregate("group_projection_softmax", (0, 1, 2, 3))
        with self.assertRaises(ValueError):
            self.aggregate("group_projection_softmax", (0, 1, 2, 3), beta=.9)
        rounded = {0: .1, 1: .2, 2: .3, 3: .3999999999999999}
        result = self.aggregate("group_projection_softmax", (0, 1, 2, 3), samples=rounded)
        self.assertEqual(result[1]["actual_outside_mass"], 0.)
        self.assertEqual(result[1]["configured_group_beta"], 1.)

    def test_invalid_ids_parameters_uploads_and_parameter_sets_rejected(self):
        for ids in ("", "0,,1", "0,0", "x", "-1", "20", [], [True], [0.5]):
            with self.assertRaises(ValueError):
                normalize_group_ids(ids, 20)
        for beta in (float("nan"), float("inf"), -.1, 1.1):
            with self.assertRaises(ValueError):
                validate_group_options(None, beta, "group_projection_softmax", 20)
        for mode in ("avg", "projection_softmax", "group_avg_all", "group_avg_only"):
            with self.assertRaises(ValueError):
                validate_group_options(None, .35, mode, 20)
        with self.assertRaises(ValueError):
            validate_group_options("0", None, "avg", 20)
        for kwargs in (dict(meta_split="split.json"), dict(collect=True)):
            with self.assertRaises(ValueError):
                validate_group_options(None, None, "group_avg_all", 20, **kwargs)
        with self.assertRaises(ValueError):
            aggregate_group_updates(self.initial, lambda: iter([(0, .1, self.posts[0])]),
                                    self.samples, [0, 1], "group_avg_all")
        posts = copy.deepcopy(self.posts)
        posts[1].pop("head.weight")
        with self.assertRaises(ValueError):
            self.aggregate("group_avg_all", posts=posts)

    def test_inputs_and_all_rng_streams_protected_even_when_factory_draws_or_raises(self):
        old = copy.deepcopy((self.initial, self.posts))
        def factory():
            random.random(); np.random.rand(); torch.rand(1)
            for cid in self.order:
                yield cid, self.samples[cid], self.posts[cid]
        for mode in GROUP_MODES:
            before = rng_state()
            aggregate_group_updates(self.initial, factory, self.samples, [0, 1], mode)
            assert_rng(self, before)
        for actual, expected in [(self.initial, old[0]), *zip(self.posts, old[1])]:
            for name in actual:
                torch.testing.assert_close(actual[name], expected[name], atol=0, rtol=0)
        def failed():
            torch.rand(1); random.random(); np.random.rand()
            raise RuntimeError("read failed")
            yield
        before = rng_state()
        with self.assertRaises(RuntimeError):
            aggregate_group_updates(self.initial, failed, self.samples, [0, 1], "group_avg_all")
        assert_rng(self, before)

    def test_macro_summary_best_is_best_round_average_not_average_member_bests(self):
        members1 = [dict(client_id=0, post_local_acc=1., test_samples=1),
                    dict(client_id=1, post_local_acc=0., test_samples=9)]
        members2 = [dict(client_id=0, post_local_acc=0., test_samples=1),
                    dict(client_id=1, post_local_acc=1., test_samples=9)]
        stats = accuracy_statistics(members1, "post_local_acc")
        self.assertEqual(stats["macro"], .5)
        self.assertEqual(stats["sample_weighted"], .1)
        self.assertEqual(stats["worst"], 0.)
        self.assertEqual(stats["std"], .5)
        history = [dict(round=i + 1, group_post_local_acc=.5, members=members1 if i % 2 == 0 else members2)
                   for i in range(12)]
        summary = group_accuracy_summary(history)
        self.assertEqual(summary["group_post_local"]["best"], .5)
        self.assertEqual(summary["group_post_local"]["best_round"], 1)
        self.assertEqual(summary["members"]["0"]["best"], 1.)
        self.assertEqual(summary["members"]["1"]["best_round"], 2)
        self.assertEqual(summary["group_post_local"]["last10_count"], 10)


class GroupLauncherTests(unittest.TestCase):
    def test_three_modes_canonical_group_and_public_parameters(self):
        output = io.StringIO()
        with patch.object(sys, "argv", ["run_target_proj.py", "--dry-run", "--modes", *GROUP_MODES,
                    "--target_group_ids", "7,6,4,3,2,1,0"]), redirect_stdout(output):
            launcher.main()
        lines = output.getvalue().splitlines()
        self.assertEqual(len(lines), len(GROUP_MODES))
        for line in lines:
            self.assertIn("--target_group_ids 0,1,2,3,4,6,7", line)
            self.assertIn("group0_1_2_3_4_6_7", line)
            self.assertNotIn("target0_seed", line)
            self.assertNotIn("--target_group_weight", line)
            self.assertNotIn("--projection_self_weight", line)
            for token in ("-gr 100", "-ls 5", "-lr 0.005", "-lbs 16", "-nc 20", "-m Decom_CNN-5-512"):
                self.assertIn(token, line)

    def test_explicit_mass_and_bad_combinations(self):
        output = io.StringIO()
        with patch.object(sys, "argv", ["run_target_proj.py", "--dry-run", "--modes",
                    "group_projection_softmax", "--target_group_weight", ".35"]), redirect_stdout(output):
            launcher.main()
        self.assertIn("--target_group_weight 0.35", output.getvalue())
        cases = (["--modes", "group_avg_all", "--target_group_weight", ".35"],
                 ["--modes", "group_projection_softmax", "--projection_self_weight", ".125"],
                 ["--modes", "group_projection_softmax", "--target_group_weight", "nan"],
                 ["--modes", "group_avg_only", "--meta_c0_split", "split.json"],
                 ["--modes", "projection", "--target_group_ids", "0,1"])
        for options in cases:
            with patch.object(sys, "argv", ["run_target_proj.py", "--dry-run", *options]), \
                    patch.object(sys, "stderr", io.StringIO()), self.assertRaises(SystemExit):
                launcher.main()

    def test_formal_array_has_three_modes_distinct_gpus_dirs_and_no_beta_scan(self):
        import re
        text = (Path(__file__).resolve().parents[1] / "system" / "target_group_commands.sh").read_text()
        self.assertIn("declare -a COMMANDS=(", text)
        commands = re.findall(r'^\s+"([^"]+)"$', text, re.MULTILINE)
        self.assertEqual(len(commands), 3)
        directories, gpus, names = [], [], []
        for command, mode in zip(commands, GROUP_MODES[:3]):
            args = shlex.split(command)
            for flag, value in (("--target_proj_mode", mode), ("--target_group_ids", "0,1,2,3,4,6,7"),
                ("-gr", "100"), ("-ls", "5"), ("-lr", "0.005"), ("-lbs", "16"), ("-nc", "20"),
                ("-ncl", "100"), ("-m", "Decom_CNN-5-512"), ("-data", "Cifar100"),
                ("-pt", "pat"), ("-cpc", "20"), ("--seed", "0"), ("-regular_lamda", "1e-3")):
                self.assertEqual(args[args.index(flag) + 1], value)
            for flag in ("--target_group_weight", "--projection_self_weight", "--meta_c0_split", "--apa_self_weight"):
                self.assertNotIn(flag, args)
            directories.append(tuple(args[args.index(flag) + 1] for flag in
                ("-sfn", "--h5_result_root", "--final-model-root")))
            names.append(args[args.index("-exp_name") + 1])
            gpus.append(args[args.index("-did") + 1])
        self.assertEqual(len(set(directories)), 3)
        self.assertEqual(len(set(names)), 3)
        self.assertEqual(len(set(gpus)), 3)


if __name__ == "__main__":
    unittest.main()

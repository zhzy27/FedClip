"""Strict ProjectionSoftmax ablation: same weights, completely untouched deltas."""

import copy
import importlib.util
import math
from pathlib import Path
import random
import subprocess
import sys
import unittest
from unittest import mock

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "system"))
import utils.projection_variants as variants
from utils.target_projection import aggregate_target_updates


def parameters(values, dtype=torch.float64):
    return {"conv.weight": torch.tensor(values[:2], dtype=dtype), "fc.bias": torch.tensor(values[2:], dtype=dtype)}


def flatten(values):
    return torch.cat([value.reshape(-1) for value in values.values()])


class SoftmaxOnlyTests(unittest.TestCase):
    def setUp(self):
        self.initial = parameters([0., 0., 0.])
        self.target = parameters([1., 0., 0.])

    def run_mode(self, uploads, mode="softmax_only", initial=None):
        return variants.aggregate_projection_weighting(initial or self.initial, uploads[0][2], lambda: iter(uploads), 0, mode)

    def test_weights_conserve_target_and_helper_mass_including_negative_helpers(self):
        uploads = [(cid, .05, parameters([(-1.) ** cid, cid, 0.])) for cid in range(20)]
        _, metrics, rows, _, _ = self.run_mode(uploads)
        weights = [row["aggregation_weight"] for row in rows]
        self.assertEqual(weights[0], .05)
        self.assertAlmostEqual(math.fsum(weights[1:]), .95, places=15)
        self.assertAlmostEqual(math.fsum(weights), 1., places=15)
        self.assertTrue(all(weight > 0 for weight in weights[1:]))
        self.assertGreater(metrics["conflict_client_count"], 0)
        self.assertAlmostEqual(metrics["mean_helper_weight"], np.mean(weights[1:]), places=15)
        self.assertAlmostEqual(metrics["std_helper_weight"], np.std(weights[1:]), places=15)

    def test_uniform_cosines_equal_balanced_sample_avg(self):
        uploads = [(cid, .25, parameters([float(cid + 1), 0., 0.])) for cid in range(4)]
        result, metrics, rows, _, _ = self.run_mode(uploads)
        avg = aggregate_target_updates(self.initial, uploads[0][2], uploads, 0, "avg")[0]
        torch.testing.assert_close(flatten(result), flatten(avg), rtol=0, atol=0)
        self.assertEqual([row["aggregation_weight"] for row in rows], [.25] * 4)
        self.assertAlmostEqual(metrics["effective_helper_count"], 3.)

    def test_negative_parallel_component_and_orthogonal_component_both_survive(self):
        helper = parameters([-2., 3., 0.])
        uploads = [(0, .2, self.target), (1, .8, helper)]
        result, metrics, rows, _, matrices = self.run_mode(uploads)
        torch.testing.assert_close(flatten(result), torch.tensor([-1.4, 2.4, 0.], dtype=torch.float64))
        self.assertEqual(rows[1]["projection_coefficient"], 0.)
        self.assertEqual(rows[1]["removed_component_norm"], 0.)
        self.assertAlmostEqual(rows[1]["projected_update_norm"], math.sqrt(13))
        self.assertEqual(metrics["removed_update_ratio"], 0.)
        self.assertEqual(matrices["projection_scope"], "none")
        self.assertEqual(matrices["projection_enabled"], 0)

    def test_projected_control_has_identical_cosines_weights_and_only_correction_differs(self):
        initial = parameters([7., -3., 5.])
        target = parameters([8., -3., 5.])
        uploads = [(0, .2, target), (1, .3, parameters([5., 0., 5.])), (2, .5, parameters([8., -1., 5.]))]
        plain = self.run_mode(uploads, initial=initial)
        projected = self.run_mode(uploads, "projection_softmax", initial)
        for left, right in zip(plain[2], projected[2]):
            for key in ("cosine_before_projection", "dot_before_projection", "raw_similarity_score",
                        "aggregation_weight", "sample_weight", "temperature", "softmax_shift_max"):
                self.assertEqual(left[key], right[key])
        correction = sum(row["aggregation_weight"] * row["projection_coefficient"] for row in projected[2])
        expected_difference = -correction * (flatten(target) - flatten(initial))
        torch.testing.assert_close(flatten(projected[0]) - flatten(plain[0]), expected_difference, rtol=1e-12, atol=1e-12)
        expected = sum(row["aggregation_weight"] * flatten(upload[2]) for row, upload in zip(plain[2], uploads))
        torch.testing.assert_close(flatten(plain[0]), expected, rtol=1e-12, atol=1e-12)

    def test_never_calls_projection_kernel_and_reuses_same_softmax_function(self):
        uploads = [(0, .25, self.target), (1, .75, parameters([-1., 2., 0.]))]
        before = copy.deepcopy(uploads)
        with mock.patch.object(variants, "aggregate_target_updates", side_effect=AssertionError("Projection called")), \
                mock.patch.object(variants, "projection_similarity_weights", wraps=variants.projection_similarity_weights) as weights:
            self.run_mode(uploads)
        self.assertEqual(weights.call_args.args[2], "projection_softmax")
        for original, current in zip(before, uploads):
            torch.testing.assert_close(flatten(original[2]), flatten(current[2]), rtol=0, atol=0)

    def test_tiny_and_zero_updates_share_exact_cosines_and_weights_with_control(self):
        for tiny in (0., 1e-15, 1e-8):
            uploads = [(0, .2, parameters([tiny, 0., 0.])),
                       (1, .1, parameters([-tiny, tiny, 0.])), (2, .7, parameters([tiny, tiny, 0.]))]
            plain = self.run_mode(uploads)
            projected = self.run_mode(uploads, "projection_softmax")
            for left, right in zip(plain[2], projected[2]):
                self.assertEqual(left["cosine_before_projection"], right["cosine_before_projection"])
                self.assertEqual(left["aggregation_weight"], right["aggregation_weight"])
            self.assertTrue(torch.isfinite(flatten(plain[0])).all())

    def test_second_pass_rng_restored(self):
        calls, states = [], []

        def factory():
            calls.append(1)
            for cid in range(2):
                torch.rand(2)
                np.random.rand(2)
                random.random()
                yield cid, .5, self.target
            if len(calls) == 1:
                states.append((torch.get_rng_state(), np.random.get_state(), random.getstate()))

        variants.aggregate_projection_weighting(self.initial, self.target, factory, 0, "softmax_only")
        torch.testing.assert_close(torch.get_rng_state(), states[0][0], rtol=0, atol=0)
        np.testing.assert_array_equal(np.random.get_state()[1], states[0][1][1])
        self.assertEqual(np.random.get_state()[2:], states[0][1][2:])
        self.assertEqual(random.getstate(), states[0][2])

    def test_missing_or_changed_uploads_rejected(self):
        for uploads in ([(1, 1., self.target)], [(0, .5, self.target)] * 2, [(0, .5, self.target)],
                        [(0, float("nan"), self.target)]):
            with self.assertRaises(ValueError):
                self.run_mode(uploads)
        for changed in ([(0, .5, self.target)], [(0, .5, self.target), (1, .4, self.target)]):
            factory = mock.Mock(side_effect=[iter([(0, .5, self.target), (1, .5, self.target)]), iter(changed)])
            with self.assertRaises(ValueError):
                variants.aggregate_projection_weighting(self.initial, self.target, factory, 0, "softmax_only")

    def test_existing_projection_weighting_is_bitwise_equal_to_previous_commit(self):
        root = Path(__file__).resolve().parents[1]
        code = subprocess.check_output(["git", "show", "c12bfc5:system/utils/projection_variants.py"], cwd=root, text=True)
        previous = {}
        exec(compile(code, "c12bfc5/projection_variants.py", "exec"), previous)
        for dtype in (torch.float32, torch.float64):
            for seed in range(3):
                generator = torch.Generator().manual_seed(seed)
                initial = {"conv.weight": torch.randn(7, dtype=dtype, generator=generator)}
                uploads = [(cid, (cid + 1) / 210, {"conv.weight": torch.randn(7, dtype=dtype, generator=generator)})
                           for cid in range(20)]
                for mode in ("projection_softmax", "projection_relu"):
                    arguments = (initial, uploads[0][2], lambda: iter(uploads), 0, mode)
                    actual = variants.aggregate_projection_weighting(*arguments)
                    old = previous["aggregate_projection_weighting"](*arguments)
                    torch.testing.assert_close(flatten(actual[0]), flatten(old[0]), rtol=0, atol=0)
                    self.assertEqual(actual[1:], old[1:])


class SoftmaxOnlyScriptTests(unittest.TestCase):
    def test_dedicated_script_uses_frozen_shared_launcher(self):
        path = Path(__file__).resolve().parents[1] / "system" / "run_softmax_only.py"
        spec = importlib.util.spec_from_file_location("softmax_only_launcher", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        with mock.patch.object(sys, "argv", [str(path), "--device-id", "3", "--dry-run"]), \
                mock.patch.object(module.subprocess, "run") as run:
            module.main()
        command = run.call_args.args[0]
        self.assertEqual(command[0], sys.executable)
        self.assertEqual(Path(command[1]).name, "run_target_proj.py")
        self.assertEqual(command[2:], ["--modes", "softmax_only", "--rounds", "100", "--device-id", "3", "--dry-run"])


if __name__ == "__main__":
    unittest.main()

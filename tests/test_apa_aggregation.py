"""Causal APA basis, proxy-gradient sign, scalar optimizer and persistent server state."""

from contextlib import redirect_stdout
import copy
import csv
import importlib.util
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
from utils.apa_aggregation import aggregate_apa, apa_proxy_gradient, apa_weight_step, validate_apa_options
from utils.target_projection import aggregate_target_updates
import test_target_projection as fixtures


def params(x, y=0., dtype=torch.float64):
    return {"conv.weight": torch.tensor([x], dtype=dtype), "fc.bias": torch.tensor([y], dtype=dtype)}


class APAMathTests(unittest.TestCase):
    def test_round_zero_is_bitwise_avg_and_has_no_gradient_or_self_override(self):
        for dtype in (torch.float32, torch.float64):
            initial, target = params(3, -2, dtype), params(8, 3, dtype)
            uploads = [(2, .5, params(-4, 7, dtype)), (0, .2, target), (1, .3, params(1, -8, dtype))]
            samples = {cid: weight for cid, weight, _ in uploads}
            cache = mock.Mock()
            result, metrics, rows, weights, velocity = aggregate_apa(
                initial, target, iter(uploads), 0, samples, 0, cache_basis=cache)
            old = aggregate_target_updates(initial, target, iter(uploads), 0, "avg")[0]
            for name in result:
                torch.testing.assert_close(result[name], old[name], rtol=0, atol=0)
            self.assertEqual(weights.tolist(), [.2, .3, .5])
            self.assertEqual(velocity.tolist(), [0., 0., 0.])
            self.assertEqual(metrics["apa_weight_update_enabled"], 0)
            self.assertEqual(metrics["apa_weight_update_norm"], 0.)
            self.assertIsNone(metrics["apa_grad_norm"])
            self.assertTrue(all(row["apa_grad"] is None for row in rows))
            self.assertEqual(cache.call_count, 3)

    def test_manual_gradient_matches_autograd_in_full_parameter_space(self):
        bases = [params(2, 3), params(-1, 4), params(5, -2)]
        weights = torch.tensor([.2, .3, .5], dtype=torch.float64, requires_grad=True)
        mixture = {name: sum(weights[index] * basis[name] for index, basis in enumerate(bases)) for name in bases[0]}
        post = params(1, 6)
        loss = .5 * sum((mixture[name] - post[name]).square().sum() for name in mixture)
        loss.backward()
        gradient, proxy, residual_norm = apa_proxy_gradient(mixture, post, enumerate(bases), [0, 1, 2])
        torch.testing.assert_close(gradient, weights.grad, rtol=0, atol=0)
        self.assertAlmostEqual(proxy, loss.item())
        self.assertAlmostEqual(residual_norm ** 2, 2 * proxy)
        self.assertFalse(gradient.requires_grad)

    def test_beneficial_helper_increases_harmful_decreases_and_proxy_falls(self):
        weights = torch.tensor([.5, .25, .25], dtype=torch.float64)
        basis = [params(0), params(1), params(-1)]
        gradient, proxy, _ = apa_proxy_gradient(params(0), params(1), enumerate(basis), [0, 1, 2])
        updated, _, _, _ = apa_weight_step(weights, torch.zeros(3), gradient, 0)
        self.assertGreater(updated[1], weights[1])
        self.assertLess(updated[2], weights[2])
        new_position = updated[1] - updated[2]
        self.assertLess(.5 * (new_position.item() - 1) ** 2, proxy)

    def test_momentum_matches_two_explicit_sgd_steps(self):
        weights = torch.tensor([.5, .25, .25], dtype=torch.float64)
        g1, g2 = torch.tensor([.2, -1., 1.], dtype=torch.float64), torch.tensor([-.1, .3, -.3], dtype=torch.float64)
        first, velocity, raw, _ = apa_weight_step(weights, torch.zeros(3), g1, 0)
        torch.testing.assert_close(velocity, g1, rtol=0, atol=0)
        torch.testing.assert_close(raw, weights - .01 * g1, rtol=0, atol=0)
        _, velocity2, raw2, _ = apa_weight_step(first, velocity, g2, 0)
        torch.testing.assert_close(velocity2, .9 * g1 + g2, rtol=0, atol=0)
        torch.testing.assert_close(raw2, first - .01 * (.9 * g1 + g2), rtol=0, atol=0)

    def test_clip_then_self_then_normalize_and_self_is_not_final_half(self):
        updated, _, raw, fallback = apa_weight_step([.2, .3, .5], [0., 0., 0.], [100., -200., 0.], 0)
        self.assertLess(raw[0], 0)
        self.assertGreater(raw[1], 1)
        torch.testing.assert_close(updated, torch.tensor([.25, .5, .25], dtype=torch.float64), rtol=0, atol=0)
        self.assertFalse(fallback)

    def test_zero_sum_fallback_is_uniform(self):
        updated, _, _, fallback = apa_weight_step([.2, .3, .5], [0., 0., 0.], [100.] * 3, 0, self_weight=0)
        self.assertTrue(fallback)
        self.assertEqual(updated.tolist(), [1 / 3] * 3)

    def test_finite_nonnegative_sum_one_over_extreme_updates(self):
        generator = torch.Generator().manual_seed(42)
        weights, velocity = torch.full((20,), .05, dtype=torch.float64), torch.zeros(20, dtype=torch.float64)
        for _ in range(30):
            gradient = torch.randn(20, generator=generator, dtype=torch.float64) * 1e6
            weights, velocity, _, _ = apa_weight_step(weights, velocity, gradient, 0)
            self.assertTrue(torch.isfinite(weights).all())
            self.assertTrue(((weights >= 0) & (weights <= 1)).all())
            self.assertAlmostEqual(weights.sum().item(), 1., places=15)

    def test_zero_residual_has_zero_gradient_but_does_not_clear_momentum(self):
        gradient, loss, norm = apa_proxy_gradient(params(1, 2), params(1, 2), [(0, params(3, 4)), (1, params(8, -9))], [0, 1])
        self.assertEqual(gradient.tolist(), [0., 0.])
        self.assertEqual((loss, norm), (0., 0.))
        _, velocity, _, _ = apa_weight_step([.5, .5], [1., 2.], gradient, 0)
        torch.testing.assert_close(velocity, torch.tensor([.9, 1.8], dtype=torch.float64), rtol=0, atol=0)

    def test_previous_basis_used_before_current_uploads_and_new_weight_aggregates_current(self):
        events = []
        previous = [(0, params(0)), (1, params(2)), (2, params(-2))]

        def bases():
            for item in previous:
                events.append("basis")
                yield item

        def uploads():
            for cid, post in enumerate([params(1), params(-100), params(100)]):
                self.assertEqual(events[:3], ["basis"] * 3)
                events.append("upload")
                yield cid, 1 / 3, post

        result, metrics, rows, weights, _ = aggregate_apa(params(0), params(1), uploads(), 0,
            {cid: 1 / 3 for cid in range(3)}, 1, weights=[.5, .25, .25], velocity=[0.] * 3, bases=bases())
        self.assertEqual([row["apa_grad"] for row in rows], [0., -2., 2.])
        self.assertGreater(weights[1], .25)  # Current upload has the opposite sign; cannot generate this gradient.
        self.assertAlmostEqual(result["conv.weight"].item(), weights[0] - 100 * weights[1] + 100 * weights[2])
        self.assertEqual(metrics["apa_basis_loop_round"], 0)

    def test_missing_stale_state_or_invalid_basis_is_rejected(self):
        uploads = [(0, .5, params(1)), (1, .5, params(2))]
        with self.assertRaises(ValueError):
            aggregate_apa(params(0), params(1), uploads, 0, {0: .5, 1: .5}, 1)
        with self.assertRaises(ValueError):
            aggregate_apa(params(0), params(1), uploads, 0, {0: .5, 1: .5}, 0, weights=[.5, .5])
        for bases in ([(0, params(1))], [(0, params(1)), (0, params(2))],
                      [(0, params(1)), (1, {"wrong": torch.zeros(1)})]):
            with self.assertRaises(ValueError):
                apa_proxy_gradient(params(0), params(1), bases, [0, 1])

    def test_nan_inf_and_bad_options_fail_without_mutating_state(self):
        weights, velocity = torch.tensor([.5, .5]), torch.tensor([0., 0.])
        for invalid in (float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                apa_weight_step(weights, velocity, [invalid, 1.], 0)
            with self.assertRaises(ValueError):
                apa_proxy_gradient(params(invalid), params(1), [(0, params(1))], [0])
        self.assertEqual(weights.tolist(), [.5, .5])
        self.assertEqual(velocity.tolist(), [0., 0.])
        for args in ((-1, .9, .5), (.01, 1, .5), (.01, .9, -1), (.01, .9, float("nan"))):
            with self.assertRaises(ValueError):
                validate_apa_options(*args)

    def test_inputs_and_rng_unchanged_by_proxy_optimization_and_aggregation(self):
        before_rng = (random.getstate(), np.random.get_state(), torch.get_rng_state())
        before = [params(0), params(1), [(0, .5, params(2)), (1, .5, params(3))],
                  torch.tensor([.5, .5]), torch.zeros(2), [(0, params(1)), (1, params(-1))]]
        originals = copy.deepcopy(before)
        aggregate_apa(before[0], before[1], before[2], 0, {0: .5, 1: .5}, 1,
                      weights=before[3], velocity=before[4], bases=before[5])
        self.assertEqual(random.getstate(), before_rng[0])
        np.testing.assert_array_equal(np.random.get_state()[1], before_rng[1][1])
        torch.testing.assert_close(torch.get_rng_state(), before_rng[2], rtol=0, atol=0)
        for actual, old in zip(before[:2], originals[:2]):
            for key in old:
                torch.testing.assert_close(actual[key], old[key], rtol=0, atol=0)
        for index in (3, 4):
            torch.testing.assert_close(before[index], originals[index], rtol=0, atol=0)
        for actual, old in zip(before[2], originals[2]):
            for key in old[2]:
                torch.testing.assert_close(actual[2][key], old[2][key], rtol=0, atol=0)


class APAServerTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        self.fixture = fixtures.ProjectionIntegrationTests()
        self.fixture.setUp()

    def test_three_rounds_persist_exact_prior_bases_and_independent_logs(self):
        with tempfile.TemporaryDirectory() as folder, redirect_stdout(io.StringIO()):
            server = self.fixture.make_server(folder, "apa")
            server.target_client_id = 0
            server.global_rounds = 2
            server.Budget, server.rs_test_acc, server.auto_break = [], [], False
            server.clients = [self.fixture.make_client(folder, cid) for cid in range(3)]
            server._capture_pre_local_parameters = mock.Mock(side_effect=AssertionError("No pre snapshot for APA"))
            sentinel = Path(folder) / "projection_softmax_metrics.csv"
            sentinel.write_text("old experiment")
            original_aggregate = server._aggregate_apa
            previous_uploads = []

            def inspect_aggregate(global_params, target_post, uploads):
                current = list(uploads)
                if server.cur_ground:
                    for cid, basis in server._load_apa_bases():
                        for key in basis:
                            torch.testing.assert_close(basis[key], previous_uploads[-1][cid][key], rtol=0, atol=0)
                    residual = {key: global_params[key] - target_post[key] for key in target_post}
                    expected = [sum((previous_uploads[-1][cid][key].double() * residual[key].double()).sum().item()
                                    for key in residual) for cid in range(3)]
                previous_uploads.append({cid: {key: value.detach().clone() for key, value in post.items()}
                                         for cid, _, post in current})
                result = original_aggregate(global_params, target_post, iter(current))
                if server.cur_ground:
                    np.testing.assert_allclose([row["apa_grad"] for row in result[2]], expected, rtol=1e-14, atol=1e-14)
                return result

            server._aggregate_apa = inspect_aggregate
            server.train()
            self.assertEqual(sentinel.read_text(), "old experiment")
            self.assertEqual(server.apa_basis_round, 2)
            self.assertEqual(server.apa_weights.shape, (3,))
            self.assertEqual(len(list((Path(folder) / "apa_basis").glob("*/*.pt"))), 6)
            for client in server.clients:
                self.assertEqual(client.train_time_cost["num_rounds"], 3)
            with open(Path(folder) / "apa_history.json") as stream:
                history = json.load(stream)
            self.assertEqual([row["apa_weight_update_enabled"] for row in history["history"]], [0, 1, 1])
            self.assertEqual(history["summary"], server._local_accuracy_summary())
            self.assertTrue(all(row["target_post_local_acc"] is not None for row in history["history"]))
            self.assertTrue(all(client["apa_grad"] is None for client in history["history"][0]["clients"]))
            for suffix, count in (("metrics", 3), ("weights", 9)):
                with open(Path(folder) / f"apa_{suffix}.csv") as stream:
                    self.assertEqual(len(list(csv.DictReader(stream))), count)
            destination = Path(folder) / "export"
            server.final_model_dir = lambda: str(destination)
            with mock.patch.object(fixtures.FakeServer, "export_final_models", create=True,
                                   side_effect=lambda: destination.mkdir()):
                server.export_final_models()
            self.assertTrue((destination / "apa_weights.csv").is_file())
            self.assertFalse((destination / "apa_basis").exists())

    def test_stale_basis_rejected_without_committing_optimizer_state(self):
        with tempfile.TemporaryDirectory() as folder, redirect_stdout(io.StringIO()):
            server = self.fixture.make_server(folder, "apa")
            server.aggregate_parameters_avg()
            before_w, before_v = server.apa_weights.clone(), server.apa_velocity.clone()
            server.cur_ground = 1
            path = server._apa_basis_path(0, 0)
            saved = torch.load(path, weights_only=True)
            saved["loop_round"] = -1
            torch.save(saved, path)
            with self.assertRaisesRegex(RuntimeError, "stale"):
                server.aggregate_parameters_avg()
            torch.testing.assert_close(server.apa_weights, before_w, rtol=0, atol=0)
            torch.testing.assert_close(server.apa_velocity, before_v, rtol=0, atol=0)
            self.assertEqual(server.apa_basis_round, 0)


class APALauncherTests(unittest.TestCase):
    def test_dedicated_script_only_delegates_public_configuration(self):
        path = Path(__file__).resolve().parents[1] / "system" / "run_apa.py"
        spec = importlib.util.spec_from_file_location("apa_launcher_test", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        with mock.patch.object(sys, "argv", [str(path), "--device-id", "4", "--dry-run"]), \
                mock.patch.object(module.subprocess, "run") as run:
            module.main()
        command = run.call_args.args[0]
        self.assertEqual(Path(command[1]).name, "run_target_proj.py")
        self.assertEqual(command[2:], ["--modes", "apa", "--rounds", "100", "--device-id", "4",
                                      "--apa_server_lr", "0.01", "--apa_momentum", "0.9", "--apa_self_weight", "0.5", "--dry-run"])


if __name__ == "__main__":
    unittest.main()

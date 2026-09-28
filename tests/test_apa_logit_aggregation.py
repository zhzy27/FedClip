"""Centered logit mathematics, causal basis, six-round synthetic server smoke."""

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
from utils.apa_logit_aggregation import (
    aggregate_apa_logit, apa_logit_proxy_gradient, centered_logit_gradient, helper_probabilities,
)
from utils.target_projection import aggregate_target_updates
import test_target_projection as fixtures


def params(x, y=0., dtype=torch.float64):
    return {"weight": torch.tensor([x, y], dtype=dtype)}


class LogitMathTests(unittest.TestCase):
    def test_manual_gradient_matches_autograd_and_centered_basis_formula(self):
        basis = [params(3, -2), params(2, 4), params(-1, 5), params(4, -3)]
        z = torch.tensor([-.5, .3, .8], dtype=torch.float64, requires_grad=True)
        q = torch.softmax(z, dim=0)
        wh = sum(q[j] * basis[j + 1]["weight"] for j in range(3))
        mixture = {"weight": .05 * basis[0]["weight"] + .95 * wh}
        post = params(-2, 1)
        residual = mixture["weight"] - post["weight"]
        loss = .5 * residual.square().sum()
        loss.backward()
        raw, centered, gradient, _, proxy, _ = apa_logit_proxy_gradient(mixture, post, enumerate(basis), [0, 1, 2, 3], 0, q)
        direct = torch.stack([.95 * q[j] * torch.dot(basis[j + 1]["weight"] - wh, residual) for j in range(3)])
        torch.testing.assert_close(gradient, z.grad, rtol=1e-13, atol=1e-13)
        torch.testing.assert_close(gradient, direct, rtol=1e-13, atol=1e-13)
        torch.testing.assert_close(centered, raw - (q.detach() * raw).sum(), rtol=0, atol=0)
        self.assertAlmostEqual(proxy, loss.item())
        self.assertFalse(gradient.requires_grad)
        self.assertEqual(gradient.dtype, torch.float64)

    def test_common_translation_with_corresponding_target_preserves_relative_gradient(self):
        basis = [params(0), params(-2, 1), params(3, 2)]
        q = torch.tensor([.5, .5], dtype=torch.float64)
        server, post = params(.475, 1.425), params(-.525, 2.425)
        original = apa_logit_proxy_gradient(server, post, enumerate(basis), [0, 1, 2], 0, q)
        offset = torch.tensor([1e8, -1e8], dtype=torch.float64)
        shifted = [basis[0]] + [{"weight": b["weight"] + offset} for b in basis[1:]]
        # Shift both server mixture and endpoint by .95*C, holding residual fixed.
        shifted_server = {"weight": server["weight"] + .95 * offset}
        shifted_post = {"weight": post["weight"] + .95 * offset}
        actual = apa_logit_proxy_gradient(shifted_server, shifted_post, enumerate(shifted), [0, 1, 2], 0, q)
        self.assertGreater(actual[0].abs().max().item(), 1e8)
        torch.testing.assert_close(actual[1], original[1], rtol=0, atol=1e-7)
        torch.testing.assert_close(actual[2], original[2], rtol=0, atol=1e-7)

    def test_uniform_initialization_round_zero_bitwise_avg_both_dtypes(self):
        q = helper_probabilities(torch.zeros(19))
        self.assertEqual(q.dtype, torch.float64)
        torch.testing.assert_close(q, torch.full((19,), 1 / 19, dtype=torch.float64), rtol=0, atol=0)
        for dtype in (torch.float32, torch.float64):
            posts = [params(cid / 3, -cid / 7, dtype) for cid in range(20)]
            uploads = [(cid, .05, posts[cid]) for cid in reversed(range(20))]
            initial = params(-2, 1, dtype)
            result, metrics, rows, logits = aggregate_apa_logit(initial, posts[0], iter(uploads), 0,
                {cid: .05 for cid in range(20)}, 0)
            avg = aggregate_target_updates(initial, posts[0], iter(uploads), 0, "avg")[0]
            torch.testing.assert_close(result["weight"], avg["weight"], rtol=0, atol=0)
            self.assertEqual([row["apa_weight"] for row in rows], [.05] * 20)
            self.assertEqual(logits.tolist(), [0.] * 19)
            self.assertAlmostEqual(metrics["effective_helper_count"], 19.)
            self.assertEqual(metrics["apa_logit_update_enabled"], 0)
            self.assertTrue(all(row["apa_logit_grad"] is None for row in rows))

    def test_beneficial_and_harmful_helpers_change_in_correct_direction_and_lower_proxy(self):
        basis = [params(0), params(1), params(-1)]
        uploads = [(cid, 1 / 3, value) for cid, value in enumerate(basis)]
        _, _, rows, updated = aggregate_apa_logit(params(0), params(1), uploads, 0,
            {cid: 1 / 3 for cid in range(3)}, 1, logits=[0., 0.], bases=enumerate(basis))
        self.assertGreater(updated[0], 0.)
        self.assertLess(updated[1], 0.)
        self.assertGreater(rows[1]["apa_q"], .5)
        self.assertLess(rows[2]["apa_q"], .5)
        self.assertGreater(rows[1]["apa_weight"], .475)
        self.assertLess(rows[2]["apa_weight"], .475)
        position = rows[1]["apa_weight"] - rows[2]["apa_weight"]
        self.assertLess(.5 * (position - 1.) ** 2, .5)

    def test_hundreds_and_thousands_common_bias_does_not_collapse_simplex(self):
        for offset in (600., 6000.):
            basis = [params(offset)] + [params(offset + (cid - 10) / 3) for cid in range(1, 20)]
            z = torch.zeros(19, dtype=torch.float64)
            for round_number in range(1, 7):
                _, metrics, rows, next_z = aggregate_apa_logit(params(1), params(0),
                    [(cid, .05, p) for cid, p in enumerate(basis)], 0,
                    {cid: .05 for cid in range(20)}, round_number, logits=z, bases=enumerate(basis))
                self.assertEqual(metrics["target_weight"], .05)
                self.assertAlmostEqual(metrics["helper_total_weight"], .95, places=14)
                self.assertGreater(metrics["min_helper_weight"], 0.)
                self.assertLess(metrics["max_helper_weight"], .95)
                self.assertGreater(metrics["apa_raw_gradient_norm"], 100 * metrics["apa_centered_gradient_norm"])
                gradient = torch.tensor([row["apa_logit_grad"] for row in rows[1:]], dtype=torch.float64)
                torch.testing.assert_close(next_z, z - .01 * gradient, rtol=0, atol=0)
                z = next_z

    def test_previous_basis_read_before_current_uploads(self):
        events = []
        def bases():
            for cid, value in enumerate([0, 2, -2]):
                events.append("basis")
                yield cid, params(value)
        def uploads():
            for cid, value in enumerate([1, -100, 100]):
                self.assertEqual(events[:3], ["basis"] * 3)
                events.append("upload")
                yield cid, 1 / 3, params(value)
        result, _, rows, _ = aggregate_apa_logit(params(0), params(1), uploads(), 0,
            {cid: 1 / 3 for cid in range(3)}, 1, logits=[0., 0.], bases=bases())
        self.assertEqual([row["apa_raw_grad"] for row in rows[1:]], [-2., 2.])
        self.assertGreater(rows[1]["apa_weight"], .475)
        self.assertAlmostEqual(result["weight"][0].item(), .05 - 100 * rows[1]["apa_weight"] + 100 * rows[2]["apa_weight"])

    def test_zero_residual_leaves_logits_exactly_unchanged(self):
        logits = torch.tensor([.2, -.3], dtype=torch.float64)
        _, metrics, rows, updated = aggregate_apa_logit(params(1), params(1),
            [(cid, 1 / 3, params(cid)) for cid in range(3)], 0, {cid: 1 / 3 for cid in range(3)},
            1, logits=logits, bases=[(cid, params(cid)) for cid in range(3)])
        torch.testing.assert_close(logits, updated, rtol=0, atol=0)
        self.assertEqual(metrics["apa_logit_update_norm"], 0.)
        for row in rows[1:]:
            self.assertEqual([row[k] for k in ("apa_raw_grad", "apa_centered_grad", "apa_logit_grad")], [0.] * 3)

    def test_softmax_stable_offset_and_underflow_fails_without_pruning(self):
        torch.testing.assert_close(helper_probabilities([10000., 10001.]), helper_probabilities([0., 1.]), rtol=0, atol=0)
        for logits in ([float("nan"), 0.], [float("inf"), 0.], [-10000., 10000.], []):
            with self.assertRaises(ValueError):
                helper_probabilities(logits)

    def test_invalid_state_basis_upload_and_lr_rejected(self):
        samples = {0: .5, 1: .5}
        uploads = [(0, .5, params(0)), (1, .5, params(1))]
        for kwargs in ({"lr": -1}, {"lr": float("nan")}, {"logits": [0.]}, {"bases": []}):
            with self.assertRaises(ValueError):
                aggregate_apa_logit(params(0), params(1), uploads, 0, samples, 0, **kwargs)
        for bases in ([], [(0, params(0)), (0, params(1))], [(0, params(0)), (1, params(float("nan")))],
                      [(0, params(0)), (1, {"bad": torch.zeros(1)})]):
            with self.assertRaises(ValueError):
                aggregate_apa_logit(params(0), params(1), uploads, 0, samples, 1, logits=[0.], bases=bases)
        for bad in (uploads[:1], uploads + uploads[:1], [(0, .5, params(0)), (1, .5, params(float("inf")))]):
            with self.assertRaises(ValueError):
                aggregate_apa_logit(params(0), params(1), bad, 0, samples, 0)

    def test_input_and_rng_are_unchanged(self):
        initial, post = params(0), params(1)
        uploads = [(cid, 1 / 3, params(cid)) for cid in range(3)]
        basis = [(cid, params(-cid)) for cid in range(3)]
        z = torch.tensor([.1, -.1], dtype=torch.float64)
        originals = copy.deepcopy((initial, post, uploads, basis, z))
        rng = random.getstate(), np.random.get_state(), torch.get_rng_state()
        aggregate_apa_logit(initial, post, uploads, 0, {cid: 1 / 3 for cid in range(3)}, 1, logits=z, bases=basis)
        for actual, expected in zip((initial, post), originals[:2]):
            torch.testing.assert_close(actual["weight"], expected["weight"], rtol=0, atol=0)
        for items, old_items in ((uploads, originals[2]), (basis, originals[3])):
            for item, old in zip(items, old_items):
                torch.testing.assert_close(item[-1]["weight"], old[-1]["weight"], rtol=0, atol=0)
        torch.testing.assert_close(z, originals[-1], rtol=0, atol=0)
        self.assertEqual(random.getstate(), rng[0])
        np.testing.assert_array_equal(np.random.get_state()[1], rng[1][1])
        torch.testing.assert_close(torch.get_rng_state(), rng[2], rtol=0, atol=0)


class LogitServerTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        self.fixture = fixtures.ProjectionIntegrationTests()
        self.fixture.setUp()

    def test_twenty_clients_six_rounds_local_training_basis_and_independent_logs(self):
        with tempfile.TemporaryDirectory() as folder, redirect_stdout(io.StringIO()):
            server = self.fixture.make_server(folder, "apa_logit")
            server.num_clients, server.target_client_id, server.global_rounds = 20, 0, 5
            server.Budget, server.rs_test_acc, server.auto_break = [], [], False
            server.clients = [self.fixture.make_client(folder, cid) for cid in range(20)]
            for client in server.clients:
                self.fixture.save_item(fixtures.TinyModel(low_rank=True), client.role, "model", folder)
                batches = [(torch.tensor([[1., -.5], [-.2, .3]]), torch.tensor([client.id % 2, 1]))]
                client.load_train_data = lambda batches=batches: batches
            old_file = Path(folder) / "apa_history.json"
            old_file.write_text("old APA baseline")
            old_aggregate = server._aggregate_apa_logit
            previous = []
            def inspect(initial, target, uploads):
                current = list(uploads)
                if server.cur_ground:
                    for cid, b in server._load_apa_logit_bases():
                        for key in b:
                            torch.testing.assert_close(b[key], previous[-1][cid][key], rtol=0, atol=0)
                previous.append({cid: {k: v.clone() for k, v in p.items()} for cid, _, p in current})
                result = old_aggregate(initial, target, iter(current))
                if server.cur_ground == 0:
                    avg = aggregate_target_updates(initial, target, iter(current), 0, "avg")[0]
                    for name in avg:
                        torch.testing.assert_close(result[0][name], avg[name], rtol=0, atol=0)
                return result
            server._aggregate_apa_logit = inspect
            server._aggregate_apa = mock.Mock(side_effect=AssertionError("Old APA path must not be used"))
            server.train()
            self.assertEqual(old_file.read_text(), "old APA baseline")
            self.assertEqual(server.apa_logits.shape, (19,))
            self.assertEqual(server.apa_logits.dtype, torch.float64)
            self.assertFalse(hasattr(server, "apa_velocity"))
            self.assertEqual(server.apa_logit_basis_round, 5)
            self.assertEqual(len(list((Path(folder) / "apa_logit_basis").glob("*/*.pt"))), 40)
            history = json.loads((Path(folder) / "apa_logit_history.json").read_text())["history"]
            self.assertEqual([row["apa_logit_update_enabled"] for row in history], [0, 1, 1, 1, 1, 1])
            self.assertAlmostEqual(history[0]["effective_helper_count"], 19.)
            for row in history:
                self.assertEqual(row["target_weight"], .05)
                self.assertAlmostEqual(row["helper_total_weight"], .95, places=14)
                self.assertGreater(row["min_helper_weight"], 0.)
                self.assertIsNotNone(row["target_post_local_acc"])
                self.assertIsNotNone(row["target_client_test_acc"])
            for client in server.clients:
                self.assertEqual(client.train_time_cost["num_rounds"], 6)
            with (Path(folder) / "apa_logit_weights.csv").open() as stream:
                self.assertEqual(len(list(csv.DictReader(stream))), 120)
            destination = Path(folder) / "export"
            server.final_model_dir = lambda: str(destination)
            with mock.patch.object(fixtures.FakeServer, "export_final_models", create=True,
                                   side_effect=lambda: destination.mkdir()):
                server.export_final_models()
            self.assertTrue((destination / "apa_logit_history.json").is_file())
            self.assertFalse((destination / "apa_history.json").exists())

    def test_stale_basis_does_not_commit_logit_state(self):
        with tempfile.TemporaryDirectory() as folder, redirect_stdout(io.StringIO()):
            server = self.fixture.make_server(folder, "apa_logit")
            server.aggregate_parameters_avg()
            before = server.apa_logits.clone()
            server.cur_ground = 1
            path = server._apa_logit_basis_path(0, 0)
            saved = torch.load(path, weights_only=True)
            saved["loop_round"] = -1
            torch.save(saved, path)
            with self.assertRaisesRegex(RuntimeError, "stale"):
                server.aggregate_parameters_avg()
            torch.testing.assert_close(server.apa_logits, before, rtol=0, atol=0)
            self.assertEqual(server.apa_logit_basis_round, 0)


class LogitLauncherTests(unittest.TestCase):
    def test_dedicated_script_delegates_smoke_default_and_explicit_full_run(self):
        path = Path(__file__).resolve().parents[1] / "system" / "run_apa_logit.py"
        spec = importlib.util.spec_from_file_location("apa_logit_launcher", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        for extra, rounds in (([], "5"), (["--rounds", "100"], "100")):
            with mock.patch.object(sys, "argv", [str(path), "--device-id", "3", "--dry-run", *extra]), \
                    mock.patch.object(module.subprocess, "run") as run:
                module.main()
            command = run.call_args.args[0]
            self.assertEqual(Path(command[1]).name, "run_target_proj.py")
            self.assertEqual(command[2:], ["--modes", "apa_logit", "--rounds", rounds, "--device-id", "3",
                                          "--apa_logit_lr", "0.01", "--dry-run"])


if __name__ == "__main__":
    unittest.main()

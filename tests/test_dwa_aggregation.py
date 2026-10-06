"""C0 DWA distance mathematics, protected guidance epoch and smoke integration."""

from contextlib import redirect_stdout
import copy
import csv
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
from utils.dwa_aggregation import (
    aggregate_dwa, guidance_distance_weights, squared_parameter_distance,
)
from utils.target_projection import EPS, aggregate_target_updates
import test_target_projection as fixtures


def params(values, head=0., dtype=torch.float64):
    return {"base.weight": torch.tensor(values, dtype=dtype), "head.bias": torch.tensor([head], dtype=dtype)}


def guide(parameters, loop_round=0):
    return dict(parameters=parameters, client_id=0, loop_round=loop_round, source_post_local_round=loop_round)


def aggregate(initial, posts, guidance, mode, loop_round=0, uploads_factory=None, **kwargs):
    samples = {cid: 1 / len(posts) for cid in range(len(posts))}
    uploads_factory = uploads_factory or (lambda: ((cid, p, posts[cid]) for cid, p in samples.items()))
    return aggregate_dwa(initial, posts[0], uploads_factory, guidance, 0, samples,
                         loop_round, {cid: loop_round for cid in samples}, mode, **kwargs)


def rng_snapshot():
    return (random.getstate(), np.random.get_state(), torch.get_rng_state(),
            torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [])


def assert_rng_equal(test, before):
    after = rng_snapshot()
    test.assertEqual(before[0], after[0])
    np.testing.assert_array_equal(before[1][1], after[1][1])
    test.assertEqual(before[1][2:], after[1][2:])
    torch.testing.assert_close(before[2], after[2], rtol=0, atol=0)
    for left, right in zip(before[3], after[3]):
        torch.testing.assert_close(left, right, rtol=0, atol=0)


class DWAMathTests(unittest.TestCase):
    def test_equal_distances_uniform_twenty_weights_and_mass(self):
        for distance in (0., 2., 1e200):
            weights, q, _, summary = guidance_distance_weights({cid: distance for cid in range(1, 20)})
            self.assertEqual(weights[0], .05)
            for cid in range(1, 20):
                self.assertAlmostEqual(weights[cid], .05, places=15)
                self.assertAlmostEqual(q[cid], 1 / 19, places=15)
            self.assertAlmostEqual(sum(weights.values()), 1.)
            self.assertAlmostEqual(summary["helper_total_weight"], .95)
            self.assertAlmostEqual(summary["effective_helper_count"], 19.)
            self.assertAlmostEqual(summary["same_label_helper_weight"], .15)
            self.assertAlmostEqual(summary["cross_label_helper_weight"], .8)

    def test_inverse_squared_distance_not_squared_again_or_softmax(self):
        weights, q, _, _ = guidance_distance_weights({1: 1., 2: 4.}, epsilon=1e-12)
        self.assertAlmostEqual(q[1], (1 / (1 + 1e-12)) / (1 / (1 + 1e-12) + 1 / (4 + 1e-12)), places=15)
        self.assertAlmostEqual(q[1] / q[2], 4., places=10)
        self.assertAlmostEqual(weights[1] + weights[2], .95)
        self.assertEqual(squared_parameter_distance(params([0., 0.], head=1), params([3., 4.], head=3)), 29.)
        # Subtract in float64 before squaring/reducing, even for float32 models.
        self.assertEqual(squared_parameter_distance(params([1e20], dtype=torch.float32),
                                                    params([-1e20], dtype=torch.float32)),
                         (float(torch.tensor(1e20, dtype=torch.float32)) * 2) ** 2)

    def test_zero_distance_and_extreme_normalization_are_finite(self):
        for distances, epsilon in (({1: 0., 2: 0.}, 1e-12), ({1: 0., 2: 1.}, 1e-12),
                                   ({1: 0., 2: 1e308}, 1e-12), ({1: 1e308, 2: 1e308}, 1e308),
                                   ({1: 0., 2: 1e-320}, 1e-320)):
            weights, q, _, _ = guidance_distance_weights(distances, epsilon=epsilon)
            self.assertTrue(all(np.isfinite(v) and v > 0 for v in weights.values()))
            self.assertAlmostEqual(sum(q.values()), 1.)
            self.assertAlmostEqual(sum(weights.values()), 1.)

    def test_plain_aggregation_is_weighted_ordinary_models_and_never_projects(self):
        initial = params([10., -2.], head=5)
        posts = [params([11., -2.], head=6), params([8., 3.], head=-2), params([12., -1.], head=4)]
        guidance = guide(params([13., -2.], head=6))
        with mock.patch("utils.dwa_aggregation.aggregate_target_updates", side_effect=AssertionError("No Projection in plain DWA")):
            result, metrics, rows = aggregate(initial, posts, guidance, "dwa_soft")
        weights = {row["client_id"]: row["aggregation_weight"] for row in rows}
        for name in result:
            expected = sum(post[name] * weights[cid] for cid, post in enumerate(posts))
            torch.testing.assert_close(result[name], expected, rtol=0, atol=0)
        self.assertEqual(metrics["projection_enabled"], 0)
        self.assertEqual(metrics["guidance_target_post_squared_distance"], 4.)
        self.assertIsNone(rows[0]["dwa_q"])

    def test_both_modes_same_weights_project_only_conflicting_global_delta(self):
        initial = params([4., 7.], head=1)
        posts = [params([5., 7.], head=1), params([2., 10.], head=1), params([7., 9.], head=1)]
        guidance = guide(params([100., -30.], head=9))  # Not the projection direction.
        plain, _, rows_a = aggregate(initial, posts, guidance, "dwa_soft")
        projected, metrics, rows_b = aggregate(initial, posts, guidance, "dwa_soft_projection")
        for a, b in zip(rows_a, rows_b):
            for key in ("guidance_squared_distance", "dwa_q", "aggregation_weight"):
                self.assertEqual(a[key], b[key])
        weights = {row["client_id"]: row["aggregation_weight"] for row in rows_a}
        target = posts[0]["base.weight"] - initial["base.weight"]
        correction = weights[1] * (-2 / (1 + EPS))
        torch.testing.assert_close(projected["base.weight"], plain["base.weight"] - correction * target, rtol=1e-14, atol=1e-14)
        self.assertEqual([row["conflict"] for row in rows_b], [0, 1, 0])
        self.assertAlmostEqual(rows_b[1]["dot_after_projection"], -2 + 2 / (1 + EPS))
        self.assertEqual(metrics["projection_epsilon"], EPS)
        reference = aggregate_target_updates(initial, posts[0],
            [(cid, weights[cid], post) for cid, post in enumerate(posts)], 0, "projection")[0]
        for name in reference:
            torch.testing.assert_close(projected[name], reference[name], rtol=0, atol=0)

    def test_zero_target_delta_finite_and_no_removal(self):
        initial = params([1., 2.], head=1)
        posts = [copy.deepcopy(initial), params([-3., 9.], head=2)]
        plain, _, _ = aggregate(initial, posts, guide(initial), "dwa_soft")
        projected, metrics, rows = aggregate(initial, posts, guide(initial), "dwa_soft_projection")
        for name in plain:
            torch.testing.assert_close(plain[name], projected[name], rtol=1e-14, atol=1e-14)
        self.assertEqual(metrics["removed_update_ratio"], 0.)
        self.assertTrue(all(row["conflict"] == 0 for row in rows))

    def test_guidance_upload_rounds_parameter_names_and_shapes_checked(self):
        initial = params([0., 0.])
        posts = [params([1., 0.]), params([0., 1.])]
        for field in ("loop_round", "source_post_local_round", "client_id"):
            stale = guide(initial)
            stale[field] = 1
            with self.assertRaises(ValueError):
                aggregate(initial, posts, stale, "dwa_soft")
        with self.assertRaisesRegex(ValueError, "uploads.*round"):
            aggregate_dwa(initial, posts[0], lambda: iter([(0, .5, posts[0]), (1, .5, posts[1])]),
                guide(initial), 0, {0: .5, 1: .5}, 0, {0: 0, 1: -1}, "dwa_soft")
        for bad in ({"wrong": torch.zeros(1)}, params([1.]), params([float("nan"), 0.])):
            with self.assertRaises(ValueError):
                aggregate(initial, posts, guide(bad), "dwa_soft")
        with self.assertRaises(ValueError):
            aggregate(initial, posts, guide(initial), "dwa_soft", uploads_factory=lambda: iter([(0, .5, posts[0])]))
        for epsilon in (0., -1., float("inf"), float("nan")):
            with self.assertRaises(ValueError):
                guidance_distance_weights({1: 1.}, epsilon=epsilon)

    def test_inputs_and_rng_unchanged(self):
        initial, posts = params([0., 0.]), [params([1., 0.]), params([-1., 2.])]
        guidance = guide(params([2., 0.]))
        original = copy.deepcopy((initial, posts, guidance))
        before = rng_snapshot()
        for mode in ("dwa_soft", "dwa_soft_projection"):
            aggregate(initial, posts, guidance, mode)
            assert_rng_equal(self, before)
        for actual, old in [(initial, original[0]), *zip(posts, original[1]), (guidance["parameters"], original[2]["parameters"])]:
            for name in old:
                torch.testing.assert_close(actual[name], old[name], rtol=0, atol=0)


class StochasticTiny(fixtures.TinyModel):
    def forward(self, x):
        if self.training:
            self.running_stat.add_(1.)
            random.random()
            np.random.rand()
            if torch.cuda.is_available():
                torch.rand(2, device="cuda")
            x = torch.nn.functional.dropout(x, .25, True)
        return super().forward(x)


class DWAGuidanceTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        self.fixture = fixtures.ProjectionIntegrationTests()
        self.fixture.setUp()

    def test_guidance_one_complete_epoch_matches_normal_loss_sgd_and_clip(self):
        with tempfile.TemporaryDirectory() as folder:
            client = self.fixture.make_client(folder, 0)
            client.local_epochs = 5
            normal = fixtures.TinyModel(low_rank=True, offset=.2)
            self.fixture.save_item(normal, client.role, "model", folder)
            batches = client.load_train_data() * 3
            client.load_train_data = lambda: copy.deepcopy(batches)
            client.load_test_data = mock.Mock(side_effect=AssertionError("No test data for guidance"))
            expected = copy.deepcopy(normal).train()
            optimizer = torch.optim.SGD(expected.parameters(), lr=client.learning_rate)
            for x, y in batches:
                optimizer.zero_grad()
                ce, reg = client._objective(expected, x, y)
                (ce + reg).backward()
                torch.nn.utils.clip_grad_norm_(expected.parameters(), 10.)
                optimizer.step()
            expected.recover_larger_model()
            before = rng_snapshot()
            payload = client.build_dwa_guidance(4, 4)
            assert_rng_equal(self, before)
            for name, value in expected.named_parameters():
                torch.testing.assert_close(payload["parameters"][name], value, rtol=0, atol=0)
            metadata = payload["metadata"]
            self.assertEqual(metadata["guidance_epochs"], 1)
            self.assertEqual(metadata["guidance_train_batches"], 3)
            self.assertEqual(metadata["guidance_train_samples"], 6)
            self.assertEqual(metadata["guidance_lr"], client.learning_rate)
            self.assertEqual(metadata["guidance_extra_upload_bytes"], sum(p.numel() * p.element_size() for p in expected.parameters()))
            self.assertEqual(client.train_time_cost["num_rounds"], 0)
            self.assertFalse(hasattr(client, "last_train_time_cost"))
            restored = self.fixture.load_item(client.role, "model", folder)
            for name, value in restored.state_dict().items():
                torch.testing.assert_close(value, normal.state_dict()[name], rtol=0, atol=0)
            client.load_test_data.assert_not_called()

    def test_copy_buffers_rng_and_next_normal_train_unchanged(self):
        with tempfile.TemporaryDirectory() as folder, redirect_stdout(io.StringIO()):
            client = self.fixture.make_client(folder, 0)
            normal = StochasticTiny(low_rank=True)
            self.fixture.save_item(normal, client.role, "model", folder)
            # Even a shared in-memory loaded model must stay untouched.
            with mock.patch.object(client, "_load_model", return_value=normal):
                before = rng_snapshot()
                client.build_dwa_guidance(0, 0)
                assert_rng_equal(self, before)
                self.assertEqual(normal.running_stat.item(), 17.)
                self.assertTrue(all(p.grad is None for p in normal.parameters()))
            client.train(current_round=1)
            after_guidance = self.fixture.load_item(client.role, "model", folder)
            self.fixture.save_item(normal, client.role, "model", folder)
            random.setstate(before[0]); np.random.set_state(before[1]); torch.set_rng_state(before[2])
            if before[3]:
                torch.cuda.set_rng_state_all(before[3])
            client.train(current_round=1)
            reference = self.fixture.load_item(client.role, "model", folder)
            for name, value in reference.state_dict().items():
                torch.testing.assert_close(after_guidance.state_dict()[name], value, rtol=0, atol=0)

    def test_failure_and_stale_source_restore_rng_and_never_save_checkpoint(self):
        with tempfile.TemporaryDirectory() as folder:
            client = self.fixture.make_client(folder, 0)
            normal = StochasticTiny(low_rank=True)
            self.fixture.save_item(normal, client.role, "model", folder)
            for fail_stage in ("loader", "recovery"):
                before = rng_snapshot()
                if fail_stage == "loader":
                    def fail():
                        random.random(); np.random.rand(); torch.rand(5)
                        raise RuntimeError("synthetic guidance failure")
                    patcher = mock.patch.object(client, "load_train_data", side_effect=fail)
                else:
                    patcher = mock.patch.object(StochasticTiny, "recover_larger_model", side_effect=RuntimeError("synthetic guidance failure"))
                with patcher, mock.patch.object(self.fixture.client_module, "save_item") as save:
                    with self.assertRaisesRegex(RuntimeError, "synthetic guidance failure"):
                        client.build_dwa_guidance(0, 0)
                    save.assert_not_called()
                assert_rng_equal(self, before)
            with self.assertRaises(ValueError):
                client.build_dwa_guidance(1, 0)
            client.load_train_data = lambda: []
            with self.assertRaisesRegex(RuntimeError, "non-empty"):
                client.build_dwa_guidance(0, 0)


class DWAServerTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        self.fixture = fixtures.ProjectionIntegrationTests()
        self.fixture.setUp()

    def test_twenty_clients_three_rounds_five_epoch_normal_guidance_logs_and_export(self):
        for mode in ("dwa_soft", "dwa_soft_projection"):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as folder, redirect_stdout(io.StringIO()):
                server = self.fixture.make_server(folder, mode)
                server.target_client_id, server.num_clients, server.global_rounds = 0, 20, 2
                server.Budget, server.rs_test_acc, server.auto_break = [], [], False
                server.clients = [self.fixture.make_client(folder, cid) for cid in range(20)]
                for client in server.clients:
                    client.local_epochs = 5
                    client.learning_rate = .005
                    client.args.regular_lamda = .001
                    self.fixture.save_item(fixtures.TinyModel(low_rank=True), client.role, "model", folder)
                    client.build_dwa_guidance = mock.Mock(wraps=client.build_dwa_guidance)
                target = server.clients[0]
                ordinary_test = target.test_post_local
                events = []
                def check_accuracy():
                    events.append(("ordinary_test", server.cur_ground))
                    self.assertEqual(target.build_dwa_guidance.call_count, server.cur_ground)
                    return ordinary_test()
                target.test_post_local = check_accuracy
                sentinel = Path(folder) / "apa_history.json"
                sentinel.write_text("existing experiment")
                server.train()
                self.assertEqual(sentinel.read_text(), "existing experiment")
                self.assertEqual(events, [("ordinary_test", n) for n in range(3)])
                self.assertEqual(target.build_dwa_guidance.call_count, 3)
                for client in server.clients:
                    self.assertEqual(client.train_time_cost["num_rounds"], 3)
                    if client.id:
                        client.build_dwa_guidance.assert_not_called()
                history = json.loads((Path(folder) / f"{mode}_history.json").read_text())["history"]
                for row in history:
                    self.assertEqual(row["target_weight"], .05)
                    self.assertAlmostEqual(row["helper_total_weight"], .95)
                    self.assertEqual(row["guidance_source_post_local_round"], row["loop_round"])
                    self.assertEqual(row["guidance_epochs"], 1)
                    self.assertEqual(row["guidance_lr"], .005)
                    self.assertGreater(row["guidance_extra_upload_bytes"], 0)
                    self.assertIsNotNone(row["target_post_local_acc"])
                    self.assertIsNotNone(row["target_client_test_acc"])
                with (Path(folder) / f"{mode}_weights.csv").open() as stream:
                    self.assertEqual(len(list(csv.DictReader(stream))), 60)
                destination = Path(folder) / "export"
                server.final_model_dir = lambda: str(destination)
                with mock.patch.object(fixtures.FakeServer, "export_final_models", create=True,
                                       side_effect=lambda: destination.mkdir()):
                    server.export_final_models()
                self.assertTrue((destination / f"{mode}_history.json").is_file())
                self.assertFalse((destination / "dwa_guidance.pt").exists())
                with self.assertRaisesRegex(RuntimeError, "fresh guidance"):
                    server.aggregate_parameters_avg()

    def test_last_ten_summary_uses_ordinary_local_accuracy_and_earliest_best(self):
        server = self.fixture.make_server("memory", "dwa_soft")
        accuracies = [.9, .9] + [n / 100 for n in range(10)]
        server.target_proj_history = [dict(round=n + 1, target_post_local_acc=acc, target_client_test_acc=1.)
                                      for n, acc in enumerate(accuracies)]
        summary = server._local_accuracy_summary()
        self.assertEqual(summary["final_target_local_acc"], .09)
        self.assertEqual(summary["best_target_local_acc"], .9)
        self.assertEqual(summary["best_target_local_round"], 1)
        self.assertAlmostEqual(summary["last10_target_local_acc"], .045)
        self.assertEqual(summary["last10_target_local_count"], 10)


if __name__ == "__main__":
    unittest.main()

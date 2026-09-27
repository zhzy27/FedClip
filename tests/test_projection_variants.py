"""Mathematical and server integration checks for the five isolated ablations."""

from contextlib import redirect_stdout
import copy
import csv
import io
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "system"))
from utils.layer_mask import logical_layer_groups
from utils.projection_variants import (
    PROJECTION_VARIANT_MODES, LOCAL_PROJECTION_MODES, LAYER_PROJECTION_MODES, SOURCE_MODES,
    aggregate_projection_variant, aggregate_source_projection, source_helper_ids, source_weights,
    validate_source_config, print_projection_variant,
)
from utils.target_projection import EPS, aggregate_target_updates
import test_target_projection as fixtures


def params(x, y, dtype=torch.float64):
    return {"conv.weight": torch.tensor(x, dtype=dtype), "fc.weight": torch.tensor(y, dtype=dtype)}


def assert_params(test, actual, expected, exact=False):
    for key in actual:
        torch.testing.assert_close(actual[key], expected[key], **({"rtol": 0, "atol": 0} if exact else {}))


class ProjectionVariantMathTests(unittest.TestCase):
    def setUp(self):
        self.global_params = params([0., 0.], [0.])
        self.pre = params([100., 200.], [300.])
        self.target = params([101., 200.], [302.])
        self.groups = logical_layer_groups(self.global_params)

    def aggregate(self, mode, helper, helper_pre=None, groups=None):
        return aggregate_projection_variant(self.global_params, self.target,
            [(0, 0.25, self.target, self.pre), (1, 0.75, helper, helper_pre or self.pre)],
            0, mode, target_pre=self.pre, groups=groups or self.groups)

    def test_local_no_conflict_is_bitwise_avg_and_retains_pre_structure(self):
        helper = params([102., 203.], [301.])
        avg = aggregate_target_updates(self.global_params, self.target,
                [(0, .25, self.target), (1, .75, helper)], 0, "avg")[0]
        for mode in LOCAL_PROJECTION_MODES:
            result, metrics, _, _, _ = self.aggregate(mode, helper)
            assert_params(self, result, avg, exact=True)
            self.assertEqual(metrics["conflict_client_count"], 0)
            self.assertGreater(result["conv.weight"][0].item(), 100)

    def test_local_single_conflict_matches_analytic_correction(self):
        helper = params([97., 204.], [299.])
        result, _, rows, _, _ = self.aggregate("projection_local", helper)
        target_delta = {key: self.target[key] - self.pre[key] for key in self.pre}
        coefficient = -5 / (5 + EPS)
        expected = {key: .25 * self.target[key] + .75 * helper[key] - .75 * coefficient * target_delta[key]
                    for key in helper}
        assert_params(self, result, expected)
        self.assertAlmostEqual(rows[1]["full_local_dot"], -5)
        self.assertAlmostEqual(rows[1]["projection_coefficient"], coefficient)

    def test_local_uses_each_clients_own_pre_not_global_or_target_pre(self):
        helper_pre, helper = params([1000., 500.], [600.]), params([997., 504.], [599.])
        _, _, rows, _, _ = self.aggregate("projection_local", helper, helper_pre)
        self.assertEqual(rows[1]["full_local_dot"], -5)

    def test_layer_global_corrects_only_negative_parallel_component(self):
        target, helper = params([1., 0.], [2.]), params([-2., 3.], [4.])
        result, _, _, layers, _ = aggregate_projection_variant(self.global_params, target,
            [(0, .25, target, None), (1, .75, helper, None)], 0, "layer_projection_global", groups=self.groups)
        avg = aggregate_target_updates(self.global_params, target,
            [(0, .25, target), (1, .75, helper)], 0, "avg")[0]
        torch.testing.assert_close(result["fc.weight"], avg["fc.weight"], rtol=0, atol=0)
        self.assertEqual(result["conv.weight"][1].item(), 2.25)  # Orthogonal component survives.
        self.assertAlmostEqual(result["conv.weight"][0].item(), .25)
        entry = next(row for row in layers if row["client_id"] == 1 and row["layer"] == "conv")
        self.assertLess(abs(entry["dot_after_projection"]), 3e-12)

    def test_layer_local_negative_layer_retains_orthogonal_and_positive_layers(self):
        helper = params([98., 203.], [304.])
        result, _, _, layers, _ = self.aggregate("layer_projection_local", helper)
        avg = {key: .25 * self.target[key] + .75 * helper[key] for key in helper}
        torch.testing.assert_close(result["fc.weight"], avg["fc.weight"], rtol=0, atol=0)
        self.assertEqual(result["conv.weight"][1].item(), 202.25)
        self.assertAlmostEqual(result["conv.weight"][0].item(), 100.25)
        self.assertLess(abs(layers[2]["dot_after_projection"]), 3e-12)

    def test_zero_target_layers_are_finite_for_global_and_local(self):
        for mode in LAYER_PROJECTION_MODES:
            target = self.pre if mode in LOCAL_PROJECTION_MODES else self.global_params
            result, metrics, _, _, matrices = aggregate_projection_variant(self.global_params, target,
                [(0, .25, target, self.pre), (1, .75, self.target, self.pre)], 0, mode,
                target_pre=self.pre, groups=self.groups)
            self.assertTrue(all(torch.isfinite(value).all() for value in result.values()))
            self.assertEqual(metrics["total_conflict_entries"], 0)
            json.dumps(matrices, allow_nan=False)

    def test_one_global_group_matches_legacy_projection(self):
        target, helper = params([1., 2.], [3.]), params([-3., -2.], [1.])
        old = aggregate_target_updates(self.global_params, target,
            [(0, .25, target), (1, .75, helper)], 0, "projection")[0]
        new = aggregate_projection_variant(self.global_params, target,
            [(0, .25, target, None), (1, .75, helper, None)], 0, "layer_projection_global",
            groups={"all": list(target)})[0]
        assert_params(self, new, old)

    def test_one_local_group_matches_full_local_projection(self):
        helper = params([98., 203.], [299.])
        full = self.aggregate("projection_local", helper)[0]
        grouped = self.aggregate("layer_projection_local", helper, groups={"all": list(helper)})[0]
        assert_params(self, full, grouped, exact=True)

    def test_positive_global_updates_are_bitwise_avg_and_inputs_unchanged(self):
        target, helper = params([1., 2.], [3.]), params([2., 3.], [4.])
        uploads = [(0, .25, target, None), (1, .75, helper, None)]
        originals = copy.deepcopy(uploads)
        result = aggregate_projection_variant(self.global_params, target, uploads, 0,
            "layer_projection_global", groups=self.groups)[0]
        avg = aggregate_target_updates(self.global_params, target,
            [(cid, p, post) for cid, p, post, _ in uploads], 0, "avg")[0]
        assert_params(self, result, avg, exact=True)
        for before, after in zip(originals, uploads):
            assert_params(self, before[2], after[2], exact=True)

    def test_layer_weight_and_bias_share_coefficient(self):
        initial = {"conv.weight": torch.zeros(1), "conv.bias": torch.zeros(1)}
        target = {"conv.weight": torch.tensor([1.]), "conv.bias": torch.tensor([2.])}
        helper = {"conv.weight": torch.tensor([-3.]), "conv.bias": torch.tensor([1.])}
        result, _, _, rows, _ = aggregate_projection_variant(initial, target,
            [(0, .5, target, None), (1, .5, helper, None)], 0, "layer_projection_global",
            groups=logical_layer_groups(initial))
        self.assertEqual(len(rows), 2)
        coefficient = -1 / (5 + EPS)
        assert_params(self, result, {key: .5 * target[key] + .5 * helper[key] - .5 * coefficient * target[key]
                                    for key in initial})

    def test_invalid_snapshots_groups_weights_and_missing_target_fail(self):
        uploads = [(0, .25, self.target, self.pre), (1, .75, self.target, self.pre)]
        for bad_uploads, groups in ((uploads, {"conv": ["conv.weight"]}),
                (uploads + uploads, self.groups), (uploads[1:], self.groups),
                ([(0, -1., self.target, self.pre)], self.groups),
                ([(0, 1., self.target, None)], self.groups)):
            with self.assertRaises(ValueError):
                aggregate_projection_variant(self.global_params, self.target, bad_uploads, 0,
                    "layer_projection_local", target_pre=self.pre, groups=groups)


class SourceProjectionTests(unittest.TestCase):
    def setUp(self):
        self.samples = {cid: (cid + 1) / 210 for cid in range(20)}
        self.initial, self.target = params([3., 2.], [1.]), params([4., 4.], [4.])
        self.uploads = [(cid, p, self.target if cid == 0 else params([cid - 3., cid + 2.], [-cid * .5]))
                        for cid, p in self.samples.items()]

    def test_source_ids_and_fixed_target_and_helper_mass(self):
        for mode, ids in zip(SOURCE_MODES, (set(range(1, 4)), set(range(4, 20)))):
            self.assertEqual(source_helper_ids(mode), ids)
            q = source_weights(self.samples, 0, ids)
            self.assertEqual(q[0], self.samples[0])
            self.assertAlmostEqual(sum(q[cid] for cid in ids), 1 - self.samples[0], places=15)
            self.assertTrue(all(q[cid] == 0 for cid in self.samples.keys() - ids - {0}))
            a, b = sorted(ids)[:2]
            self.assertAlmostEqual(q[a] / q[b], self.samples[a] / self.samples[b])

    def test_excluded_huge_parameters_cannot_change_aggregation(self):
        for mode in SOURCE_MODES:
            original = aggregate_source_projection(self.initial, self.target, self.uploads, 0, self.samples, mode)
            selected = source_helper_ids(mode)
            changed = [(cid, p, post if cid in selected or cid == 0 else params([1e100, -1e100], [1e100]))
                       for cid, p, post in self.uploads]
            result = aggregate_source_projection(self.initial, self.target, changed, 0, self.samples, mode)
            assert_params(self, result[0], original[0], exact=True)
            for row in result[2]:
                self.assertEqual(row["selected_as_helper"], int(row["client_id"] in selected))
                if row["client_id"] not in selected | {0}:
                    self.assertEqual(row["removed_component_norm"], 0)

    def test_all_helpers_reproduce_legacy_projection_bitwise(self):
        expected = aggregate_target_updates(self.initial, self.target, self.uploads, 0, "projection")
        actual = aggregate_source_projection(self.initial, self.target, self.uploads, 0, self.samples,
            SOURCE_MODES[0], selected_helpers=set(range(1, 20)))
        assert_params(self, actual[0], expected[0], exact=True)
        for key, value in expected[1].items():
            self.assertEqual(actual[1][key], value)

    def test_source_config_guards_every_fixed_partition_field(self):
        valid = dict(target_proj_mode=SOURCE_MODES[0], dataset="Cifar100", partition="pat",
                     class_per_client=20, target_client_id=0, num_clients=20)
        validate_source_config(SimpleNamespace(**valid))
        for mode in SOURCE_MODES:
            for field, value in dict(dataset="Cifar10", partition="dir", class_per_client=10,
                                     target_client_id=1, num_clients=10).items():
                with self.subTest(mode=mode, field=field), self.assertRaises(ValueError):
                    validate_source_config(SimpleNamespace(**(valid | {"target_proj_mode": mode, field: value})))

    def test_zero_selected_mass_rejected(self):
        with self.assertRaises(ValueError):
            source_weights({0: .5, 1: 0., 2: .5}, 0, {1})


class LegacyCommitRegressionTests(unittest.TestCase):
    def test_previous_commit_projection_parameters_and_diagnostics_are_bitwise_identical(self):
        root = Path(__file__).resolve().parents[1]
        code = subprocess.check_output(["git", "show", "a2ab191:system/utils/target_projection.py"], cwd=root, text=True)
        previous = {}
        exec(compile(code, "a2ab191/target_projection.py", "exec"), previous)
        for dtype in (torch.float32, torch.float64):
            for seed in range(5):
                generator = torch.Generator().manual_seed(seed)
                initial = {key: torch.randn(8, generator=generator, dtype=dtype) for key in ("conv.weight", "fc.weight")}
                uploads = [(cid, (cid + 1) / 210,
                    {key: torch.randn(8, generator=generator, dtype=dtype) for key in initial}) for cid in range(20)]
                actual = aggregate_target_updates(initial, uploads[0][2], uploads, 0, "projection")
                old = previous["aggregate_target_updates"](initial, uploads[0][2], uploads, 0, "projection")
                assert_params(self, actual[0], old[0], exact=True)
                self.assertEqual(actual[1:], old[1:])


class ProjectionVariantServerTests(unittest.TestCase):
    def test_all_five_modes_train_save_and_only_local_modes_take_snapshots(self):
        torch.set_num_threads(2)
        for mode in PROJECTION_VARIANT_MODES:
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
                fixture = fixtures.ProjectionIntegrationTests()
                fixture.setUp()
                server = fixture.make_server(directory, mode)
                server.target_client_id = 0
                count = 20 if mode in SOURCE_MODES else 3
                server.num_clients = count
                server.clients = [fixture.make_client(directory, cid) for cid in range(count)]
                for client in server.clients:
                    fixture.save_item(fixtures.TinyModel(low_rank=True, offset=client.id * .02), client.role, "model", directory)
                    client.train = mock.Mock(side_effect=client.train)
                server.layer_groups = logical_layer_groups(dict(fixtures.TinyModel().named_parameters()))
                server.Budget, server.rs_test_acc, server.auto_break = [], [], False
                server._capture_pre_local_parameters = mock.Mock(side_effect=server._capture_pre_local_parameters)
                sentinel = Path(directory) / "layer_mask_metrics.csv"
                sentinel.write_text("prior result")
                server.train()
                self.assertEqual(server._capture_pre_local_parameters.call_count, 2 if mode in LOCAL_PROJECTION_MODES else 0)
                for client in server.clients:
                    self.assertEqual(client.train.call_count, 2)
                self.assertEqual(sentinel.read_text(), "prior result")
                with open(Path(directory) / f"{mode}_matrices.json") as stream:
                    saved = json.load(stream)
                self.assertEqual(len(saved["history"]), 2)
                self.assertTrue(all(row["target_post_local_acc"] is not None for row in saved["history"]))
                for suffix, count_rows in (("metrics", 2), ("clients", count * 2)):
                    with open(Path(directory) / f"{mode}_{suffix}.csv") as stream:
                        self.assertEqual(len(list(csv.DictReader(stream))), count_rows)
                destination = Path(directory) / "export"
                server.final_model_dir = lambda: str(destination)
                with mock.patch.object(fixtures.FakeServer, "export_final_models", create=True,
                                       side_effect=lambda: destination.mkdir()):
                    server.export_final_models()
                self.assertTrue((destination / f"{mode}_matrices.json").is_file())
                self.assertFalse((destination / "layer_mask_metrics.csv").exists())

    def test_full_twenty_by_five_terminal_projection_matrices(self):
        initial = {f"{layer}.weight": torch.zeros(2) for layer in ("conv1", "conv2", "fc1", "fc2", "fc3")}
        target = {name: torch.ones(2) for name in initial}
        _, _, clients, _, matrices = aggregate_projection_variant(initial, target,
            [(cid, .05, target, None) for cid in range(20)], 0, "layer_projection_global",
            groups=logical_layer_groups(initial))
        output = io.StringIO()
        with redirect_stdout(output):
            print_projection_variant(101, "layer_projection_global", matrices, clients)
        for key in ("cosine", "conflict"):
            block = output.getvalue().split(f"{key} matrix")[1].split("[layer_projection_global]")[0]
            for cid in range(20):
                self.assertIn(f"Client {cid} ", block)
            self.assertIn("fc3", block)


if __name__ == "__main__":
    unittest.main()

"""Joint helper budgets, unchanged masks/anchors, and isolated persisted diagnostics."""

from contextlib import redirect_stdout
import copy
import csv
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "system"))
from utils.layer_mask import aggregate_layer_mask, logical_layer_groups
from utils.layer_mask_budget import aggregate_layer_mask_budget, clip_helper_budget
import test_layer_mask as fixtures


def vector(values, dtype=torch.float64):
    return {"layer.weight": torch.tensor(values, dtype=dtype)}


class LayerMaskBudgetMathTests(unittest.TestCase):
    def test_under_budget_is_bitwise_equal_to_original_layer_mask(self):
        pre, target = vector([100000.0, 100000.0], torch.float32), vector([100010.0, 100001.0], torch.float32)
        initial = vector([0.0, 0.0], torch.float32)
        uploads = [(0, 0.2, target, pre), (1, 0.3, vector([0.01, 0.02], torch.float32), initial),
                   (2, 0.5, vector([0.02, 0.01], torch.float32), initial)]
        groups = logical_layer_groups(initial)
        old = aggregate_layer_mask(initial, target, pre, iter(uploads), 0, groups)
        new = aggregate_layer_mask_budget(initial, target, pre, iter(uploads), 0, groups)
        torch.testing.assert_close(new[0]["layer.weight"], old[0]["layer.weight"], rtol=0, atol=0)
        self.assertEqual(new[4]["budget_layers"][0]["budget_scale"], 1.0)
        self.assertGreater(new[4]["budget_layers"][0]["raw_helper_norm"], 0.0)
        for key, value in old[4].items():
            self.assertEqual(new[4][key], value)
        for key, value in old[1].items():
            self.assertEqual(new[1][key], value)
        self.assertEqual(new[2:4], old[2:4])

    def test_twice_target_norm_scales_half_and_keeps_whole_anchor(self):
        initial, pre, target, helper = vector([0]), vector([99]), vector([100]), vector([4])
        result, metrics, _, _, matrices = aggregate_layer_mask_budget(initial, target, pre,
            [(0, 0.5, target, pre), (1, 0.5, helper, initial)], 0, logical_layer_groups(initial))
        budget = matrices["budget_layers"][0]
        self.assertAlmostEqual(budget["budget_scale"], 0.5, places=12)
        self.assertAlmostEqual(budget["clipped_helper_norm"], 1.0, delta=1e-12)
        self.assertAlmostEqual(result["layer.weight"].item(), 101.0, delta=1e-12)
        self.assertEqual(metrics["budget_active_layers"], 1)
        self.assertEqual(budget["raw_helper_norm"], 2.0)

    def test_joint_weighted_sum_is_clipped_once_and_direction_is_preserved(self):
        initial, pre, target = vector([0, 0]), vector([10, 20]), vector([11, 20])
        uploads = [(0, 0.5, target, pre), (1, 0.25, vector([2, 10]), initial),
                   (2, 0.25, vector([2, -6]), initial)]
        originals = copy.deepcopy(uploads)
        result, _, _, _, matrices = aggregate_layer_mask_budget(
            initial, target, pre, iter(uploads), 0, logical_layer_groups(initial))
        row = matrices["budget_layers"][0]
        self.assertEqual(matrices["mask_matrix"], [[1], [1], [1]])
        self.assertAlmostEqual(row["raw_helper_norm"], 2 ** 0.5)
        scale = 1 / (2 ** 0.5 + 1e-12)
        expected_helper = torch.tensor([scale, scale], dtype=torch.float64)
        torch.testing.assert_close(result["layer.weight"], target["layer.weight"] + expected_helper)
        self.assertLessEqual(row["clipped_helper_norm"], row["target_local_norm"])
        self.assertGreaterEqual(row["budget_scale"], 0)
        for upload, original in zip(uploads, originals):
            for index in (2, 3):
                torch.testing.assert_close(upload[index]["layer.weight"], original[index]["layer.weight"], rtol=0, atol=0)

    def test_equal_budget_does_not_shrink_or_amplify(self):
        target = vector([3, 4])
        for helper in (vector([3, 4]), vector([0.3, 0.4])):
            clipped, rows, _ = clip_helper_budget(helper, target, logical_layer_groups(target))
            self.assertEqual(rows[0]["budget_scale"], 1.0)
            torch.testing.assert_close(clipped["layer.weight"], helper["layer.weight"], rtol=0, atol=0)

    def test_weight_bias_share_budget_and_other_layers_are_unchanged(self):
        target = {"conv.weight": torch.tensor([3.0]), "conv.bias": torch.tensor([4.0]),
                  "head.weight": torch.tensor([10.0])}
        helper = {"conv.weight": torch.tensor([6.0]), "conv.bias": torch.tensor([8.0]),
                  "head.weight": torch.tensor([1.0])}
        clipped, rows, metrics = clip_helper_budget(helper, target, logical_layer_groups(target))
        self.assertAlmostEqual(rows[0]["budget_scale"], 0.5)
        self.assertEqual(rows[1]["budget_scale"], 1.0)
        for name in target:
            torch.testing.assert_close(clipped[name], helper[name] * (0.5 if name.startswith("conv") else 1.0))
        self.assertAlmostEqual(metrics["raw_helper_norm"], 101 ** 0.5)
        self.assertAlmostEqual(metrics["clipped_helper_norm"], 26 ** 0.5)

    def test_zero_and_tiny_norm_special_cases_are_finite(self):
        for target, helper, expected in ((0, 1, 0), (0, 0, 1), (1, 0, 1),
                                         (1, 1e-13, 1), (1e-13, 1, 0), (1e-13, 1e-14, 0)):
            with self.subTest(target=target, helper=helper):
                local, raw = vector([target]), vector([helper])
                clipped, rows, metrics = clip_helper_budget(raw, local, logical_layer_groups(raw))
                self.assertEqual(rows[0]["budget_scale"], expected)
                self.assertEqual(clipped["layer.weight"].item(), helper * expected)
                json.dumps({"layers": rows, "metrics": metrics}, allow_nan=False)

    def test_zero_target_and_negative_helpers_follow_original_mask(self):
        initial, target = vector([0]), vector([0])
        _, _, _, _, matrices = aggregate_layer_mask_budget(initial, target, initial,
            [(0, 0.5, target, initial), (1, 0.5, vector([1]), initial)], 0, logical_layer_groups(initial))
        self.assertEqual(matrices["mask_matrix"], [[1], [0]])
        self.assertEqual(matrices["budget_layers"][0]["raw_helper_norm"], 0)
        target = vector([1])
        result, _, _, _, matrices = aggregate_layer_mask_budget(initial, target, initial,
            [(0, 0.5, target, initial), (1, 0.5, vector([-10]), initial)], 0, logical_layer_groups(initial))
        self.assertEqual(result["layer.weight"].item(), 1)
        self.assertEqual(matrices["mask_matrix"], [[1], [0]])


class LayerMaskBudgetServerTests(unittest.TestCase):
    def test_training_uses_budget_and_writes_isolated_logs_and_exports(self):
        fixture = fixtures.LayerMaskServerTests()
        fixture.setUp()
        with tempfile.TemporaryDirectory() as folder:
            server = fixture.make_server(folder)
            server.target_proj_mode = "layer_mask_budget"
            server.args.target_proj_mode = "layer_mask_budget"
            server.clients = [fixture.fixture.make_client(folder, cid) for cid in range(3)]
            server.Budget, server.rs_test_acc, server.auto_break = [], [], False
            for client in server.clients:
                client.train = mock.Mock(side_effect=client.train)
            # Preserve existing LayerMask output even if it shares a manually chosen directory.
            sentinel = Path(folder) / "layer_mask_matrices.json"
            sentinel.write_text("existing layer_mask run", encoding="utf-8")
            output = io.StringIO()
            with redirect_stdout(output):
                server.train()
            self.assertEqual(sentinel.read_text(encoding="utf-8"), "existing layer_mask run")
            self.assertIn("[LayerMask][Round 2] Mask matrix", output.getvalue())
            self.assertIn("[LayerBudget][Round 2]", output.getvalue())
            self.assertIn("budget_active_layers =", output.getvalue())
            for client in server.clients:
                self.assertEqual(client.train.call_count, 2)
            with open(Path(folder) / "layer_mask_budget_matrices.json") as stream:
                history = json.load(stream)["history"]
            self.assertEqual([entry["round"] for entry in history], [1, 2])
            for entry in history:
                self.assertEqual(entry["budget_beta"], 1.0)
                self.assertIn("target_client_test_acc", entry)
                self.assertEqual(entry["mask_matrix"][0], [1])
                self.assertIn("cosine_matrix", entry)
                for layer in entry["budget_layers"]:
                    self.assertLessEqual(layer["clipped_helper_norm"], layer["target_local_norm"] + 1e-7)
            for suffix, count in (("layers", 2), ("metrics", 2), ("clients", 6), ("cosines", 6)):
                with open(Path(folder) / f"layer_mask_budget_{suffix}.csv", newline="") as stream:
                    self.assertEqual(len(list(csv.DictReader(stream))), count)
            destination = Path(folder) / "export"
            server.final_model_dir = lambda: str(destination)
            with mock.patch.object(fixtures.fixtures.FakeServer, "export_final_models", create=True,
                                   side_effect=lambda: destination.mkdir()):
                server.export_final_models()
            self.assertTrue((destination / "layer_mask_budget_layers.csv").is_file())
            self.assertFalse((destination / "layer_mask_matrices.json").exists())


if __name__ == "__main__":
    unittest.main()

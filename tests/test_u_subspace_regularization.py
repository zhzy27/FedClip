import importlib.util
from importlib.machinery import ModuleSpec
import sys
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch


ROOT = Path(__file__).resolve().parents[1]
CLIENT_PATH = ROOT / "system" / "flcore" / "clients" / "clientCLIP.py"


def load_client_class():
    clientbase = types.ModuleType("flcore.clients.clientbase")
    clientbase.Client = object
    clientbase.load_item = lambda *args, **kwargs: None
    clientbase.save_item = lambda *args, **kwargs: None
    clip_utils = types.ModuleType("utils.get_clip_text_encoder")
    clip_utils.get_clip_class_embeddings = lambda *args, **kwargs: None
    clip_utils.get_clip_class_depth_embeddings = lambda *args, **kwargs: None
    with patch.dict(
        sys.modules,
        {
            "flcore.clients.clientbase": clientbase,
            "utils.get_clip_text_encoder": clip_utils,
        },
    ):
        spec = importlib.util.spec_from_file_location(
            "clientclip_subspace_test", CLIENT_PATH
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    return module.clientCLIP


clientCLIP = load_client_class()

for module_name in ("sklearn", "sklearn.preprocessing", "sklearn.metrics"):
    module = sys.modules.get(module_name)
    if module is not None and getattr(module, "__spec__", None) is None:
        module.__spec__ = ModuleSpec(module_name, loader=None)


class FactorModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight_u = torch.nn.Parameter(
            torch.tensor(
                [[1.0, 0.0], [0.0, 1.0], [0.0, 0.0], [0.0, 0.0]]
            )
        )
        self.weight_v = torch.nn.Parameter(torch.randn(2, 3))
        self.bias = torch.nn.Parameter(torch.zeros(4))


class USubspaceRegularizationTest(unittest.TestCase):
    def make_client(self, asymmetric=0, u_ratio=0.3, v_ratio=1.0):
        client = clientCLIP.__new__(clientCLIP)
        client.id = 3
        client.learning_rate = 0.005
        client.args = SimpleNamespace(
            use_asymmetric_lr=asymmetric,
            u_lr_ratio=u_ratio,
            v_lr_ratio=v_ratio,
            u_subspace_reg=0,
            u_subspace_diag=0,
            u_subspace_lambda=0.1,
        )
        return client

    def test_disabled_asymmetry_keeps_single_sgd_group(self):
        model = FactorModel()
        optimizer = self.make_client(asymmetric=0)._build_optimizer(model)
        self.assertEqual(len(optimizer.param_groups), 1)
        self.assertEqual(optimizer.param_groups[0]["lr"], 0.005)
        self.assertEqual(
            sum(len(group["params"]) for group in optimizer.param_groups),
            len(list(model.parameters())),
        )

    def test_asymmetric_learning_rates_are_assigned_by_factor(self):
        model = FactorModel()
        optimizer = self.make_client(
            asymmetric=1, u_ratio=0.3, v_ratio=1.0
        )._build_optimizer(model)
        learning_rates = {
            id(parameter): group["lr"]
            for group in optimizer.param_groups
            for parameter in group["params"]
        }
        self.assertAlmostEqual(learning_rates[id(model.weight_u)], 0.0015)
        self.assertAlmostEqual(learning_rates[id(model.weight_v)], 0.005)
        self.assertAlmostEqual(learning_rates[id(model.bias)], 0.005)

    def test_subspace_loss_is_differentiable_and_detects_rotation(self):
        model = FactorModel()
        client = self.make_client()
        start = client._capture_u_start_subspaces(model)
        initial_loss = client._u_subspace_loss(model, start)
        self.assertLess(abs(initial_loss.item()), 1e-6)

        with torch.no_grad():
            model.weight_u[2, 0] = 0.5
        rotated_loss = client._u_subspace_loss(model, start)
        self.assertGreater(rotated_loss.item(), 0.0)
        q_end, _ = torch.linalg.qr(model.weight_u.detach(), mode="reduced")
        explicit_loss = torch.linalg.matrix_norm(
            q_end @ q_end.T
            - start["weight_u"] @ start["weight_u"].T,
            ord="fro",
        ) ** 2 / (2 * q_end.shape[1])
        self.assertAlmostEqual(
            rotated_loss.item(), explicit_loss.item(), places=6
        )
        rotated_loss.backward()
        self.assertIsNotNone(model.weight_u.grad)
        self.assertTrue(torch.isfinite(model.weight_u.grad).all())
        self.assertGreater(
            client._u_subspace_drift_norm(model, start), 0.0
        )

    def test_missing_u_factor_raises(self):
        client = self.make_client()
        with self.assertRaisesRegex(RuntimeError, "contains no weight_u"):
            client._capture_u_start_subspaces(torch.nn.Linear(3, 2))

    def test_gradient_diagnostics_are_read_only_and_finite(self):
        model = FactorModel()
        client = self.make_client()
        start = client._capture_u_start_subspaces(model)
        with torch.no_grad():
            model.weight_u[2, 0] = 0.25

        base_loss = torch.sum(model.weight_u ** 2)
        subspace_loss = client._u_subspace_loss(model, start)
        stats = client._gradient_diagnostics(
            base_loss,
            subspace_loss,
            [model.weight_u],
            weight=0.3,
        )

        self.assertIsNone(model.weight_u.grad)
        for value in stats.values():
            self.assertTrue(torch.isfinite(torch.tensor(value)))
        self.assertGreater(stats["u_base_grad_norm"], 0.0)
        self.assertGreater(stats["u_sub_grad_norm"], 0.0)
        self.assertAlmostEqual(
            stats["u_sub_weighted_grad_norm"],
            0.3 * stats["u_sub_grad_norm"],
            places=7,
        )

    def test_layer_diagnostics_report_rotation_and_factor_changes(self):
        model = FactorModel()
        client = self.make_client(asymmetric=1)
        client.args.u_subspace_reg = 1
        start_subspaces = client._capture_u_start_subspaces(model)
        factor_starts = client._capture_factor_starts(
            model, start_subspaces
        )
        with torch.no_grad():
            model.weight_u[2, 0] = 0.5
            model.weight_v.add_(0.1)

        rows = client._u_subspace_layer_diagnostics(
            model,
            start_subspaces,
            factor_starts,
            current_round=7,
        )

        self.assertEqual(len(rows), 1)
        row = rows[0]
        self.assertEqual(row["round"], 7)
        self.assertEqual(row["client_id"], 3)
        self.assertEqual(row["layer_name"], "weight_u")
        self.assertEqual(row["rank"], 2)
        self.assertGreater(row["subspace_drift_norm"], 0.0)
        self.assertGreater(row["principal_angle_max_deg"], 0.0)
        self.assertGreater(row["R_U"], 0.0)
        self.assertGreater(row["R_V"], 0.0)
        self.assertTrue(torch.isfinite(torch.tensor(row["u_sigma_min"])))
        self.assertTrue(
            torch.isfinite(torch.tensor(row["u_condition_number"]))
        )


if __name__ == "__main__":
    unittest.main()

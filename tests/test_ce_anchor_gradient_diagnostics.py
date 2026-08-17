import copy
import csv
import importlib.util
import math
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
SYSTEM_ROOT = REPO_ROOT / "system"
sys.path.insert(0, str(SYSTEM_ROOT))

from utils.ce_anchor_diagnostics import (  # noqa: E402
    collect_ce_anchor_gradient_diagnostics,
)


class FactorBase(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight_u = torch.nn.Parameter(torch.randn(4, 2))
        self.weight_v = torch.nn.Parameter(torch.randn(2, 3))

    def forward(self, inputs):
        return inputs @ (self.weight_u @ self.weight_v).T


class DenseBase(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(3, 4)

    def forward(self, inputs):
        return self.linear(inputs)


class TinyModel(torch.nn.Module):
    def __init__(self, base):
        super().__init__()
        self.base = base
        self.head = torch.nn.Linear(4, 3)

    def forward_losses(self, inputs, labels, anchors):
        features = self.base(inputs)
        logits = self.head(features)
        return (
            torch.nn.functional.cross_entropy(logits, labels),
            torch.nn.functional.mse_loss(features, anchors),
        )


def make_batch():
    torch.manual_seed(17)
    inputs = torch.randn(6, 3)
    labels = torch.tensor([0, 1, 2, 1, 0, 2])
    anchors = torch.randn(6, 4)
    return inputs, labels, anchors


class CEAnchorGradientDiagnosticsTest(unittest.TestCase):
    def test_disabled_mode_does_not_call_autograd_grad(self):
        with mock.patch("torch.autograd.grad") as grad_mock:
            result = collect_ce_anchor_gradient_diagnostics(False)
        self.assertIsNone(result)
        grad_mock.assert_not_called()

    def test_diagnostics_preserve_loss_grad_fields_and_normal_backward(self):
        torch.manual_seed(3)
        diagnostic_model = TinyModel(FactorBase())
        control_model = copy.deepcopy(diagnostic_model)
        inputs, labels, anchors = make_batch()

        ce_loss, anchor_loss = diagnostic_model.forward_losses(
            inputs, labels, anchors
        )
        total_loss = ce_loss + 0.7 * anchor_loss
        loss_before = total_loss.detach().clone()
        metrics = collect_ce_anchor_gradient_diagnostics(
            True,
            ce_loss,
            anchor_loss,
            diagnostic_model.base.named_parameters(),
            mse_lambda=0.7,
        )

        self.assertTrue(torch.equal(loss_before, total_loss.detach()))
        self.assertTrue(
            all(parameter.grad is None for parameter in diagnostic_model.parameters())
        )
        total_loss.backward()

        control_ce, control_anchor = control_model.forward_losses(
            inputs, labels, anchors
        )
        (control_ce + 0.7 * control_anchor).backward()
        for diagnostic_parameter, control_parameter in zip(
            diagnostic_model.parameters(), control_model.parameters()
        ):
            self.assertTrue(
                torch.allclose(
                    diagnostic_parameter.grad,
                    control_parameter.grad,
                    atol=1e-7,
                    rtol=1e-6,
                )
            )
        self.assertIn("ce_anchor_grad_cos", metrics)

    def test_metrics_are_finite_and_cosines_are_clamped(self):
        torch.manual_seed(5)
        model = TinyModel(FactorBase())
        ce_loss, anchor_loss = model.forward_losses(*make_batch())
        metrics = collect_ce_anchor_gradient_diagnostics(
            True,
            ce_loss,
            anchor_loss,
            model.base.named_parameters(),
            mse_lambda=1.0,
        )

        for name, value in metrics.items():
            if name == "gradient_conflict":
                self.assertIn(value, (0, 1))
            else:
                self.assertTrue(math.isfinite(value), name)
        for name in (
            "ce_anchor_grad_cos",
            "u_ce_anchor_grad_cos",
            "v_ce_anchor_grad_cos",
        ):
            self.assertGreaterEqual(metrics[name], -1.0)
            self.assertLessEqual(metrics[name], 1.0)

    def test_zero_mse_lambda_keeps_raw_anchor_measurement(self):
        torch.manual_seed(7)
        model = TinyModel(FactorBase())
        ce_loss, anchor_loss = model.forward_losses(*make_batch())
        metrics = collect_ce_anchor_gradient_diagnostics(
            True,
            ce_loss,
            anchor_loss,
            model.base.named_parameters(),
            mse_lambda=0.0,
        )

        self.assertGreater(metrics["anchor_grad_norm"], 0.0)
        self.assertEqual(metrics["weighted_anchor_grad_norm"], 0.0)
        self.assertEqual(metrics["anchor_to_ce_grad_ratio"], 0.0)
        self.assertEqual(metrics["u_anchor_to_ce_grad_ratio"], 0.0)
        self.assertEqual(metrics["v_anchor_to_ce_grad_ratio"], 0.0)

    def test_missing_factor_groups_are_reported_as_nan(self):
        torch.manual_seed(11)
        model = TinyModel(DenseBase())
        ce_loss, anchor_loss = model.forward_losses(*make_batch())
        metrics = collect_ce_anchor_gradient_diagnostics(
            True,
            ce_loss,
            anchor_loss,
            model.base.named_parameters(),
            mse_lambda=1.0,
        )

        self.assertTrue(math.isfinite(metrics["ce_grad_norm"]))
        for prefix in ("u", "v"):
            for suffix in (
                "ce_grad_norm",
                "anchor_grad_norm",
                "anchor_to_ce_grad_ratio",
                "ce_anchor_grad_cos",
            ):
                self.assertTrue(math.isnan(metrics[f"{prefix}_{suffix}"]))


class CEAnchorServerDiagnosticsTest(unittest.TestCase):
    @staticmethod
    def _load_server_class():
        client_module = types.ModuleType("flcore.clients.clientCLIP")
        client_module.clientCLIP = object
        clientbase_module = types.ModuleType("flcore.clients.clientbase")
        clientbase_module.load_item = mock.MagicMock()
        clientbase_module.save_item = mock.MagicMock()
        serverbase_module = types.ModuleType("flcore.servers.serverbase")
        serverbase_module.Server = object
        models_module = types.ModuleType("flcore.trainmodel.models")
        models_module.Model_Distribe = object
        stubs = {
            "flcore.clients.clientCLIP": client_module,
            "flcore.clients.clientbase": clientbase_module,
            "flcore.servers.serverbase": serverbase_module,
            "flcore.trainmodel.models": models_module,
        }
        module_path = SYSTEM_ROOT / "flcore" / "servers" / "serverCLIP.py"
        spec = importlib.util.spec_from_file_location(
            "ce_anchor_server_test_module", module_path
        )
        module = importlib.util.module_from_spec(spec)
        with mock.patch.dict(sys.modules, stubs):
            spec.loader.exec_module(module)
        return module.FedCLIP

    def test_csv_output_and_capacity_groups_use_current_clients(self):
        server_class = self._load_server_class()
        server = server_class.__new__(server_class)
        rows = []
        for client_id, capacity in enumerate((0.9, 0.15, 0.37, 0.25, 0.9, 0.15)):
            row = {
                field: 0.25 for field in server_class._CE_ANCHOR_DIAG_FIELDS
            }
            row.update(
                {
                    "round": 4,
                    "client_id": client_id,
                    "capacity_ratio": capacity,
                    "gradient_conflict": client_id % 2,
                }
            )
            rows.append(row)
        server.selected_clients = [
            SimpleNamespace(last_ce_anchor_diag=row) for row in rows
        ]

        with tempfile.TemporaryDirectory() as temp_dir:
            server.save_folder_name = str(Path(temp_dir) / "run")
            log_dir = Path(temp_dir) / "logs"
            with mock.patch.dict(
                os.environ, {"FEDCLIP_TRAIN_LOG_DIR": str(log_dir)}
            ):
                with mock.patch("builtins.print"):
                    server._record_ce_anchor_diagnostics(4)

            csv_path = log_dir / "ce_anchor_gradient_diagnostics.csv"
            self.assertTrue(csv_path.is_file())
            with csv_path.open(newline="", encoding="utf-8") as csv_file:
                saved_rows = list(csv.DictReader(csv_file))
            self.assertEqual(len(saved_rows), len(rows))
            self.assertEqual(
                [int(row["client_id"]) for row in saved_rows],
                list(range(len(rows))),
            )

        groups = server._capacity_groups(rows)
        grouped_ids = [
            int(row["client_id"])
            for group_name in ("low", "mid", "high")
            for row in groups[group_name]
        ]
        self.assertEqual(set(grouped_ids), set(range(len(rows))))
        self.assertEqual(
            [row["capacity_ratio"] for row in groups["low"]], [0.15, 0.15]
        )


if __name__ == "__main__":
    unittest.main()

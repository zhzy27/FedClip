import copy
import importlib.util
from importlib.machinery import ModuleSpec
import math
import sys
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
SYSTEM_ROOT = REPO_ROOT / "system"
sys.path.insert(0, str(SYSTEM_ROOT))

from utils.factor_loss_diagnostics import (  # noqa: E402
    DIAGNOSTIC_FIELDS,
    collect_factor_loss_diagnostics,
    gradient_clip_diagnostics,
    named_factor_parameters,
    scaled_u_gradients,
)


class FactorBase(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight_u = torch.nn.Parameter(torch.randn(4, 2))
        self.weight_v = torch.nn.Parameter(torch.randn(2, 3))

    def forward(self, inputs):
        return inputs @ (self.weight_u @ self.weight_v).T


class FactorModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.base = FactorBase()
        self.head = torch.nn.Linear(4, 3)
        self.ratio_LR = 0.25

    def forward_losses(self, inputs, labels, anchors):
        features = self.base(inputs)
        logits = self.head(features)
        return (
            torch.nn.functional.cross_entropy(logits, labels),
            torch.nn.functional.mse_loss(features, anchors),
            torch.sum((self.base.weight_u @ self.base.weight_v) ** 2),
        )


class FactorBNBase(FactorBase):
    def __init__(self):
        super().__init__()
        self.bn = torch.nn.BatchNorm1d(4)

    def forward(self, inputs):
        return self.bn(super().forward(inputs))


def make_batch():
    torch.manual_seed(31)
    return (
        torch.randn(6, 3),
        torch.tensor([0, 1, 2, 0, 2, 1]),
        torch.randn(6, 4),
    )


class FactorLossDiagnosticsTest(unittest.TestCase):
    def test_diagnostics_do_not_pollute_grad_and_wpath_matches_direct_product(self):
        torch.manual_seed(13)
        model = FactorModel()
        ce_loss, anchor_loss, reg_loss = model.forward_losses(*make_batch())
        rows, gradients = collect_factor_loss_diagnostics(
            model=model,
            ce_loss=ce_loss,
            anchor_loss=anchor_loss,
            regularization_loss=reg_loss,
            anchor_coefficient=0.8,
            regularization_coefficient=1e-3,
            round_number=1,
            client_id=2,
            capacity=0.25,
            u_lr=0.0015,
            v_lr=0.005,
        )

        self.assertTrue(all(parameter.grad is None for parameter in model.parameters()))
        self.assertEqual(rows[0]["layer"], "__overall__")
        self.assertEqual(len(rows), 2)
        self.assertTrue(set(DIAGNOSTIC_FIELDS).issubset(rows[0]))
        self.assertTrue(set(DIAGNOSTIC_FIELDS).issubset(rows[1]))
        for field in (
            "u_ce_grad_norm",
            "u_anchor_grad_norm",
            "u_reg_grad_norm",
            "v_ce_grad_norm",
            "v_anchor_grad_norm",
            "v_reg_grad_norm",
            "u_ce_anchor_cos",
            "v_ce_anchor_cos",
            "wpath_u_ce_anchor_cos",
            "wpath_v_ce_anchor_cos",
        ):
            self.assertTrue(math.isfinite(rows[0][field]), field)

        u_ce_path = gradients["ce"]["base.weight_u"] @ model.base.weight_v
        u_anchor_path = (
            gradients["anchor"]["base.weight_u"] @ model.base.weight_v
        )
        expected_u_cos = torch.nn.functional.cosine_similarity(
            u_ce_path.flatten(), u_anchor_path.flatten(), dim=0
        ).item()
        self.assertAlmostEqual(
            rows[0]["wpath_u_ce_anchor_cos"], expected_u_cos, places=6
        )

    def test_unit_scales_reproduce_original_u_gradient(self):
        torch.manual_seed(17)
        model = FactorModel()
        ce_loss, anchor_loss, reg_loss = model.forward_losses(*make_batch())
        named_u = [
            (name, parameter)
            for name, parameter in named_factor_parameters(model)
            if name.endswith("weight_u")
        ]
        combined = scaled_u_gradients(
            ce_loss,
            anchor_loss,
            reg_loss,
            named_u,
            anchor_coefficient=0.8,
            regularization_coefficient=0.01,
            ce_scale=1.0,
            anchor_scale=1.0,
            reg_scale=1.0,
        )
        total_loss = ce_loss + 0.8 * anchor_loss + 0.01 * reg_loss
        total_loss.backward()
        for name, parameter in named_u:
            self.assertTrue(
                torch.allclose(combined[name], parameter.grad, atol=1e-6, rtol=1e-5)
            )

    def test_slow_all_is_point_three_of_original_u_gradient(self):
        torch.manual_seed(19)
        model = FactorModel()
        ce_loss, anchor_loss, reg_loss = model.forward_losses(*make_batch())
        named_u = [
            (name, parameter)
            for name, parameter in named_factor_parameters(model)
            if name.endswith("weight_u")
        ]
        combined = scaled_u_gradients(
            ce_loss,
            anchor_loss,
            reg_loss,
            named_u,
            anchor_coefficient=1.0,
            regularization_coefficient=1e-3,
            ce_scale=0.3,
            anchor_scale=0.3,
            reg_scale=0.3,
        )
        (ce_loss + anchor_loss + 1e-3 * reg_loss).backward()
        for name, parameter in named_u:
            self.assertTrue(
                torch.allclose(
                    combined[name],
                    0.3 * parameter.grad,
                    atol=1e-6,
                    rtol=1e-5,
                )
            )

    @staticmethod
    def _load_client_class():
        clientbase_module = types.ModuleType("flcore.clients.clientbase")
        clientbase_module.Client = object
        clientbase_module.load_item = mock.MagicMock()
        clientbase_module.save_item = mock.MagicMock()
        clip_module = types.ModuleType("utils.get_clip_text_encoder")
        clip_module.get_clip_class_embeddings = mock.MagicMock()
        clip_module.get_clip_class_depth_embeddings = mock.MagicMock()
        sklearn_module = types.ModuleType("sklearn")
        preprocessing_module = types.ModuleType("sklearn.preprocessing")
        preprocessing_module.label_binarize = mock.MagicMock()
        stubs = {
            "flcore.clients.clientbase": clientbase_module,
            "utils.get_clip_text_encoder": clip_module,
            "sklearn": sklearn_module,
            "sklearn.preprocessing": preprocessing_module,
        }
        module_path = SYSTEM_ROOT / "flcore" / "clients" / "clientCLIP.py"
        spec = importlib.util.spec_from_file_location(
            "factor_loss_client_test_module", module_path
        )
        module = importlib.util.module_from_spec(spec)
        with mock.patch.dict(sys.modules, stubs):
            spec.loader.exec_module(module)
        return module.clientCLIP

    def test_virtual_steps_restore_parameters_and_do_not_touch_grad(self):
        client_class = self._load_client_class()
        client = client_class.__new__(client_class)
        client.device = torch.device("cpu")
        client.use_resnet_multilevel_clip = False
        client.mse_fn = torch.nn.MSELoss()
        client.loss = torch.nn.CrossEntropyLoss()
        client.learning_rate = 0.005
        client.args = SimpleNamespace(virtual_step_scale=1.0)
        client.clip_text_features = torch.randn(3, 4)

        torch.manual_seed(23)
        model = FactorModel()
        ce_loss, anchor_loss, reg_loss = model.forward_losses(*make_batch())
        _, gradients = collect_factor_loss_diagnostics(
            model,
            ce_loss,
            anchor_loss,
            reg_loss,
            1.0,
            1e-3,
            1,
            0,
            0.25,
            0.0015,
            0.005,
        )
        original = copy.deepcopy(model.state_dict())
        probe_inputs, probe_labels, _ = make_batch()
        results = client._virtual_step_diagnostics(
            model,
            (probe_inputs, probe_labels),
            gradients,
            0.0015,
            0.005,
        )
        common_step_reference = client._virtual_step_diagnostics(
            model,
            (probe_inputs, probe_labels),
            gradients,
            client.learning_rate,
            client.learning_rate,
        )

        for name, value in model.state_dict().items():
            self.assertTrue(torch.equal(value, original[name]), name)
        self.assertTrue(all(parameter.grad is None for parameter in model.parameters()))
        self.assertTrue(all(math.isfinite(value) for value in results.values()))
        for source_name in ("ce", "anchor"):
            for group_name in ("u", "v"):
                for target_name in ("ce", "anchor"):
                    common_field = (
                        f"virtual_common_{source_name}_to_{group_name}_delta_"
                        f"{target_name}"
                    )
                    reference_field = (
                        f"virtual_{source_name}_to_{group_name}_delta_"
                        f"{target_name}"
                    )
                    self.assertAlmostEqual(
                        results[common_field],
                        common_step_reference[reference_field],
                        places=12,
                    )

    def test_gradient_clip_diagnostics_use_true_pre_clip_global_norm(self):
        first = torch.nn.Parameter(torch.zeros(2))
        second = torch.nn.Parameter(torch.zeros(1))
        first.grad = torch.tensor([3.0, 4.0])
        second.grad = torch.tensor([12.0])
        expected_pre_clip_norm = math.sqrt(3.0 ** 2 + 4.0 ** 2 + 12.0 ** 2)

        returned_norm = torch.nn.utils.clip_grad_norm_(
            [first, second], max_norm=10.0
        )
        values = gradient_clip_diagnostics(returned_norm, max_norm=10.0)
        actual_post_clip_norm = math.sqrt(
            float(torch.sum(first.grad.double() ** 2).item())
            + float(torch.sum(second.grad.double() ** 2).item())
        )

        self.assertAlmostEqual(
            values["pre_clip_total_grad_norm"],
            expected_pre_clip_norm,
            places=6,
        )
        self.assertEqual(values["clip_was_active"], 1.0)
        self.assertLess(values["clip_coef"], 1.0)
        self.assertAlmostEqual(
            values["post_clip_total_grad_norm"],
            actual_post_clip_norm,
            places=6,
        )

    def test_scheduled_diagnostic_restores_training_mode_and_bn_statistics(self):
        client_class = self._load_client_class()
        client = client_class.__new__(client_class)
        client.id = 0
        client.device = torch.device("cpu")
        client.use_resnet_multilevel_clip = False
        client.mse_fn = torch.nn.MSELoss()
        client.loss = torch.nn.CrossEntropyLoss()
        client.learning_rate = 0.005
        client.clip_text_features = torch.randn(3, 4)
        client.args = SimpleNamespace(
            virtual_step_scale=1.0,
            enable_virtual_step_diagnostics=1,
            mse_lamda=1.0,
            is_regular=0,
            regular_lamda=1e-3,
        )
        torch.manual_seed(37)
        model = FactorModel()
        model.base = FactorBNBase()
        model.train()
        running_mean = model.base.bn.running_mean.clone()
        running_var = model.base.bn.running_var.clone()
        inputs, labels, _ = make_batch()

        rows = client._run_loss_diagnostics(
            model,
            (inputs, labels),
            (inputs.flip(0), labels.flip(0)),
            current_round=0,
            actual_u_lr=0.0015,
            actual_v_lr=0.005,
        )

        self.assertTrue(model.training)
        self.assertTrue(torch.equal(model.base.bn.running_mean, running_mean))
        self.assertTrue(torch.equal(model.base.bn.running_var, running_var))
        self.assertTrue(rows)

    def test_disabled_schedule_does_not_select_diagnostic_path(self):
        client_class = self._load_client_class()
        client = client_class.__new__(client_class)
        client.id = 0
        client.args = SimpleNamespace(
            enable_ce_anchor_diagnostics=0,
            enable_virtual_step_diagnostics=0,
            diagnostic_rounds="1",
            diagnostic_client_ids="0",
        )
        self.assertFalse(client._diagnostic_target(0))

    def test_set_parameters_reports_missing_local_model_shell(self):
        client_class = self._load_client_class()
        client = client_class.__new__(client_class)
        client.id = 8
        client.role = "Client_8"
        client.device = torch.device("cpu")
        client.save_folder_name = "missing_run"
        client.args = SimpleNamespace(aggregation_mode="avg", d_max=0.7)
        train_globals = client_class.set_parameters.__globals__

        with mock.patch.dict(
            train_globals,
            {"load_item": mock.MagicMock(return_value=None)},
        ):
            with self.assertRaisesRegex(
                RuntimeError, "Missing local model shell for Client_8"
            ):
                client.set_parameters()

    def test_all_new_features_disabled_preserve_original_training_step(self):
        client_class = self._load_client_class()
        torch.manual_seed(29)
        trained_model = FactorModel()
        control_model = copy.deepcopy(trained_model)
        inputs, labels, _ = make_batch()
        clip_anchors = torch.randn(3, 4)

        client = client_class.__new__(client_class)
        client.id = 0
        client.role = "Client_0"
        client.device = torch.device("cpu")
        client.learning_rate = 0.005
        client.local_epochs = 1
        client.train_slow = False
        client.use_resnet_multilevel_clip = False
        client.resnet_clip_aligners = None
        client.mse_fn = torch.nn.MSELoss()
        client.loss = torch.nn.CrossEntropyLoss()
        client.clip_text_features = clip_anchors
        client.save_folder_name = "unused"
        client.train_samples = len(labels)
        client.train_time_cost = {"num_rounds": 0, "total_cost": 0.0}
        client.args = SimpleNamespace(
            enable_ce_anchor_diagnostics=0,
            enable_virtual_step_diagnostics=0,
            use_loss_specific_u_scaling=0,
            u_ce_grad_scale=1.0,
            u_anchor_grad_scale=1.0,
            u_reg_grad_scale=1.0,
            use_asymmetric_lr=0,
            mse_lamda=1.0,
            is_regular=0,
            regular_lamda=1e-3,
        )
        client.load_train_data = lambda: [(inputs.clone(), labels.clone())]

        diagnostic_mock = mock.MagicMock()
        scaling_mock = mock.MagicMock()
        train_globals = client_class.train.__globals__
        sklearn_module = sys.modules.get("sklearn")
        if sklearn_module is not None and getattr(sklearn_module, "__spec__", None) is None:
            sklearn_module.__spec__ = ModuleSpec("sklearn", loader=None)
        with mock.patch.dict(
            train_globals,
            {
                "load_item": mock.MagicMock(return_value=trained_model),
                "save_item": mock.MagicMock(),
                "collect_factor_loss_diagnostics": diagnostic_mock,
                "scaled_u_gradients": scaling_mock,
            },
        ):
            with mock.patch("builtins.print"):
                client.train(current_round=0)

        optimizer = torch.optim.SGD(control_model.parameters(), lr=0.005)
        optimizer.zero_grad()
        control_features = control_model.base(inputs)
        control_logits = control_model.head(control_features)
        control_loss = torch.nn.functional.cross_entropy(control_logits, labels)
        control_loss += torch.nn.functional.mse_loss(
            control_features, clip_anchors[labels]
        )
        control_loss.backward()
        torch.nn.utils.clip_grad_norm_(list(control_model.parameters()), 10.0)
        optimizer.step()

        diagnostic_mock.assert_not_called()
        scaling_mock.assert_not_called()
        for name, parameter in trained_model.state_dict().items():
            self.assertTrue(
                torch.allclose(parameter, control_model.state_dict()[name]), name
            )


if __name__ == "__main__":
    unittest.main()

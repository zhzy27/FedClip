import ast
import copy
import contextlib
import csv
import importlib.util
import io
import math
import sys
import tempfile
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SERVER_PATH = REPOSITORY_ROOT / "system" / "flcore" / "servers" / "serverCLIP.py"
MAIN_PATH = REPOSITORY_ROOT / "system" / "main.py"


def _module(name, **attributes):
    value = types.ModuleType(name)
    for attribute_name, attribute_value in attributes.items():
        setattr(value, attribute_name, attribute_value)
    return value


def _package(name):
    value = _module(name)
    value.__path__ = []
    return value


def _load_fedclip_class():
    dependency_stubs = {
        "flcore": _package("flcore"),
        "flcore.clients": _package("flcore.clients"),
        "flcore.servers": _package("flcore.servers"),
        "flcore.trainmodel": _package("flcore.trainmodel"),
        "utils": _package("utils"),
        "flcore.clients.clientCLIP": _module(
            "flcore.clients.clientCLIP",
            clientCLIP=object,
        ),
        "flcore.clients.clientbase": _module(
            "flcore.clients.clientbase",
            load_item=lambda *args, **kwargs: None,
            save_item=lambda *args, **kwargs: None,
        ),
        "flcore.servers.serverbase": _module(
            "flcore.servers.serverbase",
            Server=object,
        ),
        "flcore.trainmodel.models": _module(
            "flcore.trainmodel.models",
            Model_Distribe=object,
        ),
        "utils.get_clip_text_encoder": _module(
            "utils.get_clip_text_encoder",
            get_clip_class_embeddings=lambda *args, **kwargs: (None, None),
        ),
    }
    with mock.patch.dict(sys.modules, dependency_stubs):
        spec = importlib.util.spec_from_file_location(
            "serverCLIP_personalized_direction_test",
            SERVER_PATH,
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    return module.FedCLIP


FedCLIP = _load_fedclip_class()


class _FakeFactorizedConv(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_u = torch.nn.Parameter(torch.ones(1, 1))
        self.conv_v = torch.nn.Parameter(torch.ones(1, 1))


class _FakeFactorizedLinear(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight_u = torch.nn.Parameter(torch.ones(1, 1))
        self.weight_v = torch.nn.Parameter(torch.ones(1, 1))


class _ScopeLowRankModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(1, 1, 1, bias=True)
        self.conv2 = _FakeFactorizedConv()
        self.fc1 = _FakeFactorizedLinear()
        self.fc2 = _FakeFactorizedLinear()
        self.classifier = torch.nn.Linear(1, 1, bias=True)
        self.layer_norm = torch.nn.LayerNorm(1)


class _ScopeFullModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(1, 1, 1, bias=True)
        self.conv2 = torch.nn.Conv2d(1, 1, 1, bias=True)
        self.fc1 = torch.nn.Linear(1, 1, bias=True)
        self.fc2 = torch.nn.Linear(1, 1, bias=True)
        self.classifier = torch.nn.Linear(1, 1, bias=True)
        self.layer_norm = torch.nn.LayerNorm(1)


class PersonalizedDirectionSelectionTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rank_r = 4
        cls.eigvals = torch.tensor(
            [16.0, 9.0, 4.0, 1.0],
            dtype=torch.float64,
        )
        coefficient = 1.0 / math.sqrt(2.0)
        cls.eigvecs = torch.tensor(
            [
                [0.0, coefficient, coefficient, 0.0],
                [1.0, 0.0, 0.0, 0.0],
                [0.0, coefficient, -coefficient, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=torch.float64,
        )

    def select(self, rank_num, force_u1=True):
        return FedCLIP._select_personalized_directions(
            self.eigvals,
            self.eigvecs,
            self.rank_r,
            rank_num,
            force_u1,
        )

    def select_energy(
        self,
        energy_threshold=0.8,
        force_u1=False,
        eigvals=None,
        eigvecs=None,
        rank_r=None,
    ):
        eigvals = self.eigvals if eigvals is None else eigvals
        eigvecs = self.eigvecs if eigvecs is None else eigvecs
        rank_r = self.rank_r if rank_r is None else rank_r
        return FedCLIP._select_personalized_directions(
            eigvals,
            eigvecs,
            rank_r,
            personalized_rank_num=999,
            force_u1=force_u1,
            rank_mode="energy",
            energy_threshold=energy_threshold,
        )

    def orthogonal_layer_inputs(self):
        normalized_eigvals = self.eigvals / self.eigvals.sum()
        alpha_tensor = self.eigvecs.square() @ normalized_eigvals
        weighted_updates = (
            torch.diag(torch.sqrt(normalized_eigvals)) @ self.eigvecs.t()
        )
        updates = [
            weighted_updates[:, client_idx]
            / torch.sqrt(alpha_tensor[client_idx])
            for client_idx in range(self.eigvecs.shape[0])
        ]
        alpha = [float(value.item()) for value in alpha_tensor]
        return updates, alpha

    @staticmethod
    def _fill_scope_model(model, major_values, other_value):
        with torch.no_grad():
            for name, param in model.named_parameters():
                param.fill_(major_values.get(name, other_value))
        return model

    def _run_scope_aggregation(
        self,
        scope,
        cross_layer_mode=None,
        direction_selection_mode=None,
    ):
        major_names = [
            "conv1.weight",
            "conv2.weight",
            "fc1.weight",
            "fc2.weight",
            "classifier.weight",
        ]
        start_values = [
            {
                "conv1.weight": 1.0,
                "conv2.weight": 10.0,
                "fc1.weight": 20.0,
                "fc2.weight": 30.0,
                "classifier.weight": 40.0,
            },
            {
                "conv1.weight": 5.0,
                "conv2.weight": 11.0,
                "fc1.weight": 21.0,
                "fc2.weight": 31.0,
                "classifier.weight": 50.0,
            },
        ]
        global_values = {
            "conv1.weight": 100.0,
            "conv2.weight": 110.0,
            "fc1.weight": 120.0,
            "fc2.weight": 130.0,
            "classifier.weight": 140.0,
        }
        start_models = [
            self._fill_scope_model(
                _ScopeFullModel(),
                values,
                float(client_index * 10),
            )
            for client_index, values in enumerate(start_values)
        ]
        uploaded_models = [copy.deepcopy(model) for model in start_models]
        with torch.no_grad():
            for model in uploaded_models:
                params = dict(model.named_parameters())
                for name in major_names:
                    params[name].add_(2.0)
        global_model = self._fill_scope_model(
            _ScopeFullModel(),
            global_values,
            -100.0,
        )

        server = FedCLIP.__new__(FedCLIP)
        server.args = SimpleNamespace(
            aggregation_mode="sign_projection_no_group_renorm",
            personalized_rank_selection=1,
            personalized_rank_num=1,
            personalized_rank_force_u1=1,
            personalized_rank_mode="fixed",
            personalized_rank_energy=0.8,
            personalized_g_scale=0,
            local_update_views=1,
            personalized_repeatability_threshold=-1.0,
            personalized_coeff_mode="same_sign",
            personalized_tail_scale=1.0,
            projection_energy=1.0,
            projection_k_max=2,
            projection_norm_scale_max=2.0,
        )
        if cross_layer_mode is not None:
            server.args.personalized_cross_layer_client_mode = (
                cross_layer_mode
            )
            server.args.personalized_cross_layer_client_topk = 1
        if direction_selection_mode is not None:
            server.args.personalized_direction_selection_mode = (
                direction_selection_mode
            )
            server.args.personalized_extra_topk = 1
        if scope is not None:
            server.args.projection_layer_scope = scope
        server.device = torch.device("cpu")
        server.role = "Server"
        server.save_folder_name = "memory"
        server.uploaded_ids = [0, 1]
        server.uploaded_weights = [0.25, 0.75]
        server.num_clients = 2
        server.cur_ground = 21
        server.clients = [
            SimpleNamespace(role=f"Client_{index}", save_folder_name="memory")
            for index in range(server.num_clients)
        ]
        server.personal_residuals = {}
        server.client_start_full_weights = {
            client_id: {
                name: param.detach().clone()
                for name, param in model.named_parameters()
            }
            for client_id, model in enumerate(start_models)
        }
        server._recover_if_needed = lambda model: model
        server._projectable_weight_names_from_low_rank_model = lambda model: {
            "conv2.weight",
            "fc1.weight",
            "fc2.weight",
        }
        server._is_sign_projection_diagnostic_round = lambda: False

        uploaded_by_role = {
            f"Client_{client_id}": model
            for client_id, model in enumerate(uploaded_models)
        }
        saved_models = {}

        def fake_load_item(role, item_name, item_path):
            if role == "Server" and item_name == "model":
                return copy.deepcopy(global_model)
            if role in uploaded_by_role and item_name == "model":
                return copy.deepcopy(uploaded_by_role[role])
            raise AssertionError(f"Unexpected load: {role}/{item_name}")

        def fake_save_item(item, role, item_name, item_path):
            saved_models[item_name] = copy.deepcopy(item)

        method_globals = FedCLIP._aggregate_sign_projection_variant.__globals__
        console = io.StringIO()
        with contextlib.redirect_stdout(console), mock.patch.dict(
            method_globals,
            {"load_item": fake_load_item, "save_item": fake_save_item},
        ):
            server.aggregate_sign_projection_no_group_renorm()

        return {
            "saved_models": saved_models,
            "start_values": start_values,
            "global_values": global_values,
            "major_names": major_names,
            "console": console.getvalue(),
        }

    def run_diagnostic_layer(
        self,
        *,
        rank_mode="energy",
        energy_threshold=0.8,
        g_scale=1,
        force_u1=False,
        rank_num=2,
        projection_k_max=1,
        norm_restore=True,
        updates=None,
        updates_b=None,
        alpha=None,
        repeatability_threshold=-1.0,
        coeff_mode="same_sign",
        tail_scale=1.0,
        m_filter_mode=None,
        dominance_threshold=0.7,
        conflict_handling=None,
        local_update_views=None,
        personalized_rank_selection=1,
        mode_name="sign_projection_no_group_renorm",
        input_kind=None,
        norm_scale_max=2.0,
        projection_layer_scope="low_rank",
        direction_selection_mode=None,
        extra_topk=1,
        start_weights=None,
    ):
        if updates is None or alpha is None:
            updates, alpha = self.orthogonal_layer_inputs()
        server = FedCLIP.__new__(FedCLIP)
        if local_update_views is None:
            local_update_views = 2 if updates_b is not None else 1
        server.args = SimpleNamespace(
            aggregation_mode=mode_name,
            personalized_rank_selection=personalized_rank_selection,
            personalized_rank_num=rank_num,
            personalized_rank_force_u1=int(force_u1),
            personalized_rank_mode=rank_mode,
            personalized_rank_energy=energy_threshold,
            personalized_g_scale=g_scale,
            local_update_views=local_update_views,
            personalized_repeatability_threshold=repeatability_threshold,
            personalized_coeff_mode=coeff_mode,
            personalized_tail_scale=tail_scale,
            projection_energy=1.0,
            projection_k_max=projection_k_max,
            projection_norm_scale_max=norm_scale_max,
        )
        if m_filter_mode is not None:
            server.args.personalized_m_filter_mode = m_filter_mode
            server.args.personalized_dominance_threshold = dominance_threshold
        if conflict_handling is not None:
            server.args.personalized_conflict_handling = conflict_handling
        if direction_selection_mode is not None:
            server.args.personalized_direction_selection_mode = (
                direction_selection_mode
            )
            server.args.personalized_extra_topk = extra_topk
        server.device = torch.device("cpu")
        server.uploaded_ids = list(range(len(updates)))
        server.cur_ground = 1
        server._projection_diagnostic_paths_printed = False

        with tempfile.TemporaryDirectory() as temporary_directory:
            server.projection_client_diagnostic_csv = str(
                Path(temporary_directory) / "clients.csv"
            )
            server.projection_direction_diagnostic_csv = str(
                Path(temporary_directory) / "directions.csv"
            )
            console_output = io.StringIO()
            layer_kwargs = {
                "delta_param_dicts_b": (
                    None
                    if updates_b is None
                    else [{"layer": update} for update in updates_b]
                ),
                "log_diagnostics": True,
                "console_diagnostics": True,
                "group_renorm": False,
                "norm_restore": norm_restore,
                "mode_name": mode_name,
                "projection_layer_scope": projection_layer_scope,
            }
            if input_kind is not None:
                layer_kwargs["input_kind"] = input_kind
            if start_weights is not None:
                layer_kwargs["start_param_dicts"] = [
                    {"layer": start_weight}
                    for start_weight in start_weights
                ]
            with contextlib.redirect_stdout(console_output):
                personalized, average = server._sign_personalized_update_for_layer(
                    "layer",
                    [{"layer": update} for update in updates],
                    alpha,
                    updates[0].shape,
                    **layer_kwargs,
                )
            with open(
                server.projection_client_diagnostic_csv,
                newline="",
                encoding="utf-8",
            ) as file:
                client_rows = list(csv.DictReader(file))
            with open(
                server.projection_direction_diagnostic_csv,
                newline="",
                encoding="utf-8",
            ) as file:
                direction_rows = list(csv.DictReader(file))
        return (
            personalized,
            average,
            client_rows,
            direction_rows,
            console_output.getvalue(),
        )

    def collect_cross_layer_state(
        self,
        selection_mode="model_only",
        capture_diagnostics=False,
    ):
        updates, alpha = self.orthogonal_layer_inputs()
        start_weights = [
            torch.tensor(
                [1.0, 8.0, 2.0, 3.0],
                dtype=updates[0].dtype,
            ),
            torch.tensor(
                [2.0, 1.0, 7.0, 3.0],
                dtype=updates[0].dtype,
            ),
            torch.tensor(
                [3.0, 2.0, 1.0, 9.0],
                dtype=updates[0].dtype,
            ),
            torch.tensor(
                [4.0, 6.0, 2.0, 1.0],
                dtype=updates[0].dtype,
            ),
        ]
        server = FedCLIP.__new__(FedCLIP)
        server.args = SimpleNamespace(
            aggregation_mode="sign_projection_no_group_renorm",
            personalized_rank_selection=1,
            personalized_rank_num=2,
            personalized_rank_force_u1=0,
            personalized_rank_mode="fixed",
            personalized_rank_energy=0.8,
            personalized_direction_selection_mode=selection_mode,
            personalized_extra_topk=1,
            personalized_g_scale=0,
            local_update_views=1,
            personalized_repeatability_threshold=-1.0,
            personalized_coeff_mode="same_sign",
            personalized_tail_scale=1.0,
            personalized_m_filter_mode="none",
            personalized_conflict_handling="zero",
            projection_energy=1.0,
            projection_k_max=4,
            projection_norm_scale_max=2.0,
        )
        server.device = torch.device("cpu")
        server.uploaded_ids = list(range(len(updates)))
        server.cur_ground = 21
        server._projection_diagnostic_paths_printed = False
        personalized, average, state = (
            server._sign_personalized_update_for_layer(
                "layer",
                [{"layer": update} for update in updates],
                alpha,
                updates[0].shape,
                start_param_dicts=[
                    {"layer": start_weight}
                    for start_weight in start_weights
                ],
                log_diagnostics=False,
                console_diagnostics=False,
                group_renorm=False,
                norm_restore=True,
                mode_name="sign_projection_no_group_renorm",
                return_cross_layer_state=True,
                capture_cross_layer_diagnostics=capture_diagnostics,
            )
        )
        return server, personalized, average, state

    def test_model_guided_selection_formulas_and_candidate_filter(self):
        delta = torch.tensor(
            [[2.0, 1.0, 1e-10, 3.0]],
            dtype=torch.float64,
        )
        start = torch.tensor(
            [[1.0, 4.0, 100.0, 2.0]],
            dtype=torch.float64,
        )
        model_only = FedCLIP._select_model_guided_directions(
            delta,
            start,
            "model_only",
            extra_topk=1,
        )
        self.assertEqual(
            torch.nonzero(
                model_only["selected_direction_mask"][0],
                as_tuple=False,
            ).flatten().tolist(),
            [0, 1],
        )
        self.assertFalse(
            bool(model_only["tail_candidate_mask"][0, 2].item())
        )

        joint_delta = torch.tensor(
            [[1.0, 5.0, 2.0, 4.0]],
            dtype=torch.float64,
        )
        joint_start = torch.tensor(
            [[1.0, 1.0, 4.0, 1.5]],
            dtype=torch.float64,
        )
        joint = FedCLIP._select_model_guided_directions(
            joint_delta,
            joint_start,
            "model_delta_joint",
            extra_topk=1,
        )
        expected_product = (
            joint["delta_energy_ratio"] * joint["model_energy_ratio"]
        )
        expected_tail = int(torch.argmax(expected_product[0, 1:]).item()) + 1
        selected = torch.nonzero(
            joint["selected_direction_mask"][0],
            as_tuple=False,
        ).flatten().tolist()
        self.assertEqual(selected, [0, expected_tail])
        self.assertLessEqual(len(selected), 2)

    def test_start_weight_snapshot_uses_recovered_effective_weight(self):
        start_model = torch.nn.Linear(2, 2, bias=False)
        with torch.no_grad():
            start_model.weight.fill_(3.0)
        server = FedCLIP.__new__(FedCLIP)
        server.client_start_full_weights = {}

        def recover_effective_weight(model):
            with torch.no_grad():
                model.weight.mul_(2.0)

        server._recover_if_needed = recover_effective_weight
        client = SimpleNamespace(
            id=4,
            role="Client_4",
            save_folder_name="memory",
        )
        method_globals = FedCLIP._snapshot_client_start_full_weights.__globals__
        with mock.patch.dict(
            method_globals,
            {"load_item": lambda *args, **kwargs: start_model},
        ):
            server._snapshot_client_start_full_weights(client)

        torch.testing.assert_close(
            server.client_start_full_weights[4]["weight"],
            torch.full((2, 2), 6.0),
            rtol=0.0,
            atol=0.0,
        )
        with torch.no_grad():
            start_model.weight.fill_(99.0)
        self.assertTrue(
            bool(
                (
                    server.client_start_full_weights[4]["weight"] == 6.0
                ).all()
            )
        )

    def test_model_guided_selection_always_keeps_only_u1_without_tail(self):
        delta = torch.tensor(
            [[0.0, 1e-12, -1e-12], [2.0, 0.0, 1e-11]],
            dtype=torch.float64,
        )
        start = torch.tensor(
            [[0.0, 100.0, 200.0], [3.0, 4.0, 5.0]],
            dtype=torch.float64,
        )
        for mode in ("model_only", "model_delta_joint"):
            with self.subTest(mode=mode):
                result = FedCLIP._select_model_guided_directions(
                    delta,
                    start,
                    mode,
                    extra_topk=5,
                )
                self.assertTrue(
                    bool(result["selected_direction_mask"][:, 0].all())
                )
                torch.testing.assert_close(
                    result["selected_direction_counts"],
                    torch.ones(2, dtype=torch.long),
                    rtol=0.0,
                    atol=0.0,
                )

    def test_model_guided_selection_and_reconstruction_are_sign_flip_invariant(self):
        delta = torch.tensor(
            [[2.0, -3.0, 1.0], [-1.0, 4.0, -2.0]],
            dtype=torch.float64,
        )
        start = torch.tensor(
            [[1.0, 5.0, -2.0], [3.0, -1.0, 4.0]],
            dtype=torch.float64,
        )
        signs = torch.tensor([[-1.0, 1.0, -1.0]], dtype=torch.float64)
        alpha = torch.tensor([0.25, 0.75], dtype=torch.float64)
        directions = torch.eye(3, dtype=torch.float64)

        for mode in ("model_only", "model_delta_joint"):
            with self.subTest(mode=mode):
                original = FedCLIP._select_model_guided_directions(
                    delta,
                    start,
                    mode,
                    extra_topk=1,
                )
                flipped = FedCLIP._select_model_guided_directions(
                    delta * signs,
                    start * signs,
                    mode,
                    extra_topk=1,
                )
                torch.testing.assert_close(
                    original["selected_direction_mask"],
                    flipped["selected_direction_mask"],
                    rtol=0.0,
                    atol=0.0,
                )

                def reconstruct(coefficients, basis, selected_mask):
                    target_coefficients = []
                    for target_idx in range(coefficients.shape[0]):
                        same_sign = (
                            coefficients[target_idx].unsqueeze(0)
                            * coefficients
                        ) > 0
                        aggregated = (
                            same_sign.to(coefficients.dtype)
                            * alpha.unsqueeze(1)
                            * coefficients
                        ).sum(dim=0)
                        target_coefficients.append(
                            (aggregated * selected_mask[target_idx]) @ basis
                        )
                    return torch.stack(target_coefficients)

                reconstruction = reconstruct(
                    delta,
                    directions,
                    original["selected_direction_mask"],
                )
                flipped_reconstruction = reconstruct(
                    delta * signs,
                    directions * signs.t(),
                    flipped["selected_direction_mask"],
                )
                torch.testing.assert_close(
                    reconstruction,
                    flipped_reconstruction,
                    rtol=0.0,
                    atol=1e-12,
                )

    def test_model_guided_integration_uses_start_weights_and_logs_metrics(self):
        updates, alpha = self.orthogonal_layer_inputs()
        start_weights = [
            torch.tensor([1.0, 8.0, 2.0, 3.0], dtype=updates[0].dtype),
            torch.tensor([2.0, 1.0, 7.0, 3.0], dtype=updates[0].dtype),
            torch.tensor([3.0, 2.0, 1.0, 9.0], dtype=updates[0].dtype),
            torch.tensor([4.0, 6.0, 2.0, 1.0], dtype=updates[0].dtype),
        ]
        routed = self.run_diagnostic_layer(
            updates=updates,
            alpha=alpha,
            start_weights=start_weights,
            direction_selection_mode="model_only",
            extra_topk=1,
            m_filter_mode="dominant_side",
            conflict_handling="self",
        )
        baseline = self.run_diagnostic_layer(
            updates=updates,
            alpha=alpha,
            start_weights=start_weights,
            direction_selection_mode="model_only",
            extra_topk=1,
            m_filter_mode="none",
            conflict_handling="zero",
        )
        client_rows = routed[2]
        direction_rows = routed[3]
        for routed_value, baseline_value in zip(routed[0], baseline[0]):
            torch.testing.assert_close(
                routed_value,
                baseline_value,
                rtol=0.0,
                atol=0.0,
            )
        self.assertTrue(all(row["fallback_used"] == "0" for row in client_rows))
        self.assertTrue(all(row["u1_selected"] == "1" for row in client_rows))
        self.assertTrue(all(row["selected_count"] in {"1", "2"} for row in client_rows))
        self.assertTrue(all(row["selected_tail_count"] in {"0", "1"} for row in client_rows))
        self.assertTrue(all(row["personalized_direction_selection_mode"] == "model_only" for row in direction_rows))
        self.assertTrue(all(row["q_i_k"] != "" for row in direction_rows))
        for row in direction_rows:
            client_idx = int(row["client_id"])
            direction_idx = int(row["direction_index"])
            self.assertAlmostEqual(
                abs(float(row["q_i_k"])),
                abs(float(start_weights[client_idx][direction_idx].item())),
                places=5,
            )
        self.assertTrue(all(row["dominance_ratio"] == "" for row in direction_rows))
        self.assertTrue(all(row["private_direction_count"] == "0" for row in client_rows))

    def test_explicit_delta_selection_mode_is_bitwise_compatible(self):
        implicit = self.run_diagnostic_layer(
            m_filter_mode="dominant_side",
            conflict_handling="self",
        )
        explicit = self.run_diagnostic_layer(
            m_filter_mode="dominant_side",
            conflict_handling="self",
            direction_selection_mode="delta",
        )
        for implicit_value, explicit_value in zip(implicit[0], explicit[0]):
            torch.testing.assert_close(
                implicit_value,
                explicit_value,
                rtol=0.0,
                atol=0.0,
            )
        torch.testing.assert_close(
            implicit[1],
            explicit[1],
            rtol=0.0,
            atol=0.0,
        )

    def test_cross_layer_mode_none_is_bitwise_compatible(self):
        implicit = self._run_scope_aggregation(None)["saved_models"]
        explicit = self._run_scope_aggregation(
            None,
            cross_layer_mode="none",
        )["saved_models"]
        self.assertEqual(set(implicit), set(explicit))
        for item_name in implicit:
            implicit_parameters = dict(
                implicit[item_name].named_parameters()
            )
            explicit_parameters = dict(
                explicit[item_name].named_parameters()
            )
            self.assertEqual(
                set(implicit_parameters),
                set(explicit_parameters),
            )
            for name in implicit_parameters:
                torch.testing.assert_close(
                    implicit_parameters[name],
                    explicit_parameters[name],
                    rtol=0.0,
                    atol=0.0,
                )

    def test_cross_layer_contribution_uses_tail_scaling_and_excludes_self(self):
        projections = torch.tensor(
            [
                [1.0, 1.0],
                [1.0, 2.0],
                [1.0, 3.0],
            ],
            dtype=torch.float64,
        )
        selected = torch.ones_like(projections, dtype=torch.bool)
        coefficients = torch.tensor(
            [
                [1.0, 2.3],
                [1.0, 2.3],
                [1.0, 2.3],
            ],
            dtype=torch.float64,
        )
        final_coefficients = coefficients.clone()
        final_coefficients[:, 1] *= 0.5
        alpha = torch.tensor([0.2, 0.3, 0.5], dtype=torch.float64)
        contribution = FedCLIP._cross_layer_client_contribution(
            projections,
            selected,
            coefficients,
            final_coefficients,
            alpha,
        )
        self.assertEqual(float(contribution[0, 0]), 0.0)
        self.assertAlmostEqual(float(contribution[0, 1]), 0.09)
        self.assertAlmostEqual(float(contribution[0, 2]), 0.5625)

    def test_cross_layer_scores_normalize_each_layer_before_sum(self):
        layer_contributions = [
            torch.tensor(
                [
                    [0.0, 1000.0, 1.0],
                    [0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0],
                ]
            ),
            torch.tensor(
                [
                    [0.0, 0.0, 1.0],
                    [0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0],
                ]
            ),
            torch.tensor(
                [
                    [0.0, 0.0, 1.0],
                    [0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0],
                ]
            ),
        ]
        result = FedCLIP._select_cross_layer_client_consensus(
            layer_contributions,
            topk=1,
        )
        self.assertGreater(
            float(result["cross_layer_scores"][0, 2]),
            float(result["cross_layer_scores"][0, 1]),
        )
        self.assertTrue(bool(result["selected_mask"][0, 2]))
        self.assertFalse(bool(result["selected_mask"][0, 1]))
        self.assertFalse(bool(torch.diag(result["selected_mask"]).any()))

    def test_cross_layer_consensus_does_not_fill_invalid_or_zero_candidates(self):
        one_valid = torch.tensor(
            [
                [0.0, 3.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
            ]
        )
        result = FedCLIP._select_cross_layer_client_consensus(
            [one_valid],
            topk=4,
        )
        self.assertEqual(
            torch.nonzero(
                result["selected_mask"][0],
                as_tuple=False,
            ).flatten().tolist(),
            [1],
        )
        all_zero = FedCLIP._select_cross_layer_client_consensus(
            [torch.zeros_like(one_valid)],
            topk=3,
        )
        self.assertFalse(bool(all_zero["selected_mask"].any()))
        self.assertTrue(
            bool(torch.isfinite(all_zero["cross_layer_scores"]).all())
        )

    def test_cross_layer_reconstruction_limits_tail_but_not_u1(self):
        projections = torch.tensor(
            [
                [1.0, 1.0],
                [1.0, 2.0],
                [1.0, 3.0],
            ],
            dtype=torch.float64,
        )
        alpha = torch.tensor([0.2, 0.3, 0.5], dtype=torch.float64)
        original_coefficients = torch.tensor(
            [[1.0, 2.3], [1.0, 2.3], [1.0, 2.3]],
            dtype=torch.float64,
        )
        state = {
            "name": "layer",
            "target_shape": torch.Size([2]),
            "direction_projections": projections,
            "selected_direction_mask": torch.ones_like(
                projections,
                dtype=torch.bool,
            ),
            "coefficient_mode_values_before": original_coefficients,
            "selected_output_coefficients_before": (
                original_coefficients.clone()
            ),
            "selected_unscaled_coefficients_before": (
                original_coefficients.clone()
            ),
            "target_strengths": torch.ones_like(projections),
            "alpha_tensor": alpha,
            "left_directions": torch.eye(2, dtype=torch.float64),
            "h": torch.tensor(
                [[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]],
                dtype=torch.float64,
            ),
            "sigma": torch.ones(2, dtype=torch.float64),
            "weighted_unit_vecs": [
                torch.tensor([1.0, 0.0], dtype=torch.float64),
                torch.tensor([0.0, 1.0], dtype=torch.float64),
                torch.tensor([0.0, 0.0], dtype=torch.float64),
            ],
            "average_delta": torch.tensor([1.0, 1.0], dtype=torch.float64),
            "group_renorm": False,
            "norm_restore": False,
            "gamma_max": 2.0,
            "personalized_g_scale": False,
            "personalized_tail_scale": 1.0,
            "contribution": torch.zeros((3, 3), dtype=torch.float64),
            "diagnostic_locals": None,
        }
        consensus_mask = torch.tensor(
            [
                [False, True, False],
                [False, False, True],
                [True, False, False],
            ]
        )
        server = FedCLIP.__new__(FedCLIP)
        reconstructed = (
            server._apply_cross_layer_client_consensus_to_layer(
                state,
                consensus_mask,
            )
        )
        self.assertEqual(float(reconstructed[0][0]), 1.0)
        self.assertAlmostEqual(float(reconstructed[0][1]), 0.6)

        empty_tail = (
            server._apply_cross_layer_client_consensus_to_layer(
                state,
                torch.zeros_like(consensus_mask),
            )
        )
        self.assertEqual(float(empty_tail[0][0]), 1.0)
        self.assertEqual(float(empty_tail[0][1]), 0.0)

    def test_cross_layer_collection_runs_for_both_model_modes(self):
        for mode in ("model_only", "model_delta_joint"):
            with self.subTest(mode=mode):
                _, personalized, _, state = (
                    self.collect_cross_layer_state(mode)
                )
                self.assertIsNotNone(state)
                self.assertEqual(len(personalized), 4)
                self.assertTrue(
                    bool(torch.isfinite(state["contribution"]).all())
                )

    def test_cross_layer_delta_collection_uses_personalized_direction_mask(self):
        _, personalized, _, state = self.collect_cross_layer_state("delta")
        self.assertIsNotNone(state)
        self.assertEqual(len(personalized), 4)
        self.assertEqual(
            state["selected_direction_mask"].sum(dim=1).tolist(),
            [2, 2, 2, 2],
        )
        expected = FedCLIP._cross_layer_client_contribution(
            state["direction_projections"],
            state["selected_direction_mask"],
            state["coefficient_mode_values_before"],
            state["selected_output_coefficients_before"],
            state["alpha_tensor"],
        )
        torch.testing.assert_close(
            state["contribution"],
            expected,
            rtol=0.0,
            atol=0.0,
        )

    def test_cross_layer_delta_zero_energy_fallback_has_zero_contribution(self):
        server = FedCLIP.__new__(FedCLIP)
        server.args = SimpleNamespace(
            aggregation_mode="sign_projection_no_group_renorm",
            personalized_rank_selection=1,
            personalized_rank_num=2,
            personalized_rank_force_u1=1,
            personalized_rank_mode="energy",
            personalized_rank_energy=0.8,
            personalized_direction_selection_mode="delta",
            personalized_extra_topk=1,
            personalized_g_scale=0,
            local_update_views=1,
            personalized_repeatability_threshold=-1.0,
            personalized_coeff_mode="same_sign",
            personalized_tail_scale=1.0,
            personalized_m_filter_mode="none",
            personalized_conflict_handling="zero",
            projection_energy=1.0,
            projection_k_max=2,
            projection_norm_scale_max=2.0,
        )
        server.device = torch.device("cpu")
        server.uploaded_ids = [0, 1]
        server.cur_ground = 21
        updates = [
            torch.tensor([1.0, 0.0], dtype=torch.float64),
            torch.zeros(2, dtype=torch.float64),
        ]
        personalized, average, state = (
            server._sign_personalized_update_for_layer(
                "layer",
                [{"layer": update} for update in updates],
                [0.5, 0.5],
                updates[0].shape,
                log_diagnostics=False,
                console_diagnostics=False,
                group_renorm=False,
                norm_restore=True,
                mode_name="sign_projection_no_group_renorm",
                return_cross_layer_state=True,
            )
        )
        self.assertTrue(bool(state["fallback_used"][1]))
        self.assertFalse(bool(state["selected_direction_mask"][1].any()))
        self.assertEqual(float(state["contribution"][1].sum()), 0.0)
        torch.testing.assert_close(
            personalized[1],
            average,
            rtol=0.0,
            atol=0.0,
        )

        consensus = FedCLIP._select_cross_layer_client_consensus(
            [state["contribution"]],
            topk=1,
        )
        constrained = server._apply_cross_layer_client_consensus_to_layer(
            state,
            consensus["selected_mask"],
        )
        torch.testing.assert_close(
            constrained[1],
            average,
            rtol=0.0,
            atol=0.0,
        )
        torch.testing.assert_close(
            constrained[1],
            personalized[1],
            rtol=0.0,
            atol=0.0,
        )

    def test_cross_layer_outer_two_stage_runs_for_both_model_modes(self):
        for mode in ("model_only", "model_delta_joint"):
            with self.subTest(mode=mode):
                result = self._run_scope_aggregation(
                    "low_rank",
                    cross_layer_mode="consensus_topk",
                    direction_selection_mode=mode,
                )
                self.assertIn("model", result["saved_models"])
                for model in result["saved_models"].values():
                    self.assertTrue(
                        all(
                            bool(torch.isfinite(parameter).all())
                            for parameter in model.parameters()
                        )
                    )

    def test_cross_layer_outer_two_stage_runs_for_delta(self):
        result = self._run_scope_aggregation(
            "low_rank",
            cross_layer_mode="consensus_topk",
            direction_selection_mode="delta",
        )
        self.assertIn("model", result["saved_models"])
        for model in result["saved_models"].values():
            self.assertTrue(
                all(
                    bool(torch.isfinite(parameter).all())
                    for parameter in model.parameters()
                )
            )

    def test_cross_layer_conflict_configurations_are_rejected(self):
        for m_filter_mode, conflict_handling in (
            ("dominant_side", "zero"),
            ("none", "self"),
        ):
            with self.subTest(
                m_filter_mode=m_filter_mode,
                conflict_handling=conflict_handling,
            ):
                server = FedCLIP.__new__(FedCLIP)
                server.uploaded_ids = [0, 1]
                server.num_clients = 2
                server.args = SimpleNamespace(
                    aggregation_mode="sign_projection_no_group_renorm",
                    personalized_rank_selection=1,
                    personalized_direction_selection_mode="model_only",
                    personalized_extra_topk=1,
                    personalized_cross_layer_client_mode="consensus_topk",
                    personalized_cross_layer_client_topk=1,
                    personalized_coeff_mode="same_sign",
                    personalized_m_filter_mode=m_filter_mode,
                    personalized_dominance_threshold=0.7,
                    personalized_conflict_handling=conflict_handling,
                    personalized_tail_scale=1.0,
                    personalized_rank_mode="fixed",
                    personalized_rank_num=2,
                    personalized_rank_force_u1=1,
                    personalized_g_scale=0,
                    local_update_views=1,
                    personalized_repeatability_threshold=-1.0,
                )
                with contextlib.redirect_stdout(io.StringIO()):
                    with self.assertRaisesRegex(
                        ValueError,
                        "consensus_topk",
                    ):
                        server.aggregate_sign_projection_no_group_renorm()

    def test_cross_layer_diagnostics_are_complete_and_finite(self):
        server, _, _, state = self.collect_cross_layer_state(
            "model_only",
            capture_diagnostics=True,
        )
        consensus = FedCLIP._select_cross_layer_client_consensus(
            [state["contribution"], state["contribution"] * 0.5],
            topk=2,
        )
        with tempfile.TemporaryDirectory() as temporary_directory:
            server.projection_client_diagnostic_csv = str(
                Path(temporary_directory) / "clients.csv"
            )
            server.projection_direction_diagnostic_csv = str(
                Path(temporary_directory) / "directions.csv"
            )
            server.projection_cross_layer_client_diagnostic_csv = str(
                Path(temporary_directory) / "cross_layer.csv"
            )
            server._projection_diagnostic_paths_printed = False
            server._cross_layer_diagnostic_path_printed = False
            server._apply_cross_layer_client_consensus_to_layer(
                state,
                consensus["selected_mask"],
            )
            with contextlib.redirect_stdout(io.StringIO()):
                server._emit_cross_layer_layer_diagnostics(
                    state["diagnostic_locals"]
                )
                server._write_cross_layer_client_diagnostics(
                    consensus,
                    topk=2,
                )
            with open(
                server.projection_direction_diagnostic_csv,
                newline="",
                encoding="utf-8",
            ) as file:
                direction_rows = list(csv.DictReader(file))
            with open(
                server.projection_cross_layer_client_diagnostic_csv,
                newline="",
                encoding="utf-8",
            ) as file:
                cross_layer_rows = list(csv.DictReader(file))

        required_direction_fields = {
            "source_client_allowed_by_cross_layer_consensus",
            "num_same_sign_clients_before_consensus",
            "num_same_sign_clients_after_consensus",
            "tail_coeff_before_consensus",
            "tail_coeff_after_consensus",
        }
        self.assertTrue(direction_rows)
        self.assertTrue(
            required_direction_fields.issubset(direction_rows[0])
        )
        for row in direction_rows:
            self.assertEqual(
                row["personalized_cross_layer_client_mode"],
                "consensus_topk",
            )
            for field in (
                "num_same_sign_clients_before_consensus",
                "num_same_sign_clients_after_consensus",
                "tail_coeff_before_consensus",
                "tail_coeff_after_consensus",
            ):
                self.assertTrue(math.isfinite(float(row[field])))
            if row["direction_index"] == "0":
                self.assertEqual(
                    row["num_same_sign_clients_before_consensus"],
                    row["num_same_sign_clients_after_consensus"],
                )
                self.assertAlmostEqual(
                    float(row["tail_coeff_before_consensus"]),
                    float(row["tail_coeff_after_consensus"]),
                )

        required_cross_fields = {
            "round",
            "target_client",
            "source_client",
            "cross_layer_score",
            "selected_in_consensus",
            "consensus_rank",
            "consensus_topk",
            "valid_layer_count",
            "num_unique_layer_top1_clients",
            "mean_pairwise_layer_cosine",
            "mean_pairwise_layer_top3_jaccard",
            "num_layers_with_nonzero_tail",
            "consensus_client_ids",
        }
        self.assertEqual(len(cross_layer_rows), 16)
        self.assertTrue(
            required_cross_fields.issubset(cross_layer_rows[0])
        )
        for row in cross_layer_rows:
            for field in (
                "cross_layer_score",
                "mean_pairwise_layer_cosine",
                "mean_pairwise_layer_top3_jaccard",
            ):
                self.assertTrue(math.isfinite(float(row[field])))

    def test_repeatability_formula_identical_zero_and_opposite(self):
        identical = FedCLIP._direction_repeatability(
            torch.tensor([1.0, -2.0]),
            torch.tensor([1.0, -2.0]),
        )
        torch.testing.assert_close(
            identical,
            torch.ones_like(identical),
            rtol=0.0,
            atol=1e-6,
        )

        zero_or_orthogonal = FedCLIP._direction_repeatability(
            torch.tensor([1.0, 0.0]),
            torch.tensor([0.0, 1.0]),
        )
        torch.testing.assert_close(
            zero_or_orthogonal,
            torch.zeros_like(zero_or_orthogonal),
            rtol=0.0,
            atol=0.0,
        )

        opposite = FedCLIP._direction_repeatability(
            torch.tensor([1.0, -2.0]),
            torch.tensor([-1.0, 2.0]),
        )
        self.assertTrue(bool((opposite < 0.0).all()))
        self.assertTrue(bool(torch.isfinite(opposite).all()))

        extreme = FedCLIP._direction_repeatability(
            torch.tensor([1e30, 1e-30, -1e30], dtype=torch.float32),
            torch.tensor([1e30, 0.0, 1e30], dtype=torch.float32),
        )
        self.assertTrue(bool(torch.isfinite(extreme).all()))
        self.assertGreater(float(extreme[0]), 0.999)
        self.assertEqual(float(extreme[1]), 0.0)
        self.assertLess(float(extreme[2]), -0.999)

    def test_dominant_side_filter_matches_weighted_energy_formula(self):
        projections = torch.tensor(
            [
                [2.0, -1.0, 1e-10],
                [1.0, 3.0, -1e-10],
                [-4.0, -2.0, 0.0],
            ],
            dtype=torch.float64,
        )
        alpha = torch.tensor([0.2, 0.3, 0.5], dtype=torch.float64)
        raw_mask = torch.ones_like(projections, dtype=torch.bool)
        (
            positive_energy,
            negative_energy,
            dominance_ratio,
            dominant_sign,
            keep_mask,
            balanced_mask,
            weak_side_mask,
        ) = FedCLIP._dominant_side_filter(
            projections,
            alpha,
            raw_mask,
            dominance_threshold=0.7,
            sign_epsilon=1e-8,
        )

        torch.testing.assert_close(
            positive_energy,
            torch.tensor([1.1, 2.7, 0.0], dtype=torch.float64),
        )
        torch.testing.assert_close(
            negative_energy,
            torch.tensor([8.0, 2.2, 0.0], dtype=torch.float64),
        )
        torch.testing.assert_close(
            dominance_ratio[:2],
            torch.tensor([8.0 / 9.1, 2.7 / 4.9], dtype=torch.float64),
        )
        self.assertEqual(dominant_sign.tolist(), [-1, 1, 0])
        self.assertEqual(keep_mask[:, 0].tolist(), [False, False, True])
        self.assertFalse(bool(keep_mask[:, 1:].any()))
        self.assertTrue(bool(balanced_mask[:, 1:].all()))
        self.assertEqual(weak_side_mask[:, 0].tolist(), [True, True, False])
        self.assertTrue(bool(torch.isfinite(dominance_ratio).all()))

    def test_dominant_side_filter_balanced_and_near_zero_are_not_kept(self):
        projections = torch.tensor(
            [[1.0, 1e-10], [-1.0, -1e-10]],
            dtype=torch.float64,
        )
        raw_mask = torch.ones_like(projections, dtype=torch.bool)
        result = FedCLIP._dominant_side_filter(
            projections,
            torch.tensor([0.5, 0.5], dtype=torch.float64),
            raw_mask,
            dominance_threshold=0.7,
            sign_epsilon=1e-8,
        )
        _, _, ratio, dominant_sign, keep, balanced, weak = result
        self.assertFalse(bool(keep.any()))
        self.assertTrue(bool(balanced.all()))
        self.assertFalse(bool(weak.any()))
        self.assertEqual(dominant_sign.tolist(), [0, 0])
        self.assertAlmostEqual(float(ratio[0]), 0.5)
        self.assertEqual(float(ratio[1]), 0.0)

    def test_dominant_side_equal_threshold_keeps_only_dominant_client(self):
        projections = torch.tensor([[1.0], [-2.0]], dtype=torch.float32)
        raw_mask = torch.ones_like(projections, dtype=torch.bool)
        threshold = float(torch.tensor(0.8, dtype=torch.float32).item())
        result = FedCLIP._dominant_side_filter(
            projections,
            torch.tensor([0.5, 0.5], dtype=torch.float32),
            raw_mask,
            dominance_threshold=threshold,
        )
        _, _, ratio, dominant_sign, keep, balanced, weak = result
        self.assertEqual(float(ratio[0]), threshold)
        self.assertEqual(dominant_sign.tolist(), [-1])
        self.assertEqual(keep[:, 0].tolist(), [False, True])
        self.assertFalse(bool(balanced.any()))
        self.assertEqual(weak[:, 0].tolist(), [True, False])

    def test_dominance_threshold_validation_is_strict_only_when_enabled(self):
        server = FedCLIP.__new__(FedCLIP)
        for threshold in (0.5, 0.0, -0.1, 1.0001):
            with self.subTest(mode="dominant_side", threshold=threshold):
                server.args = SimpleNamespace(
                    personalized_m_filter_mode="dominant_side",
                    personalized_dominance_threshold=threshold,
                )
                with self.assertRaisesRegex(ValueError, "0.5 < threshold <= 1.0"):
                    server._personalized_dominance_threshold()

        for threshold in (0.500001, 0.6, 0.7, 0.8, 1.0):
            with self.subTest(mode="dominant_side", threshold=threshold):
                server.args = SimpleNamespace(
                    personalized_m_filter_mode="dominant_side",
                    personalized_dominance_threshold=threshold,
                )
                self.assertEqual(
                    server._personalized_dominance_threshold(),
                    threshold,
                )

        server.args = SimpleNamespace(
            personalized_m_filter_mode="none",
            personalized_dominance_threshold=0.5,
        )
        self.assertEqual(server._personalized_dominance_threshold(), 0.5)

    def test_conflict_balanced_direction_zero_vs_self(self):
        updates = [torch.tensor([1.0, 0.0]), torch.tensor([-1.0, 0.0])]
        common = dict(
            updates=updates,
            alpha=[0.5, 0.5],
            rank_mode="energy",
            energy_threshold=0.8,
            projection_k_max=1,
            g_scale=0,
            norm_restore=False,
            m_filter_mode="dominant_side",
            dominance_threshold=0.7,
        )
        zero_result = self.run_diagnostic_layer(
            **common,
            conflict_handling="zero",
        )
        self_result = self.run_diagnostic_layer(
            **common,
            conflict_handling="self",
        )

        self.assertTrue(
            all(torch.equal(value, torch.zeros(2)) for value in zero_result[0])
        )
        for actual, expected in zip(self_result[0], updates):
            torch.testing.assert_close(actual, expected, rtol=0.0, atol=1e-6)
        for row in self_result[2]:
            self.assertEqual(row["raw_direction_count"], "1")
            self.assertEqual(row["shared_direction_count"], "0")
            self.assertEqual(row["private_direction_count"], "1")
            self.assertEqual(row["zeroed_direction_count"], "0")
            self.assertEqual(row["fallback_used"], "0")
            self.assertEqual(row["dominance_empty_after_filter"], "1")
            self.assertAlmostEqual(float(row["shared_local_energy_ratio"]), 0.0)
            self.assertAlmostEqual(float(row["private_local_energy_ratio"]), 1.0)
            self.assertAlmostEqual(
                float(row["final_total_retained_energy_ratio"]),
                1.0,
            )
            self.assertAlmostEqual(float(row["shared_reconstruction_norm"]), 0.0)
            self.assertAlmostEqual(
                float(row["private_reconstruction_norm"]),
                1.0,
                places=6,
            )
            self.assertAlmostEqual(
                float(row["final_reconstruction_norm_before_restore"]),
                1.0,
                places=6,
            )
            self.assertTrue(
                math.isfinite(float(row["final_update_cosine_with_delta_avg"]))
            )
        private_rows = [
            row
            for row in self_result[3]
            if row["conflict_route"] == "private_self"
        ]
        self.assertEqual(len(private_rows), 2)
        for row in private_rows:
            self.assertAlmostEqual(
                float(row["conflict_routed_coeff_before_g"]),
                float(row["a_self"]),
            )

    def test_conflict_dominant_side_shared_and_weak_side_routes(self):
        updates = [torch.tensor([1.0, 0.0]), torch.tensor([-2.0, 0.0])]
        common = dict(
            updates=updates,
            alpha=[0.5, 0.5],
            rank_mode="energy",
            energy_threshold=0.8,
            projection_k_max=1,
            g_scale=0,
            norm_restore=False,
            m_filter_mode="dominant_side",
            dominance_threshold=0.7,
        )
        zero_result = self.run_diagnostic_layer(
            **common,
            conflict_handling="zero",
        )
        self_result = self.run_diagnostic_layer(
            **common,
            conflict_handling="self",
        )

        torch.testing.assert_close(zero_result[0][0], torch.zeros(2))
        torch.testing.assert_close(
            self_result[0][0],
            updates[0],
            rtol=0.0,
            atol=1e-6,
        )
        torch.testing.assert_close(
            zero_result[0][1],
            self_result[0][1],
            rtol=0.0,
            atol=0.0,
        )
        zero_rows = {int(row["client_id"]): row for row in zero_result[3]}
        self_rows = {int(row["client_id"]): row for row in self_result[3]}
        self.assertEqual(zero_rows[0]["conflict_route"], "zeroed")
        self.assertEqual(self_rows[0]["conflict_route"], "private_self")
        self.assertEqual(zero_rows[1]["conflict_route"], "shared")
        self.assertEqual(self_rows[1]["conflict_route"], "shared")
        for rows in (zero_rows, self_rows):
            self.assertAlmostEqual(
                float(rows[1]["conflict_routed_coeff_before_g"]),
                float(rows[1]["coeff_same_sign"]),
            )
        self.assertAlmostEqual(
            float(self_rows[0]["conflict_routed_coeff_before_g"]),
            float(self_rows[0]["a_self"]),
        )

    def test_conflict_self_does_not_keep_near_zero_coefficient(self):
        personalized, _, client_rows, direction_rows, _ = (
            self.run_diagnostic_layer(
                updates=[
                    torch.tensor([1e-10, 0.0]),
                    torch.tensor([-1.0, 0.0]),
                ],
                alpha=[0.5, 0.5],
                rank_mode="energy",
                energy_threshold=0.8,
                projection_k_max=1,
                g_scale=0,
                norm_restore=False,
                m_filter_mode="dominant_side",
                dominance_threshold=0.7,
                conflict_handling="self",
            )
        )
        torch.testing.assert_close(personalized[0], torch.zeros(2))
        client_zero = next(
            row for row in client_rows if int(row["client_id"]) == 0
        )
        self.assertEqual(client_zero["private_direction_count"], "0")
        self.assertEqual(client_zero["zeroed_direction_count"], "1")
        direction_zero = next(
            row for row in direction_rows if int(row["client_id"]) == 0
        )
        self.assertEqual(direction_zero["conflict_route"], "zeroed")

    def test_conflict_zero_default_is_bitwise_compatible(self):
        implicit = self.run_diagnostic_layer(
            m_filter_mode="dominant_side",
            conflict_handling=None,
        )
        explicit = self.run_diagnostic_layer(
            m_filter_mode="dominant_side",
            conflict_handling="zero",
        )
        self.assertTrue(torch.equal(implicit[1], explicit[1]))
        self.assertTrue(
            all(
                torch.equal(left, right)
                for left, right in zip(implicit[0], explicit[0])
            )
        )

    def test_conflict_self_is_ignored_when_m_filter_none(self):
        zero_result = self.run_diagnostic_layer(
            m_filter_mode="none",
            conflict_handling="zero",
        )
        self_result = self.run_diagnostic_layer(
            m_filter_mode="none",
            conflict_handling="self",
        )
        self.assertTrue(torch.equal(zero_result[1], self_result[1]))
        self.assertTrue(
            all(
                torch.equal(left, right)
                for left, right in zip(zero_result[0], self_result[0])
            )
        )

    def test_dominant_side_integration_filters_weak_client_without_refill(self):
        updates = [torch.tensor([1.0, 0.0]), torch.tensor([-2.0, 0.0])]
        personalized, average, client_rows, direction_rows, console = (
            self.run_diagnostic_layer(
                updates=updates,
                alpha=[0.5, 0.5],
                rank_mode="energy",
                energy_threshold=0.8,
                projection_k_max=1,
                g_scale=0,
                norm_restore=False,
                m_filter_mode="dominant_side",
                dominance_threshold=0.7,
            )
        )

        torch.testing.assert_close(average, torch.tensor([-0.5, 0.0]))
        torch.testing.assert_close(personalized[0], torch.zeros(2))
        self.assertGreater(float(torch.norm(personalized[1])), 0.0)
        rows = {int(row["client_id"]): row for row in client_rows}
        self.assertEqual(rows[0]["selected_count_raw"], "1")
        self.assertEqual(rows[0]["selected_count_after_m_filter"], "0")
        self.assertEqual(rows[0]["selected_count_after_filter"], "0")
        self.assertEqual(rows[0]["selection_count_monotonic_check"], "1")
        self.assertEqual(rows[1]["selected_count_after_m_filter"], "1")
        self.assertEqual(rows[0]["dominance_weak_side_filtered_count"], "1")
        self.assertEqual(rows[0]["fallback_used"], "0")
        self.assertEqual(rows[0]["dominance_empty_after_filter"], "1")
        self.assertAlmostEqual(float(rows[0]["retained_raw_local_energy_ratio"]), 0.0)
        self.assertAlmostEqual(float(rows[1]["retained_raw_local_energy_ratio"]), 1.0)
        self.assertEqual(
            rows[0]["retained_local_energy_ratio_in_range_check"],
            "1",
        )
        for row in rows.values():
            self.assertLessEqual(
                int(row["selected_count_after_filter"]),
                int(row["selected_count_raw"]),
            )
            self.assertGreaterEqual(float(row["retained_local_energy_ratio"]), 0.0)
            self.assertLessEqual(float(row["retained_local_energy_ratio"]), 1.0 + 1e-6)
        self.assertGreater(float(rows[0]["update_norm_before_m_filter"]), 0.0)
        self.assertEqual(
            float(rows[0]["update_norm_after_m_filter_before_restore"]),
            0.0,
        )
        self.assertTrue(
            math.isfinite(float(rows[0]["cosine_before_m_filter_with_delta_avg"]))
        )
        self.assertTrue(
            math.isfinite(float(rows[0]["cosine_after_m_filter_with_delta_avg"]))
        )
        first_direction = next(
            row
            for row in direction_rows
            if int(row["client_id"]) == 0
            and int(row["direction_index"]) == 0
        )
        self.assertAlmostEqual(float(first_direction["dominance_ratio"]), 0.8)
        self.assertEqual(first_direction["dominance_filtered_weak_side"], "1")
        self.assertEqual(first_direction["selected_after_m_filter"], "0")
        self.assertIn("m_filter_mode=dominant_side", console)

    def test_dominant_side_integration_filters_balanced_direction_to_zero(self):
        personalized, _, client_rows, direction_rows, _ = self.run_diagnostic_layer(
            updates=[torch.tensor([1.0, 0.0]), torch.tensor([-1.0, 0.0])],
            alpha=[0.5, 0.5],
            rank_mode="energy",
            energy_threshold=0.8,
            force_u1=True,
            projection_k_max=1,
            g_scale=0,
            norm_restore=True,
            m_filter_mode="dominant_side",
            dominance_threshold=0.7,
        )
        self.assertTrue(all(torch.equal(value, torch.zeros(2)) for value in personalized))
        self.assertTrue(all(row["fallback_used"] == "0" for row in client_rows))
        self.assertTrue(
            all(row["dominance_balanced_filtered_count"] == "1" for row in client_rows)
        )
        self.assertEqual(
            {row["layer_dominance_balanced_filtered_direction_count"] for row in direction_rows},
            {"1"},
        )

    def test_dominance_filter_default_none_is_bitwise_compatible(self):
        implicit = self.run_diagnostic_layer(m_filter_mode=None)
        explicit = self.run_diagnostic_layer(m_filter_mode="none")
        self.assertTrue(torch.equal(implicit[1], explicit[1]))
        self.assertTrue(
            all(
                torch.equal(left, right)
                for left, right in zip(implicit[0], explicit[0])
            )
        )

    def test_repeatability_filter_all_directions_falls_back_to_delta_avg(self):
        updates, alpha = self.orthogonal_layer_inputs()
        personalized, average, client_rows, direction_rows, _ = (
            self.run_diagnostic_layer(
                updates=updates,
                updates_b=[-update for update in updates],
                alpha=alpha,
                rank_mode="energy",
                energy_threshold=0.8,
                force_u1=False,
                repeatability_threshold=0.0,
            )
        )
        self.assertTrue(
            all(torch.equal(client_update, average) for client_update in personalized)
        )
        self.assertTrue(
            all(row["selected_count_after"] == "0" for row in client_rows)
        )
        self.assertTrue(all(row["fallback_used"] == "1" for row in client_rows))
        self.assertTrue(
            all(float(row["gamma_raw"]) == 1.0 for row in client_rows)
        )
        selected_rows = [
            row
            for row in direction_rows
            if int(row["selected_before_repeatability"]) == 1
        ]
        self.assertTrue(selected_rows)
        self.assertTrue(
            all(float(row["repeatability_normalized"]) < 0.0 for row in selected_rows)
        )
        self.assertTrue(
            all(int(row["selected_after_repeatability"]) == 0 for row in selected_rows)
        )

    def test_repeatability_threshold_default_is_exactly_compatible(self):
        updates, alpha = self.orthogonal_layer_inputs()
        single_view = self.run_diagnostic_layer(
            updates=updates,
            alpha=alpha,
            repeatability_threshold=-1.0,
        )
        dual_view = self.run_diagnostic_layer(
            updates=updates,
            updates_b=[-update for update in updates],
            alpha=alpha,
            repeatability_threshold=-1.0,
        )
        self.assertTrue(torch.equal(single_view[1], dual_view[1]))
        self.assertTrue(
            all(
                torch.equal(single_value, dual_value)
                for single_value, dual_value in zip(
                    single_view[0],
                    dual_view[0],
                )
            )
        )

    def test_zero_b_view_and_epsilon_guarded_diagnostics_remain_finite(self):
        updates, alpha = self.orthogonal_layer_inputs()
        personalized, average, client_rows, direction_rows, _ = (
            self.run_diagnostic_layer(
                updates=updates,
                updates_b=[torch.zeros_like(update) for update in updates],
                alpha=alpha,
                repeatability_threshold=-1.0,
                coeff_mode="avg",
            )
        )
        self.assertTrue(bool(torch.isfinite(average).all()))
        self.assertTrue(
            all(bool(torch.isfinite(update).all()) for update in personalized)
        )
        for row in direction_rows:
            for field in (
                "a_A_raw",
                "a_B_raw",
                "a_A_normalized",
                "a_B_normalized",
                "repeatability_raw",
                "repeatability_normalized",
                "coeff_same_sign",
                "coeff_self",
                "coeff_avg",
                "same_sign_over_self_abs",
                "same_sign_over_avg_abs",
                "final_coeff_before_restore",
                "final_coeff",
                "sign_amplification",
                "final_ratio",
            ):
                self.assertTrue(math.isfinite(float(row[field])), field)
        for row in client_rows:
            for field in (
                "energy_ratio_before",
                "energy_ratio_after",
                "update_norm_before_restore",
                "update_norm_after_restore",
                "cosine_with_delta_avg",
                "cosine_with_client_A",
                "gamma_raw",
                "gamma_used",
            ):
                self.assertTrue(math.isfinite(float(row[field])), field)

    def test_coefficient_modes_match_same_sign_self_and_avg_formulas(self):
        updates = [
            torch.tensor([1.2, -0.7, 0.3], dtype=torch.float64),
            torch.tensor([-0.4, 1.1, 0.8], dtype=torch.float64),
            torch.tensor([0.9, 0.2, -1.3], dtype=torch.float64),
        ]
        alpha = [0.2, 0.3, 0.5]
        results = {}
        for mode in ("same_sign", "self", "avg"):
            results[mode] = self.run_diagnostic_layer(
                rank_mode="fixed",
                rank_num=len(updates),
                force_u1=True,
                g_scale=0,
                coeff_mode=mode,
                norm_restore=False,
                updates=updates,
                alpha=alpha,
            )

        rows_by_mode = {
            mode: {
                (int(row["client_id"]), int(row["direction_index"])): row
                for row in result[3]
            }
            for mode, result in results.items()
        }
        client_ids = sorted({key[0] for key in rows_by_mode["same_sign"]})
        direction_ids = sorted({key[1] for key in rows_by_mode["same_sign"]})
        for direction_id in direction_ids:
            projections = [
                float(
                    rows_by_mode["same_sign"][(client_id, direction_id)][
                        "a_A_raw"
                    ]
                )
                for client_id in client_ids
            ]
            self.assertTrue(all(abs(value) > 1e-8 for value in projections))
            expected_avg = sum(
                weight * projection
                for weight, projection in zip(alpha, projections)
            )
            for client_position, client_id in enumerate(client_ids):
                key = (client_id, direction_id)
                same_row = rows_by_mode["same_sign"][key]
                self_row = rows_by_mode["self"][key]
                avg_row = rows_by_mode["avg"][key]
                target_projection = projections[client_position]
                expected_same_sign = sum(
                    weight * projection
                    for weight, projection in zip(alpha, projections)
                    if target_projection * projection > 0.0
                )
                self.assertAlmostEqual(
                    float(same_row["coeff_same_sign"]),
                    expected_same_sign,
                    places=6,
                )
                self.assertAlmostEqual(
                    float(self_row["coeff_self"]),
                    target_projection,
                    places=6,
                )
                self.assertAlmostEqual(
                    float(avg_row["coeff_avg"]),
                    expected_avg,
                    places=6,
                )
                self.assertAlmostEqual(
                    float(same_row["final_coeff_before_restore"]),
                    expected_same_sign,
                    places=6,
                )
                self.assertAlmostEqual(
                    float(self_row["final_coeff_before_restore"]),
                    target_projection,
                    places=6,
                )
                self.assertAlmostEqual(
                    float(avg_row["final_coeff_before_restore"]),
                    expected_avg,
                    places=6,
                )

    def test_tail_scale_zero_is_exact_k1_and_one_is_full_update(self):
        tail_zero = self.run_diagnostic_layer(
            rank_mode="fixed",
            rank_num=self.rank_r,
            force_u1=True,
            tail_scale=0.0,
        )
        k1 = self.run_diagnostic_layer(
            rank_mode="fixed",
            rank_num=1,
            force_u1=True,
            tail_scale=1.0,
        )
        self.assertTrue(
            all(
                torch.equal(tail_value, k1_value)
                for tail_value, k1_value in zip(tail_zero[0], k1[0])
            )
        )
        shared_k1 = self.run_diagnostic_layer(
            personalized_rank_selection=0,
            projection_k_max=1,
            tail_scale=1.0,
        )
        self.assertTrue(
            all(
                torch.equal(tail_value, shared_value)
                for tail_value, shared_value in zip(tail_zero[0], shared_k1[0])
            )
        )

        explicit_one = self.run_diagnostic_layer(
            rank_mode="fixed",
            rank_num=self.rank_r,
            force_u1=True,
            tail_scale=1.0,
        )
        default_one = self.run_diagnostic_layer(
            rank_mode="fixed",
            rank_num=self.rank_r,
            force_u1=True,
        )
        self.assertTrue(
            all(
                torch.equal(explicit_value, default_value)
                for explicit_value, default_value in zip(
                    explicit_one[0],
                    default_one[0],
                )
            )
        )
        self.assertTrue(
            any(
                not torch.equal(full_value, k1_value)
                for full_value, k1_value in zip(explicit_one[0], k1[0])
            )
        )
        quarter = self.run_diagnostic_layer(
            rank_mode="fixed",
            rank_num=self.rank_r,
            force_u1=True,
            tail_scale=0.25,
            norm_restore=False,
        )
        zero_without_restore = self.run_diagnostic_layer(
            rank_mode="fixed",
            rank_num=self.rank_r,
            force_u1=True,
            tail_scale=0.0,
            norm_restore=False,
        )
        one_without_restore = self.run_diagnostic_layer(
            rank_mode="fixed",
            rank_num=self.rank_r,
            force_u1=True,
            tail_scale=1.0,
            norm_restore=False,
        )
        for quarter_value, zero_value, one_value in zip(
            quarter[0],
            zero_without_restore[0],
            one_without_restore[0],
        ):
            expected_quarter = zero_value + 0.25 * (one_value - zero_value)
            torch.testing.assert_close(
                quarter_value,
                expected_quarter,
                rtol=1e-6,
                atol=1e-7,
            )
        for row in explicit_one[3]:
            self.assertEqual(float(row["tail_scale"]), 1.0)

    def test_projection_layer_scope_low_rank_keeps_legacy_names(self):
        server = FedCLIP.__new__(FedCLIP)
        server.args = SimpleNamespace()
        full_model = _ScopeFullModel()
        low_rank_model = _ScopeLowRankModel()
        expected = {"conv2.weight", "fc1.weight", "fc2.weight"}

        self.assertEqual(server._projection_layer_scope(), "low_rank")
        self.assertEqual(
            server._get_projectable_weight_names(
                full_model,
                low_rank_model,
                "low_rank",
            ),
            expected,
        )

    def test_projection_layer_scope_all_weight_selects_only_matrix_weights(self):
        server = FedCLIP.__new__(FedCLIP)
        names = server._get_projectable_weight_names(
            _ScopeFullModel(),
            _ScopeLowRankModel(),
            "all_weight",
        )
        self.assertEqual(
            names,
            {
                "conv1.weight",
                "conv2.weight",
                "fc1.weight",
                "fc2.weight",
                "classifier.weight",
            },
        )
        self.assertNotIn("conv1.bias", names)
        self.assertNotIn("classifier.bias", names)
        self.assertNotIn("layer_norm.weight", names)
        self.assertNotIn("layer_norm.bias", names)

    def test_projection_layer_scope_plus_classifier_excludes_first_conv(self):
        server = FedCLIP.__new__(FedCLIP)
        names = server._get_projectable_weight_names(
            _ScopeFullModel(),
            _ScopeLowRankModel(),
            "low_rank_plus_classifier",
        )
        self.assertEqual(
            names,
            {
                "conv2.weight",
                "fc1.weight",
                "fc2.weight",
                "classifier.weight",
            },
        )
        self.assertNotIn("conv1.weight", names)

    def test_all_weight_scope_uses_client_start_delta_and_adds_start_once(self):
        result = self._run_scope_aggregation("all_weight")
        saved = result["saved_models"]

        self.assertAlmostEqual(float(saved["model"].conv1.weight.item()), 102.0)
        self.assertAlmostEqual(
            float(saved["model"].classifier.weight.item()),
            142.0,
        )
        self.assertAlmostEqual(float(saved["model_0"].conv1.weight.item()), 3.0)
        self.assertAlmostEqual(float(saved["model_1"].conv1.weight.item()), 7.0)
        self.assertAlmostEqual(
            float(saved["model_0"].classifier.weight.item()),
            42.0,
        )
        self.assertAlmostEqual(
            float(saved["model_1"].classifier.weight.item()),
            52.0,
        )

    def test_plus_classifier_scope_keeps_first_conv_on_weighted_avg(self):
        result = self._run_scope_aggregation("low_rank_plus_classifier")
        saved = result["saved_models"]

        for item_name in ("model", "model_0", "model_1"):
            self.assertAlmostEqual(
                float(saved[item_name].conv1.weight.item()),
                6.0,
            )
        self.assertAlmostEqual(
            float(saved["model"].classifier.weight.item()),
            142.0,
        )
        self.assertAlmostEqual(
            float(saved["model_0"].classifier.weight.item()),
            42.0,
        )
        self.assertAlmostEqual(
            float(saved["model_1"].classifier.weight.item()),
            52.0,
        )

    def test_omitted_scope_is_bitwise_equal_to_explicit_low_rank(self):
        omitted = self._run_scope_aggregation(None)["saved_models"]
        explicit = self._run_scope_aggregation("low_rank")["saved_models"]

        self.assertEqual(set(omitted), set(explicit))
        for item_name in omitted:
            omitted_params = dict(omitted[item_name].named_parameters())
            explicit_params = dict(explicit[item_name].named_parameters())
            self.assertEqual(set(omitted_params), set(explicit_params))
            for name in omitted_params:
                self.assertTrue(
                    torch.equal(omitted_params[name], explicit_params[name]),
                    msg=f"{item_name}/{name} changed under the default scope",
                )

    def test_weight_mode_ignores_projection_layer_scope(self):
        server = FedCLIP.__new__(FedCLIP)
        server.args = SimpleNamespace(projection_layer_scope="all_weight")
        self.assertEqual(
            server._projection_layer_scope_for_mode(
                "sign_projection_weight",
                "weight",
            ),
            "low_rank",
        )

    def test_delta_scope_diagnostic_fields_are_explicit(self):
        _, _, client_rows, direction_rows, console = self.run_diagnostic_layer(
            projection_layer_scope="all_weight",
        )
        required_client_fields = {
            "layer_name",
            "projection_layer_scope",
            "projection_input_kind",
            "selected_count",
            "selected_direction_ids",
            "selected_energy_ratio",
            "singular_values",
            "singular_energy_ratios",
            "cumulative_energy",
            "norm_delta_avg",
            "projected_norm_before_restore",
            "projected_norm_after_restore",
            "final_to_delta_avg_norm_ratio",
            "cos_final_delta_avg",
            "gamma",
            "gamma_capped",
        }
        self.assertTrue(required_client_fields.issubset(client_rows[0]))
        self.assertEqual(client_rows[0]["projection_layer_scope"], "all_weight")
        self.assertEqual(client_rows[0]["projection_input_kind"], "delta")
        self.assertNotEqual(client_rows[0]["singular_values"], "")
        self.assertTrue(
            all(
                row["projection_layer_scope"] == "all_weight"
                for row in direction_rows
            )
        )
        self.assertIn("projection_layer_scope=all_weight", console)

    def test_weight_mode_identical_weights_return_identical_weight(self):
        weight = torch.tensor([1.5, -0.5, 2.0, 0.25])
        weights = [weight.clone() for _ in range(3)]
        personalized, average, client_rows, direction_rows, console = (
            self.run_diagnostic_layer(
                updates=weights,
                alpha=[0.2, 0.3, 0.5],
                rank_mode="fixed",
                rank_num=3,
                force_u1=True,
                g_scale=0,
                coeff_mode="same_sign",
                mode_name="sign_projection_weight",
                input_kind="weight",
            )
        )
        torch.testing.assert_close(average, weight, rtol=1e-6, atol=1e-6)
        for result in personalized:
            torch.testing.assert_close(result, weight, rtol=1e-6, atol=1e-6)
        self.assertTrue(client_rows)
        for row in client_rows:
            self.assertEqual(row["projection_input_kind"], "weight")
            self.assertEqual(row["average_reference_semantics"], "avg_weight")
            self.assertEqual(
                row["personalized_output_semantics"],
                "aggregated_weight",
            )
            self.assertEqual(row["norm_restore_reference"], "avg_weight")
            self.assertEqual(row["personalized_writeback_semantics"], "copy_absolute")
            self.assertEqual(row["global_writeback_semantics"], "copy_average_weight")
            self.assertEqual(row["start_weight_added"], "0")
            self.assertEqual(row["norm_delta_avg"], "")
            self.assertNotEqual(row["avg_weight_norm"], "")
            self.assertNotEqual(row["weight_norm"], "")
        self.assertTrue(
            all(row["projection_input_kind"] == "weight" for row in direction_rows)
        )
        self.assertIn("input_kind=weight", console)
        self.assertNotIn("norm_delta_avg=", console)

    def test_weight_mode_avg_coeff_full_rank_reconstructs_average_weight(self):
        weights = [
            torch.tensor([2.0, 0.5, -1.0]),
            torch.tensor([-0.25, 1.5, 0.75]),
            torch.tensor([0.5, -1.0, 2.5]),
        ]
        alpha = [0.2, 0.3, 0.5]
        expected_average = sum(
            weight * value for weight, value in zip(alpha, weights)
        )
        personalized, average, client_rows, _, _ = self.run_diagnostic_layer(
            updates=weights,
            alpha=alpha,
            rank_mode="fixed",
            rank_num=len(weights),
            force_u1=False,
            g_scale=0,
            coeff_mode="avg",
            norm_restore=False,
            mode_name="sign_projection_weight",
            input_kind="weight",
        )
        torch.testing.assert_close(
            average,
            expected_average,
            rtol=1e-6,
            atol=1e-6,
        )
        for result in personalized:
            torch.testing.assert_close(
                result,
                expected_average,
                rtol=1e-5,
                atol=1e-6,
            )
        self.assertTrue(
            all(
                int(row["selected_count"]) == int(row["rank_R"])
                for row in client_rows
            )
        )
        restored_personalized, _, _, _, _ = self.run_diagnostic_layer(
            updates=weights,
            alpha=alpha,
            rank_mode="fixed",
            rank_num=len(weights),
            force_u1=False,
            g_scale=0,
            coeff_mode="avg",
            norm_restore=True,
            mode_name="sign_projection_weight",
            input_kind="weight",
        )
        for result in restored_personalized:
            torch.testing.assert_close(
                result,
                expected_average,
                rtol=1e-5,
                atol=1e-6,
            )

    def test_weight_mode_self_and_same_sign_use_weight_projections(self):
        weights = [
            torch.tensor([1.2, -0.7, 0.3, 0.9]),
            torch.tensor([-0.4, 1.1, 0.8, -0.2]),
            torch.tensor([0.9, 0.2, -1.3, 0.6]),
        ]
        alpha = [0.2, 0.3, 0.5]
        self_result = self.run_diagnostic_layer(
            updates=weights,
            alpha=alpha,
            rank_mode="fixed",
            rank_num=len(weights),
            force_u1=False,
            g_scale=0,
            coeff_mode="self",
            norm_restore=False,
            mode_name="sign_projection_weight",
            input_kind="weight",
        )
        for reconstructed, expected in zip(self_result[0], weights):
            torch.testing.assert_close(
                reconstructed,
                expected,
                rtol=1e-5,
                atol=1e-6,
            )

        same_sign_result = self.run_diagnostic_layer(
            updates=weights,
            alpha=alpha,
            rank_mode="fixed",
            rank_num=len(weights),
            force_u1=False,
            g_scale=0,
            coeff_mode="same_sign",
            norm_restore=False,
            mode_name="sign_projection_weight",
            input_kind="weight",
        )
        rows = {
            (int(row["client_id"]), int(row["direction_index"])): row
            for row in same_sign_result[3]
        }
        client_ids = sorted({key[0] for key in rows})
        direction_ids = sorted({key[1] for key in rows})
        for direction_id in direction_ids:
            projections = [
                float(rows[(client_id, direction_id)]["a_A_weight"])
                for client_id in client_ids
            ]
            self.assertTrue(all(abs(value) > 1e-7 for value in projections))
            for client_position, client_id in enumerate(client_ids):
                target_projection = projections[client_position]
                expected_coefficient = sum(
                    coefficient * projection
                    for coefficient, projection in zip(alpha, projections)
                    if target_projection * projection > 0.0
                )
                row = rows[(client_id, direction_id)]
                self.assertAlmostEqual(
                    float(row["coeff_same_sign"]),
                    expected_coefficient,
                    places=6,
                )
                self.assertAlmostEqual(
                    float(row["final_coeff_before_restore"]),
                    expected_coefficient,
                    places=6,
                )

    def test_weight_mode_supports_energy_direction_selection(self):
        weights = [
            torch.tensor([2.0, 0.5, -1.0]),
            torch.tensor([-0.25, 1.5, 0.75]),
            torch.tensor([0.5, -1.0, 2.5]),
        ]
        _, _, client_rows, _, _ = self.run_diagnostic_layer(
            updates=weights,
            alpha=[0.2, 0.3, 0.5],
            rank_mode="energy",
            energy_threshold=0.75,
            force_u1=False,
            g_scale=0,
            coeff_mode="self",
            mode_name="sign_projection_weight",
            input_kind="weight",
        )
        for row in client_rows:
            self.assertEqual(row["personalized_rank_mode"], "energy")
            self.assertGreaterEqual(float(row["selected_energy_ratio"]), 0.75)
            self.assertGreaterEqual(int(row["selected_count"]), 1)

    def test_weight_mode_norm_restore_uses_average_weight_norm(self):
        weights = [
            torch.tensor([3.0, 1.0, 0.25]),
            torch.tensor([1.0, 2.0, 1.0]),
            torch.tensor([2.0, 0.5, 1.5]),
        ]
        alpha = [0.2, 0.3, 0.5]
        expected_average = sum(
            weight * value for weight, value in zip(alpha, weights)
        )
        _, average, client_rows, _, _ = self.run_diagnostic_layer(
            updates=weights,
            alpha=alpha,
            rank_mode="fixed",
            rank_num=1,
            force_u1=True,
            g_scale=0,
            coeff_mode="self",
            norm_restore=True,
            norm_scale_max=100.0,
            mode_name="sign_projection_weight",
            input_kind="weight",
        )
        expected_norm = float(torch.norm(expected_average).item())
        fake_starts = [
            torch.full_like(weight, 20.0 + client_index)
            for client_index, weight in enumerate(weights)
        ]
        wrong_average_delta = sum(
            coefficient * (weight - start)
            for coefficient, weight, start in zip(alpha, weights, fake_starts)
        )
        wrong_reference_norm = float(torch.norm(wrong_average_delta).item())
        torch.testing.assert_close(
            average,
            expected_average,
            rtol=1e-6,
            atol=1e-6,
        )
        for row in client_rows:
            norm_before = float(row["projected_weight_norm_before_restore"])
            self.assertGreater(norm_before, 1e-8)
            expected_gamma = expected_norm / (norm_before + 1e-12)
            wrong_gamma = wrong_reference_norm / (norm_before + 1e-12)
            self.assertAlmostEqual(float(row["gamma_raw"]), expected_gamma, places=5)
            self.assertNotAlmostEqual(float(row["gamma_raw"]), wrong_gamma, places=3)
            self.assertEqual(row["gamma_capped"], "0")
            self.assertAlmostEqual(
                float(row["projected_weight_norm_after_restore"]),
                expected_norm,
                places=5,
            )
            self.assertAlmostEqual(
                float(row["final_to_avg_weight_norm_ratio"]),
                1.0,
                places=5,
            )
            self.assertEqual(row["update_norm_before_restore"], "")
            self.assertEqual(row["update_norm_after_restore"], "")

    def test_explicit_delta_input_kind_is_bitwise_legacy_compatible(self):
        default_result = self.run_diagnostic_layer(
            rank_mode="fixed",
            rank_num=self.rank_r,
            force_u1=True,
        )
        explicit_result = self.run_diagnostic_layer(
            rank_mode="fixed",
            rank_num=self.rank_r,
            force_u1=True,
            input_kind="delta",
        )
        self.assertTrue(torch.equal(default_result[1], explicit_result[1]))
        self.assertTrue(
            all(
                torch.equal(default_value, explicit_value)
                for default_value, explicit_value in zip(
                    default_result[0],
                    explicit_result[0],
                )
            )
        )
        for row in explicit_result[2]:
            self.assertEqual(row["projection_input_kind"], "delta")
            self.assertEqual(row["average_reference_semantics"], "delta_avg")
            self.assertEqual(row["personalized_writeback_semantics"], "add_to_client_start")
            self.assertNotEqual(row["norm_delta_avg"], "")
            self.assertEqual(row["avg_weight_norm"], "")

    def test_delta_mode_outer_path_retains_legacy_additive_writeback(self):
        def make_model(weight, bias):
            model = torch.nn.Linear(2, 2, bias=True)
            with torch.no_grad():
                model.weight.copy_(weight)
                model.bias.copy_(bias)
            return model

        old_global_weight = torch.tensor(
            [[10.0, 11.0], [12.0, 13.0]]
        )
        old_global_bias = torch.tensor([7.0, 8.0])
        client_start_weight = torch.tensor(
            [[2.0, -3.0], [4.0, 1.0]]
        )
        client_delta = torch.tensor(
            [[0.5, -1.0], [2.0, 0.25]]
        )
        uploaded_weight = client_start_weight + client_delta
        uploaded_bias = torch.tensor([-2.0, 5.0])

        server = FedCLIP.__new__(FedCLIP)
        server.args = SimpleNamespace(
            aggregation_mode="sign_projection_no_group_renorm",
            personalized_rank_selection=1,
            personalized_rank_num=1,
            personalized_rank_force_u1=1,
            personalized_rank_mode="fixed",
            personalized_rank_energy=0.8,
            personalized_g_scale=1,
            local_update_views=1,
            personalized_repeatability_threshold=-1.0,
            personalized_coeff_mode="same_sign",
            personalized_tail_scale=1.0,
            projection_energy=1.0,
            projection_k_max=1,
            projection_norm_scale_max=2.0,
        )
        server.device = torch.device("cpu")
        server.role = "Server"
        server.save_folder_name = "memory"
        server.uploaded_ids = [0]
        server.uploaded_weights = [1.0]
        server.num_clients = 2
        server.cur_ground = 21
        server.clients = [
            SimpleNamespace(role=f"Client_{index}", save_folder_name="memory")
            for index in range(server.num_clients)
        ]
        server.personal_residuals = {}
        server.client_start_full_weights = {
            0: {"weight": client_start_weight.clone()}
        }
        server._recover_if_needed = lambda model: model
        server._projectable_weight_names_from_low_rank_model = (
            lambda model: {"weight"}
        )
        server._is_sign_projection_diagnostic_round = lambda: False

        global_model = make_model(old_global_weight, old_global_bias)
        uploaded_model = make_model(uploaded_weight, uploaded_bias)
        saved_models = {}

        def fake_load_item(role, item_name, item_path):
            if role == "Server" and item_name == "model":
                return copy.deepcopy(global_model)
            if role == "Client_0" and item_name == "model":
                return copy.deepcopy(uploaded_model)
            raise AssertionError(f"Unexpected load: {role}/{item_name}")

        def fake_save_item(item, role, item_name, item_path):
            saved_models[item_name] = copy.deepcopy(item)

        method_globals = FedCLIP._aggregate_sign_projection_variant.__globals__
        with contextlib.redirect_stdout(io.StringIO()), mock.patch.dict(
            method_globals,
            {"load_item": fake_load_item, "save_item": fake_save_item},
        ):
            server.aggregate_sign_projection_no_group_renorm()

        expected_global_weight = old_global_weight + client_delta
        torch.testing.assert_close(
            saved_models["model"].weight,
            expected_global_weight,
            rtol=1e-6,
            atol=1e-6,
        )
        torch.testing.assert_close(
            saved_models["model_0"].weight,
            uploaded_weight,
            rtol=1e-6,
            atol=1e-6,
        )
        torch.testing.assert_close(
            saved_models["model_1"].weight,
            expected_global_weight,
            rtol=1e-6,
            atol=1e-6,
        )
        for item_name in ("model", "model_0", "model_1"):
            torch.testing.assert_close(
                saved_models[item_name].bias,
                uploaded_bias,
                rtol=0.0,
                atol=0.0,
            )
        self.assertEqual(server.client_start_full_weights, {})

    def test_weight_mode_writeback_copies_absolute_weights(self):
        def make_model(weight, bias):
            model = torch.nn.Linear(2, 2, bias=True)
            with torch.no_grad():
                model.weight.copy_(weight)
                model.bias.copy_(bias)
            return model

        old_global_weight = torch.full((2, 2), 20.0)
        old_global_bias = torch.tensor([10.0, 11.0])
        client_weights = [
            torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
            torch.tensor([[-2.0, 1.0], [0.5, 3.0]]),
        ]
        client_biases = [torch.tensor([1.0, -1.0]), torch.tensor([3.0, 5.0])]
        alpha = [0.25, 0.75]
        expected_weight = sum(
            coefficient * weight
            for coefficient, weight in zip(alpha, client_weights)
        )
        expected_bias = sum(
            coefficient * bias
            for coefficient, bias in zip(alpha, client_biases)
        )
        expected_average_norm = torch.norm(expected_weight)
        expected_personalized_weights = []
        for weight in client_weights:
            gamma = torch.clamp(
                expected_average_norm / (torch.norm(weight) + 1e-12),
                max=2.0,
            )
            expected_personalized_weights.append(gamma * weight)

        server = FedCLIP.__new__(FedCLIP)
        server.args = SimpleNamespace(
            aggregation_mode="sign_projection_weight",
            personalized_rank_selection=1,
            personalized_rank_num=2,
            personalized_rank_force_u1=1,
            personalized_rank_mode="fixed",
            personalized_rank_energy=0.8,
            personalized_g_scale=0,
            local_update_views=1,
            personalized_repeatability_threshold=-1.0,
            personalized_coeff_mode="self",
            personalized_tail_scale=1.0,
            projection_energy=1.0,
            projection_k_max=2,
            projection_norm_scale_max=2.0,
        )
        server.device = torch.device("cpu")
        server.role = "Server"
        server.save_folder_name = "memory"
        server.uploaded_ids = [0, 1]
        server.uploaded_weights = alpha
        server.num_clients = 3
        server.cur_ground = 21
        server.clients = [
            SimpleNamespace(role=f"Client_{index}", save_folder_name="memory")
            for index in range(server.num_clients)
        ]
        server.personal_residuals = {}
        server.client_start_full_weights = {
            0: {"weight": torch.full((2, 2), 1000.0)},
            1: {"weight": torch.full((2, 2), -1000.0)},
        }
        server._recover_if_needed = lambda model: model
        server._projectable_weight_names_from_low_rank_model = (
            lambda model: {"weight"}
        )
        server._is_sign_projection_diagnostic_round = lambda: False

        global_model = make_model(old_global_weight, old_global_bias)
        uploaded_models = {
            "Client_0": make_model(client_weights[0], client_biases[0]),
            "Client_1": make_model(client_weights[1], client_biases[1]),
        }
        saved_models = {}

        def fake_load_item(role, item_name, item_path):
            if role == "Server" and item_name == "model":
                return copy.deepcopy(global_model)
            if role in uploaded_models and item_name == "model":
                return copy.deepcopy(uploaded_models[role])
            raise AssertionError(f"Unexpected load: {role}/{item_name}")

        def fake_save_item(item, role, item_name, item_path):
            saved_models[item_name] = copy.deepcopy(item)

        method_globals = FedCLIP._aggregate_sign_projection_variant.__globals__
        with contextlib.redirect_stdout(io.StringIO()), mock.patch.dict(
            method_globals,
            {"load_item": fake_load_item, "save_item": fake_save_item},
        ):
            server.aggregate_sign_projection_weight()

        torch.testing.assert_close(
            saved_models["model"].weight,
            expected_weight,
            rtol=1e-5,
            atol=1e-6,
        )
        torch.testing.assert_close(
            saved_models["model"].bias,
            expected_bias,
            rtol=0.0,
            atol=0.0,
        )
        self.assertFalse(
            torch.allclose(
                saved_models["model"].weight,
                old_global_weight + expected_weight,
            )
        )
        for client_id in range(server.num_clients):
            personalized_model = saved_models[f"model_{client_id}"]
            expected_personalized_weight = (
                expected_personalized_weights[client_id]
                if client_id in server.uploaded_ids
                else expected_weight
            )
            torch.testing.assert_close(
                personalized_model.weight,
                expected_personalized_weight,
                rtol=1e-5,
                atol=1e-6,
            )
            torch.testing.assert_close(
                personalized_model.bias,
                expected_bias,
                rtol=0.0,
                atol=0.0,
            )
        self.assertFalse(
            torch.allclose(
                saved_models["model_0"].weight,
                torch.full((2, 2), 1000.0)
                + expected_personalized_weights[0],
            )
        )

    def test_two_view_diagnostic_csv_contains_repeatability_fields(self):
        updates, alpha = self.orthogonal_layer_inputs()
        _, _, client_rows, direction_rows, _ = self.run_diagnostic_layer(
            updates=updates,
            updates_b=[update.clone() for update in updates],
            alpha=alpha,
            repeatability_threshold=-1.0,
        )
        required_direction_fields = {
            "round",
            "client_id",
            "layer_name",
            "direction_index",
            "singular_value",
            "selected_before_repeatability",
            "selected_after_repeatability",
            "client_energy_score",
            "energy_rank",
            "a_A_raw",
            "a_B_raw",
            "a_A_normalized",
            "a_B_normalized",
            "repeatability_raw",
            "repeatability_normalized",
            "coeff_same_sign",
            "coeff_self",
            "coeff_avg",
            "final_coeff",
            "tail_scale",
            "u1_selected",
        }
        required_client_fields = {
            "selected_count_before",
            "selected_count_after",
            "energy_ratio_before",
            "energy_ratio_after",
            "update_norm_before_restore",
            "update_norm_after_restore",
            "cosine_with_delta_avg",
            "cosine_with_client_A",
            "fallback_used",
        }
        self.assertTrue(required_direction_fields.issubset(direction_rows[0]))
        self.assertTrue(required_client_fields.issubset(client_rows[0]))
        self.assertTrue(
            all(row["a_B_raw"] != "" for row in direction_rows)
        )
        self.assertTrue(
            all(
                math.isfinite(float(row["repeatability_normalized"]))
                for row in direction_rows
            )
        )

    def test_projection_diagnostic_rounds_include_21_50_and_100(self):
        server = FedCLIP.__new__(FedCLIP)
        server.args = SimpleNamespace(
            aggregation_mode="sign_projection_no_group_renorm",
            personalized_rank_selection=1,
            projection_warmup_ratio=0.2,
        )
        server.global_rounds = 100
        for diagnostic_round in (21, 50, 100):
            server.cur_ground = diagnostic_round
            self.assertTrue(server._is_sign_projection_diagnostic_round())
        server.cur_ground = 49
        self.assertFalse(server._is_sign_projection_diagnostic_round())

    def test_energy_mode_clients_choose_different_direction_counts(self):
        mask, _, selected_counts, fallback = self.select_energy(0.8, False)
        self.assertEqual(selected_counts.tolist(), [2, 1, 2, 1])
        self.assertTrue(torch.equal(mask.sum(dim=1), selected_counts))
        self.assertFalse(bool(mask[0, 0]))
        self.assertFalse(bool(mask[2, 0]))
        self.assertFalse(bool(fallback.any()))

    def test_energy_mode_meets_threshold_and_uses_minimum_prefix(self):
        threshold = 0.8
        mask, scores, selected_counts, fallback = self.select_energy(
            threshold,
            False,
        )
        self.assertFalse(bool(fallback.any()))
        for client_idx in range(scores.shape[0]):
            total_score = scores[client_idx].sum()
            selected_score = scores[client_idx, mask[client_idx]].sum()
            self.assertGreaterEqual(
                float((selected_score / total_score).item()),
                threshold,
            )
            selected_count = int(selected_counts[client_idx].item())
            if selected_count > 1:
                order = torch.argsort(
                    scores[client_idx],
                    descending=True,
                    stable=True,
                )
                previous_score = scores[
                    client_idx,
                    order[:selected_count - 1],
                ].sum()
                self.assertLess(
                    float((previous_score / total_score).item()),
                    threshold,
                )

    def test_energy_tau_one_stops_after_last_positive_score(self):
        mask, scores, selected_counts, fallback = self.select_energy(1.0, False)
        self.assertEqual(selected_counts.tolist(), [2, 1, 2, 1])
        self.assertFalse(bool(fallback.any()))
        for client_idx in range(scores.shape[0]):
            selected_score = scores[client_idx, mask[client_idx]].sum()
            self.assertAlmostEqual(
                float((selected_score / scores[client_idx].sum()).item()),
                1.0,
            )

    def test_energy_mode_has_no_ten_direction_cap(self):
        hadamard = torch.ones((1, 1), dtype=torch.float64)
        while hadamard.shape[0] < 16:
            hadamard = torch.cat(
                (
                    torch.cat((hadamard, hadamard), dim=1),
                    torch.cat((hadamard, -hadamard), dim=1),
                ),
                dim=0,
            )
        hadamard = hadamard / math.sqrt(16.0)
        eigvals = torch.ones(16, dtype=torch.float64)
        mask, _, selected_counts, fallback = self.select_energy(
            0.7,
            False,
            eigvals=eigvals,
            eigvecs=hadamard,
            rank_r=16,
        )
        self.assertTrue(torch.equal(selected_counts, torch.full((16,), 12)))
        self.assertTrue(torch.equal(mask.sum(dim=1), selected_counts))
        self.assertFalse(bool(fallback.any()))

    def test_energy_force_u1_counts_u1_toward_threshold(self):
        scores = torch.tensor([0.29, 0.40, 0.31], dtype=torch.float64)
        eigvals = torch.ones(3, dtype=torch.float64)
        eigvecs = torch.sqrt(scores).unsqueeze(0)
        mask, returned_scores, selected_counts, fallback = self.select_energy(
            0.68,
            True,
            eigvals=eigvals,
            eigvecs=eigvecs,
            rank_r=3,
        )
        self.assertEqual(selected_counts.tolist(), [2])
        self.assertEqual(
            torch.nonzero(mask[0], as_tuple=False).flatten().tolist(),
            [0, 1],
        )
        self.assertAlmostEqual(
            float(returned_scores[0, mask[0]].sum().item()),
            0.69,
        )
        self.assertFalse(bool(fallback.any()))

    def test_personalized_g_scale_switches_only_output_coefficient(self):
        g_enabled = self.run_diagnostic_layer(
            rank_mode="energy",
            g_scale=1,
            norm_restore=False,
        )
        g_disabled = self.run_diagnostic_layer(
            rank_mode="energy",
            g_scale=0,
            norm_restore=False,
        )
        enabled_rows = {
            (int(row["client_id"]), int(row["direction_id_0based"])): row
            for row in g_enabled[3]
        }
        disabled_rows = {
            (int(row["client_id"]), int(row["direction_id_0based"])): row
            for row in g_disabled[3]
        }
        self.assertEqual(enabled_rows.keys(), disabled_rows.keys())
        found_nontrivial_g = False
        for key, enabled_row in enabled_rows.items():
            disabled_row = disabled_rows[key]
            self.assertEqual(
                enabled_row["selected_by_client"],
                disabled_row["selected_by_client"],
            )
            if int(enabled_row["selected_by_client"]) == 0:
                continue
            self.assertAlmostEqual(
                float(enabled_row["output_coefficient_after_selection"]),
                float(enabled_row["g_times_b_after_selection"]),
                places=6,
            )
            self.assertAlmostEqual(
                float(disabled_row["output_coefficient_after_selection"]),
                float(disabled_row["b_sign"]),
                places=6,
            )
            self.assertAlmostEqual(
                float(enabled_row["coefficient_after_selection_and_restore"]),
                float(enabled_row["output_coefficient_after_selection"]),
                places=6,
            )
            self.assertAlmostEqual(
                float(disabled_row["coefficient_after_selection_and_restore"]),
                float(disabled_row["output_coefficient_after_selection"]),
                places=6,
            )
            g_value = float(enabled_row["g"])
            b_value = float(enabled_row["b_sign"])
            if 1e-4 < g_value < 1.0 - 1e-4 and abs(b_value) > 1e-6:
                found_nontrivial_g = True
                self.assertNotAlmostEqual(
                    float(enabled_row["output_coefficient_after_selection"]),
                    float(disabled_row["output_coefficient_after_selection"]),
                    places=6,
                )
        self.assertTrue(found_nontrivial_g)

    def test_zero_score_client_falls_back_to_exact_delta_avg(self):
        updates = [
            torch.zeros(3),
            torch.tensor([1.0, 0.0, 0.0]),
            torch.tensor([0.0, 1.0, 0.0]),
        ]
        personalized, average, client_rows, direction_rows, _ = (
            self.run_diagnostic_layer(
                rank_mode="energy",
                energy_threshold=0.8,
                g_scale=1,
                force_u1=False,
                updates=updates,
                alpha=[0.2, 0.4, 0.4],
            )
        )
        self.assertTrue(torch.equal(personalized[0], average))
        row = {int(item["client_id"]): item for item in client_rows}[0]
        self.assertEqual(row["selected_count"], "0")
        self.assertEqual(row["selected_direction_ids"], "")
        self.assertEqual(row["energy_threshold_met"], "0")
        self.assertEqual(row["zero_energy_fallback"], "1")
        self.assertEqual(float(row["gamma_raw"]), 1.0)
        self.assertEqual(float(row["gamma_used"]), 1.0)
        client_zero_direction_rows = [
            item for item in direction_rows if int(item["client_id"]) == 0
        ]
        self.assertTrue(
            all(int(item["selected_by_client"]) == 0 for item in client_zero_direction_rows)
        )
        for direction_row in client_zero_direction_rows:
            self.assertAlmostEqual(
                float(direction_row["output_coefficient_after_selection"]),
                float(direction_row["a_avg"]),
                places=6,
            )
            self.assertAlmostEqual(
                float(direction_row["coefficient_after_selection_and_restore"]),
                float(direction_row["a_avg"]),
                places=6,
            )

    def test_energy_csv_records_per_client_direction_counts(self):
        _, _, client_rows, direction_rows, console = self.run_diagnostic_layer(
            rank_mode="energy",
            energy_threshold=0.8,
            g_scale=1,
            force_u1=False,
            projection_k_max=1,
        )
        rows_by_client = {int(row["client_id"]): row for row in client_rows}
        actual_counts = [
            int(rows_by_client[client_id]["selected_count"])
            for client_id in range(4)
        ]
        self.assertEqual(actual_counts, [2, 1, 2, 1])
        self.assertEqual(len(set(actual_counts)), 2)
        for client_id, selected_count in enumerate(actual_counts):
            row = rows_by_client[client_id]
            selected_ids = (
                []
                if not row["selected_direction_ids"]
                else [int(value) for value in row["selected_direction_ids"].split(";")]
            )
            self.assertEqual(len(selected_ids), selected_count)
            self.assertEqual(row["personalized_rank_mode"], "energy")
            self.assertAlmostEqual(float(row["personalized_rank_energy"]), 0.8)
            self.assertEqual(row["personalized_g_scale"], "1")
            self.assertEqual(row["personalized_rank_num_requested"], "")
            self.assertEqual(int(row["personalized_rank_num_effective"]), selected_count)
            self.assertEqual(row["energy_threshold_met"], "1")
            self.assertEqual(row["zero_energy_fallback"], "0")
            self.assertGreaterEqual(float(row["selected_score_ratio"]), 0.8)
            self.assertTrue(math.isfinite(float(row["gamma_raw"])))
            self.assertTrue(math.isfinite(float(row["gamma_used"])))
            self.assertEqual(row["uniform_reference_kind"], "M_i")
            self.assertEqual(int(row["uniform_reference_size"]), selected_count)

            client_direction_rows = [
                item
                for item in direction_rows
                if int(item["client_id"]) == client_id
            ]
            self.assertEqual(len(client_direction_rows), self.rank_r)
            self.assertEqual(
                sum(int(item["selected_by_client"]) for item in client_direction_rows),
                selected_count,
            )
            self.assertTrue(
                all(
                    int(item["selected_count"]) == selected_count
                    for item in client_direction_rows
                )
            )
        self.assertIn("selected_count(min/mean/max)=1/1.500/2", console)
        self.assertIn("overlap_reference=uniform_top_M_i(per_client)", console)

    def test_default_fixed_g_scale_one_full_layer_is_unchanged(self):
        updates, alpha = self.orthogonal_layer_inputs()

        def aggregate(extra_args):
            server = FedCLIP.__new__(FedCLIP)
            base_args = {
                "aggregation_mode": "sign_projection_no_group_renorm",
                "personalized_rank_selection": 1,
                "personalized_rank_num": 2,
                "personalized_rank_force_u1": 1,
                "projection_energy": 0.8,
                "projection_k_max": 4,
                "projection_norm_scale_max": 2.0,
            }
            base_args.update(extra_args)
            server.args = SimpleNamespace(**base_args)
            server.device = torch.device("cpu")
            return server._sign_personalized_update_for_layer(
                "layer",
                [{"layer": update} for update in updates],
                alpha,
                updates[0].shape,
                group_renorm=False,
                norm_restore=True,
                mode_name="sign_projection_no_group_renorm",
            )

        default_personalized, default_average = aggregate({})
        explicit_personalized, explicit_average = aggregate({
            "personalized_rank_mode": "fixed",
            "personalized_rank_energy": 0.8,
            "personalized_g_scale": 1,
            "local_update_views": 1,
            "personalized_repeatability_threshold": -1.0,
            "personalized_coeff_mode": "same_sign",
            "personalized_tail_scale": 1.0,
            "personalized_m_filter_mode": "none",
            "personalized_dominance_threshold": 0.7,
        })
        self.assertTrue(torch.equal(default_average, explicit_average))
        self.assertTrue(
            all(
                torch.equal(default_value, explicit_value)
                for default_value, explicit_value in zip(
                    default_personalized,
                    explicit_personalized,
                )
            )
        )

    def test_force_u1_m2_keeps_u1_and_selects_best_remaining(self):
        mask, scores, selected_counts, fallback = self.select(2, True)
        expected = torch.tensor(
            [
                [True, True, False, False],
                [True, True, False, False],
                [True, True, False, False],
                [True, False, False, True],
            ]
        )
        self.assertTrue(torch.equal(mask, expected))
        self.assertTrue(torch.equal(selected_counts, torch.full((4,), 2)))
        self.assertFalse(bool(fallback.any()))
        torch.testing.assert_close(
            scores,
            self.eigvals.unsqueeze(0) * self.eigvecs.square(),
            rtol=0.0,
            atol=0.0,
        )

    def test_free_m2_can_exclude_u1(self):
        mask, _, selected_counts, fallback = self.select(2, False)
        expected = torch.tensor(
            [
                [False, True, True, False],
                [True, True, False, False],
                [False, True, True, False],
                [True, False, False, True],
            ]
        )
        self.assertTrue(torch.equal(mask, expected))
        self.assertFalse(bool(mask[0, 0]))
        self.assertFalse(bool(mask[2, 0]))
        self.assertTrue(torch.equal(selected_counts, torch.full((4,), 2)))
        self.assertFalse(bool(fallback.any()))

    def test_m1_force_and_free_modes(self):
        force_mask, _, _, _ = self.select(1, True)
        free_mask, _, _, _ = self.select(1, False)
        self.assertTrue(torch.equal(force_mask[:, 0], torch.ones(4, dtype=torch.bool)))
        self.assertTrue(torch.equal(force_mask.sum(dim=1), torch.ones(4, dtype=torch.long)))
        self.assertEqual(
            [int(index.item()) for index in torch.argmax(free_mask.to(torch.int64), dim=1)],
            [1, 0, 1, 3],
        )

    def test_m_at_least_rank_selects_every_direction(self):
        for force_u1 in (True, False):
            for rank_num in (self.rank_r, self.rank_r + 5):
                with self.subTest(force_u1=force_u1, rank_num=rank_num):
                    mask, _, selected_counts, fallback = self.select(
                        rank_num,
                        force_u1,
                    )
                    self.assertTrue(bool(mask.all()))
                    self.assertTrue(
                        torch.equal(
                            selected_counts,
                            torch.full((4,), self.rank_r),
                        )
                    )
                    self.assertFalse(bool(fallback.any()))

    def test_every_mask_row_has_exact_effective_count(self):
        for force_u1 in (True, False):
            for rank_num in (1, 2, 3, 4, 9):
                with self.subTest(force_u1=force_u1, rank_num=rank_num):
                    mask, scores, selected_counts, fallback = self.select(
                        rank_num,
                        force_u1,
                    )
                    self.assertEqual(mask.dtype, torch.bool)
                    self.assertTrue(bool(torch.isfinite(mask).all()))
                    self.assertTrue(bool(torch.isfinite(scores).all()))
                    expected_count = min(rank_num, self.rank_r)
                    self.assertTrue(
                        torch.equal(
                            selected_counts,
                            torch.full((4,), expected_count),
                        )
                    )
                    self.assertFalse(bool(fallback.any()))
                    self.assertTrue(
                        torch.equal(
                            mask.sum(dim=1),
                            torch.full((4,), expected_count, dtype=torch.long),
                        )
                    )
                    for row in mask:
                        selected_ids = torch.nonzero(row, as_tuple=False).flatten().tolist()
                        self.assertGreater(len(selected_ids), 0)
                        self.assertEqual(len(selected_ids), len(set(selected_ids)))
                        self.assertTrue(
                            all(0 <= direction_id < self.rank_r for direction_id in selected_ids)
                        )

    def test_default_force_u1_matches_pre_change_algorithm(self):
        server = FedCLIP.__new__(FedCLIP)
        server.args = SimpleNamespace()
        self.assertTrue(server._personalized_rank_force_u1())

        for rank_num in (1, 2, 3, 4, 9):
            with self.subTest(rank_num=rank_num):
                legacy_mask, legacy_scores, legacy_effective = self.legacy_select(
                    rank_num
                )
                default_mask, default_scores, default_counts, default_fallback = (
                    FedCLIP._select_personalized_directions(
                        self.eigvals,
                        self.eigvecs,
                        self.rank_r,
                        rank_num,
                    )
                )
                explicit_mask, explicit_scores, explicit_counts, explicit_fallback = self.select(
                    rank_num,
                    True,
                )
                self.assertTrue(torch.equal(default_mask, legacy_mask))
                self.assertTrue(torch.equal(explicit_mask, legacy_mask))
                self.assertTrue(torch.equal(default_scores, legacy_scores))
                self.assertTrue(torch.equal(explicit_scores, legacy_scores))
                expected_counts = torch.full((4,), legacy_effective)
                self.assertTrue(torch.equal(default_counts, expected_counts))
                self.assertTrue(torch.equal(explicit_counts, expected_counts))
                self.assertFalse(bool(default_fallback.any()))
                self.assertFalse(bool(explicit_fallback.any()))

    def test_cli_force_u1_parameter_defaults_to_one(self):
        tree = ast.parse(MAIN_PATH.read_text(encoding="utf-8"))
        matching_calls = []
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call) or not node.args:
                continue
            if not isinstance(node.func, ast.Attribute) or node.func.attr != "add_argument":
                continue
            first_argument = node.args[0]
            if (
                isinstance(first_argument, ast.Constant)
                and first_argument.value == "--personalized_rank_force_u1"
            ):
                matching_calls.append(node)

        self.assertEqual(len(matching_calls), 1)
        keywords = {keyword.arg: keyword.value for keyword in matching_calls[0].keywords}
        self.assertIsInstance(keywords["type"], ast.Name)
        self.assertEqual(keywords["type"].id, "int")
        self.assertEqual(ast.literal_eval(keywords["choices"]), [0, 1])
        self.assertEqual(ast.literal_eval(keywords["default"]), 1)

    def test_energy_mode_cli_parameters_have_compatible_defaults(self):
        tree = ast.parse(MAIN_PATH.read_text(encoding="utf-8"))
        arguments = {}
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call) or not node.args:
                continue
            if not isinstance(node.func, ast.Attribute) or node.func.attr != "add_argument":
                continue
            first_argument = node.args[0]
            if isinstance(first_argument, ast.Constant):
                arguments[first_argument.value] = {
                    keyword.arg: keyword.value for keyword in node.keywords
                }

        aggregation_mode = arguments["--aggregation_mode"]
        self.assertIn(
            "sign_projection_weight",
            ast.literal_eval(aggregation_mode["choices"]),
        )
        layer_scope = arguments["--projection_layer_scope"]
        self.assertEqual(
            ast.literal_eval(layer_scope["choices"]),
            ["low_rank", "low_rank_plus_classifier", "all_weight"],
        )
        self.assertEqual(ast.literal_eval(layer_scope["default"]), "low_rank")
        rank_mode = arguments["--personalized_rank_mode"]
        self.assertEqual(ast.literal_eval(rank_mode["choices"]), ["fixed", "energy"])
        self.assertEqual(ast.literal_eval(rank_mode["default"]), "fixed")
        rank_energy = arguments["--personalized_rank_energy"]
        self.assertEqual(ast.literal_eval(rank_energy["default"]), 0.8)
        direction_selection_mode = arguments[
            "--personalized_direction_selection_mode"
        ]
        self.assertEqual(
            ast.literal_eval(direction_selection_mode["choices"]),
            ["delta", "model_only", "model_delta_joint"],
        )
        self.assertEqual(
            ast.literal_eval(direction_selection_mode["default"]),
            "delta",
        )
        extra_topk = arguments["--personalized_extra_topk"]
        self.assertEqual(ast.literal_eval(extra_topk["default"]), 1)
        cross_layer_mode = arguments[
            "--personalized_cross_layer_client_mode"
        ]
        self.assertEqual(
            ast.literal_eval(cross_layer_mode["choices"]),
            ["none", "consensus_topk"],
        )
        self.assertEqual(
            ast.literal_eval(cross_layer_mode["default"]),
            "none",
        )
        cross_layer_topk = arguments[
            "--personalized_cross_layer_client_topk"
        ]
        self.assertEqual(
            ast.literal_eval(cross_layer_topk["default"]),
            5,
        )
        g_scale = arguments["--personalized_g_scale"]
        self.assertEqual(ast.literal_eval(g_scale["choices"]), [0, 1])
        self.assertEqual(ast.literal_eval(g_scale["default"]), 1)
        local_views = arguments["--local_update_views"]
        self.assertEqual(ast.literal_eval(local_views["choices"]), [1, 2])
        self.assertEqual(ast.literal_eval(local_views["default"]), 1)
        repeatability = arguments["--personalized_repeatability_threshold"]
        self.assertEqual(ast.literal_eval(repeatability["default"]), -1.0)
        coeff_mode = arguments["--personalized_coeff_mode"]
        self.assertEqual(
            ast.literal_eval(coeff_mode["choices"]),
            ["same_sign", "self", "avg"],
        )
        self.assertEqual(ast.literal_eval(coeff_mode["default"]), "same_sign")
        tail_scale = arguments["--personalized_tail_scale"]
        self.assertEqual(ast.literal_eval(tail_scale["default"]), 1.0)
        m_filter_mode = arguments["--personalized_m_filter_mode"]
        self.assertEqual(
            ast.literal_eval(m_filter_mode["choices"]),
            ["none", "dominant_side"],
        )
        self.assertEqual(ast.literal_eval(m_filter_mode["default"]), "none")
        dominance_threshold = arguments["--personalized_dominance_threshold"]
        self.assertEqual(ast.literal_eval(dominance_threshold["default"]), 0.7)
        dominance_help = ast.literal_eval(dominance_threshold["help"])
        for expected_text in (
            "0.5 < threshold <= 1",
            "P_k >= threshold",
            "dominant-sign side",
            "0.6/0.7/0.8",
            "40%/30%/20%",
        ):
            self.assertIn(expected_text, dominance_help)
        conflict_handling = arguments["--personalized_conflict_handling"]
        self.assertEqual(
            ast.literal_eval(conflict_handling["choices"]),
            ["zero", "self"],
        )
        self.assertEqual(ast.literal_eval(conflict_handling["default"]), "zero")

    def test_free_mode_diagnostics_use_uniform_top_m_reference(self):
        normalized_eigvals = self.eigvals / self.eigvals.sum()
        alpha_tensor = self.eigvecs.square() @ normalized_eigvals
        weighted_updates = (
            torch.diag(torch.sqrt(normalized_eigvals)) @ self.eigvecs.t()
        )
        updates = [
            weighted_updates[:, client_idx]
            / torch.sqrt(alpha_tensor[client_idx])
            for client_idx in range(self.eigvecs.shape[0])
        ]
        alpha = [float(value.item()) for value in alpha_tensor]

        server = FedCLIP.__new__(FedCLIP)
        server.args = SimpleNamespace(
            aggregation_mode="sign_projection_no_group_renorm",
            personalized_rank_selection=1,
            personalized_rank_num=2,
            personalized_rank_force_u1=0,
            projection_energy=1.0,
            projection_k_max=20,
            projection_norm_scale_max=2.0,
        )
        server.device = torch.device("cpu")
        server.uploaded_ids = [0, 1, 2, 3]
        server.cur_ground = 1
        server._projection_diagnostic_paths_printed = False

        with tempfile.TemporaryDirectory() as temporary_directory:
            server.projection_client_diagnostic_csv = str(
                Path(temporary_directory) / "clients.csv"
            )
            server.projection_direction_diagnostic_csv = str(
                Path(temporary_directory) / "directions.csv"
            )
            console_output = io.StringIO()
            with contextlib.redirect_stdout(console_output):
                personalized, _ = server._sign_personalized_update_for_layer(
                    "layer",
                    [{"layer": update} for update in updates],
                    alpha,
                    updates[0].shape,
                    log_diagnostics=True,
                    console_diagnostics=True,
                    group_renorm=False,
                    norm_restore=True,
                    mode_name="sign_projection_no_group_renorm",
                )

            with open(
                server.projection_client_diagnostic_csv,
                newline="",
                encoding="utf-8",
            ) as file:
                client_rows = list(csv.DictReader(file))
            with open(
                server.projection_direction_diagnostic_csv,
                newline="",
                encoding="utf-8",
            ) as file:
                direction_rows = list(csv.DictReader(file))

        self.assertTrue(all(torch.isfinite(update).all() for update in personalized))
        self.assertEqual(len(client_rows), 4)
        self.assertEqual(len(direction_rows), 16)
        rows_by_client = {int(row["client_id"]): row for row in client_rows}
        self.assertEqual(rows_by_client[0]["selected_direction_ids_0based"], "1;2")
        self.assertEqual(rows_by_client[0]["u1_selected"], "0")
        self.assertEqual(rows_by_client[0]["personalized_rank_force_u1"], "0")
        self.assertEqual(rows_by_client[0]["selected_K"], "4")
        self.assertEqual(rows_by_client[0]["uniform_reference_kind"], "M")
        self.assertEqual(rows_by_client[0]["uniform_reference_size"], "2")
        self.assertAlmostEqual(
            float(rows_by_client[0]["uniform_K_overlap_ratio"]),
            0.5,
        )
        self.assertAlmostEqual(
            float(rows_by_client[0]["uniform_M_overlap_ratio"]),
            0.5,
        )
        self.assertIn("personalized_rank_force_u1", direction_rows[0])
        self.assertEqual(direction_rows[0]["uniform_reference_kind"], "M")
        self.assertEqual(direction_rows[0]["uniform_reference_size"], "2")
        client_zero_direction_rows = sorted(
            (
                row
                for row in direction_rows
                if int(row["client_id"]) == 0
            ),
            key=lambda row: int(row["direction_id_0based"]),
        )
        client_zero_score_order = sorted(
            range(self.rank_r),
            key=lambda direction_id: (
                -float(client_zero_direction_rows[direction_id]["direction_score"]),
                direction_id,
            ),
        )
        expected_u1_rank = client_zero_score_order.index(0) + 1
        self.assertEqual(
            int(rows_by_client[0]["u1_score_rank_1based"]),
            expected_u1_rank,
        )
        u1_selection_rate = sum(
            int(row["u1_selected"]) for row in client_rows
        ) / len(client_rows)
        rendered_console = console_output.getvalue()
        self.assertIn("force_u1=False", rendered_console)
        self.assertIn(
            f"u1_selection_rate={u1_selection_rate:.6f}",
            rendered_console,
        )
        self.assertIn("overlap_reference=uniform_top_M(2)", rendered_console)

    def legacy_select(self, rank_num):
        direction_scores = (
            self.eigvals[:self.rank_r].unsqueeze(0)
            * self.eigvecs[:, :self.rank_r].square()
        )
        effective_rank_num = min(rank_num, self.rank_r)
        selected_direction_mask = torch.zeros_like(
            direction_scores,
            dtype=torch.bool,
        )
        selected_direction_mask[:, 0] = True
        if effective_rank_num > 1:
            client_top_directions = torch.argsort(
                direction_scores[:, 1:],
                dim=1,
                descending=True,
                stable=True,
            )[:, :effective_rank_num - 1] + 1
            selected_direction_mask.scatter_(1, client_top_directions, True)
        return selected_direction_mask, direction_scores, effective_rank_num


if __name__ == "__main__":
    unittest.main()

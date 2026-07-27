import importlib.util
import copy
import sys
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
            "serverCLIP_coeff_lowrank_sparse_test",
            SERVER_PATH,
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    return module.FedCLIP


FedCLIP = _load_fedclip_class()


class CoeffLowRankSparseDecompositionTest(unittest.TestCase):
    def decompose(self, delta_matrix, rho_c=0.1, rho_p=0.05):
        return FedCLIP._coeff_lowrank_sparse_decompose(
            delta_matrix,
            rho_c=rho_c,
            rho_p=rho_p,
            max_iters=15,
            tolerance=1e-5,
        )

    def test_zero_delta_is_finite(self):
        result = self.decompose(torch.zeros(12, 4, dtype=torch.float64))

        for key in (
            "u",
            "s",
            "vh",
            "z",
            "c",
            "p",
            "e",
            "z_clean",
            "clean_delta_matrix",
        ):
            self.assertTrue(torch.isfinite(result[key]).all())
        self.assertEqual(result["clean_delta_matrix"].shape, (12, 4))
        self.assertTrue(
            torch.equal(
                result["clean_delta_matrix"],
                torch.zeros(12, 4, dtype=torch.float64),
            )
        )

    def test_full_svd_reconstructs_delta_matrix(self):
        generator = torch.Generator().manual_seed(7)
        delta_matrix = torch.randn(
            31,
            5,
            generator=generator,
            dtype=torch.float64,
        )
        result = self.decompose(delta_matrix)

        reconstructed = result["u"] @ result["z"]
        self.assertTrue(
            torch.allclose(
                reconstructed,
                delta_matrix,
                atol=1e-10,
                rtol=1e-9,
            )
        )
        self.assertLess(result["svd_reconstruction_error"], 1e-9)

    def test_coefficients_reconstruct_as_c_plus_p_plus_e(self):
        generator = torch.Generator().manual_seed(11)
        delta_matrix = torch.randn(
            19,
            6,
            generator=generator,
            dtype=torch.float64,
        )
        result = self.decompose(delta_matrix)

        self.assertTrue(
            torch.allclose(
                result["z"],
                result["c"] + result["p"] + result["e"],
                atol=1e-12,
                rtol=1e-12,
            )
        )
        self.assertLess(result["coeff_reconstruction_error"], 1e-12)

    def test_zero_thresholds_preserve_raw_delta(self):
        generator = torch.Generator().manual_seed(17)
        delta_matrix = torch.randn(
            23,
            4,
            generator=generator,
            dtype=torch.float64,
        )
        result = self.decompose(delta_matrix, rho_c=0.0, rho_p=0.0)

        self.assertTrue(
            torch.allclose(
                result["clean_delta_matrix"],
                delta_matrix,
                atol=1e-10,
                rtol=1e-9,
            )
        )

    def test_extreme_thresholds_remove_c_and_p(self):
        generator = torch.Generator().manual_seed(23)
        delta_matrix = torch.randn(
            17,
            5,
            generator=generator,
            dtype=torch.float64,
        )
        result = self.decompose(
            delta_matrix,
            rho_c=1e6,
            rho_p=1e6,
        )

        self.assertLess(torch.linalg.vector_norm(result["c"]).item(), 1e-12)
        self.assertLess(torch.linalg.vector_norm(result["p"]).item(), 1e-12)
        self.assertLess(
            torch.linalg.vector_norm(result["clean_delta_matrix"]).item(),
            1e-12,
        )

    def test_output_columns_restore_original_layer_shape(self):
        generator = torch.Generator().manual_seed(29)
        layer_shape = (2, 3, 2)
        delta_matrix = torch.randn(
            12,
            3,
            generator=generator,
            dtype=torch.float32,
        )
        result = self.decompose(delta_matrix)

        self.assertEqual(result["clean_delta_matrix"].shape, (12, 3))
        for column in range(delta_matrix.shape[1]):
            restored = result["clean_delta_matrix"][:, column].reshape(
                layer_shape
            )
            self.assertEqual(restored.shape, layer_shape)


class CoeffLowRankSparseRegressionTest(unittest.TestCase):
    def test_aggregation_uses_each_clients_actual_start_weight(self):
        class SingleWeightModel(torch.nn.Module):
            def __init__(self, values):
                super().__init__()
                self.layer = torch.nn.Linear(2, 1, bias=False)
                with torch.no_grad():
                    self.layer.weight.copy_(
                        torch.tensor([values], dtype=torch.float32)
                    )

        endpoints = {
            "Client_0": SingleWeightModel([12.0, 14.0]),
            "Client_1": SingleWeightModel([23.0, 25.0]),
        }
        global_model = SingleWeightModel([100.0, 100.0])
        saved_global = {}
        captured_personalized_updates = {}

        def fake_load_item(role, name, folder):
            if role == "Server":
                return copy.deepcopy(global_model)
            return copy.deepcopy(endpoints[role])

        def fake_save_item(model, role, name, folder):
            if role == "Server" and name == "model":
                saved_global["model"] = copy.deepcopy(model)

        server = FedCLIP.__new__(FedCLIP)
        server.device = torch.device("cpu")
        server.role = "Server"
        server.save_folder_name = "unused"
        server.num_clients = 2
        server.uploaded_ids = [0, 1]
        server.uploaded_weights = [0.25, 0.75]
        server.clients = [
            SimpleNamespace(
                id=0,
                role="Client_0",
                save_folder_name="unused_client_0",
            ),
            SimpleNamespace(
                id=1,
                role="Client_1",
                save_folder_name="unused_client_1",
            ),
        ]
        server.client_start_full_weights = {
            0: {"layer.weight": torch.tensor([[10.0, 10.0]])},
            1: {"layer.weight": torch.tensor([[20.0, 20.0]])},
        }
        server.personal_residuals = {}
        server.args = SimpleNamespace(
            coeff_rho_c=0.0,
            coeff_rho_p=0.0,
            coeff_decomp_iters=15,
            coeff_decomp_tol=1e-5,
            coeff_decomp_warmup_ratio=0.2,
        )
        server.cur_ground = 21
        server.global_rounds = 100
        server.coeff_sparse_diagnostic_csv = "unused.csv"
        server._coeff_sparse_diagnostic_path_printed = True
        server._projectable_weight_names_from_low_rank_model = (
            lambda model: {"layer.weight"}
        )
        server._recover_if_needed = lambda model: model
        server._append_projection_diagnostic_rows = lambda path, rows: None
        server._save_sign_personalized_models = (
            lambda model, updates: captured_personalized_updates.update(
                copy.deepcopy(updates)
            )
        )

        function_globals = (
            FedCLIP.aggregate_coeff_lowrank_sparse.__globals__
        )
        with mock.patch.dict(
            function_globals,
            {
                "load_item": fake_load_item,
                "save_item": fake_save_item,
            },
        ):
            server.aggregate_coeff_lowrank_sparse()

        expected_delta_0 = torch.tensor([[2.0, 4.0]])
        expected_delta_1 = torch.tensor([[3.0, 5.0]])
        self.assertTrue(
            torch.allclose(
                captured_personalized_updates[0]["layer.weight"],
                expected_delta_0,
                atol=1e-5,
            )
        )
        self.assertTrue(
            torch.allclose(
                captured_personalized_updates[1]["layer.weight"],
                expected_delta_1,
                atol=1e-5,
            )
        )
        expected_global = 0.25 * endpoints[
            "Client_0"
        ].layer.weight + 0.75 * endpoints["Client_1"].layer.weight
        self.assertTrue(
            torch.allclose(
                saved_global["model"].layer.weight,
                expected_global,
                atol=1e-5,
            )
        )

    def test_old_aggregation_modes_are_still_selected_verbatim(self):
        old_modes = (
            "avg",
            "delta_avg",
            "projection",
            "consensus_projection",
            "sign_personalized_projection",
            "sign_projection_norm_restore",
            "sign_projection_no_group_renorm",
            "sign_projection_weight",
        )
        server = FedCLIP.__new__(FedCLIP)
        for mode in old_modes:
            server.args = SimpleNamespace(aggregation_mode=mode)
            self.assertEqual(server._aggregation_mode(), mode)

    def test_new_mode_and_parameters_are_exposed_by_main(self):
        main_source = MAIN_PATH.read_text(encoding="utf-8")
        self.assertIn('"coeff_lowrank_sparse"', main_source)
        for argument in (
            "--coeff_rho_c",
            "--coeff_rho_p",
            "--coeff_decomp_iters",
            "--coeff_decomp_tol",
            "--coeff_decomp_warmup_ratio",
        ):
            self.assertIn(argument, main_source)


if __name__ == "__main__":
    unittest.main()

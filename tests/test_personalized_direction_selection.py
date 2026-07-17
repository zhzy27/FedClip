import ast
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
        alpha=None,
    ):
        if updates is None or alpha is None:
            updates, alpha = self.orthogonal_layer_inputs()
        server = FedCLIP.__new__(FedCLIP)
        server.args = SimpleNamespace(
            aggregation_mode="sign_projection_no_group_renorm",
            personalized_rank_selection=1,
            personalized_rank_num=rank_num,
            personalized_rank_force_u1=int(force_u1),
            personalized_rank_mode=rank_mode,
            personalized_rank_energy=energy_threshold,
            personalized_g_scale=g_scale,
            projection_energy=1.0,
            projection_k_max=projection_k_max,
            projection_norm_scale_max=2.0,
        )
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
            with contextlib.redirect_stdout(console_output):
                personalized, average = server._sign_personalized_update_for_layer(
                    "layer",
                    [{"layer": update} for update in updates],
                    alpha,
                    updates[0].shape,
                    log_diagnostics=True,
                    console_diagnostics=True,
                    group_renorm=False,
                    norm_restore=norm_restore,
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
        return (
            personalized,
            average,
            client_rows,
            direction_rows,
            console_output.getvalue(),
        )

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

        rank_mode = arguments["--personalized_rank_mode"]
        self.assertEqual(ast.literal_eval(rank_mode["choices"]), ["fixed", "energy"])
        self.assertEqual(ast.literal_eval(rank_mode["default"]), "fixed")
        rank_energy = arguments["--personalized_rank_energy"]
        self.assertEqual(ast.literal_eval(rank_energy["default"]), 0.8)
        g_scale = arguments["--personalized_g_scale"]
        self.assertEqual(ast.literal_eval(g_scale["choices"]), [0, 1])
        self.assertEqual(ast.literal_eval(g_scale["default"]), 1)

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

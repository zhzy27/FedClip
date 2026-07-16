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

    def test_force_u1_m2_keeps_u1_and_selects_best_remaining(self):
        mask, scores, effective_rank_num = self.select(2, True)
        expected = torch.tensor(
            [
                [True, True, False, False],
                [True, True, False, False],
                [True, True, False, False],
                [True, False, False, True],
            ]
        )
        self.assertTrue(torch.equal(mask, expected))
        self.assertEqual(effective_rank_num, 2)
        torch.testing.assert_close(
            scores,
            self.eigvals.unsqueeze(0) * self.eigvecs.square(),
            rtol=0.0,
            atol=0.0,
        )

    def test_free_m2_can_exclude_u1(self):
        mask, _, effective_rank_num = self.select(2, False)
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
        self.assertEqual(effective_rank_num, 2)

    def test_m1_force_and_free_modes(self):
        force_mask, _, _ = self.select(1, True)
        free_mask, _, _ = self.select(1, False)
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
                    mask, _, effective_rank_num = self.select(rank_num, force_u1)
                    self.assertTrue(bool(mask.all()))
                    self.assertEqual(effective_rank_num, self.rank_r)

    def test_every_mask_row_has_exact_effective_count(self):
        for force_u1 in (True, False):
            for rank_num in (1, 2, 3, 4, 9):
                with self.subTest(force_u1=force_u1, rank_num=rank_num):
                    mask, scores, effective_rank_num = self.select(
                        rank_num,
                        force_u1,
                    )
                    self.assertEqual(mask.dtype, torch.bool)
                    self.assertTrue(bool(torch.isfinite(mask).all()))
                    self.assertTrue(bool(torch.isfinite(scores).all()))
                    expected_count = min(rank_num, self.rank_r)
                    self.assertEqual(effective_rank_num, expected_count)
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
                default_mask, default_scores, default_effective = (
                    FedCLIP._select_personalized_directions(
                        self.eigvals,
                        self.eigvecs,
                        self.rank_r,
                        rank_num,
                    )
                )
                explicit_mask, explicit_scores, explicit_effective = self.select(
                    rank_num,
                    True,
                )
                self.assertTrue(torch.equal(default_mask, legacy_mask))
                self.assertTrue(torch.equal(explicit_mask, legacy_mask))
                self.assertTrue(torch.equal(default_scores, legacy_scores))
                self.assertTrue(torch.equal(explicit_scores, legacy_scores))
                self.assertEqual(default_effective, legacy_effective)
                self.assertEqual(explicit_effective, legacy_effective)

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

import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
SYSTEM_ROOT = REPO_ROOT / "system"
if str(SYSTEM_ROOT) not in os.sys.path:
    os.sys.path.insert(0, str(SYSTEM_ROOT))

from utils.agg_path_diagnostics import (  # noqa: E402
    aggregation_human_round,
    aggregation_path_consistency_rows,
    collect_agg_path_updates,
    diagnostic_round_selected,
    global_truncation_rows,
    prelocal_source_aggregation_round,
    resolve_diagnostic_output_dir,
    weight_to_svd_matrix,
)


def make_entry(path, rank, weight_name="base.layer.weight"):
    return {
        "u_path": path.clone(),
        "v_path": path.clone(),
        "rank": rank,
        "weight_name": weight_name,
    }


class AggregationPathDiagnosticTests(unittest.TestCase):
    def test_different_factor_ranks_recover_the_same_full_w_shape(self):
        start_rank1 = {
            "base.layer.weight_u": torch.zeros(4, 1),
            "base.layer.weight_v": torch.ones(1, 3),
        }
        end_rank1 = {
            "base.layer.weight_u": torch.ones(4, 1),
            "base.layer.weight_v": torch.full((1, 3), 2.0),
        }
        start_rank2 = {
            "base.layer.weight_u": torch.zeros(4, 2),
            "base.layer.weight_v": torch.ones(2, 3),
        }
        end_rank2 = {
            "base.layer.weight_u": torch.ones(4, 2),
            "base.layer.weight_v": torch.full((2, 3), 2.0),
        }

        update_rank1 = collect_agg_path_updates(start_rank1, end_rank1)
        update_rank2 = collect_agg_path_updates(start_rank2, end_rank2)

        self.assertEqual(update_rank1["base.layer"]["u_path"].shape, (4, 3))
        self.assertEqual(update_rank2["base.layer"]["u_path"].shape, (4, 3))
        self.assertEqual(update_rank1["base.layer"]["rank"], 1)
        self.assertEqual(update_rank2["base.layer"]["rank"], 2)

    def test_identical_directions_have_unit_consistency(self):
        direction = torch.tensor([[1.0, -2.0], [3.0, 4.0]])
        updates = {
            0: {"layer": make_entry(direction, 1)},
            1: {"layer": make_entry(direction * 2.0, 1)},
            2: {"layer": make_entry(direction * 0.5, 2)},
        }
        rows, _ = aggregation_path_consistency_rows(
            5, updates, [0, 1, 2], [0.2, 0.3, 0.5]
        )
        layer_row = next(row for row in rows if row["layer"] == "layer")
        self.assertAlmostEqual(layer_row["S_U"], 1.0, places=6)
        self.assertAlmostEqual(layer_row["S_V"], 1.0, places=6)
        self.assertAlmostEqual(layer_row["same_rank_u_cos"], 1.0, places=6)
        self.assertAlmostEqual(layer_row["cross_rank_u_cos"], 1.0, places=6)
        self.assertEqual(layer_row["same_rank_pair_count"], 1)
        self.assertEqual(layer_row["cross_rank_pair_count"], 2)

    def test_opposite_directions_reduce_consistency(self):
        direction = torch.tensor([[1.0, 2.0], [-3.0, 4.0]])
        updates = {
            0: {"layer": make_entry(direction, 1)},
            1: {"layer": make_entry(-direction, 1)},
        }
        rows, _ = aggregation_path_consistency_rows(
            1, updates, [0, 1], [0.5, 0.5]
        )
        layer_row = next(row for row in rows if row["layer"] == "layer")
        self.assertLess(layer_row["S_U"], 1e-6)
        self.assertAlmostEqual(layer_row["same_rank_u_cos"], -1.0, places=6)

    def test_retained_energy_is_monotonic_and_full_rank_is_exact(self):
        weight = torch.diag(torch.tensor([4.0, 2.0, 1.0]))
        original = weight.clone()
        rows = global_truncation_rows(
            7,
            {"base.layer.weight": weight},
            {
                "base.layer": {
                    "weight_name": "base.layer.weight",
                    "ranks": [1, 2, 3],
                }
            },
        )
        energies = [row["retained_energy"] for row in rows]
        self.assertEqual(energies, sorted(energies))
        self.assertAlmostEqual(energies[-1], 1.0, places=7)
        self.assertAlmostEqual(
            rows[-1]["relative_truncation_error"], 0.0, places=7
        )
        self.assertTrue(torch.equal(weight, original))

    def test_convolution_matrixization_matches_decom_cov(self):
        weight = torch.arange(24, dtype=torch.float32).reshape(2, 3, 2, 2)
        expected = weight.permute(0, 2, 1, 3).reshape(4, 6)
        actual = weight_to_svd_matrix(weight)
        self.assertTrue(torch.equal(actual, expected))

    def test_collection_and_consistency_do_not_modify_input_tensors(self):
        start = {
            "layer.weight_u": torch.randn(4, 2),
            "layer.weight_v": torch.randn(2, 3),
        }
        end = {
            "layer.weight_u": torch.randn(4, 2),
            "layer.weight_v": torch.randn(2, 3),
        }
        start_before = {name: value.clone() for name, value in start.items()}
        end_before = {name: value.clone() for name, value in end.items()}
        updates = collect_agg_path_updates(start, end)
        aggregation_path_consistency_rows(
            1, {0: updates}, [0], [1.0]
        )
        for name in start:
            self.assertTrue(torch.equal(start[name], start_before[name]))
            self.assertTrue(torch.equal(end[name], end_before[name]))

    def test_output_directory_defaults_to_train_log_environment(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            with mock.patch.dict(
                os.environ, {"FEDCLIP_TRAIN_LOG_DIR": temp_dir}, clear=False
            ):
                resolved = resolve_diagnostic_output_dir("", "fallback")
            self.assertTrue(os.path.samefile(resolved, temp_dir))

    def test_human_rounds_align_path_truncation_and_next_prelocal_send(self):
        configured = "1,5,10,100"
        aggregation_loops = [
            loop_round
            for loop_round in range(101)
            if diagnostic_round_selected(
                aggregation_human_round(loop_round), configured
            )
        ]
        prelocal_send_loops = [
            send_round
            for send_round in range(1, 101)
            if diagnostic_round_selected(
                prelocal_source_aggregation_round(send_round), configured
            )
        ]

        self.assertEqual(aggregation_loops, [0, 4, 9, 99])
        self.assertEqual(prelocal_send_loops, [1, 5, 10, 100])
        self.assertEqual(
            [aggregation_human_round(item) for item in aggregation_loops],
            [1, 5, 10, 100],
        )
        self.assertEqual(
            [
                prelocal_source_aggregation_round(item)
                for item in prelocal_send_loops
            ],
            [1, 5, 10, 100],
        )
        self.assertEqual(prelocal_source_aggregation_round(0), "initial")

    def test_train_order_is_send_then_prelocal_then_local_train(self):
        source = (SYSTEM_ROOT / "flcore" / "servers" / "serverCLIP.py").read_text(
            encoding="utf-8"
        )
        train_start = source.index("    def train(self):")
        train_end = source.index("    def _record_factor_update_stats", train_start)
        train_source = source[train_start:train_end]
        send_index = train_source.index("self.send_parameters()")
        prelocal_index = train_source.index("self._record_prelocal_download_accuracy(i)")
        local_train_index = train_source.index("client.train(current_round=i)")
        self.assertLess(send_index, prelocal_index)
        self.assertLess(prelocal_index, local_train_index)
        self.assertIn(
            "if self._should_record_prelocal_download(i):", train_source
        )

    def test_disabled_diagnostics_leave_calls_guarded(self):
        source = (SYSTEM_ROOT / "flcore" / "servers" / "serverCLIP.py").read_text(
            encoding="utf-8"
        )
        self.assertIn(
            "if not self.enable_agg_path_diagnostics:\n            return False",
            source,
        )
        client_source = (
            SYSTEM_ROOT / "flcore" / "clients" / "clientCLIP.py"
        ).read_text(encoding="utf-8")
        self.assertIn(
            'if not bool(getattr(self.args, "enable_agg_path_diagnostics", 0)):',
            client_source,
        )


if __name__ == "__main__":
    unittest.main()

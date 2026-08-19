import os
import unittest
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


REPO_ROOT = Path(__file__).resolve().parents[1]
SYSTEM_ROOT = REPO_ROOT / "system"
if str(SYSTEM_ROOT) not in os.sys.path:
    os.sys.path.insert(0, str(SYSTEM_ROOT))

from utils.anchor_mechanism_diagnostics import (  # noqa: E402
    build_anchor_configuration,
    collect_model_class_prototypes,
    prototype_class_summary_rows,
    prototype_client_rows,
    prototype_human_round,
    prototype_local_drift_rows,
    prototype_round_selected,
    prototype_summary_row,
)


def true_anchors():
    return torch.tensor(
        [
            [3.0, 4.0, 0.0],
            [0.0, 2.0, 0.0],
            [1.0, 2.0, 2.0],
        ]
    )


class FeatureModel(nn.Module):
    def __init__(self, with_batch_norm=False):
        super().__init__()
        if with_batch_norm:
            self.base = nn.Sequential(nn.BatchNorm1d(3), nn.Identity())
        else:
            self.base = nn.Identity()


def client_result(client_id, capacity, prototypes, training=None):
    anchors = true_anchors()
    return {
        "client_id": client_id,
        "capacity": capacity,
        "prototypes": {
            class_id: {
                "prototype": prototype.clone(),
                "sample_count": count,
            }
            for class_id, (prototype, count) in prototypes.items()
        },
        "training_anchors": anchors if training is None else training,
        "true_clip_anchors": anchors,
    }


class AnchorMechanismDiagnosticTests(unittest.TestCase):
    def test_clip_mode_is_exactly_the_original_anchor_tensor(self):
        anchors = true_anchors()
        configuration = build_anchor_configuration(anchors, mode="clip")
        self.assertIs(configuration["anchors"], anchors)
        self.assertTrue(torch.equal(configuration["anchors"], anchors))

    def test_shared_random_is_identical_across_clients(self):
        left = build_anchor_configuration(
            true_anchors(), "shared_random", 2026, client_id=0
        )
        right = build_anchor_configuration(
            true_anchors(), "shared_random", 2026, client_id=19
        )
        self.assertTrue(torch.equal(left["anchors"], right["anchors"]))
        self.assertEqual(left["hash"], right["hash"])

    def test_shared_random_is_fixed_across_round_like_reconstruction(self):
        first = build_anchor_configuration(
            true_anchors(), "shared_random", 2026, client_id=4
        )
        later = build_anchor_configuration(
            true_anchors(), "shared_random", 2026, client_id=4
        )
        self.assertTrue(torch.equal(first["anchors"], later["anchors"]))

    def test_client_random_is_fixed_for_one_client(self):
        first = build_anchor_configuration(
            true_anchors(), "client_random", 2026, client_id=7
        )
        later = build_anchor_configuration(
            true_anchors(), "client_random", 2026, client_id=7
        )
        self.assertTrue(torch.equal(first["anchors"], later["anchors"]))

    def test_client_random_differs_between_clients(self):
        left = build_anchor_configuration(
            true_anchors(), "client_random", 2026, client_id=1
        )
        right = build_anchor_configuration(
            true_anchors(), "client_random", 2026, client_id=2
        )
        self.assertFalse(torch.equal(left["anchors"], right["anchors"]))

    def test_random_anchor_norm_matches_true_clip_norm(self):
        anchors = true_anchors()
        for mode in ("shared_random", "client_random"):
            generated = build_anchor_configuration(
                anchors, mode, 2026, client_id=5
            )["anchors"]
            self.assertTrue(
                torch.allclose(
                    torch.linalg.vector_norm(generated, dim=-1),
                    torch.linalg.vector_norm(anchors, dim=-1),
                    atol=1e-6,
                    rtol=1e-6,
                )
            )

    def test_shuffled_clip_uses_a_derangement(self):
        configuration = build_anchor_configuration(
            true_anchors(), "shuffled_clip", 2026
        )
        permutation = configuration["permutation"]
        self.assertEqual(sorted(permutation.tolist()), [0, 1, 2])
        self.assertTrue(torch.all(permutation != torch.arange(3)))

    def test_cifar100_seed_2026_has_no_fixed_point(self):
        anchors = torch.arange(800, dtype=torch.float32).reshape(100, 8)
        permutation = build_anchor_configuration(
            anchors, "shuffled_clip", 2026
        )["permutation"]
        self.assertTrue(torch.all(permutation != torch.arange(100)))

    def test_shuffled_clip_preserves_permuted_gram_geometry(self):
        anchors = true_anchors()
        configuration = build_anchor_configuration(
            anchors, "shuffled_clip", 2026
        )
        permutation = configuration["permutation"]
        shuffled = configuration["anchors"]
        expected = anchors[permutation]
        self.assertTrue(
            torch.allclose(
                shuffled @ shuffled.T,
                expected @ expected.T,
                atol=0.0,
                rtol=0.0,
            )
        )

    def test_prototype_computation_matches_artificial_class_means(self):
        features = torch.tensor(
            [[1.0, 0.0, 0.0], [3.0, 0.0, 0.0], [0.0, 2.0, 0.0]]
        )
        labels = torch.tensor([0, 0, 1])
        loader = DataLoader(TensorDataset(features, labels), batch_size=2)
        prototypes = collect_model_class_prototypes(
            FeatureModel(), loader, torch.device("cpu"), num_classes=3
        )
        self.assertTrue(
            torch.equal(prototypes[0]["prototype"], torch.tensor([2.0, 0.0, 0.0]))
        )
        self.assertTrue(
            torch.equal(prototypes[1]["prototype"], torch.tensor([0.0, 2.0, 0.0]))
        )
        self.assertEqual(prototypes[0]["sample_count"], 2)

    def test_absent_class_is_skipped(self):
        features = torch.tensor([[1.0, 0.0, 0.0], [3.0, 0.0, 0.0]])
        labels = torch.tensor([0, 0])
        prototypes = collect_model_class_prototypes(
            FeatureModel(),
            DataLoader(TensorDataset(features, labels), batch_size=2),
            torch.device("cpu"),
            num_classes=3,
        )
        self.assertEqual(set(prototypes), {0})

    def test_pre_and_post_use_the_same_one_based_human_round(self):
        configured = "1,5,10,100"
        selected_loops = [
            loop_round
            for loop_round in range(100)
            if prototype_round_selected(loop_round, configured)
        ]
        self.assertEqual(selected_loops, [0, 4, 9, 99])
        self.assertEqual(
            [prototype_human_round(item) for item in selected_loops],
            [1, 5, 10, 100],
        )

    def test_prototype_diagnostics_preserve_parameters_bn_and_mode(self):
        model = FeatureModel(with_batch_norm=True)
        model.train()
        before = {name: value.clone() for name, value in model.state_dict().items()}
        features = torch.tensor([[1.0, 2.0, 3.0], [3.0, 2.0, 1.0]])
        labels = torch.tensor([0, 1])
        collect_model_class_prototypes(
            model,
            DataLoader(TensorDataset(features, labels), batch_size=2),
            torch.device("cpu"),
            num_classes=3,
        )
        self.assertTrue(model.training)
        for name, value in model.state_dict().items():
            self.assertTrue(torch.equal(value, before[name]), name)

    def test_disabled_diagnostics_leave_collection_calls_guarded(self):
        source = (
            SYSTEM_ROOT / "flcore" / "servers" / "serverCLIP.py"
        ).read_text(encoding="utf-8")
        self.assertIn(
            "if not self.enable_semantic_prototype_diagnostics:\n"
            "            return False",
            source,
        )
        self.assertIn(
            'if self._semantic_prototype_diagnostic_target(i, "prelocal"):',
            source,
        )
        self.assertIn(
            'if self._semantic_prototype_diagnostic_target(i, "postlocal"):',
            source,
        )

    def test_cross_client_summary_and_local_drift_are_correct(self):
        pre = {
            0: client_result(0, 0.15, {0: (torch.tensor([1.0, 0.0, 0.0]), 2)}),
            1: client_result(1, 0.15, {0: (torch.tensor([1.0, 0.0, 0.0]), 3)}),
        }
        rows = prototype_client_rows(1, "prelocal", pre)
        summary = prototype_summary_row(1, "prelocal", pre, rows)
        self.assertAlmostEqual(summary["overall_same_class_cos"], 1.0, places=7)
        self.assertEqual(summary["same_capacity_pair_count"], 1)
        class_rows = prototype_class_summary_rows(1, "prelocal", pre)
        self.assertEqual(len(class_rows), 1)
        self.assertEqual(class_rows[0]["class_id"], 0)
        self.assertAlmostEqual(
            class_rows[0]["same_capacity_cos"], 1.0, places=7
        )
        self.assertEqual(class_rows[0]["same_capacity_pair_count"], 1)

        post = {
            0: client_result(0, 0.15, {0: (torch.tensor([0.0, 1.0, 0.0]), 2)}),
            1: pre[1],
        }
        drift_rows = prototype_local_drift_rows(1, pre, post)
        client_zero = next(
            row
            for row in drift_rows
            if row["record_type"] == "client_class" and row["client_id"] == 0
        )
        self.assertAlmostEqual(client_zero["local_proto_drift"], 1.0, places=7)

    def test_anchor_generation_does_not_advance_global_torch_rng(self):
        torch.manual_seed(123)
        before = torch.random.get_rng_state().clone()
        build_anchor_configuration(
            true_anchors(), "client_random", 2026, client_id=8
        )
        after = torch.random.get_rng_state()
        self.assertTrue(torch.equal(before, after))


if __name__ == "__main__":
    unittest.main()

import copy
import os
import random
import unittest
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


REPO_ROOT = Path(__file__).resolve().parents[1]
SYSTEM_ROOT = REPO_ROOT / "system"
if str(SYSTEM_ROOT) not in os.sys.path:
    os.sys.path.insert(0, str(SYSTEM_ROOT))

from utils.anchor_mechanism_diagnostics import (  # noqa: E402
    collect_model_class_prototypes,
    feature_scale_summary_rows,
)
from utils.feature_auxiliary_diagnostics import (  # noqa: E402
    AUX_GRADIENT_SCALE_FIELDS,
    build_global_feature_anchor,
    collect_aux_gradient_scale_diagnostic,
    feature_auxiliary_loss,
    feature_contrastive_logits,
    resolve_feature_aux_target_norm,
)


class TinyFeatureModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.base = nn.Sequential(nn.BatchNorm1d(3), nn.Linear(3, 3, bias=False))
        self.head = nn.Linear(3, 3)


class FeatureAuxiliaryLossTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.features = torch.randn(6, 4, requires_grad=True)
        self.anchors = torch.randn(3, 4)
        self.labels = torch.tensor([0, 1, 2, 0, 2, 1])
        self.mse_fn = nn.MSELoss()

    def test_mse_matches_original_implementation_exactly(self):
        expected = self.mse_fn(self.features, self.anchors[self.labels])
        actual = feature_auxiliary_loss(
            self.features,
            self.labels,
            self.anchors,
            mode="mse",
            mse_fn=self.mse_fn,
        )
        self.assertTrue(torch.equal(actual, expected))

    def test_z2_is_mse_to_zero_and_does_not_access_anchors(self):
        expected = self.mse_fn(self.features, torch.zeros_like(self.features))
        actual = feature_auxiliary_loss(
            self.features,
            self.labels,
            class_anchors=object(),
            mode="z2",
            mse_fn=self.mse_fn,
        )
        self.assertTrue(torch.equal(actual, expected))

    def test_z1_is_mean_absolute_feature_value(self):
        expected = self.features.abs().mean()
        actual = feature_auxiliary_loss(
            self.features,
            self.labels,
            class_anchors=object(),
            mode="z1",
        )
        self.assertTrue(torch.equal(actual, expected))

    def test_global_direction_losses_ignore_positive_feature_scale(self):
        global_anchor = torch.randn(4)
        for mode in ("global_dir_l1", "global_dir_l2"):
            with self.subTest(mode=mode):
                baseline = feature_auxiliary_loss(
                    self.features,
                    self.labels,
                    mode=mode,
                    global_anchor=global_anchor,
                )
                scaled = feature_auxiliary_loss(
                    self.features * 7.3,
                    self.labels,
                    mode=mode,
                    global_anchor=global_anchor,
                )
                self.assertTrue(
                    torch.allclose(baseline, scaled, atol=1e-7, rtol=1e-7)
                )

    def test_global_point_losses_are_zero_at_global_anchor(self):
        global_anchor = torch.randn(4)
        features = global_anchor.unsqueeze(0).expand(5, -1).clone()
        labels = torch.zeros(5, dtype=torch.long)
        for mode in ("global_point_l1", "global_point_l2"):
            with self.subTest(mode=mode):
                loss = feature_auxiliary_loss(
                    features,
                    labels,
                    mode=mode,
                    global_anchor=global_anchor,
                )
                self.assertEqual(float(loss), 0.0)

    def test_radial_losses_are_zero_at_target_norm(self):
        target_norm = 3.25
        features = F.normalize(torch.randn(5, 4), dim=1) * target_norm
        labels = torch.zeros(5, dtype=torch.long)
        for mode in ("radial_l1", "radial_l2"):
            with self.subTest(mode=mode):
                loss = feature_auxiliary_loss(
                    features,
                    labels,
                    mode=mode,
                    target_norm=target_norm,
                )
                self.assertTrue(torch.allclose(loss, torch.zeros_like(loss), atol=1e-6))

    def test_global_anchor_is_shared_scaled_and_rng_isolated(self):
        reference = torch.tensor(
            [[3.0, 4.0, 0.0, 0.0], [0.0, 0.0, 0.0, 2.0]]
        )
        expected_norm = (5.0 + 2.0) / 2.0

        torch.manual_seed(31)
        np.random.seed(31)
        random.seed(31)
        torch_before = torch.random.get_rng_state().clone()
        numpy_before = copy.deepcopy(np.random.get_state())
        python_before = random.getstate()

        first, first_norm = build_global_feature_anchor(
            reference, seed=17, target_norm=-1
        )
        second, second_norm = build_global_feature_anchor(
            reference.clone(), seed=17, target_norm=-1
        )

        self.assertTrue(torch.equal(first, second))
        self.assertEqual(first_norm, expected_norm)
        self.assertEqual(second_norm, expected_norm)
        self.assertAlmostEqual(float(torch.linalg.vector_norm(first)), expected_norm, places=6)
        self.assertTrue(torch.equal(torch.random.get_rng_state(), torch_before))
        self.assertEqual(random.getstate(), python_before)
        numpy_after = np.random.get_state()
        self.assertEqual(numpy_before[0], numpy_after[0])
        self.assertTrue(np.array_equal(numpy_before[1], numpy_after[1]))
        self.assertEqual(numpy_before[2:], numpy_after[2:])

    def test_explicit_target_norm_overrides_clip_anchor_norm(self):
        reference = torch.randn(3, 4)
        self.assertEqual(
            resolve_feature_aux_target_norm(reference, configured_norm=2.75),
            2.75,
        )
        anchor, target = build_global_feature_anchor(
            reference, seed=9, target_norm=2.75
        )
        self.assertEqual(target, 2.75)
        self.assertAlmostEqual(float(torch.linalg.vector_norm(anchor)), 2.75, places=6)

    def test_all_new_losses_produce_finite_feature_gradients(self):
        global_anchor = torch.randn(4)
        modes = (
            "z1",
            "global_dir_l1",
            "global_dir_l2",
            "global_point_l1",
            "global_point_l2",
            "radial_l1",
            "radial_l2",
        )
        for mode in modes:
            with self.subTest(mode=mode):
                features = self.features.detach().clone().requires_grad_(True)
                loss = feature_auxiliary_loss(
                    features,
                    self.labels,
                    mode=mode,
                    global_anchor=global_anchor,
                    target_norm=2.5,
                )
                gradient = torch.autograd.grad(loss, features)[0]
                self.assertTrue(torch.isfinite(loss))
                self.assertTrue(torch.all(torch.isfinite(gradient)))

    def test_cosine_is_invariant_to_positive_feature_and_anchor_scale(self):
        baseline = feature_auxiliary_loss(
            self.features, self.labels, self.anchors, mode="cosine"
        )
        scaled = feature_auxiliary_loss(
            self.features * 3.7,
            self.labels,
            self.anchors * 9.2,
            mode="cosine",
        )
        self.assertTrue(torch.allclose(baseline, scaled, atol=1e-7, rtol=1e-7))

    def test_contrastive_logits_shape_and_positive_label(self):
        anchors = torch.eye(3)
        features = anchors.clone()
        labels = torch.arange(3)
        logits = feature_contrastive_logits(features, anchors, temperature=0.1)
        self.assertEqual(tuple(logits.shape), (3, 3))
        self.assertTrue(torch.equal(torch.argmax(logits, dim=1), labels))
        expected = F.cross_entropy(logits, labels)
        actual = feature_auxiliary_loss(
            features,
            labels,
            anchors,
            mode="contrastive",
            contrastive_temperature=0.1,
        )
        self.assertTrue(torch.equal(actual, expected))

    def test_contrastive_is_invariant_to_positive_anchor_norm_scale(self):
        baseline = feature_auxiliary_loss(
            self.features,
            self.labels,
            self.anchors,
            mode="contrastive",
            contrastive_temperature=0.2,
        )
        scaled = feature_auxiliary_loss(
            self.features * 2.5,
            self.labels,
            self.anchors * torch.tensor([[2.0], [5.0], [11.0]]),
            mode="contrastive",
            contrastive_temperature=0.2,
        )
        self.assertTrue(torch.allclose(baseline, scaled, atol=1e-7, rtol=1e-7))

    def test_none_is_a_differentiable_zero(self):
        loss = feature_auxiliary_loss(
            self.features,
            self.labels,
            class_anchors=object(),
            mode="none",
        )
        self.assertEqual(float(loss.detach()), 0.0)
        loss.backward()
        self.assertTrue(torch.equal(self.features.grad, torch.zeros_like(self.features)))

    def test_gradient_scale_diagnostic_does_not_touch_parameters_or_grad(self):
        model = TinyFeatureModel()
        model.eval()
        inputs = torch.randn(5, 3)
        labels = torch.tensor([0, 1, 2, 0, 1])
        features = model.base(inputs)
        logits = model.head(features)
        ce_loss = F.cross_entropy(logits, labels)
        aux_loss = feature_auxiliary_loss(
            features,
            labels,
            torch.eye(3),
            mode="cosine",
        )
        before = copy.deepcopy(model.state_dict())
        rng_before = torch.random.get_rng_state().clone()
        row = collect_aux_gradient_scale_diagnostic(
            ce_loss,
            aux_loss,
            list(model.base.named_parameters()),
            round_number=1,
            client_id=4,
            aux_loss_mode="cosine",
            aux_coefficient=1.0,
            features=features,
            target_feature_norm=2.0,
            global_anchor=torch.ones(3),
        )
        self.assertTrue(set(AUX_GRADIENT_SCALE_FIELDS).issubset(row))
        self.assertAlmostEqual(row["target_feature_norm"], 2.0)
        self.assertAlmostEqual(row["global_anchor_norm"], 3.0 ** 0.5)
        self.assertTrue(all(parameter.grad is None for parameter in model.parameters()))
        self.assertTrue(torch.equal(torch.random.get_rng_state(), rng_before))
        for name, value in model.state_dict().items():
            self.assertTrue(torch.equal(value, before[name]), name)

    def test_feature_scale_summary_matches_direct_statistics(self):
        class IdentityFeatureModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.base = nn.Identity()

        features = torch.tensor([[3.0, 4.0], [0.0, 2.0]])
        labels = torch.tensor([0, 1])
        prototypes, scale = collect_model_class_prototypes(
            IdentityFeatureModel(),
            DataLoader(TensorDataset(features, labels), batch_size=1),
            torch.device("cpu"),
            num_classes=2,
            return_feature_scale=True,
        )
        self.assertEqual(set(prototypes), {0, 1})
        self.assertAlmostEqual(scale["mean_feature_norm"], 3.5, places=7)
        self.assertAlmostEqual(scale["std_feature_norm"], 1.5, places=7)
        self.assertAlmostEqual(
            scale["mean_feature_sq"], float(torch.mean(features ** 2)), places=7
        )
        rows = feature_scale_summary_rows(
            1,
            "prelocal",
            {
                2: {
                    "aux_loss_mode": "z2",
                    "feature_scale": scale,
                }
            },
        )
        self.assertEqual(rows[0]["aux_loss_mode"], "z2")
        self.assertEqual(rows[0]["sample_count"], 2)


if __name__ == "__main__":
    unittest.main()

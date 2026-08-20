import copy
import os
import unittest
from pathlib import Path

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
    collect_aux_gradient_scale_diagnostic,
    feature_auxiliary_loss,
    feature_contrastive_logits,
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
        )
        self.assertTrue(set(AUX_GRADIENT_SCALE_FIELDS).issubset(row))
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

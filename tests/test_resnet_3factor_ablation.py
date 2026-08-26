import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
SYSTEM_ROOT = REPO_ROOT / "system"
sys.path.insert(0, str(SYSTEM_ROOT))

from flcore.trainmodel.SVD_resnet import FactorizedConv, _sample_ordered_rank
from utils.resnet_3factor_ablation import (
    effective_resnet_rank_dropout_mode,
    effective_resnet_rho,
    is_resnet_3factor_target,
    resnet_aggregation_method_name,
    sample_weighted_average_parameter_dicts_,
)


def make_args(a=1, r=1, d=1, asymmetric=1):
    return SimpleNamespace(
        algorithm="FedCLIP",
        model_family="Decom_resnet18_5",
        resnet_personalized_agg=a,
        resnet_legacy_rho=r,
        resnet_ordered_dropout=d,
        use_asymmetric_lr=asymmetric,
        local_learning_rate=0.005,
    )


class ResNetThreeFactorAblationTest(unittest.TestCase):
    def test_111_routes_to_legacy_resnet_behavior(self):
        args = make_args(1, 1, 1)
        self.assertTrue(is_resnet_3factor_target(args))
        self.assertEqual(resnet_aggregation_method_name(args), "aggregate_parameters_v_svd_res")
        self.assertEqual(effective_resnet_rho(args), 0.1)
        self.assertEqual(effective_resnet_rank_dropout_mode(args), "original")

    def test_000_routes_to_simple_v_behavior(self):
        args = make_args(0, 0, 0)
        self.assertEqual(resnet_aggregation_method_name(args), "aggregate_parameters_avg")
        self.assertEqual(effective_resnet_rho(args), 0.3)
        self.assertEqual(effective_resnet_rank_dropout_mode(args), "none")

    def test_resnet_rho_produces_requested_optimizer_learning_rates(self):
        base_lr = 0.005
        self.assertAlmostEqual(base_lr * effective_resnet_rho(make_args(r=1)), 0.0005)
        self.assertAlmostEqual(base_lr * effective_resnet_rho(make_args(r=0)), 0.0015)
        self.assertEqual(effective_resnet_rho(make_args(r=1, asymmetric=0)), 1.0)

    def test_ordered_dropout_samples_a_prefix_and_none_keeps_full_rank(self):
        layer = FactorizedConv(4, 4, rank_rate=1.0, kernel_size=3)
        layer.rank_dropout_mode = "none"
        self.assertEqual(_sample_ordered_rank(layer), layer.rank)

        layer.rank_dropout_mode = "original"
        sampled = torch.tensor([max(1, layer.rank // 4)])
        with mock.patch("flcore.trainmodel.SVD_resnet.torch.randint", return_value=sampled) as randint:
            self.assertEqual(_sample_ordered_rank(layer), sampled.item())
            randint.assert_called_once()
            low, high = randint.call_args.args[:2]
            self.assertEqual(low, max(1, layer.rank // 4))
            self.assertEqual(high, layer.rank + 1)

    def test_full_weight_avg_is_sample_weighted_and_includes_all_parameters(self):
        target = {
            "base.weight": torch.nn.Parameter(torch.zeros(2)),
            "head.bias": torch.nn.Parameter(torch.zeros(1)),
        }
        sources = [
            {
                "base.weight": torch.nn.Parameter(torch.tensor([1.0, 3.0])),
                "head.bias": torch.nn.Parameter(torch.tensor([2.0])),
            },
            {
                "base.weight": torch.nn.Parameter(torch.tensor([5.0, 7.0])),
                "head.bias": torch.nn.Parameter(torch.tensor([6.0])),
            },
        ]

        sample_weighted_average_parameter_dicts_(target, sources, [0.25, 0.75])

        torch.testing.assert_close(target["base.weight"], torch.tensor([4.0, 6.0]))
        torch.testing.assert_close(target["head.bias"], torch.tensor([5.0]))

    def test_non_target_models_keep_their_existing_routes(self):
        args = make_args(0, 0, 0)
        args.model_family = "Decom_CNN-5-512"
        self.assertFalse(is_resnet_3factor_target(args))
        self.assertEqual(resnet_aggregation_method_name(args), "aggregate_parameters_v_svd")

    def test_one_round_low_rank_avg_resvd_smoke(self):
        torch.manual_seed(0)
        clients = [FactorizedConv(2, 2, rank_rate=0.5, kernel_size=3) for _ in range(2)]
        recovered_weights = []

        for client_idx, client in enumerate(clients):
            target = torch.full(
                (client.dim1, client.dim2),
                fill_value=float(client_idx + 1),
            )
            effective_weight = client.conv_u @ client.conv_v
            loss = torch.mean((effective_weight - target) ** 2)
            loss.backward()
            with torch.no_grad():
                client.conv_u.add_(client.conv_u.grad, alpha=-0.0005)
                client.conv_v.add_(client.conv_v.grad, alpha=-0.005)
            recovered_weights.append((client.conv_u @ client.conv_v).detach())

        averaged = (
            0.25 * recovered_weights[0]
            + 0.75 * recovered_weights[1]
        )
        u, singular, vh = torch.linalg.svd(averaged, full_matrices=False)
        retained_rank = clients[0].rank
        balanced_u = u[:, :retained_rank] * singular[:retained_rank].sqrt().unsqueeze(0)
        balanced_v = singular[:retained_rank].sqrt().unsqueeze(1) * vh[:retained_rank]

        self.assertEqual(balanced_u.shape, clients[0].conv_u.shape)
        self.assertEqual(balanced_v.shape, clients[0].conv_v.shape)
        self.assertTrue(torch.isfinite(balanced_u).all())
        self.assertTrue(torch.isfinite(balanced_v).all())


if __name__ == "__main__":
    unittest.main()

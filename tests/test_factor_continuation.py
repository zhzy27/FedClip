import unittest

import torch

from system.utils.factor_continuation import (
    factor_rank_signature,
    resolve_capacity_ratios,
    sample_weighted_factor_average,
    validate_factor_continuation_mode,
)


class TinyFactorModel(torch.nn.Module):
    def __init__(self, rank=2, fill=0.0):
        super().__init__()
        self.weight_u = torch.nn.Parameter(torch.full((4, rank), fill))
        self.weight_v = torch.nn.Parameter(torch.full((rank, 3), fill + 1.0))
        self.bias = torch.nn.Parameter(torch.full((4,), fill + 2.0))


class FactorContinuationTests(unittest.TestCase):
    def test_factor_continuation_requires_homogeneous_capacity(self):
        with self.assertRaisesRegex(ValueError, "requires homogeneous_capacity"):
            validate_factor_continuation_mode(True, False)
        validate_factor_continuation_mode(True, True)
        validate_factor_continuation_mode(False, False)

    def test_homogeneous_capacity_assigns_one_ratio_to_every_client(self):
        ratios = resolve_capacity_ratios(
            [0.9, 0.37, 0.35, 0.25, 0.15],
            homogeneous_capacity=True,
            homogeneous_ratio=0.35,
        )
        assigned = [ratios[client_id % len(ratios)] for client_id in range(20)]
        self.assertEqual(assigned, [0.35] * 20)

    def test_disabled_homogeneous_capacity_preserves_original_assignment(self):
        defaults = [0.9, 0.37, 0.35, 0.25, 0.15]
        ratios = resolve_capacity_ratios(
            defaults,
            homogeneous_capacity=False,
            homogeneous_ratio=0.35,
        )
        self.assertEqual(ratios, defaults)
        self.assertEqual(
            [ratios[index % len(ratios)] for index in range(10)],
            defaults + defaults,
        )

    def test_factor_average_is_sample_weighted_for_u_v_and_other_params(self):
        first = TinyFactorModel(rank=2, fill=1.0)
        second = TinyFactorModel(rank=2, fill=5.0)
        averaged = sample_weighted_factor_average(
            [first, second], [0.25, 0.75]
        )
        for name, parameter in averaged.named_parameters():
            expected = (
                dict(first.named_parameters())[name] * 0.25
                + dict(second.named_parameters())[name] * 0.75
            )
            self.assertTrue(torch.allclose(parameter, expected), name)

    def test_factor_average_rejects_mismatched_ranks(self):
        with self.assertRaisesRegex(RuntimeError, "shape mismatch"):
            sample_weighted_factor_average(
                [TinyFactorModel(rank=2), TinyFactorModel(rank=3)],
                [0.5, 0.5],
            )

    def test_factor_rank_signature_reports_actual_u_rank(self):
        self.assertEqual(
            factor_rank_signature(TinyFactorModel(rank=3)),
            (("weight_u", 3),),
        )


if __name__ == "__main__":
    unittest.main()

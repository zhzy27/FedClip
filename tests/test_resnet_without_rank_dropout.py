import sys
import unittest
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
SYSTEM_ROOT = ROOT / "system"
if str(SYSTEM_ROOT) not in sys.path:
    sys.path.insert(0, str(SYSTEM_ROOT))

from flcore.trainmodel.SVD_resnet import (  # noqa: E402
    FactorizedConv,
    layer_norm,
    low_rank_resnet18_cifar,
)


class ResNetRankDropoutRemovalTests(unittest.TestCase):
    def test_factorized_conv_uses_all_configured_rank_components(self):
        torch.manual_seed(7)
        layer = FactorizedConv(
            in_channels=4,
            out_channels=6,
            rank_rate=0.5,
            padding=1,
            kernel_size=3,
        )
        layer.train()
        inputs = torch.randn(2, 4, 8, 8)

        first = layer(inputs)
        second = layer(inputs)

        self.assertTrue(torch.equal(first, second))
        self.assertFalse(hasattr(layer, "rank_dropout_enabled"))
        self.assertFalse(hasattr(layer, "rank_dropout_mode"))
        self.assertFalse(hasattr(layer, "rank_dropout_schedule"))

    def test_fedclip_resnet18_family_has_no_rank_dropout_modules(self):
        model = low_rank_resnet18_cifar(
            features=[64, 128, 256, 512],
            num_classes=100,
            zero_init_residual=False,
            groups=1,
            width_per_group=64,
            replace_stride_with_dilation=None,
            norm_layer=layer_norm,
            has_norm=True,
            bn_block_num=4,
            ratio_LR=0.5,
            input_size=32,
        )
        factorized_layers = [
            module for module in model.modules()
            if isinstance(module, FactorizedConv)
        ]

        self.assertEqual(len(factorized_layers), 16)
        self.assertFalse(any(isinstance(module, torch.nn.Dropout) for module in model.modules()))
        for layer in factorized_layers:
            self.assertFalse(hasattr(layer, "rank_dropout_enabled"))
            self.assertFalse(hasattr(layer, "rank_dropout_mode"))
            self.assertFalse(hasattr(layer, "rank_dropout_schedule"))

    def test_active_main_resnet_path_has_no_rank_dropout_arguments(self):
        main_source = (SYSTEM_ROOT / "main.py").read_text(encoding="utf-8")
        start = main_source.index('elif args.model_family == "Decom_resnet18_5"')
        end = main_source.index('elif args.model_family in ["SPU_ResNet18_1"]', start)
        fedclip_resnet_block = main_source[start:end]

        self.assertIn("low_rank_resnet18_cifar", fedclip_resnet_block)
        self.assertNotIn("rank_dropout", fedclip_resnet_block)


if __name__ == "__main__":
    unittest.main()

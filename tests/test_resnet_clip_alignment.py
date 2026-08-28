import importlib.util
import sys
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
SYSTEM_ROOT = REPO_ROOT / "system"
sys.path.insert(0, str(SYSTEM_ROOT))

from utils.resnet_clip_alignment import resolve_resnet_clip_alignment


def make_args(
    legacy=0,
    levels=4,
    anchor_mode="depth",
    weighting="equal",
    final_projector=0,
):
    return SimpleNamespace(
        resnet_clip_legacy=legacy,
        resnet_clip_levels=levels,
        resnet_clip_anchor_mode=anchor_mode,
        resnet_clip_weighting=weighting,
        resnet_clip_final_projector=final_projector,
    )


def load_client_class():
    clientbase_module = types.ModuleType("flcore.clients.clientbase")
    clientbase_module.Client = object
    clientbase_module.load_item = mock.MagicMock()
    clientbase_module.save_item = mock.MagicMock()
    clip_module = types.ModuleType("utils.get_clip_text_encoder")
    clip_module.get_clip_class_embeddings = mock.MagicMock()
    clip_module.get_clip_class_depth_embeddings = mock.MagicMock()
    sklearn_module = types.ModuleType("sklearn")
    preprocessing_module = types.ModuleType("sklearn.preprocessing")
    preprocessing_module.label_binarize = mock.MagicMock()
    stubs = {
        "flcore.clients.clientbase": clientbase_module,
        "utils.get_clip_text_encoder": clip_module,
        "sklearn": sklearn_module,
        "sklearn.preprocessing": preprocessing_module,
    }
    module_path = SYSTEM_ROOT / "flcore" / "clients" / "clientCLIP.py"
    spec = importlib.util.spec_from_file_location(
        "resnet_clip_alignment_client_test_module", module_path
    )
    module = importlib.util.module_from_spec(spec)
    with mock.patch.dict(sys.modules, stubs):
        spec.loader.exec_module(module)
    return module.clientCLIP


def make_client(args, anchor_dim=5, num_classes=3):
    client_class = load_client_class()
    client = client_class.__new__(client_class)
    client.device = torch.device("cpu")
    client.mse_fn = torch.nn.MSELoss()
    client.loss = torch.nn.CrossEntropyLoss()
    client.use_resnet_multilevel_clip = True
    client.resnet_clip_strategy = resolve_resnet_clip_alignment(args)
    client.resnet_clip_aligners = None
    client._resnet_aligner_stage_indices = ()
    client.clip_text_features = torch.zeros(num_classes, anchor_dim)
    client.clip_text_features_norm = torch.zeros(num_classes, anchor_dim)
    client.clip_text_depth_features = torch.zeros(4, num_classes, anchor_dim)
    client.clip_text_depth_features_norm = torch.zeros(4, num_classes, anchor_dim)
    return client


class IdentityBase(torch.nn.Module):
    def forward(self, inputs):
        return inputs


class IdentityModel(torch.nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.base = IdentityBase()
        self.head = torch.nn.Linear(dim, 3, bias=False)


class ResNetClipAlignmentTest(unittest.TestCase):
    def test_legacy_overrides_every_new_option(self):
        strategy = resolve_resnet_clip_alignment(
            make_args(legacy=1, levels=0, anchor_mode="final", weighting="deep")
        )
        self.assertEqual(strategy.selected_stage_indices, (0, 1, 2, 3))
        self.assertEqual(strategy.anchor_mode, "depth")
        self.assertEqual(strategy.weighting, "equal")
        self.assertTrue(strategy.final_projector)
        self.assertEqual(strategy.aligner_stage_indices, (0, 1, 2, 3))

    def test_levels_select_stages_from_deep_to_shallow(self):
        expected = {
            0: (),
            1: (3,),
            2: (2, 3),
            3: (1, 2, 3),
            4: (0, 1, 2, 3),
        }
        for levels, selected in expected.items():
            with self.subTest(levels=levels):
                self.assertEqual(
                    resolve_resnet_clip_alignment(make_args(levels=levels)).selected_stage_indices,
                    selected,
                )

    def test_levels_zero_has_zero_loss_and_no_aligner(self):
        client = make_client(make_args(levels=0))
        stages = [torch.randn(2, 5, requires_grad=True) for _ in range(4)]
        loss = client._resnet_multilevel_clip_loss(stages, torch.tensor([0, 1]))
        self.assertEqual(loss.item(), 0.0)
        self.assertIsNone(client.resnet_clip_aligners)
        loss.backward()
        self.assertIsNotNone(stages[-1].grad)

    def test_final_stage_is_direct_by_default(self):
        client = make_client(make_args(levels=1, anchor_mode="final", final_projector=0))
        stages = [torch.randn(2, dim) for dim in (2, 3, 4, 5)]
        labels = torch.tensor([0, 1])
        client.clip_text_features = torch.randn(3, 5)
        expected = torch.nn.functional.mse_loss(stages[-1], client.clip_text_features[labels])
        actual = client._resnet_multilevel_clip_loss(stages, labels)
        torch.testing.assert_close(actual, expected)
        self.assertIsNone(client.resnet_clip_aligners)

    def test_final_projector_is_created_and_used_when_enabled(self):
        client = make_client(make_args(levels=1, anchor_mode="final", final_projector=1))
        stages = [torch.ones(2, 5) for _ in range(4)]
        labels = torch.tensor([0, 1])
        client.clip_text_features = torch.ones(3, 5)
        aligners = client._ensure_resnet_clip_aligners(stages)
        self.assertEqual(len(aligners), 1)
        self.assertEqual(client._resnet_aligner_stage_indices, (3,))
        with torch.no_grad():
            aligners[0].weight.zero_()
            aligners[0].bias.zero_()
        loss = client._resnet_multilevel_clip_loss(stages, labels)
        torch.testing.assert_close(loss, torch.tensor(1.0))

    def test_depth_and_final_anchor_modes_use_different_sources(self):
        labels = torch.tensor([0, 1])
        stages = [torch.zeros(2, 5) for _ in range(4)]

        depth_client = make_client(make_args(levels=1, anchor_mode="depth"))
        depth_client.clip_text_depth_features[3].fill_(1.0)
        depth_client.clip_text_features.fill_(2.0)
        depth_loss = depth_client._resnet_multilevel_clip_loss(stages, labels)

        final_client = make_client(make_args(levels=1, anchor_mode="final"))
        final_client.clip_text_depth_features[3].fill_(1.0)
        final_client.clip_text_features.fill_(2.0)
        final_loss = final_client._resnet_multilevel_clip_loss(stages, labels)

        torch.testing.assert_close(depth_loss, torch.tensor(1.0))
        torch.testing.assert_close(final_loss, torch.tensor(4.0))

    def test_equal_and_deep_weighting_match_the_defined_math(self):
        stages = [torch.full((2, 5), float(index + 1)) for index in range(4)]
        labels = torch.tensor([0, 1])

        equal_client = make_client(
            make_args(levels=4, anchor_mode="final", weighting="equal")
        )
        equal_aligners = equal_client._ensure_resnet_clip_aligners(stages)
        for aligner in equal_aligners:
            with torch.no_grad():
                aligner.weight.copy_(torch.eye(5))
                aligner.bias.zero_()
        equal_loss = equal_client._resnet_multilevel_clip_loss(stages, labels)

        deep_client = make_client(
            make_args(levels=4, anchor_mode="final", weighting="deep")
        )
        deep_aligners = deep_client._ensure_resnet_clip_aligners(stages)
        for aligner in deep_aligners:
            with torch.no_grad():
                aligner.weight.copy_(torch.eye(5))
                aligner.bias.zero_()
        deep_loss = deep_client._resnet_multilevel_clip_loss(stages, labels)

        torch.testing.assert_close(equal_loss, torch.tensor((1 + 4 + 9 + 16) / 4))
        torch.testing.assert_close(deep_loss, torch.tensor((1 + 8 + 36 + 128) / 15))

    def test_legacy_loss_matches_the_original_four_aligner_formula(self):
        client = make_client(make_args(legacy=1))
        stages = [torch.randn(2, dim) for dim in (2, 3, 4, 5)]
        labels = torch.tensor([0, 1])
        aligners = client._ensure_resnet_clip_aligners(stages)
        expected_losses = []
        for stage_idx, (stage, aligner) in enumerate(zip(stages, aligners)):
            expected_losses.append(
                client.mse_fn(
                    aligner(stage),
                    client.clip_text_depth_features[stage_idx][labels],
                )
            )
        expected = sum(expected_losses) / 4
        actual = client._resnet_multilevel_clip_loss(stages, labels)
        torch.testing.assert_close(actual, expected)

    def test_cnn_forward_remains_single_final_feature_mse(self):
        client = make_client(make_args(levels=0), anchor_dim=4)
        client.use_resnet_multilevel_clip = False
        client.clip_text_features = torch.randn(3, 4)
        model = IdentityModel(4)
        inputs = torch.randn(5, 4)
        labels = torch.tensor([0, 1, 2, 0, 1])

        logits, anchor_loss = client._forward_clip_outputs(model, inputs, labels)
        torch.testing.assert_close(logits, model.head(inputs))
        torch.testing.assert_close(
            anchor_loss,
            torch.nn.functional.mse_loss(inputs, client.clip_text_features[labels]),
        )


if __name__ == "__main__":
    unittest.main()

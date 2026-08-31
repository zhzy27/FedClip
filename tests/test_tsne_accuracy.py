import contextlib
import importlib.util
import io
import os
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest import mock

import numpy as np
import torch
from torch import nn
from torch.utils.data import TensorDataset


ROOT = Path(__file__).resolve().parents[1]


def load_tsne():
    data_utils = types.ModuleType("utils.data_utils")
    data_utils.read_client_data = mock.Mock()
    spec = importlib.util.spec_from_file_location(
        "legacy_tsne_accuracy_test", ROOT / "system" / "T-SNE-Cifar-legacy-compatible.py"
    )
    module = importlib.util.module_from_spec(spec)
    cwd = os.getcwd()
    try:
        with mock.patch.dict(sys.modules, {"utils.data_utils": data_utils}):
            spec.loader.exec_module(module)
    finally:
        os.chdir(cwd)
    return module


class CountingBase(nn.Identity):
    def __init__(self):
        super().__init__()
        self.calls = 0

    def forward(self, x):
        self.calls += 1
        return x


class ToyClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.base = CountingBase()
        self.head = nn.Linear(2, 2, bias=False)
        with torch.no_grad():
            self.head.weight.copy_(torch.eye(2))

    def forward(self, x):
        return self.head(self.base(x))


def make_args(**overrides):
    values = dict(
        algorithm="FedCLIP", dataset="Toy", num_classes=2, num_clients=2,
        model_source="client", client_ids="0", split="test", max_batches=0,
        max_samples_per_client=0, batch_size=2, partition="pat", niid=1,
        dir_alpha=0.3, class_per_client=2, point_size=18, alpha=0.7,
        show_legend=True, max_legend_classes=20, save_excel=False,
    )
    values.update(overrides)
    return types.SimpleNamespace(**values)


class TsneAccuracyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tsne = load_tsne()

    def setUp(self):
        self.model = ToyClassifier().eval()
        self.inputs = torch.tensor([[3., 0.], [0., 3.], [3., 0.], [0., 3.], [3., 0.]])
        self.labels = torch.tensor([0, 1, 1, 1, 0])

    def collect(self, args, data, max_batches=0, evaluate=True):
        with (
            mock.patch.object(self.tsne, "resolve_model_path", return_value="Client_0_model.pt"),
            mock.patch.object(self.tsne, "torch_load_model", return_value=self.model),
            mock.patch.object(self.tsne, "read_client_data", return_value=data) as read,
        ):
            result = self.tsne.collect_one_client_features(
                args, ".", torch.device("cpu"), 0, max_batches,
                verbose=False, evaluate_accuracy=evaluate,
            )
        return result, read

    def test_train_and_test_use_the_selected_split(self):
        for split, labels, expected_correct in (
            ("train", self.labels, 4),
            ("test", 1 - self.inputs.argmax(dim=1), 0),
        ):
            with self.subTest(split=split):
                (features, truth, pred), read = self.collect(
                    make_args(split=split), TensorDataset(self.inputs, labels)
                )
                self.assertEqual(read.call_args.kwargs["is_train"], split == "train")
                self.assertEqual(np.count_nonzero(pred == truth), expected_correct)
                np.testing.assert_array_equal(features, self.inputs.numpy())

    def test_partial_batch_limit_also_limits_accuracy_and_reuses_features(self):
        (features, labels, pred), _ = self.collect(
            make_args(max_samples_per_client=3), TensorDataset(self.inputs, self.labels)
        )
        self.assertEqual((len(features), len(labels), len(pred)), (3, 3, 3))
        self.assertEqual(np.count_nonzero(pred == labels), 2)
        self.assertEqual(self.model.base.calls, 2)
        self.assertTrue(all(p.grad is None for p in self.model.parameters()))
        torch.testing.assert_close(self.model.head.weight, torch.eye(2))

    def test_batch_limit_also_limits_accuracy(self):
        (features, labels, pred), _ = self.collect(
            make_args(), TensorDataset(self.inputs, self.labels), max_batches=1
        )
        self.assertEqual(len(features), 2)
        self.assertTrue(np.all(pred == labels))

    def test_client_selection_does_not_require_classifier_predictions(self):
        with mock.patch.object(self.model.head, "forward", side_effect=AssertionError("no head call")):
            (features, _, pred), _ = self.collect(
                make_args(), TensorDataset(self.inputs, self.labels), evaluate=False
            )
        self.assertEqual(len(features), 5)
        self.assertIsNone(pred)

    def test_multiple_clients_use_total_correct_over_total_samples(self):
        data = {
            0: TensorDataset(torch.tensor([[3., 0.], [0., 3.]]), torch.tensor([0, 1])),
            1: TensorDataset(torch.tensor([[3., 0.]] * 6), torch.ones(6, dtype=torch.long)),
        }
        with (
            mock.patch.object(self.tsne, "resolve_model_path", return_value="model.pt"),
            mock.patch.object(self.tsne, "torch_load_model", return_value=self.model),
            mock.patch.object(self.tsne, "read_client_data", side_effect=lambda _, cid, **kw: data[cid]),
            contextlib.redirect_stdout(io.StringIO()) as log,
        ):
            features, labels, client_ids, pred = self.tsne.collect_legacy_features(
                make_args(client_ids="0,1"), ".", torch.device("cpu")
            )
        self.assertEqual(len(features), 8)
        self.assertAlmostEqual(float(np.mean(pred == labels)), 0.25)
        np.testing.assert_array_equal(client_ids, [0, 0, 1, 1, 1, 1, 1, 1])
        self.assertIn("25.00% (2/8)", log.getvalue())

    def test_forward_only_model_and_invalid_logits(self):
        with torch.no_grad():
            pred = self.tsne.predict_from_legacy_features(nn.Identity(), self.inputs, self.inputs, 2)
            np.testing.assert_array_equal(pred, [0, 1, 0, 1, 0])
            with self.assertRaisesRegex(ValueError, "classifier logits"):
                self.tsne.predict_from_legacy_features(nn.Identity(), self.inputs, self.inputs, 100)
            invalid = self.inputs.clone()
            invalid[0, 0] = float("nan")
            with self.assertRaisesRegex(ValueError, "NaN or Inf"):
                self.tsne.predict_from_legacy_features(nn.Identity(), invalid, invalid, 2)

    def test_csv_contains_predictions_and_plot_labels_both_formats(self):
        import pandas as pd
        from matplotlib.figure import Figure

        for split in ("train", "test"):
            with self.subTest(split=split), tempfile.TemporaryDirectory() as output_dir:
                args = make_args(split=split)
                df = self.tsne.make_dataframe(
                    np.array([[0., 0.], [1., 1.], [2., 0.]]),
                    np.array([0, 1, 0]), np.array([0, 0, 0]), args, np.array([0, 1, 1]),
                )
                self.tsne.save_outputs(df, output_dir, args)
                saved = pd.read_csv(Path(output_dir) / "legacy_tsne_all_points.csv")
                self.assertEqual(saved["prediction"].tolist(), [0, 1, 1])
                self.assertEqual(saved["correct"].tolist(), [True, True, False])
                self.assertEqual(saved["split"].tolist(), [split] * 3)

                def check_annotation(fig, path, **kwargs):
                    ax = fig.axes[0]
                    label = ax.texts[0]
                    self.assertEqual(label.get_text(), f"{split.capitalize()} accuracy: 66.67%")
                    self.assertEqual(label.get_position(), (0.02, 0.98))
                    self.assertIs(label.get_transform(), ax.transAxes)

                with mock.patch.object(Figure, "savefig", autospec=True, side_effect=check_annotation) as save:
                    self.tsne.plot_legacy_tsne(df, output_dir, args)
                self.assertEqual(save.call_count, 2)
                self.assertEqual({Path(call.args[1]).suffix for call in save.call_args_list}, {".png", ".pdf"})

    def test_prototype_methods_are_not_labeled_as_prototype_accuracy(self):
        for algorithm in ("FedTGP", "FedProto"):
            self.assertEqual(self.tsne.accuracy_label(make_args(algorithm=algorithm)), "Test head accuracy")


if __name__ == "__main__":
    unittest.main()

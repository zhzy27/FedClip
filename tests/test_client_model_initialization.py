import importlib.util
import sys
import tempfile
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
SYSTEM_ROOT = REPO_ROOT / "system"


def load_clientbase_module(model_factory):
    sklearn_module = types.ModuleType("sklearn")
    sklearn_module.metrics = types.SimpleNamespace()
    preprocessing_module = types.ModuleType("sklearn.preprocessing")
    preprocessing_module.label_binarize = mock.MagicMock()
    data_module = types.ModuleType("utils.data_utils")
    data_module.read_client_data = mock.MagicMock()
    models_module = types.ModuleType("flcore.trainmodel.models")
    models_module.BaseHeadSplit = mock.MagicMock()
    models_module.Model_Distribe = model_factory
    stubs = {
        "sklearn": sklearn_module,
        "sklearn.preprocessing": preprocessing_module,
        "utils.data_utils": data_module,
        "flcore.trainmodel.models": models_module,
    }
    module_path = SYSTEM_ROOT / "flcore" / "clients" / "clientbase.py"
    spec = importlib.util.spec_from_file_location(
        "client_model_initialization_test_module", module_path
    )
    module = importlib.util.module_from_spec(spec)
    with mock.patch.dict(sys.modules, stubs):
        spec.loader.exec_module(module)
    return module


def make_args(run_dir, resume=False):
    return SimpleNamespace(
        algorithm="FedCLIP",
        dataset="Cifar100",
        device=torch.device("cpu"),
        save_folder_name="temp/custom_root",
        save_folder_name_full=str(run_dir),
        num_classes=100,
        batch_size=16,
        local_learning_rate=0.005,
        local_epochs=5,
        few_shot=False,
        models_folder_name=None,
        resume=resume,
    )


class ClientModelInitializationTest(unittest.TestCase):
    def test_fresh_custom_temp_root_always_creates_client_model(self):
        factory = mock.MagicMock(
            side_effect=lambda args, client_id: torch.nn.Linear(2, 2)
        )
        module = load_clientbase_module(factory)
        with tempfile.TemporaryDirectory() as temp_dir:
            run_dir = Path(temp_dir) / "temp" / "custom" / "run"
            module.Client(
                make_args(run_dir),
                id=8,
                train_samples=10,
                test_samples=5,
                train_slow=False,
                send_slow=False,
            )

            model_path = run_dir / "Client_8_model.pt"
            self.assertTrue(model_path.is_file())
            self.assertIsNotNone(
                module.load_item("Client_8", "model", str(run_dir))
            )
            factory.assert_called_once()

    def test_resume_reuses_existing_model_without_reinitializing(self):
        factory = mock.MagicMock(
            side_effect=lambda args, client_id: torch.nn.Linear(2, 2)
        )
        module = load_clientbase_module(factory)
        with tempfile.TemporaryDirectory() as temp_dir:
            run_dir = Path(temp_dir) / "existing_run"
            original = torch.nn.Linear(2, 2)
            module.save_item(original, "Client_8", "model", str(run_dir))

            module.Client(
                make_args(run_dir, resume=True),
                id=8,
                train_samples=10,
                test_samples=5,
                train_slow=False,
                send_slow=False,
            )

            factory.assert_not_called()

    def test_resume_reports_missing_client_model(self):
        module = load_clientbase_module(mock.MagicMock())
        with tempfile.TemporaryDirectory() as temp_dir:
            with self.assertRaisesRegex(
                RuntimeError, "Resume run is missing Client_8_model.pt"
            ):
                module.Client(
                    make_args(Path(temp_dir), resume=True),
                    id=8,
                    train_samples=10,
                    test_samples=5,
                    train_slow=False,
                    send_slow=False,
                )


if __name__ == "__main__":
    unittest.main()

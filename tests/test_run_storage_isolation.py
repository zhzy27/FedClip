import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock


REPO_ROOT = Path(__file__).resolve().parents[1]
SYSTEM_ROOT = REPO_ROOT / "system"
sys.path.insert(0, str(SYSTEM_ROOT))

try:
    import h5py  # noqa: F401
except ModuleNotFoundError:
    sys.modules["h5py"] = mock.MagicMock()

try:
    import sklearn  # noqa: F401
except ModuleNotFoundError:
    sys.modules["sklearn"] = mock.MagicMock()
    sys.modules["sklearn.preprocessing"] = mock.MagicMock()
    sys.modules["sklearn.metrics"] = mock.MagicMock()

clientbase_stub = types.ModuleType("flcore.clients.clientbase")
clientbase_stub.load_item = mock.MagicMock()
clientbase_stub.save_item = mock.MagicMock()
sys.modules["flcore.clients.clientbase"] = clientbase_stub

from flcore.servers.serverbase import Server
from export_final_models_from_json import final_model_dir as exported_final_model_dir


def make_args(save_root, final_root):
    return SimpleNamespace(
        device="cpu",
        dataset="Cifar100",
        num_classes=100,
        global_rounds=1,
        local_epochs=1,
        batch_size=16,
        local_learning_rate=0.005,
        num_clients=20,
        join_ratio=1.0,
        random_join_ratio=False,
        few_shot=False,
        algorithm="FedCLIP",
        time_select=False,
        goal="test",
        time_threthold=1e9,
        top_cnt=20,
        auto_break=False,
        save_folder_name=str(save_root),
        eval_gap=1,
        client_drop_rate=0.0,
        train_slow_rate=0.0,
        send_slow_rate=0.0,
        resume=False,
        final_model_root=str(final_root),
        model_family="Decom_CNN-5-512",
        niid=1,
        partition="dir",
        dir_alpha=0.5,
        class_per_client=20,
    )


def create_server(args, pid):
    with mock.patch("flcore.servers.serverbase.time.time_ns", return_value=123456789):
        with mock.patch("flcore.servers.serverbase.os.getpid", return_value=pid):
            return Server(args, times=0)


class RunStorageIsolationTest(unittest.TestCase):
    def test_parallel_runs_use_distinct_work_and_final_directories(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            save_root = temp_path / "temp"
            final_root = temp_path / "final_models"
            server_a = create_server(make_args(save_root, final_root), pid=1001)
            server_b = create_server(make_args(save_root, final_root), pid=1002)

            self.assertNotEqual(server_a.run_id, server_b.run_id)
            self.assertNotEqual(server_a.save_folder_name, server_b.save_folder_name)
            self.assertTrue(Path(server_a.save_folder_name).is_dir())
            self.assertTrue(Path(server_b.save_folder_name).is_dir())
            self.assertNotEqual(server_a.final_model_dir(), server_b.final_model_dir())
            self.assertEqual(Path(server_a.final_model_dir()).parent.name, "runs")

            for server in (server_a, server_b):
                Path(server.save_folder_name, "Server_model.pt").write_bytes(b"model")
                with mock.patch("builtins.print"):
                    server.export_final_models()

            self.assertEqual(
                Path(server_a.final_model_dir(), "Server_model.pt").read_bytes(), b"model"
            )
            self.assertEqual(
                Path(server_b.final_model_dir(), "Server_model.pt").read_bytes(), b"model"
            )
            self.assertTrue(Path(server_a.final_model_dir(), "manifest.json").is_file())
            self.assertTrue(Path(server_b.final_model_dir(), "manifest.json").is_file())

    def test_custom_save_root_also_receives_a_unique_run_directory(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            args = make_args(temp_path / "custom-output", temp_path / "final_models")
            server = create_server(args, pid=os.getpid())

            expected_parent = Path(args.save_folder_name) / args.dataset / args.algorithm
            self.assertEqual(Path(server.save_folder_name).parent, expected_parent)
            self.assertEqual(Path(server.save_folder_name).name, server.run_id)

    def test_resume_requires_and_reuses_an_explicit_run_directory(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            args = make_args("temp", temp_path / "final_models")
            args.resume = True
            with self.assertRaises(ValueError):
                create_server(args, pid=1001)

            existing_run = temp_path / "temp" / "Cifar100" / "FedCLIP" / "existing_run"
            existing_run.mkdir(parents=True)
            args.save_folder_name = str(existing_run)
            server = create_server(args, pid=1001)
            self.assertEqual(Path(server.save_folder_name), existing_run)
            self.assertEqual(server.run_id, "existing_run")

    def test_legacy_json_exports_are_isolated_by_source_run(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            base_dir = Path(temp_dir)
            common_args = {
                "dataset": "Cifar100",
                "algorithm": "FedCLIP",
                "model_family": "Decom_CNN-5-512",
                "partition": "dir",
                "dir_alpha": 0.5,
                "num_classes": 100,
                "niid": 1,
                "num_clients": 20,
                "join_ratio": 1.0,
            }
            data_a = {"args": {**common_args, "save_folder_name_full": "temp/Cifar100/FedCLIP/run_a"}}
            data_b = {"args": {**common_args, "save_folder_name_full": "temp/Cifar100/FedCLIP/run_b"}}

            target_a = exported_final_model_dir(data_a, "final_models", str(base_dir))
            target_b = exported_final_model_dir(data_b, "final_models", str(base_dir))

            self.assertNotEqual(target_a, target_b)
            self.assertEqual(Path(target_a).parent.name, "runs")
            self.assertEqual(Path(target_a).name, "run_a")
            self.assertEqual(Path(target_b).name, "run_b")


if __name__ == "__main__":
    unittest.main()

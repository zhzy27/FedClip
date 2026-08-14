import importlib.util
import csv
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
SERVER_PATH = ROOT / "system" / "flcore" / "servers" / "serverCLIP.py"


def load_server_class():
    client_clip = types.ModuleType("flcore.clients.clientCLIP")
    client_clip.clientCLIP = object
    clientbase = types.ModuleType("flcore.clients.clientbase")
    clientbase.load_item = lambda *args, **kwargs: None
    clientbase.save_item = lambda *args, **kwargs: None
    serverbase = types.ModuleType("flcore.servers.serverbase")
    serverbase.Server = object
    models = types.ModuleType("flcore.trainmodel.models")
    models.Model_Distribe = object

    with patch.dict(
        sys.modules,
        {
            "flcore.clients.clientCLIP": client_clip,
            "flcore.clients.clientbase": clientbase,
            "flcore.servers.serverbase": serverbase,
            "flcore.trainmodel.models": models,
        },
    ):
        spec = importlib.util.spec_from_file_location(
            "serverclip_device_test", SERVER_PATH
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    return module.FedCLIP


FedCLIP = load_server_class()


class RecoveryCreatesCpuParameters:
    def __init__(self):
        self.recovered = False
        self.events = []

    def named_parameters(self):
        name = "layer.weight" if self.recovered else "layer.weight_v"
        return [(name, object())]

    def recover_larger_model(self):
        self.events.append("recover_to_cpu")
        self.recovered = True

    def to(self, device):
        self.events.append(f"to:{device}")
        return self


class ServerAvgDeviceAlignmentTest(unittest.TestCase):
    def test_recovered_model_is_moved_to_server_device_again(self):
        server = FedCLIP.__new__(FedCLIP)
        server.device = "cuda:0"
        model = RecoveryCreatesCpuParameters()

        result = server._recover_if_needed(model)

        self.assertIs(result, model)
        self.assertEqual(model.events, ["recover_to_cpu", "to:cuda:0"])

    def test_subspace_diagnostics_are_written_to_run_directory(self):
        server = FedCLIP.__new__(FedCLIP)
        summary = {
            "round": 2,
            "client_id": 4,
            "u_lr": 0.0015,
            "v_lr": 0.005,
            "lambda_sub": 0.3,
            "mean_subspace_drift_norm": 0.02,
            "max_subspace_drift_norm": 0.04,
            "mean_principal_angle_deg": 1.2,
            "max_principal_angle_deg": 3.4,
            "mean_R_U": 0.01,
            "mean_R_V": 0.05,
            "u_base_grad_norm": 2.0,
            "u_sub_grad_norm": 1.0,
            "u_sub_weighted_grad_norm": 0.3,
            "u_sub_to_base_grad_ratio": 0.15,
            "u_base_sub_grad_cos": -0.2,
            "mean_u_sigma_min": 0.8,
            "mean_u_condition_number": 2.5,
            "mean_pre_clip_grad_norm": 8.0,
            "max_pre_clip_grad_norm": 12.0,
            "clip_trigger_fraction": 0.25,
        }
        layer = {
            "round": 2,
            "client_id": 4,
            "layer_name": "base.fc1.weight_u",
            "rank": 8,
            "u_lr": 0.0015,
            "lambda_sub": 0.3,
            "subspace_drift_sq": 0.0004,
            "subspace_drift_norm": 0.02,
            "principal_angle_mean_deg": 1.2,
            "principal_angle_max_deg": 3.4,
            "principal_angle_median_deg": 0.9,
            "R_U": 0.01,
            "R_V": 0.05,
            "u_sigma_min": 0.8,
            "u_sigma_max": 2.0,
            "u_condition_number": 2.5,
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            server.save_folder_name = temp_dir
            server._write_u_subspace_diagnostics([summary], [layer])
            server._write_u_subspace_diagnostics([summary], [layer])

            summary_path = Path(temp_dir) / "u_subspace_round_summary.csv"
            layer_path = Path(temp_dir) / "u_subspace_layer_stats.csv"
            with summary_path.open(newline="", encoding="utf-8") as handle:
                summary_rows = list(csv.DictReader(handle))
            with layer_path.open(newline="", encoding="utf-8") as handle:
                layer_rows = list(csv.DictReader(handle))

            self.assertEqual(len(summary_rows), 2)
            self.assertEqual(len(layer_rows), 2)
            self.assertEqual(summary_rows[0]["client_id"], "4")
            self.assertEqual(
                summary_rows[0]["clip_trigger_fraction"], "0.25"
            )
            self.assertEqual(
                layer_rows[0]["layer_name"], "base.fc1.weight_u"
            )

    def test_custom_diagnostic_root_keeps_runs_isolated(self):
        server = FedCLIP.__new__(FedCLIP)
        server.args = SimpleNamespace(u_subspace_diag_dir="diagnostics")
        server.dataset = "Cifar100"
        server.algorithm = "FedCLIP"
        server.run_id = "run_123"
        server.save_folder_name = "temp/fallback"

        output_dir = server._resolve_u_subspace_diag_output_dir()

        self.assertEqual(
            output_dir,
            Path("diagnostics") / "Cifar100" / "FedCLIP" / "run_123",
        )

    def test_launcher_log_directory_is_used_when_cli_path_is_empty(self):
        server = FedCLIP.__new__(FedCLIP)
        server.args = SimpleNamespace(u_subspace_diag_dir="")
        server.save_folder_name = "temp/fallback"

        with patch.dict(
            os.environ,
            {"FEDCLIP_TRAIN_LOG_DIR": "runs/task_01"},
        ):
            output_dir = server._resolve_u_subspace_diag_output_dir()

        self.assertEqual(output_dir, Path("runs/task_01"))


if __name__ == "__main__":
    unittest.main()

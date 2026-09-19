import csv
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest import mock

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "system"))
from utils.round_costs import RoundCosts, exclude_measurement, payload_bytes, upload_payload


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Clock:
    value = 0.0

    def __call__(self):
        return self.value

    def advance(self, seconds):
        self.value += seconds


class RoundCostsTest(unittest.TestCase):
    def recorder(self, algorithm="FedCLIP"):
        self.clock = Clock()
        return RoundCosts(algorithm, "cpu", clock=self.clock, synchronize=lambda: None)

    def test_exclusive_server_local_io_and_nested_calls(self):
        costs = self.recorder()
        with costs.session():
            costs.begin_round(0, [0, 1])
            self.clock.advance(2)
            with costs.scope("server", "send_parameters"):
                self.clock.advance(3)  # Includes full-W -> SVD download preparation.
                with exclude_measurement():
                    self.clock.advance(100)
                    with exclude_measurement():
                        self.clock.advance(100)
            with costs.scope("local", "train", 0):
                self.clock.advance(5)
                with exclude_measurement():
                    self.clock.advance(100)
            with costs.scope("local", "train", 1):
                self.clock.advance(7)
            with costs.scope("server", "aggregate_parameters"):
                self.clock.advance(4)
                with costs.scope("server", "get_filters"):
                    self.clock.advance(1)
            row = costs.finish_round()
        self.assertEqual(row["local_train_sum_seconds"], 12)
        self.assertEqual(row["local_train_max_seconds"], 7)
        self.assertEqual(row["server_processing_seconds"], 10)
        self.assertEqual(row["server_processing_events"]["send_parameters"], 3)

    def test_console_writes_are_excluded_and_stream_restored(self):
        clock = Clock()
        class SlowStream(io.StringIO):
            def write(self, text):
                clock.advance(10)
                return super().write(text)
        costs = RoundCosts("FD", "cpu", clock=clock, synchronize=lambda: None)
        target = SlowStream()
        with mock.patch("sys.stdout", target):
            with costs.session():
                costs.begin_round(0, [0])
                clock.advance(2)
                print("test")
                row = costs.finish_round()
            self.assertIs(sys.stdout, target)
        self.assertEqual(row["server_processing_seconds"], 2)

    def test_payload_bytes_dtype_numpy_and_head_only(self):
        model = torch.nn.Module()
        model.base = torch.nn.Linear(20, 20)
        model.head = torch.nn.Linear(20, 2)
        model.register_buffer("not_aggregated", torch.ones(100))
        whole = sum(p.numel() * p.element_size() for p in model.parameters())
        head = sum(p.numel() * p.element_size() for p in model.head.parameters())
        for algo in ("FedCLIP", "FML", "FedMRL", "PFedAFM"):
            item = "model" if algo == "FedCLIP" else "global_model"
            self.assertEqual(payload_bytes(upload_payload(algo, item, model)), whole)
        for algo in ("LG-FedAvg", "FedGen"):
            self.assertEqual(payload_bytes(upload_payload(algo, "model", model)), head)
        for algo in ("FedProto", "FedGH", "FedTGP"):
            self.assertEqual(payload_bytes(upload_payload(algo, "protos", {0: torch.zeros(512)})), 2048)
        self.assertEqual(payload_bytes(upload_payload("FD", "logits", {0: torch.zeros(10)})), 40)
        compressed = {"w": [np.zeros((2, 1), dtype=np.float32),
                             np.ones(1, dtype=np.float64), torch.ones((1, 3), dtype=torch.float16)]}
        self.assertEqual(payload_bytes(upload_payload("FedKD", "compressed_param", compressed)), 22)
        self.assertEqual(payload_bytes(upload_payload("FedSPU", "updated_parameters", [np.ones(3)])), 24)

    def test_upload_dedup_download_exclusion_and_round_reset(self):
        costs = self.recorder()
        model = torch.nn.Linear(2, 1)
        with costs.session():
            for rnd in (0, 1):
                costs.begin_round(rnd, [0])
                with costs.scope("server", "send_parameters"):
                    costs.observe_upload(0, "model", model)
                self.assertFalse(costs.uploads)
                with costs.scope("local", "train", 0):
                    costs.observe_upload(0, "model", model)
                self.assertFalse(costs.uploads)
                costs.observe_upload(0, "model", model)
                costs.observe_upload(0, "model", model)
                row = costs.finish_round()
                self.assertEqual(row["upload_payload_bytes"], 12)
                self.assertEqual(len(row["upload_client_details"]), 1)
        self.assertEqual(len(costs.records), 2)

    def test_csv_upgrade_preserves_history_and_blank_unmeasured_values(self):
        runner = load_module("cost_runner", "system/run_compare_compute.py")
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "results.csv"
            path.write_text("algorithm,server_total_seconds\nFD,0.01\n", encoding="utf-8")
            runner.append_rows(path, [{"algorithm": "FD", "server_processing_seconds": 0.002,
                                      "upload_payload_bytes": 40}])
            with path.open(encoding="utf-8-sig", newline="") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(rows[0]["server_total_seconds"], "0.01")
            self.assertEqual(rows[0]["server_processing_seconds"], "")
            self.assertEqual(rows[1]["upload_payload_bytes"], "40")
            self.assertNotIn(None, rows[1])


class TrainingIntegrationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Stub optional dataset/model imports only; use real Client/Server wrappers,
        # torch model serialization, SGD, and SVD in this integration test.
        sklearn = types.ModuleType("sklearn")
        sklearn.metrics = types.SimpleNamespace()
        preprocessing = types.ModuleType("sklearn.preprocessing")
        preprocessing.label_binarize = mock.Mock()
        models = types.ModuleType("flcore.trainmodel.models")
        models.BaseHeadSplit = models.Model_Distribe = mock.Mock()
        stubs = {"sklearn": sklearn, "sklearn.preprocessing": preprocessing,
                 "flcore.trainmodel.models": models, "h5py": mock.Mock()}
        with mock.patch.dict(sys.modules, stubs):
            cls.clientbase = load_module("cost_client", "system/flcore/clients/clientbase.py")
            with mock.patch.dict(sys.modules, {"flcore.clients.clientbase": cls.clientbase}):
                cls.serverbase = load_module("cost_server", "system/flcore/servers/serverbase.py")

    def run_training(self, enabled):
        cb, sb = self.clientbase, self.serverbase
        costs = RoundCosts("FedCLIP", "cpu") if enabled else None
        torch.manual_seed(5)
        class ToyClient(cb.Client):
            def set_parameters(self):
                model = cb.load_item("Server", "model", self.save_folder_name)
                # Re-SVD is inside the real send_parameters boundary.
                u, s, vh = torch.linalg.svd(model.weight.detach(), full_matrices=False)
                with torch.no_grad():
                    model.weight.copy_((u * s) @ vh)
                cb.save_item(model, self.role, "model", self.save_folder_name)

            def train(self):
                model = cb.load_item(self.role, "model", self.save_folder_name)
                optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
                loss = model(torch.ones(2, 3)).square().mean()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                cb.save_item(model, self.role, "model", self.save_folder_name)

        class ToyServer(sb.Server):
            def train(self):
                for _ in range(2):
                    self.selected_clients = self.select_clients()
                    self.send_parameters()
                    for client in self.selected_clients:
                        client.train()
                    self.aggregate_parameters()
                self.save_json_file()

            def aggregate_parameters(self):
                models = [cb.load_item(c.role, "model", self.save_folder_name) for c in self.clients]
                with torch.no_grad():
                    for params in zip(*(m.parameters() for m in models)):
                        params[0].copy_((params[0] + params[1]) / 2)
                cb.save_item(models[0], "Server", "model", self.save_folder_name)

            def save_json_file(self):
                self.save_json(str(Path(self.save_folder_name) / "results.json"), {})

        with tempfile.TemporaryDirectory() as folder:
            server = ToyServer.__new__(ToyServer)
            server.args = types.SimpleNamespace(measure_local_flops=0, measure_server_compute=0)
            server._round_costs = costs
            server._local_flops_select_count = 0
            server.random_join_ratio = False
            server.num_join_clients = 2
            server.save_folder_name = folder
            server.clients = []
            for cid in range(2):
                client = ToyClient.__new__(ToyClient)
                client.id, client.role = cid, f"Client_{cid}"
                client.save_folder_name = folder
                client.send_time_cost = {"num_rounds": 0, "total_cost": 0}
                server.clients.append(client)
            cb.save_item(torch.nn.Linear(3, 2), "Server", "model", folder)
            np.random.seed(5)
            with mock.patch("sys.stdout", io.StringIO()):
                server.train()
            result = cb.load_item("Server", "model", folder).state_dict()
            payload = json.loads((Path(folder) / "results.json").read_text(encoding="utf-8"))
        return result, payload

    def test_enabled_and_disabled_training_identical_and_all_rounds_exported(self):
        without, _ = self.run_training(False)
        measured, payload = self.run_training(True)
        for name in without:
            self.assertTrue(torch.equal(without[name], measured[name]), name)
        records = payload["round_cost_records"]
        self.assertEqual([r["round"] for r in records], [0, 1])
        for row in records:
            self.assertEqual(row["upload_payload_bytes"], 2 * (3 * 2 + 2) * 4)
            self.assertEqual(len(row["local_train_client_seconds"]), 2)
            self.assertGreater(row["local_train_sum_seconds"], 0)
            self.assertGreater(row["server_processing_events"]["send_parameters"], 0)

    def test_real_save_load_delays_are_excluded(self):
        cb = self.clientbase
        clock = Clock()
        costs = RoundCosts("FedCLIP", "cpu", clock=clock, synchronize=lambda: None)
        real_save, real_load = torch.save, torch.load
        def slow_save(*args, **kwargs):
            clock.advance(50)
            return real_save(*args, **kwargs)
        def slow_load(*args, **kwargs):
            clock.advance(70)
            return real_load(*args, **kwargs)
        with tempfile.TemporaryDirectory() as folder, costs.session():
            costs.begin_round(0, [0])
            with mock.patch("torch.save", side_effect=slow_save), mock.patch("torch.load", side_effect=slow_load):
                cb.save_item(torch.nn.Linear(2, 1), "Client_0", "model", folder)
                model = cb.load_item("Client_0", "model", folder)
            clock.advance(3)
            row = costs.finish_round()
        self.assertIsInstance(model, torch.nn.Module)
        self.assertEqual(row["server_processing_seconds"], 3)
        self.assertEqual(row["upload_payload_bytes"], 12)


if __name__ == "__main__":
    unittest.main()

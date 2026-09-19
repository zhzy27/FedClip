"""Exercise real factor helpers without optional torchvision/CLIP imports."""
import ast
import copy
import importlib.util
import math
from pathlib import Path
import sys
import types
import unittest
from unittest import mock

import torch
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "system"))
from utils.round_costs import (
    RoundCosts, exclude_measurement, profile_server_layer,
    server_cost_scope, server_model_context,
)


class Clock:
    value = 0.0

    def __call__(self):
        return self.value

    def advance(self, value):
        self.value += value


def factor_helpers(filename):
    names = {"FactorizedConv", "FactorizedLinear", "Decom_COV", "Decom_LINEAR",
             "Recover_COV", "Recover_LINEAR"}
    tree = ast.parse((ROOT / "system/flcore/trainmodel" / filename).read_text(encoding="utf-8"))
    tree.body = [node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.ClassDef))
                 and node.name in names]
    namespace = dict(torch=torch, nn=nn, F=F, math=math,
                     server_cost_scope=server_cost_scope, profile_server_layer=profile_server_layer)
    exec(compile(tree, filename, "exec"), namespace)
    return namespace


def load_methods(filename, names, namespace):
    tree = ast.parse((ROOT / filename).read_text(encoding="utf-8"))
    methods = [method for cls in tree.body if isinstance(cls, ast.ClassDef)
               for method in cls.body if isinstance(method, ast.FunctionDef) and method.name in names]
    exec(compile(ast.Module(body=methods, type_ignores=[]), filename, "exec"), namespace)


class ServerBreakdownTest(unittest.TestCase):
    def test_exclusive_groups_io_calls_and_per_layer_attribution(self):
        clock = Clock()
        recorder = RoundCosts("FedCLIP", "cpu", clock=clock, synchronize=lambda: None)
        model = nn.Sequential(nn.Linear(3, 2))

        @profile_server_layer
        def decompose(layer):
            with server_cost_scope("send.svd"):
                clock.advance(2)
                with exclude_measurement():
                    clock.advance(100)
                clock.advance(1)
            with server_cost_scope("send.factor_build"):
                clock.advance(4)

        with recorder.session():
            recorder.begin_round(0, [7])
            with recorder.scope("server", "send_parameters"):
                with server_model_context(model, 7):
                    with server_cost_scope("send.decomposition_other"):
                        decompose(model[0])
                    with server_cost_scope("send.parameter_copy"):
                        clock.advance(5)
                        recorder.observe_upload(7, "model", model)
                clock.advance(1)
            self.assertFalse(recorder.uploads)
            with recorder.scope("server", "aggregate_parameters_avg"):
                with server_cost_scope("aggregate.weighted_add"):
                    clock.advance(6)
                clock.advance(2)
            clock.advance(3)
            row = recorder.finish_round()
        self.assertEqual(row["server_svd_seconds"], 3)
        self.assertEqual(row["server_svd_calls"], 1)
        self.assertEqual(row["server_send_prepare_seconds"], 13)
        self.assertEqual(row["server_aggregation_seconds"], 8)
        self.assertEqual(row["server_other_seconds"], 3)
        self.assertEqual(row["server_processing_seconds"], 24)
        detail = next(d for d in row["server_processing_details"] if d["event"] == "send.svd")
        self.assertEqual(detail, dict(event="send.svd", client_id=7, layer="0", seconds=3, calls=1))
        self.assertEqual(sum(row["server_processing_events"].values()), 24)

    def test_inactive_local_diagnostics_and_other_algorithms_not_reclassified(self):
        for algorithm in ("FedCLIP", "FD"):
            clock = Clock()
            recorder = RoundCosts(algorithm, "cpu", clock=clock, synchronize=lambda: None)
            with recorder.session():
                recorder.begin_round(0, [0])
                with recorder.scope("local", "train", 0):
                    with server_cost_scope("send.svd"):
                        clock.advance(2)
                with exclude_measurement():
                    with server_cost_scope("send.svd"):
                        clock.advance(100)
                if algorithm == "FD":
                    with server_cost_scope("send.svd"):
                        clock.advance(3)
                row = recorder.finish_round()
            self.assertNotIn("send.svd", row["server_processing_events"])
            self.assertEqual(row["local_train_sum_seconds"], 2)
            if algorithm == "FD":
                self.assertNotIn("server_svd_seconds", row)
                self.assertEqual(row["server_processing_seconds"], 3)

    def test_layer_context_restores_after_error(self):
        recorder = RoundCosts("FedCLIP", "cpu")
        @profile_server_layer
        def fail(layer):
            with server_cost_scope("send.svd"):
                raise ValueError("test")
        with recorder.session():
            recorder.begin_round(0, [0])
            with self.assertRaises(ValueError):
                with server_model_context(nn.Linear(2, 2), 0):
                    fail(nn.Linear(2, 2))
            with server_cost_scope("aggregate.zero_init"):
                pass
            row = recorder.finish_round()
        detail = next(d for d in row["server_processing_details"] if d["event"] == "aggregate.zero_init")
        self.assertIsNone(detail["client_id"])
        self.assertEqual(detail["layer"], "")

    def test_real_cnn_resnet_helpers_preserve_parameters_and_rng(self):
        self.check_real_helpers("cpu")

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA runtime unavailable")
    def test_cuda_helpers_preserve_outputs_and_measure_gpu_svd(self):
        self.check_real_helpers("cuda:0")

    def check_real_helpers(self, device):
        for filename in ("models.py", "SVD_resnet.py"):
            helpers = factor_helpers(filename)
            results = []
            for enabled in (False, True):
                torch.manual_seed(51)
                model = nn.Module()
                model.conv = nn.Conv2d(3, 4, 3, padding=1)
                if filename == "models.py":
                    model.fc = nn.Linear(12, 5)
                model = model.to(device)
                recorder = RoundCosts("FedCLIP", device)
                with recorder.session():
                    if enabled:
                        recorder.begin_round(0, [4])
                    with recorder.scope("server", "send_parameters"), server_model_context(model, 4):
                        with server_cost_scope("send.decomposition_other"):
                            model.conv = helpers["Decom_COV"](model.conv, 0.5)
                            if hasattr(model, "fc"):
                                model.fc = helpers["Decom_LINEAR"](model.fc, 0.5)
                    factors = copy.deepcopy(model.state_dict())
                    with recorder.scope("server", "aggregate_parameters_avg"), server_model_context(model, 4):
                        with server_cost_scope("aggregate.device_transfer"):
                            model = model.to(device)
                        with server_cost_scope("aggregate.recovery_other"):
                            model.conv = helpers["Recover_COV"](model.conv)
                            if hasattr(model, "fc"):
                                model.fc = helpers["Recover_LINEAR"](model.fc)
                    row = recorder.finish_round()
                results.append((factors, copy.deepcopy(model.state_dict()), torch.get_rng_state()))
                if enabled:
                    self.assertEqual(row["server_svd_calls"], 2 if filename == "models.py" else 1)
                    svds = [d for d in row["server_processing_details"] if d["event"] == "send.svd"]
                    self.assertEqual({d["layer"] for d in svds}, {"conv", "fc"} if filename == "models.py" else {"conv"})
                    self.assertTrue(all(d["client_id"] == 4 and d["calls"] == 1 for d in svds))
                    self.assertGreater(row["server_processing_events"]["aggregate.reconstruct_matmul"], 0)
                    self.assertAlmostEqual(row["server_processing_seconds"],
                        row["server_send_prepare_seconds"] + row["server_aggregation_seconds"] + row["server_other_seconds"])
            for index in (0, 1):
                for name in results[0][index]:
                    self.assertTrue(torch.equal(results[0][index][name], results[1][index][name]), (filename, name))
            self.assertTrue(torch.equal(results[0][2], results[1][2]))

    def test_actual_fedclip_send_and_avg_paths_with_small_models(self):
        helpers = factor_helpers("models.py")
        class TinyModel(nn.Module):
            ratio_LR = 0.5
            def __init__(self):
                super().__init__()
                self.fc = nn.Linear(6, 4)
                self.head = nn.Linear(4, 2)
            def decom_larger_model(self, ratio):
                self.fc = helpers["Decom_LINEAR"](self.fc, ratio)
            def recover_larger_model(self):
                self.fc = helpers["Recover_LINEAR"](self.fc)

        outcomes = []
        for enabled in (False, True):
            torch.manual_seed(72)
            store = {("Server", "model"): TinyModel()}
            def save_item(model, role, item, folder):
                with exclude_measurement():
                    store[role, item] = copy.deepcopy(model)
            def load_item(role, item, folder):
                with exclude_measurement():
                    return copy.deepcopy(store[role, item])
            ns = dict(torch=torch, copy=copy, save_item=save_item, load_item=load_item,
                      server_cost_scope=server_cost_scope, server_model_context=server_model_context)
            load_methods("system/flcore/clients/clientCLIP.py", {"set_parameters"}, ns)
            load_methods("system/flcore/servers/serverCLIP.py", {
                "_has_low_rank_params", "_recover_if_needed", "aggregate_parameters_avg",
                "_save_sample_weighted_global"}, ns)
            clients = []
            for cid in range(2):
                shell = copy.deepcopy(store["Server", "model"])
                shell.decom_larger_model(0.5)
                store[f"Client_{cid}", "model"] = shell
                clients.append(types.SimpleNamespace(id=cid, role=f"Client_{cid}", device="cpu", save_folder_name="test"))
            server = types.SimpleNamespace(clients=clients, uploaded_ids=[0, 1], uploaded_weights=[0.25, 0.75],
                                           role="Server", device="cpu", save_folder_name="test")
            for name in ("_has_low_rank_params", "_recover_if_needed", "_save_sample_weighted_global"):
                setattr(server, name, types.MethodType(ns[name], server))
            recorder = RoundCosts("FedCLIP", "cpu")
            with recorder.session(), mock.patch("builtins.print"):
                if enabled:
                    recorder.begin_round(0, [0, 1])
                with recorder.scope("server", "send_parameters"):
                    for client in clients:
                        ns["set_parameters"](client)
                # Distinct uploaded weights exercise actual sample-weighted averaging.
                with recorder.scope("local", "train", 0), torch.no_grad():
                    store["Client_0", "model"].fc.weight_u.add_(0.1)
                with recorder.scope("server", "aggregate_parameters_avg"):
                    ns["aggregate_parameters_avg"](server)
                row = recorder.finish_round()
            expected = sum(w * (store[c.role, "model"].fc.weight_u @ store[c.role, "model"].fc.weight_v)
                           for c, w in zip(clients, server.uploaded_weights))
            self.assertTrue(torch.allclose(store["Server", "model"].fc.weight, expected))
            outcomes.append((store["Server", "model"].state_dict(), torch.get_rng_state()))
            if enabled:
                self.assertEqual(row["server_svd_calls"], 2)
                for event in ("send.parameter_copy", "aggregate.deepcopy", "aggregate.weighted_add"):
                    self.assertGreater(row["server_processing_events"][event], 0)
        for name in outcomes[0][0]:
            self.assertTrue(torch.equal(outcomes[0][0][name], outcomes[1][0][name]), name)
        self.assertTrue(torch.equal(outcomes[0][1], outcomes[1][1]))

    def test_csv_standalone_columns_and_missing_historical_measurements(self):
        spec = importlib.util.spec_from_file_location("detail_runner", ROOT / "system/run_compare_compute.py")
        runner = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(runner)
        self.assertEqual(len(runner.CSV_FIELDS), len(set(runner.CSV_FIELDS)))
        self.assertTrue(all(value == "" for value in runner.fedclip_detail_columns({}).values()))
        row = runner.fedclip_detail_columns({"server_processing_details": [], "server_svd_seconds": 2.5,
            "server_svd_calls": 60, "server_processing_events": {"send.factor_build": 0.3}})
        self.assertEqual(row["server_svd_seconds"], 2.5)
        self.assertEqual(row["server_svd_calls"], 60)
        self.assertEqual(row["server_send_factor_build_seconds"], 0.3)
        self.assertEqual(row["server_aggregate_weighted_add_seconds"], 0)


if __name__ == "__main__":
    unittest.main()

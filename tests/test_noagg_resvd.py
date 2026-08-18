import copy
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


class ToyRecoverableModel(torch.nn.Module):
    total_decompose_calls = 0

    def __init__(self, value, low_rank):
        super().__init__()
        self.ratio_LR = 0.5
        self.register_buffer(
            "running_stat", torch.tensor([float(value) + 100.0])
        )
        if low_rank:
            self.weight_v = torch.nn.Parameter(torch.tensor([float(value)]))
        else:
            self.weight = torch.nn.Parameter(torch.tensor([float(value)]))

    @property
    def is_low_rank(self):
        return "weight_v" in self._parameters

    def recover_larger_model(self):
        if self.is_low_rank:
            value = self.weight_v.detach().clone()
            del self._parameters["weight_v"]
            self.register_parameter("weight", torch.nn.Parameter(value))

    def decom_larger_model(self, ratio):
        ToyRecoverableModel.total_decompose_calls += 1
        self.ratio_LR = float(ratio)
        if not self.is_low_rank:
            value = self.weight.detach().clone()
            del self._parameters["weight"]
            self.register_parameter("weight_v", torch.nn.Parameter(value))

    def scalar(self):
        parameter = self.weight_v if self.is_low_rank else self.weight
        return float(parameter.detach().item())


def make_store_functions(store, load_log=None):
    def save_item(item, role, item_name, item_path=None):
        store[(item_path, role, item_name)] = copy.deepcopy(item)

    def load_item(role, item_name, item_path=None):
        if load_log is not None:
            load_log.append((item_path, role, item_name))
        item = store.get((item_path, role, item_name))
        return None if item is None else copy.deepcopy(item)

    return load_item, save_item


def load_server_module():
    client_module = types.ModuleType("flcore.clients.clientCLIP")
    client_module.clientCLIP = object
    serverbase_module = types.ModuleType("flcore.servers.serverbase")
    serverbase_module.Server = object
    clientbase_module = types.ModuleType("flcore.clients.clientbase")
    clientbase_module.load_item = mock.MagicMock()
    clientbase_module.save_item = mock.MagicMock()
    models_module = types.ModuleType("flcore.trainmodel.models")
    models_module.Model_Distribe = mock.MagicMock()
    data_module = types.ModuleType("utils.data_utils")
    data_module.read_client_data = mock.MagicMock()
    clip_module = types.ModuleType("utils.get_clip_text_encoder")
    clip_module.get_clip_class_embeddings = mock.MagicMock()
    diagnostics_module = types.ModuleType("utils.factor_loss_diagnostics")
    diagnostics_module.DIAGNOSTIC_FIELDS = []
    sklearn_module = types.ModuleType("sklearn")
    sklearn_module.metrics = types.SimpleNamespace()
    preprocessing_module = types.ModuleType("sklearn.preprocessing")
    preprocessing_module.label_binarize = mock.MagicMock()
    matplotlib_module = types.ModuleType("matplotlib")
    pyplot_module = types.ModuleType("matplotlib.pyplot")
    seaborn_module = types.ModuleType("seaborn")
    stubs = {
        "flcore.clients.clientCLIP": client_module,
        "flcore.servers.serverbase": serverbase_module,
        "flcore.clients.clientbase": clientbase_module,
        "flcore.trainmodel.models": models_module,
        "utils.data_utils": data_module,
        "utils.get_clip_text_encoder": clip_module,
        "utils.factor_loss_diagnostics": diagnostics_module,
        "sklearn": sklearn_module,
        "sklearn.preprocessing": preprocessing_module,
        "matplotlib": matplotlib_module,
        "matplotlib.pyplot": pyplot_module,
        "seaborn": seaborn_module,
    }
    module_path = SYSTEM_ROOT / "flcore" / "servers" / "serverCLIP.py"
    spec = importlib.util.spec_from_file_location(
        "noagg_server_test_module", module_path
    )
    module = importlib.util.module_from_spec(spec)
    with mock.patch.dict(sys.modules, stubs):
        spec.loader.exec_module(module)
    return module


def load_client_module():
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
        "noagg_client_test_module", module_path
    )
    module = importlib.util.module_from_spec(spec)
    with mock.patch.dict(sys.modules, stubs):
        spec.loader.exec_module(module)
    return module


class NoAggReSVDTest(unittest.TestCase):
    def setUp(self):
        ToyRecoverableModel.total_decompose_calls = 0

    def test_avg_still_performs_sample_weighted_full_model_aggregation(self):
        module = load_server_module()
        server = module.FedCLIP.__new__(module.FedCLIP)
        server.device = torch.device("cpu")
        server.role = "Server"
        server.save_folder_name = "run"
        server.clients = [
            SimpleNamespace(id=0, role="Client_0", save_folder_name="run"),
            SimpleNamespace(id=1, role="Client_1", save_folder_name="run"),
        ]
        server.uploaded_ids = [0, 1]
        server.uploaded_weights = [0.25, 0.75]
        store = {
            ("run", "Server", "model"): ToyRecoverableModel(0, False),
            ("run", "Client_0", "model"): ToyRecoverableModel(1, True),
            ("run", "Client_1", "model"): ToyRecoverableModel(3, True),
        }
        load_item, save_item = make_store_functions(store)

        with mock.patch.object(module, "load_item", load_item), mock.patch.object(
            module, "save_item", save_item
        ), mock.patch("builtins.print"):
            server.aggregate_parameters_avg()

        self.assertAlmostEqual(
            store[("run", "Server", "model")].scalar(), 2.5
        )

    def test_client_a_upload_does_not_change_client_b_full_state(self):
        module = load_server_module()
        server = module.FedCLIP.__new__(module.FedCLIP)
        server.device = torch.device("cpu")
        server.role = "Server"
        server.save_folder_name = "run"
        server.clients = [
            SimpleNamespace(id=0, role="Client_0", save_folder_name="run"),
            SimpleNamespace(id=1, role="Client_1", save_folder_name="run"),
        ]
        server.client_full_models = {}
        store = {}
        load_item, save_item = make_store_functions(store)

        with mock.patch.object(module, "load_item", load_item), mock.patch.object(
            module, "save_item", save_item
        ), mock.patch("builtins.print"):
            server._initialize_noagg_client_models(
                ToyRecoverableModel(2, False)
            )
            client_b_before = copy.deepcopy(
                store[("run", "Server", "noagg_full_model_1")]
            )
            store[("run", "Client_0", "model")] = ToyRecoverableModel(
                7, True
            )
            server.uploaded_ids = [0]
            server.aggregate_parameters_noagg_resvd()

        self.assertAlmostEqual(
            store[("run", "Server", "noagg_full_model_0")].scalar(), 7.0
        )
        self.assertAlmostEqual(
            store[("run", "Server", "noagg_full_model_1")].scalar(),
            client_b_before.scalar(),
        )

    def test_second_round_loads_own_full_model_and_runs_resvd(self):
        module = load_client_module()
        client = module.clientCLIP.__new__(module.clientCLIP)
        client.id = 0
        client.role = "Client_0"
        client.device = torch.device("cpu")
        client.save_folder_name = "run"
        client.args = SimpleNamespace(
            aggregation_mode="noagg_resvd",
            d_max=0.0,
            enable_ce_anchor_diagnostics=1,
            enable_virtual_step_diagnostics=1,
        )
        store = {
            ("run", "Client_0", "model"): ToyRecoverableModel(1, True),
            ("run", "Server", "model"): ToyRecoverableModel(99, False),
            ("run", "Server", "noagg_full_model_0"): (
                ToyRecoverableModel(7, False)
            ),
            ("run", "Server", "noagg_full_model_1"): (
                ToyRecoverableModel(8, False)
            ),
        }
        load_log = []
        load_item, save_item = make_store_functions(store, load_log)

        with mock.patch.object(module, "load_item", load_item), mock.patch.object(
            module, "save_item", save_item
        ), mock.patch("builtins.print"):
            client.set_parameters(current_round=1)

        self.assertEqual(ToyRecoverableModel.total_decompose_calls, 1)
        self.assertAlmostEqual(
            store[("run", "Client_0", "model")].scalar(), 7.0
        )
        self.assertAlmostEqual(
            float(
                store[("run", "Client_0", "model")]
                .running_stat.item()
            ),
            107.0,
        )
        self.assertIn(
            ("run", "Server", "noagg_full_model_0"), load_log
        )
        self.assertNotIn(("run", "Server", "model"), load_log)
        self.assertIn(
            (str(Path("run") / "low_rank_start"), "Server", "model_0"),
            store,
        )

    def test_single_client_noagg_and_weighted_personalized_evaluation(self):
        module = load_server_module()
        server = module.FedCLIP.__new__(module.FedCLIP)
        server.device = torch.device("cpu")
        server.role = "Server"
        server.save_folder_name = "run"
        server.client_full_models = {}
        server.clients = [
            SimpleNamespace(
                id=0,
                role="Client_0",
                save_folder_name="run",
                test_metrics=lambda: (8, 10, 0),
                train_metrics=lambda: (4.0, 10),
            )
        ]
        server.uploaded_ids = [0]
        server.rs_test_acc = []
        server.rs_train_loss = []
        server.test_metrics = lambda: ([0], [10], [8.0], [0.0])
        server.train_metrics = lambda: ([0], [10], [4.0])
        store = {
            ("run", "Client_0", "model"): ToyRecoverableModel(4, True)
        }
        load_item, save_item = make_store_functions(store)

        with mock.patch.object(module, "load_item", load_item), mock.patch.object(
            module, "save_item", save_item
        ), mock.patch("builtins.print"):
            server.aggregate_parameters_noagg_resvd()
            accuracy = server.evaluate_noagg(epoch=1)

        self.assertAlmostEqual(accuracy, 0.8)
        self.assertEqual(server.rs_test_acc, [0.8])
        self.assertAlmostEqual(
            store[("run", "Server", "noagg_full_model_0")].scalar(), 4.0
        )


if __name__ == "__main__":
    unittest.main()

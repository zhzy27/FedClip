import importlib.util
import sys
import types
import unittest
from pathlib import Path
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


if __name__ == "__main__":
    unittest.main()

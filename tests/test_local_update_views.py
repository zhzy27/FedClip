import copy
import contextlib
import importlib.util
import io
import random
import sys
import types
import unittest
from pathlib import Path
from types import MethodType, SimpleNamespace
from unittest import mock

import numpy as np
import torch


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
CLIENT_PATH = REPOSITORY_ROOT / "system" / "flcore" / "clients" / "clientCLIP.py"


def _module(name, **attributes):
    value = types.ModuleType(name)
    for attribute_name, attribute_value in attributes.items():
        setattr(value, attribute_name, attribute_value)
    return value


def _package(name):
    value = _module(name)
    value.__path__ = []
    return value


def _load_client_module():
    dependency_stubs = {
        "flcore": _package("flcore"),
        "flcore.clients": _package("flcore.clients"),
        "utils": _package("utils"),
        "flcore.clients.clientbase": _module(
            "flcore.clients.clientbase",
            Client=object,
            load_item=lambda *args, **kwargs: None,
            save_item=lambda *args, **kwargs: None,
        ),
        "utils.get_clip_text_encoder": _module(
            "utils.get_clip_text_encoder",
            get_clip_class_embeddings=lambda *args, **kwargs: (None, None),
            get_clip_class_depth_embeddings=lambda *args, **kwargs: (None, None),
        ),
        "sklearn": _package("sklearn"),
        "sklearn.preprocessing": _module(
            "sklearn.preprocessing",
            label_binarize=lambda *args, **kwargs: None,
        ),
    }
    with mock.patch.dict(sys.modules, dependency_stubs):
        spec = importlib.util.spec_from_file_location(
            "clientCLIP_local_update_view_test",
            CLIENT_PATH,
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    return module


CLIENT_MODULE = _load_client_module()
ClientCLIP = CLIENT_MODULE.clientCLIP


class LocalUpdateViewsTest(unittest.TestCase):
    def make_client(self, views):
        client = ClientCLIP.__new__(ClientCLIP)
        client.args = SimpleNamespace(local_update_views=views)
        client.id = 3
        client.role = "Client_3"
        client.save_folder_name = "memory"
        client.device = torch.device("cpu")
        client.use_resnet_multilevel_clip = False
        client.train_slow = False
        client.local_epochs = 2
        client.train_samples = 16
        client.train_time_cost = {"num_rounds": 0, "total_cost": 0.0}
        return client

    def test_two_views_start_from_identical_independent_models(self):
        client = self.make_client(2)
        start_model = torch.nn.Linear(3, 2, bias=True)
        start_state = {
            name: value.detach().clone()
            for name, value in start_model.state_dict().items()
        }
        saved_items = {"model": copy.deepcopy(start_model)}

        def fake_load_item(role, item_name, item_path):
            return copy.deepcopy(saved_items[item_name])

        def fake_save_item(item, role, item_name, item_path):
            saved_items[item_name] = copy.deepcopy(item)

        view_inputs = []
        parameter_storage = []
        random_draws = []
        loader_orders = []
        epoch_inputs = []

        def fake_load_train_data(self, generator=None):
            order = torch.randperm(16, generator=generator).tolist()
            loader_orders.append(order)
            return order

        def fake_train_model_view(
            self,
            model,
            trainloader,
            current_round,
            max_local_epochs=None,
        ):
            epoch_inputs.append(max_local_epochs)
            view_inputs.append({
                name: value.detach().clone()
                for name, value in model.state_dict().items()
            })
            parameter_storage.append(next(model.parameters()).data_ptr())
            random_draws.append((
                random.random(),
                float(np.random.rand()),
                float(torch.rand(()).item()),
            ))
            with torch.no_grad():
                for parameter in model.parameters():
                    parameter.add_(len(view_inputs))
            return model, 0.0, max_local_epochs

        client.load_train_data = MethodType(fake_load_train_data, client)
        client._train_model_view = MethodType(fake_train_model_view, client)

        with contextlib.redirect_stdout(io.StringIO()), mock.patch.object(
            CLIENT_MODULE,
            "load_item",
            fake_load_item,
        ), mock.patch.object(CLIENT_MODULE, "save_item", fake_save_item):
            client.train(current_round=7)

        self.assertEqual(len(view_inputs), 2)
        for view_state in view_inputs:
            for name, expected in start_state.items():
                self.assertTrue(torch.equal(view_state[name], expected))
        self.assertNotEqual(parameter_storage[0], parameter_storage[1])
        self.assertEqual(epoch_inputs, [client.local_epochs, client.local_epochs])
        self.assertNotEqual(
            next(saved_items["model"].parameters()).data_ptr(),
            next(saved_items["model_view_b"].parameters()).data_ptr(),
        )
        self.assertFalse(
            torch.equal(
                next(saved_items["model"].parameters()),
                next(saved_items["model_view_b"].parameters()),
            )
        )

    def test_two_views_use_different_seeds_orders_and_random_streams(self):
        client = self.make_client(2)
        saved_items = {"model": torch.nn.Linear(2, 1)}
        loader_orders = []
        random_draws = []

        def fake_load_item(role, item_name, item_path):
            return copy.deepcopy(saved_items[item_name])

        def fake_save_item(item, role, item_name, item_path):
            saved_items[item_name] = copy.deepcopy(item)

        def fake_load_train_data(self, generator=None):
            order = torch.randperm(32, generator=generator).tolist()
            loader_orders.append(order)
            return order

        def fake_train_model_view(
            self,
            model,
            trainloader,
            current_round,
            max_local_epochs=None,
        ):
            random_draws.append((
                random.random(),
                float(np.random.rand()),
                float(torch.rand(()).item()),
            ))
            return model, 0.0, max_local_epochs

        client.load_train_data = MethodType(fake_load_train_data, client)
        client._train_model_view = MethodType(fake_train_model_view, client)

        random.seed(1234)
        np.random.seed(1234)
        torch.manual_seed(1234)
        python_state = random.getstate()
        numpy_state = np.random.get_state()
        torch_state = torch.get_rng_state().clone()

        with contextlib.redirect_stdout(io.StringIO()), mock.patch.object(
            CLIENT_MODULE,
            "load_item",
            fake_load_item,
        ), mock.patch.object(CLIENT_MODULE, "save_item", fake_save_item):
            client.train(current_round=11)

        seed_a, seed_b = client.last_local_update_view_seeds
        self.assertNotEqual(seed_a, seed_b)
        self.assertNotEqual(loader_orders[0], loader_orders[1])
        self.assertNotEqual(random_draws[0], random_draws[1])
        self.assertEqual(random.getstate(), python_state)
        current_numpy_state = np.random.get_state()
        self.assertEqual(current_numpy_state[0], numpy_state[0])
        self.assertTrue(np.array_equal(current_numpy_state[1], numpy_state[1]))
        self.assertEqual(current_numpy_state[2:], numpy_state[2:])
        self.assertTrue(torch.equal(torch.get_rng_state(), torch_state))

    def test_two_views_drive_real_dropout_training_independently(self):
        class TinyDropoutModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.base = torch.nn.Sequential(
                    torch.nn.Linear(4, 3),
                    torch.nn.ReLU(),
                    torch.nn.Dropout(p=0.5),
                )
                self.head = torch.nn.Linear(3, 2)

            def forward(self, inputs):
                return self.head(self.base(inputs))

        client = self.make_client(2)
        client.args = SimpleNamespace(
            local_update_views=2,
            mse_lamda=0.1,
            is_regular=0,
            regular_lamda=0.0,
            global_rounds=100,
        )
        client.learning_rate = 0.05
        client.loss = torch.nn.CrossEntropyLoss()
        client.mse_fn = torch.nn.MSELoss()
        client.clip_text_features = torch.tensor(
            [[0.2, -0.1, 0.4], [-0.3, 0.5, 0.1]],
            dtype=torch.float32,
        )
        inputs = torch.tensor(
            [
                [1.0, 0.5, -0.2, 0.1],
                [0.3, -0.7, 0.8, 0.2],
                [-0.4, 0.2, 0.6, 1.0],
                [0.9, -0.1, 0.4, -0.5],
                [0.1, 0.8, -0.6, 0.7],
                [-0.2, -0.4, 0.9, 0.3],
            ],
            dtype=torch.float32,
        )
        labels = torch.tensor([0, 1, 1, 0, 1, 0])
        dataset = torch.utils.data.TensorDataset(inputs, labels)
        torch.manual_seed(991)
        start_model = TinyDropoutModel()
        start_state = {
            name: value.detach().clone()
            for name, value in start_model.state_dict().items()
        }
        saved_items = {"model": copy.deepcopy(start_model)}

        def fake_load_item(role, item_name, item_path):
            return copy.deepcopy(saved_items[item_name])

        def fake_save_item(item, role, item_name, item_path):
            saved_items[item_name] = copy.deepcopy(item)

        def fake_load_train_data(self, generator=None):
            return torch.utils.data.DataLoader(
                dataset,
                batch_size=2,
                shuffle=True,
                generator=generator,
            )

        client.load_train_data = MethodType(fake_load_train_data, client)
        with contextlib.redirect_stdout(io.StringIO()), mock.patch.object(
            CLIENT_MODULE,
            "load_item",
            fake_load_item,
        ), mock.patch.object(CLIENT_MODULE, "save_item", fake_save_item):
            client.train(current_round=9)

        model_a_state = saved_items["model"].state_dict()
        model_b_state = saved_items["model_view_b"].state_dict()
        self.assertTrue(
            any(
                not torch.equal(model_a_state[name], start_state[name])
                for name in start_state
            )
        )
        self.assertTrue(
            any(
                not torch.equal(model_b_state[name], start_state[name])
                for name in start_state
            )
        )
        self.assertTrue(
            any(
                not torch.equal(model_a_state[name], model_b_state[name])
                for name in start_state
            )
        )
        self.assertTrue(
            all(torch.isfinite(value).all() for value in model_a_state.values())
        )
        self.assertTrue(
            all(torch.isfinite(value).all() for value in model_b_state.values())
        )

    def test_single_view_keeps_original_path_and_never_creates_b(self):
        client = self.make_client(1)
        start_model = torch.nn.Linear(2, 1)
        saved_items = {"model": copy.deepcopy(start_model)}
        calls = []

        def fake_load_item(role, item_name, item_path):
            return copy.deepcopy(saved_items[item_name])

        def fake_save_item(item, role, item_name, item_path):
            saved_items[item_name] = copy.deepcopy(item)

        def fake_load_train_data(self, generator=None):
            if generator is not None:
                raise AssertionError("Single-view path must not pass a generator.")
            return [0, 1]

        def fake_train_model_view(
            self,
            model,
            trainloader,
            current_round,
            max_local_epochs=None,
        ):
            calls.append(current_round)
            return model, 0.0, self.local_epochs

        client.load_train_data = MethodType(fake_load_train_data, client)
        client._train_model_view = MethodType(fake_train_model_view, client)

        with contextlib.redirect_stdout(io.StringIO()), mock.patch.object(
            CLIENT_MODULE,
            "load_item",
            fake_load_item,
        ), mock.patch.object(CLIENT_MODULE, "save_item", fake_save_item):
            client.train(current_round=5)

        self.assertEqual(calls, [5])
        self.assertNotIn("model_view_b", saved_items)
        self.assertIsNone(client.last_local_update_view_seeds)
        self.assertIsNone(client.local_update_view_b_round)


if __name__ == "__main__":
    unittest.main()

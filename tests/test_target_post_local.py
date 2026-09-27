"""Post-local target observations must be complete and have no training side effects."""

from contextlib import redirect_stdout
import copy
import csv
import io
import json
from pathlib import Path
import random
import tempfile
import unittest
from unittest import mock

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader, SequentialSampler

import test_target_projection as fixtures


class TargetPostLocalTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        self.fixture = fixtures.ProjectionIntegrationTests()
        self.fixture.setUp()

    def assert_rng_equal(self, before):
        self.assertEqual(random.getstate(), before[0])
        after = np.random.get_state()
        self.assertEqual(after[0], before[1][0])
        np.testing.assert_array_equal(after[1], before[1][1])
        self.assertEqual(after[2:], before[1][2:])
        torch.testing.assert_close(torch.get_rng_state(), before[2], rtol=0, atol=0)
        if torch.cuda.is_available():
            for actual, expected in zip(torch.cuda.get_rng_state_all(), before[3]):
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def rng(self):
        return (random.getstate(), np.random.get_state(), torch.get_rng_state(),
                torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [])

    def test_observation_preserves_checkpoint_parameters_buffers_rng_and_module_modes(self):
        with tempfile.TemporaryDirectory() as directory:
            client = self.fixture.make_client(directory, 0)
            model = fixtures.TinyModel(low_rank=True)
            model.extra = torch.nn.Sequential(torch.nn.Dropout(.5), torch.nn.BatchNorm1d(2))
            model.train()
            model.extra[1].eval()  # Preserve a mixed module state, not just root.training.
            self.fixture.save_item(model, client.role, "model", directory)
            original = copy.deepcopy(self.fixture.store)
            modes = [module.training for module in model.modules()]
            state = copy.deepcopy(model.state_dict())
            optimizer = torch.optim.SGD(model.parameters(), lr=.005, momentum=.9)
            optimizer_before = copy.deepcopy(optimizer.state_dict())

            def noisy_load():
                random.random()
                np.random.rand(3)
                torch.rand(3)
                if torch.cuda.is_available():
                    torch.rand(3, device="cuda")
                return model

            data = [(torch.tensor([1., -.5]), torch.tensor(0)), (torch.tensor([-.2, .3]), torch.tensor(1))]
            loader = DataLoader(data, batch_size=1, shuffle=False)
            self.assertIsInstance(loader.sampler, SequentialSampler)
            before = self.rng()
            with mock.patch.object(client, "_load_model", side_effect=noisy_load), \
                    mock.patch.object(client, "load_test_data", return_value=loader), \
                    mock.patch.object(self.fixture.client_module, "save_item") as save:
                result = client.test_post_local()
            self.assertEqual(result[1], 2)
            save.assert_not_called()
            self.assert_rng_equal(before)
            self.assertEqual([module.training for module in model.modules()], modes)
            self.assertEqual(optimizer.state_dict(), optimizer_before)
            for key, value in state.items():
                torch.testing.assert_close(model.state_dict()[key], value, rtol=0, atol=0)
            for key in original:
                for name, value in original[key].state_dict().items():
                    torch.testing.assert_close(self.fixture.store[key].state_dict()[name], value, rtol=0, atol=0)

    def test_failed_observation_restores_rng_and_mode_without_writes(self):
        client = self.fixture.make_client("unused", 0)
        model = fixtures.TinyModel(low_rank=True)
        model.train()

        def fail():
            torch.rand(2)
            random.random()
            np.random.rand(2)
            raise RuntimeError("synthetic loader failure")

        before = self.rng()
        with mock.patch.object(client, "_load_model", return_value=model), \
                mock.patch.object(client, "load_test_data", side_effect=fail), \
                mock.patch.object(self.fixture.client_module, "save_item") as save:
            with self.assertRaisesRegex(RuntimeError, "synthetic loader"):
                client.test_post_local()
        self.assert_rng_equal(before)
        self.assertTrue(model.training)
        save.assert_not_called()

    def test_all_101_local_rounds_measured_before_aggregation_and_summary_exported(self):
        with tempfile.TemporaryDirectory() as directory:
            server = self.fixture.make_server(directory, "projection")
            server.target_client_id = 0
            server.global_rounds, server.eval_gap = 100, 20
            server.Budget, server.rs_test_acc, server.auto_break = [], [], False
            server.clients = [self.fixture.make_client(directory, cid) for cid in range(3)]
            sequence = [.5 + ((i * 17) % 41) / 100 for i in range(101)]
            trained = []
            for client in server.clients:
                client.train = mock.Mock(side_effect=lambda current_round, cid=client.id: trained.append((current_round, cid)))
                client.test_post_local = mock.Mock()
            target = server.clients[0]

            def evaluate():
                current = server.cur_ground
                self.assertEqual(trained[-3:], [(current, cid) for cid in range(3)])
                self.assertEqual(len(server.target_proj_history), current)
                return round(sequence[current] * 100), 100, 0

            target.test_post_local.side_effect = evaluate
            target.test_downloaded_global = mock.Mock(return_value=(3, 4, 0))
            output = io.StringIO()
            with redirect_stdout(output):
                server.train()
            self.assertEqual(target.test_post_local.call_count, 101)
            for helper in server.clients[1:]:
                helper.test_post_local.assert_not_called()
            values = [row["target_post_local_acc"] for row in server.target_proj_history]
            self.assertEqual(len(values), 101)
            np.testing.assert_allclose(values, sequence, rtol=0, atol=1e-15)
            summary = server._local_accuracy_summary()
            self.assertEqual(summary["final_target_local_acc"], values[-1])
            self.assertEqual(summary["best_target_local_acc"], max(values))
            self.assertEqual(summary["best_target_local_round"], values.index(max(values)) + 1)
            for label in ("Final Client 0 post-local accuracy:", "Best Client 0 post-local accuracy:",
                          "Best Client 0 post-local round:"):
                self.assertIn(label, output.getvalue())
            with open(Path(directory) / "target_proj_metrics.json") as stream:
                saved = json.load(stream)
            self.assertEqual(saved["summary"], summary)
            self.assertEqual(saved["primary_accuracy_metric"], "target_post_local_acc")
            with open(Path(directory) / "target_proj_metrics.csv") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), 101)
            self.assertEqual(float(rows[-1]["final_target_local_acc"]), values[-1])
            self.assertEqual(int(rows[-1]["best_target_local_round"]), summary["best_target_local_round"])
            path = str(Path(directory) / "result.h5")
            server.args.save_file_paths = []

            def save_h5():
                with h5py.File(path, "w"):
                    pass
                server.args.save_file_paths.append(path)

            with mock.patch.object(fixtures.FakeServer, "save_results", side_effect=save_h5):
                server.save_results()
            with h5py.File(path) as result:
                group = result["target_projection"]
                np.testing.assert_array_equal(group["target_post_local_acc"][:], values)
                for key, value in summary.items():
                    self.assertEqual(group.attrs[key], value)
                    self.assertEqual(group[key][-1], value)

    def test_extra_observation_leaves_next_training_update_bitwise_unchanged(self):
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            client = self.fixture.make_client(directory, 0)
            self.fixture.save_item(fixtures.TinyModel(low_rank=True), client.role, "model", directory)
            before_store = copy.deepcopy(self.fixture.store)
            before_rng = self.rng()
            client.train()
            expected = copy.deepcopy(self.fixture.store)
            self.fixture.store.clear()
            self.fixture.store.update(before_store)
            random.setstate(before_rng[0])
            np.random.set_state(before_rng[1])
            torch.set_rng_state(before_rng[2])
            if torch.cuda.is_available():
                torch.cuda.set_rng_state_all(before_rng[3])
            client.test_post_local()
            client.train()
            for key in expected:
                for name, value in expected[key].state_dict().items():
                    torch.testing.assert_close(self.fixture.store[key].state_dict()[name], value, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()

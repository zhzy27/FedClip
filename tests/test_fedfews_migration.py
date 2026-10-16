"""Independent numerical invariants plus observation/non-observation equality."""
from contextlib import redirect_stdout
import copy
import io
import json
import os
from pathlib import Path
import random
import shlex
import sys
import tempfile
import unittest
from unittest.mock import patch

import h5py
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "system"))
from fedfews_compat.acceptance import Comparison, synthetic_data, torch_load
from fedfews_compat.config import parse_args
from fedfews_compat.data import audit_data
from fedfews_compat.observe import Recorder, model_state, rng_digest, rng_state, set_rng, summary
from fedfews_compat.runtime import build_server, prepare_random, run
from fedfews_compat.vendor import clientbase
from fedfews_compat.vendor.clientfedfews import clientFedFewS
from fedfews_compat.vendor.models import FedAvgCNN
from fedfews_compat.vendor.serverfedfews import FedFewS


class FedFewSMigrationTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)

    def args(self, root, *extra):
        return parse_args(["--data-root", str(root / "data"), "--output-dir", str(root / "out"),
                           "--dataset", "Cifar100/noniid_dir_20_a0p5", "--device", "cpu", "--num_clients", "3",
                           "--batch_size", "4", "--global_rounds", "1", "--rng-mode", "controlled", *extra])[0]

    def test_source_defaults_effective_config_and_protocol_gates(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            args, merged = parse_args(["--data-root", str(root), "--output-dir", str(root / "out")])
            self.assertEqual((args.num_server_models, args.smooth_mu, args.test_batch_size), (3, .01, 256))
            self.assertEqual((args.global_rounds, args.local_epochs, args.batch_size), (2000, 1, 50))
            self.assertEqual((args.num_threads, args.use_cache, args.prev, args.times), (1, True, 0, 1))
            self.assertEqual(args.momentum, .1)  # unused source option; actual SGD momentum=0
            for flag, value in [("--use_rep_mode", "True"), ("--join_ratio", ".5"), ("--smooth_mu", "nan"),
                                ("--smooth_mu", "0"), ("--client_drop_rate", ".1"), ("--few_shot", "1"),
                                ("--num_threads", "2"), ("--algorithm", "FedAvg"), ("--num_classes", "20")]:
                with self.subTest(flag=flag, value=value), self.assertRaises(ValueError):
                    self.args(root, flag, value)
            command_file = Path(__file__).resolve().parents[1] / "system/fedfews_commands.sh"
            commands = [shlex.split(line.strip()[1:-1])[2:] for line in command_file.read_text().splitlines()
                        if line.startswith('"$FEW_PYTHON system/run_fedfews.py ')]
            self.assertEqual(len(commands), 3)  # preview, formal dir05, deferred pat10
            for command in commands:
                parsed, _ = parse_args(command)
                self.assertEqual((parsed.global_rounds, parsed.local_epochs, parsed.batch_size, parsed.num_server_models), (2000, 1, 50, 3))
                self.assertEqual((parsed.momentum, parsed.learning_rate_decay, parsed.auto_break), (0 if not parsed.preview else .1, False, False))

    def test_original_rng_mode_preserves_python_numpy_and_source_torch_reset(self):
        with tempfile.TemporaryDirectory() as temp:
            args = self.args(Path(temp))
            args.rng_mode = "original"
            random.seed(73)
            np.random.seed(71)
            torch.manual_seed(99)
            before = rng_digest()
            prepare_random(args)
            after = rng_digest()
            self.assertEqual(before["python"], after["python"])
            self.assertEqual(before["numpy"], after["numpy"])
            torch.manual_seed(0)
            self.assertEqual(after["torch"], rng_digest()["torch"])

    def test_original_cnn_schema_initialization_and_alias_free_copies(self):
        torch.manual_seed(0)
        first = FedAvgCNN(3, 100, 1600)
        torch.manual_seed(0)
        second = FedAvgCNN(3, 100, 1600)
        self.assertEqual(list(first.state_dict()), ["conv1.0.weight", "conv1.0.bias", "conv2.0.weight", "conv2.0.bias", "fc1.0.weight", "fc1.0.bias", "fc.weight", "fc.bias"])
        self.assertEqual(sum(p.numel() for p in first.parameters()), 924708)
        x = torch.linspace(-1, 1, 3 * 32 * 32).reshape(1, 3, 32, 32)
        self.assertTrue(torch.equal(first(x), second(x)))
        copies = [copy.deepcopy(first) for _ in range(3)]
        self.assertTrue(all(torch.equal(a, b) for a, b in zip(copies[0].parameters(), copies[1].parameters())))
        self.assertNotEqual(copies[0].fc.weight.data_ptr(), copies[1].fc.weight.data_ptr())

    def test_reader_pkl_priority_content_order_alpha_and_rng(self):
        with tempfile.TemporaryDirectory() as temp, patch.dict(os.environ):
            root = Path(temp)
            synthetic_data(root / "data")
            args = self.args(root)
            before = rng_digest()
            audit = audit_data(args)
            self.assertEqual(before, rng_digest())
            self.assertEqual([f["count"] for f in audit["files"] if f["split"] == "train"], [7, 10, 13])
            first = audit["files"][0]
            self.assertTrue(first["npz_shadowed_by_pkl"])
            self.assertEqual(first["format"], ".pkl")
            self.assertEqual(first["labels"], dict(zip(range(7), [1] * 7)))
            args.expected_alpha = .5
            with self.assertRaisesRegex(ValueError, "directory name is not evidence"):
                audit_data(args)

    def test_unbalanced_double_weights_incremental_old_server_order_and_id_mapping(self):
        from types import SimpleNamespace
        server = FedFewS.__new__(FedFewS)
        server.K, server.mu, server.use_rep_mode = 3, .01, False
        def model(value):
            m = torch.nn.Linear(2, 1, bias=False).double()
            m.weight.data.fill_(value)
            return m
        old = [1., 2., -3.]
        server.global_model_set = [model(v) for v in old]
        losses = np.array([[1., 2., 3.], [3., 1., 2.], [2., 3., 1.]], dtype=np.float32)
        counts = np.array([2, 5, 11])
        server.clients = [SimpleNamespace(id=i, train_model_losses=losses[i]) for i in range(3)]
        server.selected_clients = server.clients
        server.uploaded_ids = [2, 0, 1]
        server.uploaded_weights = [counts[i] / counts.sum() for i in server.uploaded_ids]
        server.uploaded_models = [[model(old[k] + (i + 1) * (k + 1) * .2) for k in range(3)] for i in server.uploaded_ids]
        weighted = losses[server.uploaded_ids].astype(float) * np.array(server.uploaded_weights)[:, None]
        # Independent direct-exp formula on well-conditioned, tiny input.
        exponentials = np.exp(-weighted / server.mu)
        sums = exponentials.sum(axis=1)
        alpha = (1 / sums) / (1 / sums).sum()
        w = exponentials / sums[:, None]
        before = model_state([m for ms in server.uploaded_models for m in ms])
        with redirect_stdout(io.StringIO()):
            result = server.aggregate_parameters()
        for k in range(3):
            expected = old[k] + sum(alpha[r] * w[r, k] * (cid + 1) * (k + 1) * .2 for r, cid in enumerate(server.uploaded_ids))
            # Independent exp vs upstream logsumexp has ~5e-15 rounding here.
            self.assertAlmostEqual(server.global_model_set[k].weight[0, 0].item(), expected, delta=2e-14)
        self.assertTrue(np.allclose(np.array(result[1]).flatten(), alpha[np.argsort(server.uploaded_ids)]))
        self.assertFalse(np.allclose((alpha[:, None] * w).sum(axis=0), np.ones(3)))
        after = model_state([m for ms in server.uploaded_models for m in ms])
        self.assertTrue(all(torch.equal(before[k], after[k]) for k in before))

    def test_last_epoch_training_loss_is_not_deployment_endpoint_loss(self):
        from types import SimpleNamespace
        client = clientFedFewS.__new__(clientFedFewS)
        client.K, client.device, client.local_epochs = 3, "cpu", 2
        client.use_rep_mode, client.learning_rate_decay = False, False
        client.train_samples = 5
        client.loss = torch.nn.CrossEntropyLoss()
        torch.manual_seed(5)
        client.model_set = [torch.nn.Linear(2, 2) for _ in range(3)]
        client.optimizer_set = [torch.optim.SGD(m.parameters(), lr=.1) for m in client.model_set]
        reference_models = copy.deepcopy(client.model_set)
        batches = [(torch.tensor([[1., 2.], [-1., 1.], [.1, .2]]), torch.tensor([0, 1, 0])),
                   (torch.tensor([[.2, .8], [.9, -.3]]), torch.tensor([1, 0]))]
        client.load_train_data = lambda **kw: batches
        client.train_time_cost = dict(num_rounds=0, total_cost=0.)
        expected = torch.zeros(3)
        for k, model in enumerate(reference_models):
            opt = torch.optim.SGD(model.parameters(), lr=.1)
            for ep in range(2):
                expected[k] = 0.
                for x, y in batches:
                    opt.zero_grad()
                    loss = client.loss(model(x), y)
                    loss.backward()
                    opt.step()
                    expected[k] += loss.item() * len(y)
        client.train()
        self.assertTrue(np.array_equal(client.train_model_losses, (expected / 5).numpy()))
        idx, endpoint = client.select_best_model()
        self.assertEqual(idx, int(endpoint.argmin()))
        self.assertFalse(np.array_equal(endpoint, client.train_model_losses))
        for a, b in zip(client.model_set, reference_models):
            self.assertTrue(all(torch.equal(x, y) for x, y in zip(a.parameters(), b.parameters())))

    def test_observation_does_not_change_uninstrumented_trajectory_and_exports(self):
        with tempfile.TemporaryDirectory() as temp, patch.dict(os.environ), redirect_stdout(io.StringIO()):
            root = Path(temp)
            synthetic_data(root / "data")
            args = self.args(root)
            os.environ["PFLLIB_DATA_DIR"] = args.data_root
            prepare_random(args)
            initial = rng_state()
            raw_args = copy.deepcopy(args)
            raw_args.model = FedAvgCNN(3, 100, 1600)
            plain = FedFewS(raw_args, 0)
            plain.save_results = lambda: None
            plain.save_global_model = lambda: None
            plain.train()
            raw_states = model_state(plain.global_model_set + [m for c in plain.clients for m in c.model_set])
            raw_rng = rng_digest()
            set_rng(initial)
            observed, recorder, cleanup = build_server(args, root / "observed")
            optimizer_ids = [[id(o) for o in c.optimizer_set] for c in observed.clients]
            try:
                observed.train()
            finally:
                cleanup()
            states = model_state(observed.global_model_set + [m for c in observed.clients for m in c.model_set])
            self.assertEqual(raw_rng, rng_digest())
            self.assertTrue(all(torch.equal(raw_states[n], states[n]) for n in states))
            self.assertEqual(optimizer_ids, [[id(o) for o in c.optimizer_set] for c in observed.clients])
            self.assertEqual(len(recorder.rows), 2)
            self.assertEqual([r["loop_round"] for r in recorder.evaluations], [0, 1])
            final = torch_load(root / "observed/models/FedFewS_final.pt")
            self.assertEqual(len(final["server_models"]), 3)
            self.assertEqual(len(final["clients"]), 3)
            with h5py.File(root / "observed/metrics" / args.dataset / "FedFewS_ours_0.h5") as f:
                self.assertEqual(f["rs_model_losses"].shape, (2, 3, 3))
                self.assertEqual(f["server_received_mass"].shape, (2, 3))
                self.assertEqual(f["deployment_ce"].shape, (2, 3, 3))
                self.assertTrue(np.allclose(f["test_correct"][:].sum(axis=1) / f["test_samples"][:].sum(axis=1), f["rs_test_acc"][:]))
            self.assertTrue((root / "observed/metrics/clients.csv").is_file())
            self.assertTrue((root / "observed/metrics/summary.json").is_file())

    def test_summary_tail_means_evaluations_best_earliest(self):
        values = [1., 3., 3., 2.]
        result = summary([dict(loop_round=i * 10, completed_train_count=i * 10 + 1, test_acc_percent=v) for i, v in enumerate(values)])
        self.assertEqual(result["best_loop_round"], 10)
        self.assertEqual(result["last10_evaluations"], 2.25)
        self.assertEqual(result["final"], 2.)

    def test_acceptance_rejects_parameter_and_rng_errors(self):
        with tempfile.TemporaryDirectory() as temp:
            comparison = Comparison(Path(temp) / "states")
            expected = dict(event=0, loop_round=0, name="aggregate", payload=[1.], rng={"torch": "same"})
            comparison.reference(expected, {"p": torch.ones(2)})
            actual = copy.deepcopy(expected)
            actual["rng"]["torch"] = "different"
            comparison.migrated(actual, {"p": torch.tensor([1., 2.])})
            self.assertEqual(comparison.report()["status"], "FAIL")
            self.assertEqual(comparison.max_parameter_error, 1.)


if __name__ == "__main__":
    unittest.main()

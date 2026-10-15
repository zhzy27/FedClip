"""Two-round, full-local5 heterogeneous CNN lifecycle and shared-mode regression."""

from contextlib import redirect_stdout
import copy
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import h5py
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "system"))
from flcore.clients.clientbase import load_item, save_item
from flcore.servers.serverTargetProj import FedTargetProj
from utils.target_head import backbone_parameters, classifier_names, preserved_head_rng
from test_meta_projection import rng_state, assert_rng


def fingerprint(model):
    digest = hashlib.sha256()
    for n, p in model.state_dict().items():
        digest.update(n.encode())
        digest.update(p.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def head(model):
    return {n: dict(model.named_parameters())[n].detach().clone() for n in classifier_names(model)}


class LocalHeadCNNTests(unittest.TestCase):
    def test_shared_exact_legacy_and_two_local_modes_lifecycle_exports(self):
        torch.set_num_threads(2)
        code = subprocess.check_output(["git", "show", "715c80b:system/flcore/servers/serverTargetProj.py"], cwd=ROOT, text=True)
        legacy = {}
        exec(compile(code, "715c80b/serverTargetProj.py", "exec"), legacy)
        legacy_class = legacy["FedTargetProj"]
        generator = torch.Generator().manual_seed(51)
        x = torch.randn(16, 3, 32, 32, generator=generator)
        factory = "Hyper_CNN_512(in_features=3,num_classes=100,n_kernels=4,ratio_LR={rank},input_size=32)"
        old_cwd = os.getcwd()
        shared_records = {}
        initial_head = None
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            try:
                os.chdir(directory)
                def reader(dataset, cid, args, is_train=True, **kwargs):
                    return [(x[i], torch.tensor((cid + i % 3) % 100)) for i in range(16 if is_train else 4)]
                for label in ("legacy", "shared", "local_avg", "local_projection"):
                    is_local = label.startswith("local")
                    mode = "avg" if label == "local_avg" else "projection_softmax"
                    args = SimpleNamespace(device="cpu", dataset="Synthetic", num_classes=100, global_rounds=1,
                        local_epochs=5, batch_size=16, local_learning_rate=.005, num_clients=20, join_ratio=1.,
                        random_join_ratio=False, few_shot=0, algorithm="FedTargetProj", time_select=False,
                        goal="smoke", time_threthold=1e9, top_cnt=10, auto_break=False, eval_gap=1,
                        client_drop_rate=0., train_slow_rate=0., send_slow_rate=0., resume=False,
                        model_family="Decom_CNN-5-512", models_folder_name="", niid=1, partition="pat",
                        class_per_client=20, exp_name=label, save_file_paths=[],
                        models=[factory.format(rank=r) for r in (.9, .37, .35, .25, .15)],
                        global_model=factory.format(rank=.15), target_client_id=0, seed=0,
                        is_regular=1, regular_lamda=.001, target_proj_mode=mode, target_prox_mu=0.,
                        projection_self_weight=None if mode == "avg" else .125,
                        save_folder_name=str(root / label), final_model_root=str(root / "final" / label),
                        h5_result_root=str(root / "h5" / label), training_log_root=str(root / "logs" / label))
                    if label != "legacy":
                        args.target_head_mode = "local" if is_local else "shared"
                    if is_local:
                        # Avoid repeating mode tags under Windows' 260-char limit;
                        # real CLI logs also use a shorter explicit process root.
                        tag = mode + "_local_head"
                        args.save_folder_name = str(root / "c" / tag)
                        args.final_model_root = str(root / "f" / tag)
                        args.h5_result_root = str(root / "h" / tag)
                        args.training_log_root = str(root / "l" / tag)
                        args.training_log_dir = str(root / "l" / tag)
                    cls = legacy_class if label == "legacy" else FedTargetProj
                    with redirect_stdout(io.StringIO()), patch("flcore.clients.clientbase.read_client_data", side_effect=reader), \
                            patch("flcore.servers.serverbase.read_client_data", side_effect=reader):
                        server = cls(args, 0)
                        original_server = load_item("Server", "model", server.save_folder_name)
                        if initial_head is None:
                            initial_head = head(original_server)
                        for name, p in initial_head.items():
                            torch.testing.assert_close(head(original_server)[name], p, rtol=0, atol=0)
                        for role in ("Server", *(c.role for c in server.clients)):
                            model = load_item(role, "model", server.save_folder_name)
                            model.register_buffer("buffer_probe", torch.tensor([-7.] if role == "Server" else [3.]))
                            save_item(model, role, "model", server.save_folder_name)
                        calls, downloads, previous_heads, start_heads, records = [0] * 20, [0] * 20, {}, {}, []
                        native_send = server.send_parameters
                        def check_downloads():
                            native_send()
                            for c in server.clients:
                                downloads[c.id] += 1
                                model = load_item(c.role, "model", server.save_folder_name)
                                current = head(model)
                                if is_local:
                                    expected = previous_heads.get(c.id, initial_head)
                                    for n in expected:
                                        torch.testing.assert_close(current[n], expected[n], rtol=0, atol=0)
                                    torch.testing.assert_close(model.buffer_probe, torch.tensor([3.]), rtol=0, atol=0)
                                start_heads[c.id] = current
                            if is_local:
                                # Match the normal SVD download for each heterogeneous capacity.
                                with preserved_head_rng():
                                    expected_by_ratio = {}
                                    for c in server.clients:
                                        model = load_item(c.role, "model", server.save_folder_name)
                                        if model.ratio_LR not in expected_by_ratio:
                                            source = load_item("Server", "model", server.save_folder_name)
                                            source.decom_larger_model(model.ratio_LR)
                                            expected_by_ratio[model.ratio_LR] = dict(source.named_parameters())
                                        expected = expected_by_ratio[model.ratio_LR]
                                        for n, p in backbone_parameters(model).items():
                                            torch.testing.assert_close(p, expected[n], rtol=0, atol=0)
                        server.send_parameters = check_downloads
                        for c in server.clients:
                            native_train = c.train
                            def train_once(current_round, c=c, native_train=native_train):
                                calls[c.id] += 1
                                native_train(current_round)
                                model = load_item(c.role, "model", c.save_folder_name)
                                current = head(model)
                                self.assertTrue(any(not torch.equal(current[n], start_heads[c.id][n]) for n in current))
                                previous_heads[c.id] = current
                                self.assertEqual(c.last_normal_train_stats["local_epochs"], 5)
                                self.assertEqual(c.last_normal_train_stats["train_batches"], 5)
                                self.assertEqual(c.last_normal_train_stats["processed_train_samples"], 80)
                            c.train = train_once
                        native_aggregate = server.aggregate_parameters_avg
                        def aggregate_and_observe():
                            ordinary = [fingerprint(load_item(c.role, "model", c.save_folder_name)) for c in server.clients]
                            native_aggregate()
                            records.append(dict(uploads=ordinary, server=fingerprint(load_item("Server", "model", server.save_folder_name)), rng=rng_state()))
                            if is_local:
                                target = server.clients[0]
                                server_model = load_item("Server", "model", server.save_folder_name)
                                for n, p in initial_head.items():
                                    torch.testing.assert_close(head(server_model)[n], p, rtol=0, atol=0)
                                self.assertNotIn("fc3.weight", server._load_full_parameters(0))
                                # No extra observation writes or advances RNG, even with
                                # an intentionally incorrect server placeholder classifier.
                                paths = [Path(server.save_folder_name) / f"{role}_model.pt" for role in ("Server", "Client_0", "Client_1")]
                                hashes = [hashlib.sha256(p.read_bytes()).hexdigest() for p in paths]
                                rng = rng_state()
                                answer = target.test_downloaded_global()
                                native_load = target._load_model
                                def wrong_placeholder(role=None):
                                    model = native_load(role)
                                    if role == "Server":
                                        model.fc3.weight.fill_(1e4)
                                        model.fc3.bias.fill_(-1e4)
                                    return model
                                with torch.no_grad(), patch.object(target, "_load_model", side_effect=wrong_placeholder):
                                    self.assertEqual(target.test_downloaded_global(), answer)
                                self.assertEqual(hashes, [hashlib.sha256(p.read_bytes()).hexdigest() for p in paths])
                                assert_rng(self, rng)
                                for c in server.clients:
                                    model = load_item(c.role, "model", c.save_folder_name)
                                    for n, p in previous_heads[c.id].items():
                                        torch.testing.assert_close(head(model)[n], p, rtol=0, atol=0)
                                self.assertEqual(server.target_proj_history[-1]["target_client_test_acc"], answer[0] / answer[1])
                        server.aggregate_parameters_avg = aggregate_and_observe
                        server.train()
                        self.assertEqual(calls, [2] * 20)
                        self.assertEqual(downloads, [2] * 20)
                        self.assertEqual([c.train_time_cost["num_rounds"] for c in server.clients], [2] * 20)
                        if not is_local:
                            shared_records[label] = records
                            continue
                        summary = server._local_accuracy_summary()
                        self.assertEqual(summary["last10_target_local_count"], 2)
                        self.assertEqual(summary["last20_target_local_count"], 2)
                        values = [r["target_post_local_acc"] for r in server.target_proj_history]
                        self.assertEqual(summary["best_target_local_round"], values.index(max(values)) + 1)
                        log = Path(server.report_dir())
                        document = json.loads((log / "target_proj_metrics.json").read_text())
                        self.assertEqual(document["summary"], summary)
                        self.assertEqual(document["local_head_policy"]["classifier_parameter_names"], ["fc3.weight", "fc3.bias"])
                        self.assertEqual(document["local_head_policy"]["backbone_parameter_count"], sum(p.numel() for p in backbone_parameters(original_server).values()))
                        with (log / "target_proj_clients.csv").open() as stream:
                            rows = list(csv.DictReader(stream))
                        self.assertEqual(len(rows), 40)
                        with h5py.File(args.save_file_paths[-1]) as result:
                            group = result["target_projection"]
                            self.assertEqual(group.attrs["target_head_mode"], "local")
                            self.assertEqual(group["client_aggregation_weight"].shape, (2, 20))
                            self.assertEqual(group["target_post_local_acc"].shape, (2,))
                            self.assertEqual(group.attrs["last20_target_local_count"], 2)
                            for weights in group["client_aggregation_weight"][:]:
                                self.assertAlmostEqual(float(weights.sum()), 1.)
                                self.assertAlmostEqual(weights[0], .05 if mode == "avg" else .125)
                        final = Path(server.final_model_dir())
                        manifest = json.loads((final / "manifest.json").read_text())
                        self.assertFalse(manifest["server_checkpoint_direct_inference_valid"])
                        self.assertFalse((final / "Server_model.pt").exists())
                        artifact = torch.load(final / "Server_backbone_only.pt", weights_only=True)
                        self.assertFalse(artifact["direct_inference_valid"])
                        self.assertFalse(any(n.startswith(("fc3.", "head.")) for n in artifact["parameters"]))
                        local_model = torch.load(final / "Client_0_model.pt", weights_only=False)
                        aggregate_model = torch.load(final / "C0_aggregated_backbone_local_head.pt", weights_only=False)
                        self.assertEqual(fingerprint(local_model), fingerprint(load_item("Client_0", "model", server.save_folder_name)))
                        for n, p in head(local_model).items():
                            torch.testing.assert_close(head(aggregate_model)[n], p, rtol=0, atol=0)
                        self.assertEqual(local_model.fc3.weight.shape, (100, 512))
                        for p in (server.save_folder_name, server.report_dir(), args.save_file_paths[-1], server.final_model_dir()):
                            self.assertIn("local_head", p)
                for old, new in zip(shared_records["legacy"], shared_records["shared"]):
                    self.assertEqual(old["uploads"], new["uploads"])
                    self.assertEqual(old["server"], new["server"])
                    import random
                    import numpy as np
                    random.setstate(new["rng"][0]); np.random.set_state(new["rng"][1]); torch.set_rng_state(new["rng"][2])
                    if torch.cuda.is_available(): torch.cuda.set_rng_state_all(new["rng"][3])
                    assert_rng(self, old["rng"])
            finally:
                os.chdir(old_cwd)


if __name__ == "__main__":
    unittest.main()

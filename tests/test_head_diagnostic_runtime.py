"""Avg first: exact legacy/off/on trajectories, real CNN snapshots and offline head retraining."""

from contextlib import redirect_stdout
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "system"))
from flcore.clients.clientbase import load_item
from flcore.servers.serverTargetProj import FedTargetProj
from flcore.trainmodel.models import Hyper_CNN_512
from utils.head_diagnostic import diagnose_snapshot, write_outputs
from utils.head_snapshot import file_hash, tensor_fingerprint
from test_meta_projection import assert_rng, rng_state


class HeadDiagnosticCNNTests(unittest.TestCase):
    def test_avg_shared_local_capture_isolation_native_fresh_protocol_exports_and_cli(self):
        torch.set_num_threads(2)
        code = subprocess.check_output(["git", "show", "22ea7cf:system/flcore/servers/serverTargetProj.py"], cwd=ROOT, text=True)
        legacy = {}
        exec(compile(code, "22ea7cf/serverTargetProj.py", "exec"), legacy)
        generator = torch.Generator().manual_seed(67)
        x = torch.randn(16, 3, 32, 32, generator=generator).numpy()
        factory = "Hyper_CNN_512(in_features=3,num_classes=100,n_kernels=4,ratio_LR={rank},input_size=32)"
        original_cwd = os.getcwd()
        trajectories, snapshots, final_paths = {}, [], []
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            work = root / "s"
            work.mkdir()
            data_root = root / "dataset"
            for split, count in (("train", 16), ("test", 4)):
                folder = data_root / "Synthetic" / "pat_20" / split
                folder.mkdir(parents=True)
                for cid in range(20):
                    np.savez_compressed(folder / f"{cid}.npz", data=dict(x=x[:count], y=np.array([(cid + i % 3) % 100 for i in range(count)])))
            try:
                os.chdir(work)
                for head_mode in ("shared", "local"):
                    for phase in ("legacy", "off", "on"):
                        label = f"{head_mode}_{phase}"
                        short = head_mode[0] + str(("legacy", "off", "on").index(phase)) + "_head_diag"
                        if head_mode == "local":
                            short += "_local_head"
                        # Compact explicit roots reproduce CLI-style logging and
                        # avoid redundant Windows mode/path nesting in the test.
                        tag = f"head_diag_avg_{head_mode}" + ("_local_head" if head_mode == "local" else "")
                        args = SimpleNamespace(device="cpu", dataset="Synthetic", num_classes=100, global_rounds=1,
                            local_epochs=5, batch_size=16, local_learning_rate=.005, num_clients=20, join_ratio=1.,
                            random_join_ratio=False, few_shot=0, algorithm="FedTargetProj", time_select=False,
                            goal="smoke", time_threthold=1e9, top_cnt=10, auto_break=False, eval_gap=1,
                            client_drop_rate=0., train_slow_rate=0., send_slow_rate=0., resume=False,
                            model_family="Decom_CNN-5-512", models_folder_name="", niid=1, partition="pat",
                            class_per_client=20, exp_name=tag, save_file_paths=[], learning_rate_decay=False,
                            models=[factory.format(rank=r) for r in (.9, .37, .35, .25, .15)],
                            global_model=factory.format(rank=.15), target_client_id=0, seed=0,
                            is_regular=1, regular_lamda=.001, target_proj_mode="avg", target_prox_mu=0.,
                            target_head_mode=head_mode, save_folder_name=str(root / "c" / short),
                            final_model_root=str(root / "f" / short), h5_result_root=str(root / "h" / short),
                            training_log_root=str(root / "l" / short), training_log_dir=str(root / "l" / short))
                        if phase != "legacy":
                            args.head_diag_collect = phase == "on"
                            args.head_diag_rounds = "1,2"
                            args.head_diag_dir = str(root / "snap") if phase == "on" else ""
                        cls = legacy["FedTargetProj"] if phase == "legacy" else FedTargetProj
                        with redirect_stdout(io.StringIO()):
                            server = cls(args, 0)
                            records, calls = [], [0] * 20
                            for c in server.clients:
                                native_train = c.train
                                def train(current_round, c=c, native_train=native_train):
                                    calls[c.id] += 1
                                    native_train(current_round)
                                    self.assertEqual(c.last_normal_train_stats["local_epochs"], 5)
                                    self.assertEqual(c.last_normal_train_stats["train_batches"], 5)
                                c.train = train
                            if phase == "on":
                                native_capture = server.head_diag_collector.capture
                                def capture(correct, samples):
                                    paths = list(Path(server.save_folder_name).glob("*.pt"))
                                    hashes, state = [file_hash(p) for p in paths], rng_state()
                                    native_capture(correct, samples)
                                    self.assertEqual([file_hash(p) for p in paths], hashes)
                                    assert_rng(self, state)
                                server.head_diag_collector.capture = capture
                            native_aggregate = server.aggregate_parameters_avg
                            def aggregate():
                                states = [tensor_fingerprint(load_item(c.role, "model", c.save_folder_name).state_dict()) for c in server.clients]
                                if phase == "on":
                                    path = server.head_diag_collector.folder / f"R{server.cur_ground + 1}.pt"
                                    payload = torch.load(path, weights_only=False)
                                    self.assertEqual(payload["metadata"]["loop_round"], server.cur_ground)
                                    self.assertEqual(payload["metadata"]["model_state_sha256"], states[0])
                                    self.assertEqual(payload["metadata"]["native_test_accuracy"], server._post_local_acc)
                                    self.assertEqual(len(server.target_proj_history), server.cur_ground)
                                    self.assertTrue(hasattr(payload["model"].fc2, "weight_u"))
                                    snapshots.append(path)
                                native_aggregate()
                                records.append(dict(uploads=states, server=tensor_fingerprint(load_item("Server", "model", server.save_folder_name).state_dict()),
                                    upload_order=list(server.uploaded_ids), rng=rng_state()))
                            server.aggregate_parameters_avg = aggregate
                            # A noisy extra read proves the collector itself restores RNG.
                            if phase == "on":
                                def noisy_load(*a, **kw):
                                    random.random(); np.random.rand(); torch.rand(2)
                                    return load_item(*a, **kw)
                                with patch("utils.head_snapshot.load_item", side_effect=noisy_load):
                                    server.train()
                            else:
                                server.train()
                            self.assertEqual(calls, [2] * 20)
                            trajectories[label] = records
                            if phase == "on":
                                final_paths.extend(Path(server.save_folder_name).glob("*.pt"))
                                self.assertEqual([r["round"] for r in server.head_diag_collector.captured], [1, 2])
                for head_mode in ("shared", "local"):
                    baseline = trajectories[head_mode + "_legacy"]
                    for phase in ("off", "on"):
                        for old, new in zip(baseline, trajectories[head_mode + "_" + phase]):
                            self.assertEqual(old["uploads"], new["uploads"])
                            self.assertEqual(old["server"], new["server"])
                            self.assertEqual(old["upload_order"], new["upload_order"])
                            random.setstate(new["rng"][0]); np.random.set_state(new["rng"][1]); torch.set_rng_state(new["rng"][2])
                            if torch.cuda.is_available(): torch.cuda.set_rng_state_all(new["rng"][3])
                            assert_rng(self, old["rng"])
                before_files = {str(p): file_hash(p) for p in [*snapshots, *final_paths]}
                results = []
                with patch.object(Hyper_CNN_512, "decom_larger_model", side_effect=AssertionError("offline SVD forbidden")), \
                        patch.object(Hyper_CNN_512, "recover_larger_model", side_effect=AssertionError("offline module recovery forbidden")), \
                        patch.object(Hyper_CNN_512, "frobenius_decay", side_effect=AssertionError("offline F penalty forbidden")):
                    for path in snapshots:
                        state = rng_state()
                        result = diagnose_snapshot(path, data_root, "cpu", 1729)
                        assert_rng(self, state)
                        self.assertTrue(result["native_reproduced"])
                        self.assertTrue(result["classifier_updated"])
                        self.assertTrue(result["backbone_and_buffers_bitwise_unchanged"])
                        self.assertEqual([(r["variant"], r["head_train_epoch"]) for r in result["rows"]],
                                         [("native", 0), ("fresh_head", 0), ("fresh_head", 1), ("fresh_head", 5)])
                        self.assertEqual(len({r["head_input_mean_l2"] for r in result["rows"]}), 1)
                        self.assertEqual(len({r["fc2_recovered_weight_frobenius"] for r in result["rows"]}), 1)
                        self.assertEqual([r["feature_sample_count"] for r in result["rows"]], [16] * 4)
                        results.append(result)
                    # No warm start from the earlier three snapshots.
                    repeated = diagnose_snapshot(snapshots[-1], data_root, "cpu", 1729)
                    self.assertEqual(repeated["rows"], results[-1]["rows"])
                self.assertEqual(len({r["batch_order_sha256"] for r in results}), 1)
                self.assertEqual(len({r["source_metadata"]["public_head_sha256"] for r in results}), 1)
                output = root / "out"
                output.mkdir()
                comparisons = write_outputs(output, results)
                self.assertEqual([r["round"] for r in comparisons], [1, 2])
                with (output / "head_diagnostic.csv").open() as stream:
                    self.assertEqual(len(list(csv.DictReader(stream))), 16)
                self.assertTrue((output / "head_comparison.csv").exists())
                self.assertIn("Absolute gap reduction", (output / "comparison.md").read_text())
                self.assertEqual(before_files, {str(p): file_hash(p) for p in [*snapshots, *final_paths]})
                process = subprocess.run([sys.executable, str(ROOT / "system/head_diagnostic.py"), "--snapshots", str(root / "snap"),
                    "--dataset-root", str(data_root), "--output-dir", str(root / "cli"), "--device", "cpu"],
                    cwd=work, capture_output=True, text=True)
                self.assertEqual(process.returncode, 0, process.stderr)
                self.assertEqual(len(json.loads((root / "cli/head_diagnostic.json").read_text())["comparisons"]), 2)
                # Fail loudly on original data changes; never silently re-split.
                path = data_root / "Synthetic/pat_20/train/0.npz"
                original = path.read_bytes()
                path.write_bytes(original + b"changed")
                with self.assertRaisesRegex(ValueError, "fingerprint mismatch"):
                    diagnose_snapshot(snapshots[0], data_root, "cpu")
                path.write_bytes(original)
            finally:
                os.chdir(original_cwd)


if __name__ == "__main__":
    unittest.main()

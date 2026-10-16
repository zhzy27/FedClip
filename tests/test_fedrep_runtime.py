"""Two-round heterogeneous CNN: legacy joint equality and head5/backbone5 lifecycle."""

from contextlib import redirect_stdout
import csv
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

import h5py
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "system"))
from flcore.clients.clientbase import load_item, save_item
from flcore.servers.serverTargetProj import FedTargetProj
from flcore.trainmodel.models import Hyper_CNN_512
from utils import fedrep_training
from utils.head_snapshot import tensor_fingerprint
from utils.target_head import preserved_head_rng
from test_meta_projection import rng_state, assert_rng


class FedRepRuntimeTests(unittest.TestCase):
    def test_joint_exact_and_real_CNN_two_round_fedrep_scope_budget_and_exports(self):
        torch.set_num_threads(2)
        old_client, old_server = {}, {}
        for name, namespace in (("clients/clientTargetProj.py", old_client), ("servers/serverTargetProj.py", old_server)):
            code = subprocess.check_output(["git", "show", "7e58fda:system/flcore/" + name], cwd=ROOT, text=True)
            exec(compile(code, "7e58fda/" + name, "exec"), namespace)
        old_server["clientTargetProj"] = old_client["clientTargetProj"]
        factory = "Hyper_CNN_512(in_features=3,num_classes=100,n_kernels=4,ratio_LR={rank},input_size=32)"
        generator = torch.Generator().manual_seed(89)
        x = torch.randn(16, 3, 32, 32, generator=generator)
        original_cwd = os.getcwd()
        histories, common_initial_head = {}, None
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            try:
                os.chdir(root)
                def reader(dataset, cid, args, is_train=True, **kw):
                    return [(x[i], torch.tensor((cid + i % 3) % 100)) for i in range(16 if is_train else 4)]
                for schedule in ("legacy", "joint", "fedrep"):
                    tag = "local_head_" + ("fedrep_head5_backbone5" if schedule == "fedrep" else schedule)
                    args = SimpleNamespace(device="cpu", dataset="Synthetic", num_classes=100, global_rounds=1,
                        local_epochs=5, batch_size=16, local_learning_rate=.005, num_clients=20, join_ratio=1.,
                        random_join_ratio=False, few_shot=0, algorithm="FedTargetProj", time_select=False,
                        goal="smoke", time_threthold=1e9, top_cnt=10, auto_break=False, eval_gap=1,
                        client_drop_rate=0., train_slow_rate=0., send_slow_rate=0., resume=False,
                        model_family="Decom_CNN-5-512", models_folder_name="", niid=1, partition="pat", class_per_client=20,
                        exp_name=tag, save_file_paths=[], models=[factory.format(rank=r) for r in (.9, .37, .35, .25, .15)],
                        global_model=factory.format(rank=.15), target_client_id=0, seed=0,
                        is_regular=1, regular_lamda=.001, target_proj_mode="avg", target_prox_mu=0.,
                        target_head_mode="local", save_folder_name=str(root / "c" / tag), final_model_root=str(root / "f" / tag),
                        h5_result_root=str(root / "h" / tag), training_log_root=str(root / "l" / tag), training_log_dir=str(root / "l" / tag))
                    if schedule != "legacy":
                        args.target_local_schedule, args.fedrep_head_epochs = schedule, 5
                    cls = old_server["FedTargetProj"] if schedule == "legacy" else FedTargetProj
                    with redirect_stdout(io.StringIO()), patch("flcore.clients.clientbase.read_client_data", side_effect=reader), \
                            patch("flcore.servers.serverbase.read_client_data", side_effect=reader):
                        server = cls(args, 0)
                        model = load_item("Server", "model", server.save_folder_name)
                        initial = {n: p.detach().clone() for n, p in model.named_parameters() if n.startswith("fc3.")}
                        if common_initial_head is None: common_initial_head = initial
                        for n in initial: torch.testing.assert_close(initial[n], common_initial_head[n], rtol=0, atol=0)
                        calls, downloads, previous_heads, records, phases = [0] * 20, [0] * 20, {}, [], []
                        context = {}
                        for c in server.clients:
                            model = load_item(c.role, "model", server.save_folder_name)
                            model.register_buffer("buffer_probe", torch.tensor([c.id + 1.]))
                            save_item(model, c.role, "model", server.save_folder_name)
                            native_download = c.set_parameters
                            def download(c=c, native_download=native_download):
                                native_download()
                                downloads[c.id] += 1
                                model = load_item(c.role, "model", c.save_folder_name)
                                expected = previous_heads.get(c.id, common_initial_head)
                                for n, p in model.named_parameters():
                                    if n.startswith("fc3."): torch.testing.assert_close(p, expected[n], rtol=0, atol=0)
                            c.set_parameters = download
                            native_train = c.train
                            def train(current_round, c=c, native_train=native_train):
                                calls[c.id] += 1
                                context.update(cid=c.id, head_end=None)
                                # No recovery, new SVD or server re-download is allowed
                                # inside either local training stage or observation.
                                with patch.object(Hyper_CNN_512, "decom_larger_model", side_effect=AssertionError("SVD inside training")), \
                                        patch.object(Hyper_CNN_512, "recover_larger_model", side_effect=AssertionError("recovery inside training")), \
                                        patch.object(c, "set_parameters", side_effect=AssertionError("download inside training")):
                                    native_train(current_round)
                                model = load_item(c.role, "model", c.save_folder_name)
                                previous_heads[c.id] = {n: p.detach().clone() for n, p in model.named_parameters() if n.startswith("fc3.")}
                                self.assertTrue(all(p.requires_grad for p in model.parameters()))
                                self.assertTrue(all(m.training for m in model.modules()))
                                self.assertEqual(c.last_normal_train_stats["train_batches"], 10 if schedule == "fedrep" else 5)
                                self.assertEqual(c.last_normal_train_stats["processed_train_samples"], 160 if schedule == "fedrep" else 80)
                                torch.testing.assert_close(model.buffer_probe, torch.tensor([c.id + 1.]), rtol=0, atol=0)
                            c.train = train
                        native_phase = fedrep_training.run_phase
                        def phase(c, model, loader, parameters, stage, epochs):
                            before = {n: p.detach().clone() for n, p in model.named_parameters()}
                            if stage == "backbone":
                                self.assertEqual(context["head_end"], tensor_fingerprint(model.state_dict()))
                                self.assertEqual(context["model_id"], id(model))
                                self.assertEqual(context["loader_id"], id(loader))
                            result = native_phase(c, model, loader, parameters, stage, epochs)
                            self.assertEqual(result["train_epochs"], 5)
                            self.assertEqual(result["train_batches"], 5)
                            self.assertEqual(result["train_sample_visits"], 80)
                            for n, p in model.named_parameters():
                                fixed = not n.startswith("fc3.") if stage == "head" else n.startswith("fc3.")
                                if fixed: torch.testing.assert_close(p, before[n], rtol=0, atol=0)
                            changed = [n for n, p in model.named_parameters() if not torch.equal(p, before[n])]
                            self.assertTrue(changed)
                            self.assertTrue(all(n.startswith("fc3.") for n in changed) if stage == "head" else all(not n.startswith("fc3.") for n in changed))
                            torch.testing.assert_close(model.buffer_probe, torch.tensor([c.id + 1.]), rtol=0, atol=0)
                            if stage == "head":
                                self.assertEqual(result["train_mean_weighted_frobenius"], 0.)
                                context.update(head_end=tensor_fingerprint(model.state_dict()), model_id=id(model), loader_id=id(loader))
                            else: self.assertGreater(result["train_mean_weighted_frobenius"], 0.)
                            phases.append((server.cur_ground, c.id, stage))
                            return result
                        native_aggregate = server.aggregate_parameters_avg
                        def aggregate():
                            uploads = [tensor_fingerprint(load_item(c.role, "model", c.save_folder_name).state_dict()) for c in server.clients]
                            expected = {}
                            if schedule == "fedrep":
                                with preserved_head_rng():
                                    for cid, weight in zip(server.uploaded_ids, server.uploaded_weights):
                                        parameters = server._load_full_parameters(cid)
                                        self.assertFalse(any(n.startswith("fc3.") for n in parameters))
                                        if not expected: expected = {n: torch.zeros_like(p) for n, p in parameters.items()}
                                        for n, p in parameters.items(): expected[n].add_(p * weight)
                            native_aggregate()
                            model = load_item("Server", "model", server.save_folder_name)
                            if schedule == "fedrep":
                                for n, p in model.named_parameters():
                                    torch.testing.assert_close(p, common_initial_head[n] if n.startswith("fc3.") else expected[n], rtol=0, atol=0)
                            records.append(dict(uploads=uploads, server=tensor_fingerprint(model.state_dict()), rng=rng_state()))
                        server.aggregate_parameters_avg = aggregate
                        with patch.object(fedrep_training, "run_phase", side_effect=phase):
                            server.train()
                        self.assertEqual(calls, [2] * 20)
                        self.assertEqual(downloads, [2] * 20)
                        self.assertEqual([c.train_time_cost["num_rounds"] for c in server.clients], [2] * 20)
                        histories[schedule] = records
                        if schedule != "fedrep": continue
                        self.assertEqual(len(phases), 80)
                        for round_index in range(2):
                            for cid in range(20): self.assertEqual([p[2] for p in phases if p[:2] == (round_index, cid)], ["head", "backbone"])
                        log = Path(server.report_dir())
                        for filename, count in (("fedrep_clients.csv", 40), ("fedrep_training.csv", 80), ("fedrep_target_stages.csv", 6)):
                            with (log / filename).open() as stream: self.assertEqual(len(list(csv.DictReader(stream))), count)
                        document = json.loads((log / "fedrep_training.json").read_text())
                        self.assertEqual(document["policy"]["total_data_epochs"], 10)
                        for r in server.target_proj_history:
                            self.assertEqual(r["target_post_local_acc"], r["target_after_backbone_acc"])
                            self.assertEqual(r["total_data_epochs"], 10)
                        summary = server._local_accuracy_summary()
                        self.assertEqual(summary["last10_target_local_count"], 2)
                        self.assertEqual(summary["last20_target_local_count"], 2)
                        with h5py.File(args.save_file_paths[-1]) as file:
                            group = file["target_projection"]
                            self.assertEqual(group.attrs["target_local_schedule"], "fedrep")
                            self.assertEqual(group["fedrep_training"]["total_data_epochs"].shape, (2, 20))
                            self.assertTrue((group["fedrep_training"]["total_data_epochs"][:] == 10).all())
                            self.assertEqual(group["fedrep_training"]["target_endpoint_test_accuracy"].shape, (2, 3))
                        final = Path(server.final_model_dir())
                        manifest = json.loads((final / "manifest.json").read_text())
                        self.assertEqual(manifest["target_local_schedule"], "fedrep")
                        self.assertEqual(len(manifest["final_round_training"]), 20)
                        self.assertEqual(manifest["summary"], summary)
                        client_model = torch.load(final / "Client_0_model.pt", weights_only=False)
                        agg_model = torch.load(final / "C0_aggregated_backbone_local_head.pt", weights_only=False)
                        for a, b in zip(client_model.fc3.parameters(), agg_model.fc3.parameters()): torch.testing.assert_close(a, b, rtol=0, atol=0)
                        backbone = torch.load(final / "Server_backbone_only.pt", weights_only=True)
                        self.assertFalse(backbone["direct_inference_valid"])
                        self.assertFalse(any(n.startswith("fc3.") for n in backbone["parameters"]))
                for old, new in zip(histories["legacy"], histories["joint"]):
                    self.assertEqual(old["uploads"], new["uploads"])
                    self.assertEqual(old["server"], new["server"])
                    random.setstate(new["rng"][0]); np.random.set_state(new["rng"][1]); torch.set_rng_state(new["rng"][2])
                    if torch.cuda.is_available(): torch.cuda.set_rng_state_all(new["rng"][3])
                    assert_rng(self, old["rng"])
            finally:
                os.chdir(original_cwd)


if __name__ == "__main__":
    unittest.main()

"""Exact mu0 vs 0af37b3 trajectory and nonzero helper Prox on real heterogeneous CNN."""

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
from types import MethodType, SimpleNamespace
import unittest
from unittest.mock import patch

import h5py
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "system"))
from flcore.clients.clientbase import load_item, save_item
from flcore.servers.serverTargetProj import FedTargetProj
from utils.group_projection import preserved_group_rng
from utils.target_prox import make_prox_anchor, proximal_penalty, trainable_parameters
import utils.target_prox as prox_observation
from utils.target_prox_options import effective_prox_mu
from test_meta_projection import rng_state, assert_rng


class FeedbackHelperProxCNNTests(unittest.TestCase):
    def test_baseline_exact_zero_and_positive_helper_prox_budget_anchor_scope_state_and_exports(self):
        torch.set_num_threads(2)
        root_repo = Path(__file__).resolve().parents[1]
        previous_code = subprocess.check_output(["git", "show", "0af37b3:system/flcore/servers/serverTargetFeedback.py"],
                                                cwd=root_repo, text=True)
        old_namespace = {}
        exec(compile(previous_code, "0af37b3/serverTargetFeedback.py", "exec"), old_namespace)
        legacy_runtime = old_namespace["TargetFeedbackRuntime"]
        generator = torch.Generator().manual_seed(41)
        x = torch.randn(16, 3, 32, 32, generator=generator)
        factory = "Hyper_CNN_512(in_features=3,num_classes=100,n_kernels=4,ratio_LR={rank},input_size=32)"
        old_cwd = os.getcwd()
        trajectories = {}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            try:
                os.chdir(directory)
                def reader(dataset, cid, args, is_train=True, **kwargs):
                    return [(x[i], torch.tensor((cid + i % 3) % 100)) for i in range(16 if is_train else 4)]
                for label, mu in (("baseline", 0.), ("zero", 0.), ("helpers", .1)):
                    args = SimpleNamespace(device="cpu", dataset="Synthetic", num_classes=100, global_rounds=1,
                        local_epochs=5, batch_size=16, local_learning_rate=.005, num_clients=20, join_ratio=1.,
                        random_join_ratio=False, few_shot=0, algorithm="FedTargetProj", time_select=False,
                        goal="smoke", time_threthold=1e9, top_cnt=10, auto_break=False, eval_gap=1,
                        client_drop_rate=0., train_slow_rate=0., send_slow_rate=0., resume=False,
                        model_family="Decom_CNN-5-512", models_folder_name="", niid=1, partition="pat",
                        class_per_client=20, exp_name=label, save_file_paths=[],
                        models=[factory.format(rank=r) for r in (.9, .37, .35, .25, .15)],
                        global_model=factory.format(rank=.15), target_client_id=0, seed=0,
                        is_regular=1, regular_lamda=.001, target_proj_mode="feedback_avg", target_prox_mu=mu,
                        target_prox_scope="helpers", mu=500.,
                        save_folder_name=str(root / label), final_model_root=str(root / "final" / label),
                        h5_result_root=str(root / "h5" / label), training_log_root=str(root / "logs" / label))
                    records, downloads, calls, starts, anchors = {}, [0] * 20, [0] * 20, {}, []
                    frozen_starts = {}
                    context = {}
                    with redirect_stdout(io.StringIO()), patch("flcore.clients.clientbase.read_client_data", side_effect=reader), \
                            patch("flcore.servers.serverbase.read_client_data", side_effect=reader):
                        server = FedTargetProj(args, 0)
                        if label == "baseline":
                            # Bind the original complete runtime, including its no-observer
                            # train method, to this identically initialized shared server.
                            for name, function in vars(legacy_runtime).items():
                                if callable(function):
                                    setattr(server, name, MethodType(function, server))
                        for role in ("Server", *(c.role for c in server.clients)):
                            model = load_item(role, "model", server.save_folder_name)
                            model.register_buffer("buffer_probe", torch.tensor([-7.] if role == "Server" else [3.]))
                            model.register_parameter("frozen_probe", torch.nn.Parameter(torch.tensor([5.]), requires_grad=False))
                            save_item(model, role, "model", server.save_folder_name)
                        def full_snapshot(model):
                            return {n: p.detach().cpu().clone() for n, p in model.named_parameters()}
                        for client in server.clients:
                            native_down, native_train = client.set_parameters, client.train
                            def down(native=native_down, client=client):
                                with preserved_group_rng():
                                    source = load_item("Server", "model", server.save_folder_name)
                                    if server.cur_ground > 0 and client.id != 0:
                                        compare = records[server.cur_ground - 1]["feedback"]
                                        for name, p in source.named_parameters():
                                            torch.testing.assert_close(p.cpu(), compare[name], atol=0, rtol=0)
                                    template = client._load_model()
                                    source.decom_larger_model(template.ratio_LR)
                                    expected = {n: p.detach().clone() for n, p in source.named_parameters()}
                                native()
                                actual = client._load_model()
                                for name, p in actual.named_parameters():
                                    torch.testing.assert_close(p, expected[name], atol=0, rtol=0)
                                starts[server.cur_ground, client.id] = {n: p.detach().clone() for n, p in trainable_parameters(actual).items()}
                                frozen_starts[server.cur_ground, client.id] = actual.frozen_probe.detach().clone()
                                downloads[client.id] += 1
                            def train(current_round=0, native=native_train, client=client):
                                self.assertEqual(calls[client.id], current_round)
                                context.update(cid=client.id, loop_round=current_round)
                                if client.id == 0:
                                    self.assertEqual(effective_prox_mu(client.args, 0), 0.)
                                    with patch.object(client, "_train_with_prox", side_effect=AssertionError("C0 Prox")):
                                        result = native(current_round=current_round)
                                else:
                                    result = native(current_round=current_round)
                                calls[client.id] += 1
                                self.assertEqual(client.last_normal_train_stats["local_epochs"], 5)
                                local = client._load_model()
                                torch.testing.assert_close(local.frozen_probe, frozen_starts[current_round, client.id], atol=0, rtol=0)
                                return result
                            client.set_parameters, client.train = down, train
                        def anchor(model):
                            self.assertGreater(context["cid"], 0)
                            self.assertGreater(mu, 0)
                            actual = make_prox_anchor(model)
                            expected = starts[context["loop_round"], context["cid"]]
                            self.assertEqual(actual.keys(), expected.keys())
                            self.assertNotIn("frozen_probe", actual)
                            self.assertNotIn("buffer_probe", actual)
                            self.assertIn("fc3.weight", actual)
                            self.assertTrue(any(n.endswith("weight_u") or n.endswith("conv_u") for n in actual))
                            for name, p in actual.items():
                                torch.testing.assert_close(p, expected[name], atol=0, rtol=0)
                                self.assertFalse(p.requires_grad)
                                self.assertNotEqual(p.data_ptr(), dict(model.named_parameters())[name].data_ptr())
                            anchors.append((context["loop_round"], context["cid"], actual, {n: p.clone() for n, p in actual.items()}))
                            return actual
                        def penalty(parameters, saved_anchor, active_mu):
                            self.assertEqual(active_mu, mu)
                            self.assertIs(saved_anchor, anchors[-1][2])
                            for name in saved_anchor:
                                torch.testing.assert_close(saved_anchor[name], anchors[-1][3][name], atol=0, rtol=0)
                            return proximal_penalty(parameters, saved_anchor, active_mu)
                        native_cpu = prox_observation._cpu_local_parameters
                        def cpu_observation(client):
                            path = Path(client.save_folder_name) / f"{client.role}_model.pt"
                            checksum = hashlib.sha256(path.read_bytes()).hexdigest()
                            result = native_cpu(client)
                            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), checksum)
                            # Deliberate random draws demonstrate that observations are
                            # protected rather than merely deterministic in this fixture.
                            import random, numpy as np
                            random.random(); np.random.rand(); torch.rand(1)
                            return result
                        native_cache = server._feedback_cache_helpers
                        def cache(training):
                            before = rng_state()
                            result = native_cache(training)
                            assert_rng(self, before)
                            records[server.cur_ground] = dict(helper_ids=list(server.uploaded_ids),
                                helper_uploads={cid: server._feedback_load_helper(cid) for cid in server.uploaded_ids},
                                helper_rng=before)
                            self.assertEqual(set(server.uploaded_ids), set(range(1, 20)))
                            return result
                        server._feedback_cache_helpers = cache
                        native_aggregate = server._feedback_aggregate_first_stage
                        def aggregate():
                            result = native_aggregate()
                            intermediate = load_item("Server", "model", server.save_folder_name)
                            records[server.cur_ground]["intermediate"] = full_snapshot(intermediate)
                            return result
                        server._feedback_aggregate_first_stage = aggregate
                        native_commit = server._feedback_commit_target
                        def commit():
                            before = rng_state()
                            with preserved_group_rng():
                                records[server.cur_ground]["target_upload"] = {n: p.detach().cpu().clone() for n, p in server._load_full_parameters(0).items()}
                            result = native_commit()
                            assert_rng(self, before)
                            shared = load_item("Server", "model", server.save_folder_name)
                            records[server.cur_ground]["feedback"] = full_snapshot(shared)
                            records[server.cur_ground]["feedback_rng"] = rng_state()
                            for name, p in records[server.cur_ground]["feedback"].items():
                                torch.testing.assert_close(p, records[server.cur_ground]["target_upload"][name], atol=0, rtol=0)
                            return result
                        server._feedback_commit_target = commit
                        anchor_patch = patch("flcore.clients.clientTargetProj.make_prox_anchor", side_effect=anchor
                            if mu else AssertionError("Zero mu/C0 must not allocate a training Prox anchor"))
                        with anchor_patch, patch("flcore.clients.clientTargetProj.proximal_penalty", side_effect=penalty), \
                                patch("utils.target_prox._cpu_local_parameters", side_effect=cpu_observation):
                            server.train()
                        self.assertEqual(calls, [2] * 20)
                        self.assertEqual(downloads, [2] * 20)
                        self.assertEqual([c.train_time_cost["num_rounds"] for c in server.clients], [2] * 20)
                        if mu:
                            self.assertEqual(len(anchors), 38)
                            first_anchor = next(a for a in anchors if a[:2] == (0, 1))
                            next_anchor = next(a for a in anchors if a[:2] == (1, 1))
                            self.assertIsNot(first_anchor[2], next_anchor[2])
                            self.assertTrue(any(not torch.equal(first_anchor[2][n], next_anchor[2][n]) for n in first_anchor[2]))
                        for _, _, actual, saved in anchors:
                            for name in actual:
                                torch.testing.assert_close(actual[name], saved[name], atol=0, rtol=0)
                        if label != "baseline":
                            for row in server.feedback_history:
                                training = row["training"]
                                target = next(t for t in training if t["client_id"] == 0)
                                helpers = [t for t in training if t["client_id"] != 0]
                                self.assertEqual(target["effective_prox_mu"], 0.)
                                self.assertEqual(target["proximal_loss"], 0.)
                                self.assertEqual(target["end_proximal_loss"], 0.)
                                self.assertGreater(target["low_rank_start_distance"], 0.)
                                self.assertEqual(row["target_total_local_epochs"], 5)
                                self.assertEqual(row["server_anchor_weight"], 0.)
                                self.assertEqual(sum(t["prox_enabled"] for t in helpers), 19 if mu else 0)
                                self.assertTrue(all(t["effective_prox_mu"] == mu for t in helpers))
                                self.assertTrue(all(t["processed_train_samples"] == 80 and t["train_batches"] == 5 for t in training))
                                for t in training:
                                    self.assertAlmostEqual(t["low_rank_start_distance"] ** 2, t["low_rank_start_squared_distance"], places=12)
                                    self.assertEqual(t["end_proximal_loss"], .5 * t["effective_prox_mu"] * t["low_rank_start_squared_distance"])
                                if mu:
                                    self.assertTrue(any(t["proximal_loss"] > 0 for t in helpers))
                                    self.assertTrue(any(t["proximal_loss"] != t["end_proximal_loss"] for t in helpers))
                            report = Path(server.report_dir())
                            with open(report / "feedback_training.csv", newline="") as stream:
                                csv_rows = list(csv.DictReader(stream))
                            self.assertEqual(len(csv_rows), 40)
                            self.assertEqual(sum(float(t["proximal_loss"]) > 0 for t in csv_rows), 38 if mu else 0)
                            with open(report / "feedback_metrics.json") as stream:
                                document = json.load(stream)
                            self.assertEqual(document["target_prox_mu"], mu)
                            self.assertEqual(document["target_prox_enabled_client_ids"], list(range(1, 20)) if mu else [])
                            with h5py.File(args.save_file_paths[-1]) as file:
                                group = file["target_feedback"]
                                self.assertEqual(group["training_proximal_loss"].shape, (2, 20))
                                self.assertEqual(group.attrs["target_prox_mu"], mu)
                                ids = group["training_client_ids"][:]
                                self.assertTrue((group["training_effective_prox_mu"][:][ids == 0] == 0).all())
                            with open(Path(server.final_model_dir()) / "manifest.json") as stream:
                                manifest = json.load(stream)
                            self.assertEqual(manifest["target_prox_mu"], mu)
                            self.assertEqual(len(manifest["final_round_local_training"]), 20)
                            if mu:
                                for path in (server.save_folder_name, server.report_dir(), args.save_file_paths[-1], server.final_model_dir(), args.exp_name):
                                    self.assertIn("feedback_avg", path)
                                    self.assertIn("prox_helpers_mu0p1", path)
                        for role in ("Server", *(c.role for c in server.clients)):
                            model = load_item(role, "model", server.save_folder_name)
                            torch.testing.assert_close(model.buffer_probe, torch.tensor([-7.] if role == "Server" else [3.]), atol=0, rtol=0)
                            # Frozen local parameters stay at their actual download;
                            # legacy float32 parameter Avg can round that value.
                            expected_frozen = records[1]["feedback"]["frozen_probe"] if role == "Server" else frozen_starts[1, int(role.split("_")[1])]
                            torch.testing.assert_close(model.frozen_probe, expected_frozen, atol=0, rtol=0)
                            self.assertEqual(model.fc3.weight.shape, (100, 512))
                        if label != "helpers":
                            trajectories[label] = records
                for round_index in (0, 1):
                    old, zero = trajectories["baseline"][round_index], trajectories["zero"][round_index]
                    self.assertEqual(old["helper_ids"], zero["helper_ids"])
                    for key in ("intermediate", "target_upload", "feedback"):
                        for name, p in old[key].items():
                            torch.testing.assert_close(zero[key][name], p, atol=0, rtol=0)
                    for cid in old["helper_uploads"]:
                        for name, p in old["helper_uploads"][cid].items():
                            torch.testing.assert_close(zero["helper_uploads"][cid][name], p, atol=0, rtol=0)
                    for key in ("helper_rng", "feedback_rng"):
                        import random, numpy as np
                        self.assertEqual(old[key][0], zero[key][0])
                        np.testing.assert_array_equal(old[key][1][1], zero[key][1][1])
                        self.assertEqual(old[key][1][2:], zero[key][1][2:])
                        torch.testing.assert_close(old[key][2], zero[key][2], atol=0, rtol=0)
                        for left, right in zip(old[key][3], zero[key][3]):
                            torch.testing.assert_close(left, right, atol=0, rtol=0)
            finally:
                os.chdir(old_cwd)


if __name__ == "__main__":
    unittest.main()

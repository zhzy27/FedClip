"""Helper-first feedback: 19 uploads, one C0 local5, frozen S and next-round C, real CNN exports."""

from contextlib import redirect_stdout
import copy
import hashlib
import io
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import h5py
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "system"))
from flcore.clients.clientbase import load_item, save_item
from flcore.servers.serverTargetProj import FedTargetProj
from utils.feedback_options import FEEDBACK_MODES, feedback_tag
from utils.group_projection import preserved_group_rng
from test_meta_projection import rng_state, assert_rng


class TargetFeedbackCNNRuntimeTests(unittest.TestCase):
    def test_two_modes_two_rounds_helper_only_uploads_C0_once_real_download_next_server_and_exports(self):
        torch.set_num_threads(2)
        generator = torch.Generator().manual_seed(91)
        x = torch.randn(16, 3, 32, 32, generator=generator)
        factory = "Hyper_CNN_512(in_features=3,num_classes=100,n_kernels=4,ratio_LR={rank},input_size=32)"
        previous = os.getcwd()
        first_helper_posts, first_helper_rngs = [], []
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            try:
                os.chdir(directory)
                def reader(dataset, cid, args, is_train=True, **kwargs):
                    return [(x[i], torch.tensor((cid + i % 3) % 100)) for i in range(16 if is_train else 4)]
                for mode in FEEDBACK_MODES:
                    label = feedback_tag(mode)
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
                        feedback_anchor_weight=.125 if mode == "feedback_avg_self" else None,
                        save_folder_name=str(root / label), final_model_root=str(root / "final" / label),
                        h5_result_root=str(root / "h5" / label), training_log_root=str(root / "logs" / label))
                    output = io.StringIO()
                    with redirect_stdout(output), patch("flcore.clients.clientbase.read_client_data", side_effect=reader), \
                            patch("flcore.servers.serverbase.read_client_data", side_effect=reader):
                        server = FedTargetProj(args, 0)
                        for role in ("Server", *(c.role for c in server.clients)):
                            model = load_item(role, "model", server.save_folder_name)
                            model.register_buffer("buffer_probe", torch.tensor([-10.] if role == "Server" else [0.]))
                            save_item(model, role, "model", server.save_folder_name)
                        downloads, train_calls = [0] * 20, [0] * 20
                        target_starts, target_rng_changes = {}, []
                        state = dict(helper_phase=False)
                        def compare_params(left, right):
                            self.assertEqual(left.keys(), right.keys())
                            for name in left:
                                torch.testing.assert_close(left[name], right[name], atol=0, rtol=0)
                        begin_native = server._feedback_begin_round
                        def begin():
                            state["helper_phase"] = True
                            state["target_hash"] = hashlib.sha256((Path(server.save_folder_name) / "Client_0_model.pt").read_bytes()).hexdigest()
                            state["target_stats"] = copy.deepcopy(getattr(server.clients[0], "last_normal_train_stats", None))
                            before = rng_state()
                            begin_native()
                            assert_rng(self, before)
                        server._feedback_begin_round = begin
                        target_load_native = server.clients[0]._load_model
                        def target_load(role=None):
                            self.assertFalse(state["helper_phase"], "Old C0 checkpoint read during helper stage")
                            return target_load_native(role)
                        server.clients[0]._load_model = target_load
                        full_native = server._load_full_parameters
                        def full(cid):
                            if cid == 0:
                                self.assertFalse(state["helper_phase"], "C0 used as a helper-stage upload")
                            return full_native(cid)
                        server._load_full_parameters = full
                        for client in server.clients:
                            native_download, native_train = client.set_parameters, client.train
                            def download(native=native_download, client=client):
                                cid = client.id
                                if cid == 0:
                                    self.assertEqual(downloads[0], server.cur_ground)
                                    self.assertEqual(train_calls[0], server.cur_ground)
                                    self.assertEqual(hashlib.sha256((Path(server.save_folder_name) / "Client_0_model.pt").read_bytes()).hexdigest(), state["target_hash"])
                                    self.assertEqual(getattr(client, "last_normal_train_stats", None), state["target_stats"])
                                    state["helper_phase"] = False
                                with preserved_group_rng():
                                    source = load_item("Server", "model", server.save_folder_name)
                                    if cid == 0:
                                        expected_source = torch.load(server._feedback_slot() / "intermediate_server.pt", weights_only=False)
                                    elif server.cur_ground > 0:
                                        path = Path(server.save_folder_name) / "feedback_helper_uploads" / f"slot_{(server.cur_ground - 1) % 2}" / "feedback_server.pt"
                                        expected_source = torch.load(path, weights_only=False)
                                    else:
                                        expected_source = source
                                    compare_params(dict(source.named_parameters()), dict(expected_source.named_parameters()))
                                    template = client._load_model()
                                    source.decom_larger_model(template.ratio_LR)
                                    expected = {n: p.detach().clone() for n, p in source.named_parameters()}
                                    buffer_before = template.buffer_probe.clone()
                                native()
                                downloads[cid] += 1
                                actual = client._load_model()
                                compare_params(dict(actual.named_parameters()), expected)
                                torch.testing.assert_close(actual.buffer_probe, buffer_before, atol=0, rtol=0)
                                if cid == 0:
                                    target_starts[server.cur_ground] = expected
                            def train(current_round=0, native=native_train, client=client):
                                cid = client.id
                                self.assertEqual(train_calls[cid], current_round)
                                if cid == 0:
                                    self.assertFalse(state["helper_phase"])
                                    compare_params(dict(client._load_model().named_parameters()), target_starts[current_round])
                                    slot = server._feedback_slot()
                                    hashes = {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in slot.glob("Client_*_full.pt")}
                                    self.assertEqual(len(hashes), 19)
                                    self.assertNotIn("Client_0_full.pt", hashes)
                                    meta = (slot / "metadata.json").read_bytes()
                                before = rng_state()
                                result = native(current_round=current_round)
                                if cid == 0:
                                    target_rng_changes.append(not torch.equal(before[2], torch.get_rng_state()))
                                    self.assertEqual({path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in slot.glob("Client_*_full.pt")}, hashes)
                                    self.assertEqual((slot / "metadata.json").read_bytes(), meta)
                                train_calls[cid] += 1
                                model = client._load_model()
                                model.buffer_probe.add_(1.)
                                save_item(model, client.role, "model", server.save_folder_name)
                                return result
                            client.set_parameters, client.train = download, train
                        cache_native = server._feedback_cache_helpers
                        def cache(training):
                            self.assertEqual(set(server.uploaded_ids), set(range(1, 20)))
                            self.assertEqual(len(server.uploaded_ids), 19)
                            self.assertTrue(all(row["client_id"] != 0 for row in training))
                            before = rng_state()
                            seconds = cache_native(training)
                            assert_rng(self, before)
                            if server.cur_ground == 0:
                                first_helper_rngs.append(before)
                                first_helper_posts.append([server._feedback_load_helper(cid) for cid in (1, 4, 7, 19)])
                            with self.assertRaises(ValueError):
                                server._feedback_load_helper(0)
                            return seconds
                        server._feedback_cache_helpers = cache
                        accuracy_native = server._feedback_accuracy
                        def accuracy():
                            path = Path(server.save_folder_name) / "Client_0_model.pt"
                            checksum = hashlib.sha256(path.read_bytes()).hexdigest()
                            before = rng_state()
                            result = accuracy_native()
                            assert_rng(self, before)
                            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), checksum)
                            return result
                        server._feedback_accuracy = accuracy
                        commit_native = server._feedback_commit_target
                        def commit():
                            before = rng_state()
                            result = commit_native()
                            assert_rng(self, before)
                            with preserved_group_rng():
                                target_full = server._load_full_parameters(0)
                                shared = load_item("Server", "model", server.save_folder_name)
                                compare_params(dict(shared.named_parameters()), target_full)
                                torch.testing.assert_close(shared.buffer_probe, torch.tensor([-10.]), atol=0, rtol=0)
                            return result
                        server._feedback_commit_target = commit
                        with patch("flcore.clients.clientTargetProj.make_prox_anchor", side_effect=AssertionError("Prox enabled")), \
                                patch.object(server, "_generate_dwa_guidance", side_effect=AssertionError("guidance called")):
                            server.train()
                        self.assertEqual(train_calls, [2] * 20)
                        self.assertEqual(downloads, [2] * 20)
                        self.assertEqual([c.train_time_cost["num_rounds"] for c in server.clients], [2] * 20)
                        self.assertEqual([c.send_time_cost["num_rounds"] for c in server.clients], [2] * 20)
                        self.assertEqual(target_rng_changes, [True, True])
                        self.assertEqual(len(server.feedback_history), 2)
                        self.assertEqual(server.target_proj_history, [])
                        for row in server.feedback_history:
                            self.assertEqual(row["target_total_local_epochs"], 5)
                            self.assertEqual(row["helper_local_epochs"], 5)
                            self.assertEqual(len(row["training"]), 20)
                            self.assertEqual(len(row["clients"]), 19)
                            self.assertEqual(row["target_processed_train_samples"], 80)
                            self.assertEqual(row["helper_train_batches"], 95)
                            self.assertEqual(row["target_train_batches"], 5)
                            self.assertEqual(row["target_finetune_gain"], row["target_post_local_acc"] - row["target_intermediate_download_acc"])
                            self.assertAlmostEqual(row["server_anchor_weight"], 0. if mode == "feedback_avg" else .125)
                            self.assertAlmostEqual(row["helper_total_weight"], 1. if mode == "feedback_avg" else .875)
                            self.assertEqual(row["target_upload_in_helper_stage"], 0)
                            self.assertNotIn("target_stage1_post_local_acc", row)
                        server._feedback_helper_round = 0
                        with self.assertRaises(RuntimeError):
                            server._feedback_load_helper(1)
                        server._feedback_helper_round = 1
                        for role in ("Server", *(c.role for c in server.clients)):
                            model = load_item(role, "model", server.save_folder_name)
                            torch.testing.assert_close(model.buffer_probe, torch.tensor([-10.] if role == "Server" else [2.]), atol=0, rtol=0)
                            self.assertEqual(model.fc3.weight.shape, (100, 512))
                            self.assertTrue(all(torch.isfinite(p).all() for p in model.parameters()))
                        report = Path(server.report_dir())
                        for filename in ("feedback_metrics.csv", "feedback_metrics.json", "feedback_clients.csv", "feedback_training.csv"):
                            self.assertTrue((report / filename).is_file())
                        with open(report / "feedback_metrics.json") as stream:
                            document = json.load(stream)
                        self.assertEqual(document["schema"], 2)
                        self.assertEqual(document["primary_accuracy_metric"], "target_post_local_acc")
                        self.assertEqual(document["summary"]["target_local"]["last20_count"], 2)
                        self.assertEqual(document["target_epochs_per_round"], 5)
                        with h5py.File(args.save_file_paths[-1]) as file:
                            g = file["target_feedback"]
                            self.assertEqual(g["client_aggregation_weight"].shape, (2, 19))
                            self.assertEqual(g["training_ce_loss"].shape, (2, 20))
                            self.assertEqual(g.attrs["target_epochs_per_round"], 5)
                            self.assertEqual(g.attrs["target_local_last20_count"], 2)
                            self.assertNotIn("target_stage1_post_local_acc", g)
                        final = Path(server.final_model_dir())
                        with open(final / "manifest.json") as stream:
                            manifest = json.load(stream)
                        self.assertEqual(manifest["primary_accuracy_metric"], "target_post_local_acc")
                        self.assertEqual(manifest["target_epochs_per_round"], 5)
                        shared = load_item("Server", "model", str(final))
                        with preserved_group_rng():
                            compare_params(dict(shared.named_parameters()), server._load_full_parameters(0))
                        self.assertTrue((report / Path(args.save_file_paths[-1]).name).is_file())
                        self.assertFalse((report / "Server_model.pt").exists())
                        self.assertIn("[FeedbackTargetTrain]", output.getvalue())
                        self.assertNotIn("C0 ordinary five-epoch training FROM", output.getvalue())
                self.assertEqual(first_helper_rngs[0][0], first_helper_rngs[1][0])
                torch.testing.assert_close(first_helper_rngs[0][2], first_helper_rngs[1][2], atol=0, rtol=0)
                for left, right in zip(first_helper_posts[0], first_helper_posts[1]):
                    for name in left:
                        torch.testing.assert_close(left[name], right[name], atol=0, rtol=0)
            finally:
                os.chdir(previous)


if __name__ == "__main__":
    unittest.main()

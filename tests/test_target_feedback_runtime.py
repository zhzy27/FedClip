"""Three feedback modes: real heterogeneous CNN, actual H download/next C, local5+5 and exports."""

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
    def test_three_modes_two_rounds_original_capacity_downloads_state_budget_and_exports(self):
        torch.set_num_threads(2)
        generator = torch.Generator().manual_seed(91)
        x = torch.randn(16, 3, 32, 32, generator=generator)
        factory = "Hyper_CNN_512(in_features=3,num_classes=100,n_kernels=4,ratio_LR={rank},input_size=32)"
        previous = os.getcwd()
        first_stage_posts, first_stage_rngs = [], []
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
                        save_folder_name=str(root / label), final_model_root=str(root / "final" / label),
                        h5_result_root=str(root / "h5" / label), training_log_root=str(root / "logs" / label))
                    output = io.StringIO()
                    with redirect_stdout(output), patch("flcore.clients.clientbase.read_client_data", side_effect=reader), \
                            patch("flcore.servers.serverbase.read_client_data", side_effect=reader):
                        server = FedTargetProj(args, 0)
                        # Deliberately distinct server/client buffers; real train wrapper
                        # increments client counters to verify feedback excludes them.
                        for role in ("Server", *(c.role for c in server.clients)):
                            model = load_item(role, "model", server.save_folder_name)
                            model.register_buffer("buffer_probe", torch.tensor([-10.] if role == "Server" else [0.]))
                            save_item(model, role, "model", server.save_folder_name)
                        downloads, train_calls = [0] * 20, [0] * 20
                        stage2_starts, download_rng_changes, train_rng_changes = {}, [], []
                        def compare_params(left, right):
                            self.assertEqual(left.keys(), right.keys())
                            for name in left:
                                torch.testing.assert_close(left[name], right[name], atol=0, rtol=0)
                        for client in server.clients:
                            native_download, native_train = client.set_parameters, client.train
                            def download(native=native_download, client=client):
                                cid = client.id
                                stage2 = cid == 0 and downloads[cid] % 2 == 1
                                with preserved_group_rng():
                                    source = load_item("Server", "model", server.save_folder_name)
                                    if stage2:
                                        expected_source = torch.load(server._feedback_slot() / "intermediate_server.pt", weights_only=False)
                                    elif server.cur_ground > 0:
                                        path = Path(server.save_folder_name) / "feedback_stage1" / f"slot_{(server.cur_ground - 1) % 2}" / "feedback_server.pt"
                                        expected_source = torch.load(path, weights_only=False)
                                    else:
                                        expected_source = source
                                    compare_params(dict(source.named_parameters()), dict(expected_source.named_parameters()))
                                    target = client._load_model()
                                    source.decom_larger_model(target.ratio_LR)
                                    expected = {n: p.detach().clone() for n, p in source.named_parameters()}
                                    buffer_before = target.buffer_probe.clone()
                                before = rng_state()
                                native()
                                if stage2:
                                    download_rng_changes.append(not torch.equal(before[2], torch.get_rng_state()))
                                downloads[cid] += 1
                                actual = client._load_model()
                                compare_params(dict(actual.named_parameters()), expected)
                                torch.testing.assert_close(actual.buffer_probe, buffer_before, atol=0, rtol=0)
                                if stage2:
                                    stage2_starts[server.cur_ground] = {n: p.detach().clone() for n, p in actual.named_parameters()}
                            def train(current_round=0, native=native_train, client=client):
                                cid = client.id
                                stage2 = cid == 0 and train_calls[cid] % 2 == 1
                                if stage2:
                                    compare_params(dict(client._load_model().named_parameters()), stage2_starts[server.cur_ground])
                                    slot = server._feedback_slot()
                                    before_hash = hashlib.sha256((slot / "Client_0_full.pt").read_bytes()).hexdigest()
                                    before_low = hashlib.sha256((slot / "Client_0_low_rank.pt").read_bytes()).hexdigest()
                                    first_stats = copy.deepcopy(client.last_normal_train_stats)
                                    first_meta = json.loads((slot / "metadata.json").read_text())
                                    saved_stats = next(row for row in first_meta["training"] if row["client_id"] == 0)
                                    self.assertEqual(saved_stats["ce_loss"], first_stats["ce_loss"])
                                before = rng_state()
                                result = native(current_round=current_round)
                                if stage2:
                                    train_rng_changes.append(not torch.equal(before[2], torch.get_rng_state()))
                                    self.assertEqual(hashlib.sha256((slot / "Client_0_full.pt").read_bytes()).hexdigest(), before_hash)
                                    self.assertEqual(hashlib.sha256((slot / "Client_0_low_rank.pt").read_bytes()).hexdigest(), before_low)
                                    self.assertEqual(json.loads((slot / "metadata.json").read_text()), first_meta)
                                train_calls[cid] += 1
                                model = client._load_model()
                                model.buffer_probe.add_(1.)
                                save_item(model, client.role, "model", server.save_folder_name)
                                return result
                            client.set_parameters, client.train = download, train
                        cache_native = server._feedback_cache_first_stage
                        def cache(training):
                            before = rng_state()
                            seconds = cache_native(training)
                            assert_rng(self, before)
                            if server.cur_ground == 0:
                                first_stage_rngs.append(before)
                                first_stage_posts.append([server._feedback_load_first_stage(cid) for cid in (0, 4, 7, 19)])
                            return seconds
                        server._feedback_cache_first_stage = cache
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
                                feedback = server._load_full_parameters(0)
                                shared = load_item("Server", "model", server.save_folder_name)
                                compare_params(dict(shared.named_parameters()), feedback)
                                torch.testing.assert_close(shared.buffer_probe, torch.tensor([-10.]), atol=0, rtol=0)
                            return result
                        server._feedback_commit_target = commit
                        with patch("flcore.clients.clientTargetProj.make_prox_anchor", side_effect=AssertionError("Prox enabled")), \
                                patch.object(server, "_generate_dwa_guidance", side_effect=AssertionError("guidance called")):
                            server.train()
                        self.assertEqual(train_calls, [4] + [2] * 19)
                        self.assertEqual(downloads, [4] + [2] * 19)
                        self.assertEqual([c.train_time_cost["num_rounds"] for c in server.clients], [4] + [2] * 19)
                        self.assertEqual(download_rng_changes, [True, True])
                        self.assertEqual(train_rng_changes, [True, True])
                        self.assertEqual(len(server.feedback_history), 2)
                        self.assertEqual(server.target_proj_history, [])
                        for row in server.feedback_history:
                            self.assertEqual(row["target_total_local_epochs"], 10)
                            self.assertEqual(row["helper_local_epochs"], 5)
                            self.assertEqual(len(row["training"]), 21)
                            self.assertEqual(row["target_stage1_processed_train_samples"], 80)
                            self.assertEqual(row["target_stage2_processed_train_samples"], 80)
                            self.assertEqual(row["stage1_train_batches"], 100)
                            self.assertEqual(row["target_stage2_train_batches"], 5)
                            self.assertEqual(row["target_stage2_finetune_gain"],
                                             row["target_stage2_post_local_acc"] - row["target_intermediate_download_acc"])
                            self.assertAlmostEqual(row["target_weight"], .05 if mode == "feedback_avg" else .125)
                            self.assertAlmostEqual(row["helper_total_weight"], .95 if mode == "feedback_avg" else .875)
                            self.assertEqual(row["projection_enabled"], int(mode == "feedback_projection_self"))
                            self.assertTrue(all(r["local_epochs"] == 5 for r in row["training"]))
                        # Stale-cache reads are actively rejected, rather than reusing an old reference.
                        server._feedback_stage1_round = 0
                        with self.assertRaises(RuntimeError):
                            server._feedback_load_first_stage(0)
                        server._feedback_stage1_round = 1
                        for role in ("Server", *(c.role for c in server.clients)):
                            model = load_item(role, "model", server.save_folder_name)
                            expected = -10. if role == "Server" else 4. if role == "Client_0" else 2.
                            torch.testing.assert_close(model.buffer_probe, torch.tensor([expected]), atol=0, rtol=0)
                            self.assertEqual(model.fc3.weight.shape, (100, 512))
                            self.assertTrue(all(torch.isfinite(p).all() for p in model.parameters()))
                        report = Path(server.report_dir())
                        for filename in ("feedback_metrics.csv", "feedback_metrics.json", "feedback_clients.csv", "feedback_training.csv"):
                            self.assertTrue((report / filename).is_file())
                        with open(report / "feedback_metrics.json") as stream:
                            document = json.load(stream)
                        self.assertEqual(document["primary_accuracy_metric"], "target_stage2_post_local_acc")
                        self.assertEqual(document["summary"]["stage2"]["last20_count"], 2)
                        self.assertEqual(document["summary"]["stage2"]["final"], server.feedback_history[-1]["target_stage2_post_local_acc"])
                        with h5py.File(args.save_file_paths[-1]) as file:
                            g = file["target_feedback"]
                            self.assertEqual(g["client_aggregation_weight"].shape, (2, 20))
                            self.assertEqual(g["training_ce_loss"].shape, (2, 21))
                            self.assertEqual(g.attrs["target_epochs_per_round"], 10)
                            self.assertEqual(g.attrs["stage2_last20_count"], 2)
                            self.assertEqual(list(g["target_total_local_epochs"][:]), [10, 10])
                        final = Path(server.final_model_dir())
                        with open(final / "manifest.json") as stream:
                            manifest = json.load(stream)
                        self.assertEqual(manifest["primary_accuracy_metric"], "target_stage2_post_local_acc")
                        shared = load_item("Server", "model", str(final))
                        with preserved_group_rng():
                            compare_params(dict(shared.named_parameters()), server._load_full_parameters(0))
                        self.assertTrue((report / Path(args.save_file_paths[-1]).name).is_file())
                        self.assertFalse((report / "Server_model.pt").exists())
                        self.assertFalse((report / "target_proj_metrics.csv").exists())
                        self.assertIn("[FeedbackStage2Train]", output.getvalue())
                        self.assertIn("[FeedbackSummary][stage2]", output.getvalue())
                for index in (1, 2):
                    self.assertEqual(first_stage_rngs[0][0], first_stage_rngs[index][0])
                    torch.testing.assert_close(first_stage_rngs[0][2], first_stage_rngs[index][2], atol=0, rtol=0)
                    for left, right in zip(first_stage_posts[0], first_stage_posts[index]):
                        for name in left:
                            torch.testing.assert_close(left[name], right[name], atol=0, rtol=0)
            finally:
                os.chdir(previous)


if __name__ == "__main__":
    unittest.main()

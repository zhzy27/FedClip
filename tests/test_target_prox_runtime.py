"""Full two-round heterogeneous CNN smoke: zero equivalence, helpers and all."""

from contextlib import redirect_stdout
import copy
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
from flcore.servers.serverTargetProj import FedTargetProj
from flcore.clients.clientbase import load_item, save_item
from test_meta_projection import rng_state


class TargetProxCNNRuntimeTests(unittest.TestCase):
    def test_baseline_zero_helpers_all_real_CNN_full5_download_anchor_and_exports(self):
        torch.set_num_threads(2)
        generator = torch.Generator().manual_seed(29)
        x = torch.randn(16, 3, 32, 32, generator=generator)
        factory = "Hyper_CNN_512(in_features=3,num_classes=100,n_kernels=4,ratio_LR={rank},input_size=32)"
        previous = os.getcwd()
        baseline_checkpoints, zero_checkpoints, rngs, summaries = [], [], [], []
        aggregate_checkpoints = {"old": [], "zero": []}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            try:
                os.chdir(directory)
                def reader(dataset, cid, args, is_train=True, **kwargs):
                    return [(x[i], torch.tensor((cid + i % 2) % 100)) for i in range(16 if is_train else 4)]
                for label, mu, scope, marked in (("old", 0., "helpers", False), ("zero", 0., "helpers", True),
                                               ("helpers", .1, "helpers", True), ("all", .1, "all", True)):
                    args = SimpleNamespace(device="cpu", dataset="Synthetic", num_classes=100, global_rounds=1,
                        local_epochs=5, batch_size=16, local_learning_rate=.005, num_clients=20, join_ratio=1.,
                        random_join_ratio=False, few_shot=0, algorithm="FedTargetProj", time_select=False,
                        goal="smoke", time_threthold=1e9, top_cnt=10, auto_break=False, eval_gap=1,
                        client_drop_rate=0., train_slow_rate=0., send_slow_rate=0., resume=False,
                        model_family="Decom_CNN-5-512", models_folder_name="", niid=1, partition="pat",
                        class_per_client=20, exp_name=label, save_file_paths=[],
                        models=[factory.format(rank=r) for r in (.9, .37, .35, .25, .15)],
                        global_model=factory.format(rank=.15), target_client_id=0, seed=0,
                        is_regular=1, regular_lamda=.001, target_proj_mode="projection_softmax",
                        projection_self_weight=.125, target_prox_mu=mu, target_prox_scope=scope,
                        target_prox_experiment=marked, mu=500.,
                        save_folder_name=str(root / label), final_model_root=str(root / "final"),
                        h5_result_root=str(root / "h5"), training_log_root=str(root / "logs"))
                    with redirect_stdout(io.StringIO()), patch("flcore.clients.clientbase.read_client_data", side_effect=reader), \
                            patch("flcore.servers.serverbase.read_client_data", side_effect=reader):
                        server = FedTargetProj(args, 0)
                        for c in server.clients:
                            model = load_item(c.role, "model", c.save_folder_name)
                            model.register_buffer("untouched_probe", torch.tensor([7.]))
                            save_item(model, c.role, "model", c.save_folder_name)
                        native_send = server.send_parameters
                        expected_starts = {}
                        def record_download():
                            native_send()
                            expected_starts.clear()
                            for cid in (0, 1, 4):
                                model = load_item(f"Client_{cid}", "model", server.save_folder_name)
                                expected_starts[cid] = {n: p.detach().clone() for n, p in model.named_parameters()}
                        server.send_parameters = record_download
                        if marked:
                            from utils.target_prox import make_prox_anchor
                            native_anchor = make_prox_anchor
                            def verify_anchor(model):
                                actual = native_anchor(model)
                                # Capacity identifies one of the first-cycle clients;
                                # subsequent same-capacity clients have identical downloads.
                                cid = 0 if model.ratio_LR == .9 else 1 if model.ratio_LR == .37 else 4 if model.ratio_LR == .15 else None
                                if cid is not None:
                                    for name in actual:
                                        torch.testing.assert_close(actual[name], expected_starts[cid][name], atol=0, rtol=0)
                                return actual
                            anchor_patch = patch("flcore.clients.clientTargetProj.make_prox_anchor", side_effect=verify_anchor)
                        else:
                            anchor_patch = patch("flcore.clients.clientTargetProj.make_prox_anchor",
                                                 side_effect=AssertionError("mu zero creates no training anchor"))
                        native_aggregate = server.aggregate_parameters_avg
                        def capture_post_local():
                            if label in ("old", "zero"):
                                collection = baseline_checkpoints if label == "old" else zero_checkpoints
                                collection.append([{
                                    n: p.detach().clone() for n, p in load_item(f"Client_{cid}", "model", server.save_folder_name).named_parameters()
                                } for cid in range(20)])
                                rngs.append((label, rng_state()))
                            native_aggregate()
                            if label in aggregate_checkpoints:
                                global_model = load_item("Server", "model", server.save_folder_name)
                                aggregate_checkpoints[label].append({n: p.detach().clone() for n, p in global_model.named_parameters()})
                        server.aggregate_parameters_avg = capture_post_local
                        with anchor_patch:
                            server.train()
                        self.assertEqual([c.train_time_cost["num_rounds"] for c in server.clients], [2] * 20)
                        summaries.append(server._local_accuracy_summary())
                        for row in server.target_proj_history:
                            self.assertEqual(row["target_weight"], .125)
                            self.assertAlmostEqual(row["helper_total_weight"], .875)
                        if marked:
                            local = server.target_prox_local_history
                            self.assertEqual(len(local), 40)
                            self.assertEqual([r["actual_local_epochs"] for r in local], [5] * 40)
                            self.assertEqual([r["train_batches"] for r in local], [5] * 40)
                            count = 0 if mu == 0 else 20 if scope == "all" else 19
                            for round_number in (1, 2):
                                rows = [r for r in local if r["round"] == round_number]
                                self.assertEqual(sum(r["prox_enabled"] for r in rows), count)
                                c0 = next(r for r in rows if r["client_id"] == 0)
                                if scope == "helpers":
                                    self.assertEqual(c0["proximal_loss"], 0.)
                            if mu:
                                self.assertTrue(any(r["proximal_loss"] > 0 for r in local))
                            self.assertEqual(server._local_accuracy_summary()["last20_target_local_count"], 2)
                            log = Path(server.report_dir())
                            self.assertTrue((log / "target_prox_local.csv").is_file())
                            self.assertTrue((log / "target_prox_local.json").is_file())
                            with h5py.File(args.save_file_paths[-1]) as file:
                                group = file["target_projection"]
                                self.assertEqual(group["proximal_local_training"]["ce_loss"].shape, (2, 20))
                                self.assertEqual(group.attrs["target_prox_mu"], mu)
                                self.assertEqual(group.attrs["last20_target_local_count"], 2)
                            self.assertIn("prox_" + scope, server.save_folder_name)
                            self.assertIn("prox_" + scope, str(log))
                            self.assertIn("prox_" + scope, args.save_file_paths[-1])
                            self.assertIn("prox_" + scope, server.final_model_dir())
                        else:
                            self.assertFalse((Path(server.report_dir()) / "target_prox_local.csv").exists())
                        for c in server.clients:
                            model = load_item(c.role, "model", c.save_folder_name)
                            torch.testing.assert_close(model.untouched_probe, torch.tensor([7.]), atol=0, rtol=0)
                            self.assertEqual(model.fc3.weight.shape, (100, 512))
                        if label in ("old", "zero"):
                            summaries[-1].pop("last20_target_local_acc", None)
                            summaries[-1].pop("last20_target_local_count", None)
                self.assertEqual(summaries[0], summaries[1])
                for old_round, zero_round in zip(baseline_checkpoints, zero_checkpoints):
                    for old_client, zero_client in zip(old_round, zero_round):
                        for name in old_client:
                            torch.testing.assert_close(old_client[name], zero_client[name], atol=0, rtol=0)
                old_rng = [state for label, state in rngs if label == "old"]
                zero_rng = [state for label, state in rngs if label == "zero"]
                for old, zero in zip(old_rng, zero_rng):
                    self.assertEqual(old[0], zero[0])
                    torch.testing.assert_close(old[2], zero[2], atol=0, rtol=0)
                for old, zero in zip(aggregate_checkpoints["old"], aggregate_checkpoints["zero"]):
                    for name in old:
                        torch.testing.assert_close(old[name], zero[name], atol=0, rtol=0)
            finally:
                os.chdir(previous)


if __name__ == "__main__":
    unittest.main()

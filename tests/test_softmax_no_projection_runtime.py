"""Two rounds, 20 heterogeneous low-rank CNNs, normal local5, isolated observation/export."""

from contextlib import redirect_stdout
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
from utils.group_projection import preserved_group_rng
from utils.target_group_options import DEFAULT_GROUP_IDS
from test_meta_projection import rng_state, assert_rng


class NoProjectionCNNRuntimeTests(unittest.TestCase):
    def test_both_modes_full_local5_head_checkpoints_buffers_metrics_h5_and_manifest(self):
        torch.set_num_threads(2)
        generator = torch.Generator().manual_seed(79)
        x = torch.randn(24, 3, 32, 32, generator=generator)
        factory = "Hyper_CNN_512(in_features=3,num_classes=100,n_kernels=4,ratio_LR={rank},input_size=32)"
        previous = os.getcwd()
        first_posts, first_rngs = [], []
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            try:
                os.chdir(directory)
                def reader(dataset, cid, args, is_train=True, **kwargs):
                    # Unequal test shard lengths exercise correct-count weighted reporting.
                    return [(x[i], torch.tensor((cid + i % 3) % 100)) for i in range(16 if is_train else cid + 4)]
                for mode in ("softmax_only", "group_softmax_only"):
                    group_mode = mode.startswith("group_")
                    label = mode + ("_beta0p5" if group_mode else "_self0p125")
                    args = SimpleNamespace(device="cpu", dataset="Synthetic", num_classes=100, global_rounds=1,
                        local_epochs=5, batch_size=16, local_learning_rate=.005, num_clients=20, join_ratio=1.,
                        random_join_ratio=False, few_shot=0, algorithm="FedTargetProj", time_select=False,
                        goal="smoke", time_threthold=1e9, top_cnt=10, auto_break=False, eval_gap=1,
                        client_drop_rate=0., train_slow_rate=0., send_slow_rate=0., resume=False,
                        model_family="Decom_CNN-5-512", models_folder_name="", niid=1, partition="pat",
                        class_per_client=20, exp_name=label, save_file_paths=[],
                        models=[factory.format(rank=r) for r in (.9, .37, .35, .25, .15)],
                        global_model=factory.format(rank=.15), target_client_id=0, seed=0,
                        is_regular=1, regular_lamda=.001, target_proj_mode=mode,
                        projection_self_weight=None if group_mode else .125,
                        target_group_ids=list(DEFAULT_GROUP_IDS) if group_mode else None,
                        target_group_weight=.5 if group_mode else None,
                        target_prox_mu=0., target_prox_scope="helpers", target_prox_experiment=False,
                        save_folder_name=str(root / label), final_model_root=str(root / "final" / label),
                        h5_result_root=str(root / "h5" / label), training_log_root=str(root / "logs" / label))
                    with redirect_stdout(io.StringIO()), patch("flcore.clients.clientbase.read_client_data", side_effect=reader), \
                            patch("flcore.servers.serverbase.read_client_data", side_effect=reader):
                        server = FedTargetProj(args, 0)
                        for role in ("Server", *(client.role for client in server.clients)):
                            model = load_item(role, "model", server.save_folder_name)
                            model.register_buffer("untouched_probe", torch.tensor([3., 7.]))
                            save_item(model, role, "model", server.save_folder_name)
                        native = server.aggregate_parameters_avg
                        def verify_aggregation():
                            paths = [Path(server.save_folder_name) / f"Client_{cid}_model.pt" for cid in range(20)]
                            hashes = [hashlib.sha256(path.read_bytes()).hexdigest() for path in paths]
                            before = rng_state()
                            with preserved_group_rng():
                                if server.cur_ground == 0:
                                    first_rngs.append(before)
                                    first_posts.append([{n: p.detach().clone() for n, p in server.clients[cid]._load_model().named_parameters()}
                                                        for cid in (0, 4, 7, 19)])
                                # Extra evaluations must never change a normal model/checkpoint/RNG.
                                client = server.clients[0]
                                client.test_post_local()
                                client.test_downloaded_global()
                            assert_rng(self, before)
                            self.assertEqual([hashlib.sha256(path.read_bytes()).hexdigest() for path in paths], hashes)
                            native()
                            if group_mode:
                                assert_rng(self, before)
                            self.assertEqual([hashlib.sha256(path.read_bytes()).hexdigest() for path in paths], hashes)
                        server.aggregate_parameters_avg = verify_aggregation
                        with patch("flcore.clients.clientTargetProj.make_prox_anchor",
                                   side_effect=AssertionError("Zero Prox must use original training")):
                            server.train()
                        self.assertEqual([c.train_time_cost["num_rounds"] for c in server.clients], [2] * 20)
                        self.assertEqual([c.last_normal_train_stats["local_epochs"] for c in server.clients], [5] * 20)
                        self.assertFalse(hasattr(server, "meta_split_record"))
                        self.assertFalse(hasattr(server, "_dwa_guidance_round"))
                        report = Path(server.report_dir())
                        if group_mode:
                            self.assertEqual(len(server.group_history), 2)
                            for row in server.group_history:
                                members = row["members"]
                                expected = sum(m["post_local_correct"] for m in members) / sum(m["test_samples"] for m in members)
                                self.assertEqual(row["group_post_local_sample_weighted_acc"], expected)
                                self.assertAlmostEqual(row["actual_group_mass"], .5)
                                self.assertAlmostEqual(row["actual_outside_mass"], .5)
                                self.assertEqual(row["projection_enabled"], 0)
                                self.assertTrue(all(c["projection_coefficient"] == 0 for c in row["clients"]))
                            for filename in ("target_group_metrics.csv", "target_group_members.csv", "target_group_clients.csv"):
                                self.assertTrue((report / filename).is_file())
                            with open(report / "target_group_metrics.json") as stream:
                                document = json.load(stream)
                            self.assertEqual(document["primary_accuracy_metric"], "group_post_local_sample_weighted_acc")
                            self.assertEqual(document["summary"]["group_post_local"]["last20_count"], 2)
                        else:
                            self.assertEqual(len(server.target_proj_history), 2)
                            for row in server.target_proj_history:
                                self.assertEqual(row["target_weight"], .125)
                                self.assertAlmostEqual(row["helper_total_weight"], .875)
                                self.assertEqual(row["projection_enabled"], 0)
                            for filename in ("softmax_only_metrics.csv", "softmax_only_clients.csv", "target_proj_metrics.csv"):
                                self.assertTrue((report / filename).is_file())
                            with open(report / "softmax_only_matrices.json") as stream:
                                document = json.load(stream)
                            self.assertEqual(document["summary"]["last20_target_local_count"], 2)
                        with h5py.File(args.save_file_paths[-1]) as file:
                            g = file["target_group" if group_mode else "target_projection"]
                            self.assertEqual(g.attrs["projection_enabled"], 0)
                            self.assertEqual(g["client_aggregation_weight"].shape, (2, 20))
                            self.assertEqual(g["client_projection_coefficient"][:].sum(), 0)
                            if group_mode:
                                self.assertEqual(g["member_post_local_correct"].shape, (2, 7))
                                self.assertEqual(g.attrs["primary_accuracy_metric"], "group_post_local_sample_weighted_acc")
                            else:
                                self.assertEqual(g.attrs["last20_target_local_count"], 2)
                        for role in ("Server", *(c.role for c in server.clients)):
                            model = load_item(role, "model", server.save_folder_name)
                            torch.testing.assert_close(model.untouched_probe, torch.tensor([3., 7.]), atol=0, rtol=0)
                            self.assertEqual(model.fc3.weight.shape, (100, 512))
                            self.assertTrue(all(torch.isfinite(p).all() for p in model.parameters()))
                        with open(Path(server.final_model_dir()) / "manifest.json") as stream:
                            manifest = json.load(stream)
                        if group_mode:
                            self.assertEqual(manifest["primary_accuracy_metric"], "group_post_local_sample_weighted_acc")
                        else:
                            self.assertEqual(manifest["args"]["target_proj_mode"], "softmax_only")
                            self.assertEqual(manifest["args"]["projection_self_weight"], .125)
                        self.assertTrue((Path(server.final_model_dir()) / "Server_model.pt").is_file())
                        self.assertTrue((report / Path(args.save_file_paths[-1]).name).is_file())
                        self.assertFalse((report / "target_prox_local.csv").exists())
                        self.assertFalse((report / "Server_model.pt").exists())
                for left, right in zip(first_posts[0], first_posts[1]):
                    for name in left:
                        torch.testing.assert_close(left[name], right[name], atol=0, rtol=0)
                self.assertEqual(first_rngs[0][0], first_rngs[1][0])
                torch.testing.assert_close(first_rngs[0][2], first_rngs[1][2], atol=0, rtol=0)
            finally:
                os.chdir(previous)


if __name__ == "__main__":
    unittest.main()

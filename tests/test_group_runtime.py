"""20-client / 7-member real low-rank CNN smoke, normal epochs and immutable observations."""

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
from flcore.servers.serverTargetGroup import downloaded_group_accuracy
from utils.group_projection import GROUP_MODES, DEFAULT_GROUP_IDS
from test_meta_projection import rng_state, assert_rng


class GroupCNNRuntimeTests(unittest.TestCase):
    def test_all_three_modes_20_heterogeneous_clients_two_rounds_full_local5_and_exports(self):
        torch.set_num_threads(2)
        generator = torch.Generator().manual_seed(35)
        x = torch.randn(16, 3, 32, 32, generator=generator)
        factory = "Hyper_CNN_512(in_features=3,num_classes=100,n_kernels=4,ratio_LR={rank},input_size=32)"
        previous = os.getcwd()
        first_round_rng = []
        first_round_posts = []
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            try:
                os.chdir(directory)
                def reader(dataset, cid, args, is_train=True, **kwargs):
                    indices = range(16) if is_train else range(4)
                    return [(x[i], torch.tensor((cid + i % 2) % 100)) for i in indices]
                for mode in GROUP_MODES:
                    args = SimpleNamespace(device="cpu", dataset="Synthetic", num_classes=100, global_rounds=1,
                        local_epochs=5, batch_size=16, local_learning_rate=.005, num_clients=20, join_ratio=1.,
                        random_join_ratio=False, few_shot=0, algorithm="FedTargetProj", time_select=False,
                        goal="smoke", time_threthold=1e9, top_cnt=10, auto_break=False, eval_gap=1,
                        client_drop_rate=0., train_slow_rate=0., send_slow_rate=0., resume=False,
                        model_family="Decom_CNN-5-512", models_folder_name="", niid=1, partition="pat",
                        class_per_client=20, exp_name=mode, save_file_paths=[],
                        models=[factory.format(rank=r) for r in (.9, .37, .35, .25, .15)],
                        global_model=factory.format(rank=.15), target_client_id=0, seed=0,
                        is_regular=1, regular_lamda=.001, target_proj_mode=mode,
                        target_group_ids="0,1,2,3,4,6,7", target_group_weight=None,
                        save_folder_name=str(root / mode), final_model_root=str(root / "final"),
                        h5_result_root=str(root / "h5"), training_log_root=str(root / "logs"))
                    with redirect_stdout(io.StringIO()), patch("flcore.clients.clientbase.read_client_data", side_effect=reader), \
                            patch("flcore.servers.serverbase.read_client_data", side_effect=reader):
                        server = FedTargetProj(args, 0)
                        # Root buffers should survive both aggregation and evaluation.
                        for role in ("Server", *(c.role for c in server.clients)):
                            model = load_item(role, "model", server.save_folder_name)
                            model.register_buffer("state_probe", torch.tensor([3., 7.]))
                            save_item(model, role, "model", server.save_folder_name)
                        calls = [0] * 20
                        for client in server.clients:
                            native = client.set_parameters
                            def downlink(native=native, cid=client.id):
                                calls[cid] += 1
                                return native()
                            client.set_parameters = downlink
                        original = server._aggregate_target_group
                        def aggregate_and_verify():
                            paths = [Path(server.save_folder_name) / f"Client_{cid}_model.pt" for cid in range(20)]
                            hashes = [hashlib.sha256(p.read_bytes()).hexdigest() for p in paths]
                            before = rng_state()
                            if server.cur_ground == 0:
                                first_round_rng.append(before)
                                # Numeric equality across controls before their first aggregation.
                                with torch.no_grad():
                                    first_round_posts.append([{
                                        n: p.detach().clone() for n, p in server.clients[cid]._load_model().named_parameters()
                                    } for cid in (0, 4, 7, 19)])
                            original()
                            assert_rng(self, before)
                            self.assertEqual([hashlib.sha256(p.read_bytes()).hexdigest() for p in paths], hashes)
                            for role in ("Server", *(c.role for c in server.clients)):
                                model = load_item(role, "model", server.save_folder_name)
                                torch.testing.assert_close(model.state_probe, torch.tensor([3., 7.]), atol=0, rtol=0)
                        server._aggregate_target_group = aggregate_and_verify
                        server.train()
                        self.assertEqual(calls, [2] * 20)  # Evaluation never calls checkpoint-writing downlinks.
                        self.assertEqual([c.train_time_cost["num_rounds"] for c in server.clients], [2] * 20)
                        self.assertEqual(len(server.group_history), 2)
                        self.assertEqual(server.target_proj_history, [])
                        self.assertFalse(hasattr(server, "meta_split_record"))
                        self.assertFalse((Path(server.save_folder_name) / "target_proj_metrics.csv").exists())
                        for row in server.group_history:
                            self.assertEqual(row["target_group_ids"], list(DEFAULT_GROUP_IDS))
                            self.assertAlmostEqual(row["aggregation_mass"], 1.)
                            expected = sum(m["post_local_acc"] for m in row["members"]) / 7
                            self.assertAlmostEqual(row["group_post_local_acc"], expected)
                            self.assertEqual(len(row["clients"]), 20)
                            self.assertEqual([r["projection_coefficient"] for r in row["clients"] if r["in_target_group"]], [0.] * 7)
                        expected_mass = 1. if mode == "group_avg_only" else .35
                        self.assertAlmostEqual(server.group_history[-1]["actual_group_mass"], expected_mass)
                        with h5py.File(args.save_file_paths[-1]) as result:
                            self.assertNotIn("target_projection", result)
                            g = result["target_group"]
                            self.assertEqual(g["member_post_local_acc"].shape, (2, 7))
                            self.assertEqual(g["client_aggregation_weight"].shape, (2, 20))
                            self.assertEqual(g["group_pair_cosine_matrix"].shape, (2, 7, 7))
                            self.assertEqual(g.attrs["primary_accuracy_metric"], "group_post_local_acc")
                        self.assertFalse((Path(server.save_folder_name) / "target_group_metrics.json").exists())
                        with open(Path(server.report_dir()) / "target_group_metrics.json") as stream:
                            document = json.load(stream)
                        self.assertEqual(document["summary"]["group_post_local"]["last10_count"], 2)
                        self.assertNotIn("target_client_id", document["args"])
                        target = Path(server.final_model_dir())
                        self.assertFalse((target / "target_group_members.csv").exists())
                        self.assertTrue((Path(server.report_dir()) / "target_group_members.csv").is_file())
                        self.assertTrue((Path(server.report_dir()) / Path(args.save_file_paths[-1]).name).is_file())
                        self.assertTrue((target / "Server_model.pt").is_file())
                        final = load_item("Server", "model", server.save_folder_name)
                        self.assertEqual(final.fc3.weight.shape, (100, 512))
                        self.assertFalse(any(n.endswith("weight_v") for n, p in final.named_parameters()))
                for index in (1, 2):
                    self.assertEqual(first_round_rng[0][0], first_round_rng[index][0])
                    import numpy as np
                    np.testing.assert_array_equal(first_round_rng[0][1][1], first_round_rng[index][1][1])
                    torch.testing.assert_close(first_round_rng[0][2], first_round_rng[index][2], atol=0, rtol=0)
                    for left, right in zip(first_round_posts[0], first_round_posts[index]):
                        for name in left:
                            torch.testing.assert_close(left[name], right[name], atol=0, rtol=0)
            finally:
                os.chdir(previous)

    def test_private_download_failure_preserves_checkpoint_modes_buffers_and_rng(self):
        from test_meta_projection import SmallMetaNet
        with tempfile.TemporaryDirectory() as directory:
            model = SmallMetaNet(.6)
            model.register_buffer("state_probe", torch.tensor([5.]))
            model.train(); model.head.eval()
            server = SmallMetaNet(1.)
            client = SimpleNamespace(_load_model=lambda role=None: model if role is None else server,
                load_test_data=lambda: [(torch.randn(2, 6), torch.tensor([0, 1]))],
                _move_batch=lambda x, y: (x, y))
            checkpoint = Path(directory) / "ordinary.pt"
            torch.save(model, checkpoint)
            checksum = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
            state = copy.deepcopy(model.state_dict())
            before = rng_state()
            downloaded_group_accuracy(client)
            assert_rng(self, before)
            self.assertEqual(hashlib.sha256(checkpoint.read_bytes()).hexdigest(), checksum)
            self.assertTrue(model.training)
            self.assertFalse(model.head.training)
            for name, value in state.items():
                torch.testing.assert_close(model.state_dict()[name], value, atol=0, rtol=0)
            client.load_test_data = lambda: (_ for _ in ()).throw(RuntimeError("test read failed"))
            before = rng_state()
            with self.assertRaises(RuntimeError):
                downloaded_group_accuracy(client)
            assert_rng(self, before)
            self.assertEqual(hashlib.sha256(checkpoint.read_bytes()).hexdigest(), checksum)


if __name__ == "__main__":
    unittest.main()

"""Actual heterogeneous CNN: holdout before snapshots, final-only tests and frozen run."""

from contextlib import redirect_stdout
import copy
import hashlib
import io
import json
import os
from pathlib import Path
import random
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import h5py
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "system"))
from flcore.servers.serverTargetProj import FedTargetProj
from utils.meta_data import create_split, save_split
from utils.meta_snapshot import capture_snapshot, load_snapshot_manifest


class MetaCNNRuntimeTests(unittest.TestCase):
    def test_capture_split_snapshot_rng_and_frozen_full_run(self):
        torch.set_num_threads(2)
        generator = torch.Generator().manual_seed(33)
        data = [(torch.randn(3, 32, 32, generator=generator), torch.tensor(i % 2)) for i in range(40)]
        record = create_split(data, dataset="Synthetic", subdir="pat_2")
        factory = "Hyper_CNN_512(in_features=3,num_classes=2,n_kernels=16,ratio_LR={rank},input_size=32)"
        previous = os.getcwd()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            split_path = root / "split.json"
            save_split(split_path, record)
            args = SimpleNamespace(device="cpu", dataset="Synthetic", num_classes=2, global_rounds=1,
                local_epochs=5, batch_size=16, local_learning_rate=.005, num_clients=2, join_ratio=1.,
                random_join_ratio=False, few_shot=0, algorithm="FedTargetProj", time_select=False,
                goal="smoke", time_threthold=1e9, top_cnt=10, auto_break=False, eval_gap=1,
                client_drop_rate=0., train_slow_rate=0., send_slow_rate=0., resume=False,
                model_family="Decom_CNN-5-512", models_folder_name="", niid=1, partition="pat",
                class_per_client=2, exp_name="meta_smoke", save_file_paths=[],
                models=[factory.format(rank=.9), factory.format(rank=.15)], global_model=factory.format(rank=.15),
                target_client_id=0, seed=0, is_regular=1, regular_lamda=.001,
                target_proj_mode="projection", meta_c0_split=str(split_path), meta_collect_snapshots=True,
                meta_snapshot_rounds="1,2", meta_snapshot_dir=str(root / "snapshots"), meta_weight_file="",
                save_folder_name=str(root / "baseline"), final_model_root=str(root / "final"), h5_result_root=str(root / "h5"))
            try:
                os.chdir(directory)
                def reader(dataset, cid, args, is_train=True, **kwargs):
                    return data
                with redirect_stdout(io.StringIO()), patch("flcore.clients.clientbase.read_client_data", side_effect=reader), \
                        patch("flcore.servers.serverbase.read_client_data", side_effect=reader):
                    server = FedTargetProj(args, 0)
                    target = server.clients[0]
                    self.assertEqual(target.train_samples, 32)
                    self.assertEqual(len(target._meta_c0_train_data), 32)
                    self.assertEqual(len(server.clients[1].load_train_data().dataset), 40)
                    ordinary_test = target.test_post_local
                    count = []
                    def final_only_test():
                        self.assertEqual([c.train_time_cost["num_rounds"] for c in server.clients], [2, 2])
                        count.append(server.cur_ground)
                        return ordinary_test()
                    target.test_post_local = final_only_test
                    def capture_and_verify(s):
                        paths = [Path(s.save_folder_name) / f"Client_{cid}_model.pt" for cid in range(2)]
                        hashes = [hashlib.sha256(path.read_bytes()).hexdigest() for path in paths]
                        rng = random.getstate(), np.random.get_state(), torch.get_rng_state()
                        capture_snapshot(s)
                        self.assertEqual([hashlib.sha256(path.read_bytes()).hexdigest() for path in paths], hashes)
                        self.assertEqual(random.getstate(), rng[0])
                        np.testing.assert_array_equal(np.random.get_state()[1], rng[1][1])
                        torch.testing.assert_close(torch.get_rng_state(), rng[2], rtol=0, atol=0)
                    with patch("utils.meta_snapshot.capture_snapshot", side_effect=capture_and_verify):
                        server.train()
                    self.assertEqual(len(count), 2)
                    manifest = load_snapshot_manifest(root / "snapshots")
                    self.assertEqual([row["round"] for row in manifest["snapshots"]], [1, 2])
                    self.assertEqual(manifest["split"]["split_id"], record["split_id"])
                    with h5py.File(args.save_file_paths[-1]) as result:
                        self.assertEqual(len(result["target_projection"]["target_post_local_acc"]), 2)
                        self.assertEqual(result["target_projection"].attrs["last10_target_local_count"], 2)
                    artifact = dict(schema=1, kind="meta_projection_fixed", client_ids=[0, 1], weights=[.7, .3],
                        config=manifest["config"], capacities=manifest["capacities"], split=record,
                        selection=dict(test_used=False, criterion="synthetic_validation_fixture"))
                    path = root / "weights.json"
                    path.write_text(json.dumps(artifact), encoding="utf-8")
                    frozen_args = copy.deepcopy(args)
                    frozen_args.save_folder_name = str(root / "frozen")
                    frozen_args.meta_weight_file = str(path)
                    frozen_args.target_proj_mode = "meta_projection_fixed"
                    frozen_args.meta_collect_snapshots = False
                    frozen_args.save_file_paths = []
                    frozen = FedTargetProj(frozen_args, 0)
                    before = dict(frozen.meta_fixed_weights)
                    frozen._generate_dwa_guidance = lambda: self.fail("No DWA in frozen meta mode")
                    with patch("utils.meta_virtual.adapted_validation_loss", side_effect=AssertionError("No virtual training in frozen mode")):
                        frozen.train()
                    self.assertEqual(frozen.meta_fixed_weights, before)
                    self.assertEqual([c.train_time_cost["num_rounds"] for c in frozen.clients], [2, 2])
                    self.assertTrue(all(row["meta_weights_frozen"] == 1 for row in frozen.target_proj_history))
                    self.assertEqual(frozen._local_accuracy_summary()["last10_target_local_count"], 2)
                    self.assertTrue((Path(frozen.final_model_dir()) / "meta_fixed_weights.json").is_file())
                    # Dataset split stays identical when the training seed changes.
                    frozen_args.seed = 3
                    frozen_args.save_folder_name = str(root / "seed3")
                    other_seed = FedTargetProj(frozen_args, 0)
                    self.assertEqual(other_seed.meta_split_record, record)
                    self.assertEqual(other_seed.meta_fixed_weights, before)
            finally:
                os.chdir(previous)


if __name__ == "__main__":
    unittest.main()

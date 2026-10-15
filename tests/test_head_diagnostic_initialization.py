"""All four configurations copy the same existing public server classifier."""

from contextlib import redirect_stdout
import io
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "system"))
from flcore.servers.serverTargetProj import FedTargetProj


class HeadInitializationTests(unittest.TestCase):
    def test_four_public_initializations_match_without_any_training(self):
        torch.set_num_threads(2)
        factory = "Hyper_CNN_512(in_features=3,num_classes=100,n_kernels=4,ratio_LR={rank},input_size=32)"
        data = [(torch.zeros(3, 32, 32), torch.tensor(0))]
        before = os.getcwd()
        hashes = []
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "s").mkdir()
            for split in ("train", "test"):
                folder = root / "dataset/Synthetic/pat_20" / split
                folder.mkdir(parents=True)
                np.savez_compressed(folder / "0.npz", data=dict(x=np.zeros((1, 3, 32, 32), dtype=np.float32), y=np.zeros(1, dtype=np.int64)))
            try:
                os.chdir(root / "s")
                for mode in ("avg", "projection_softmax"):
                    for head in ("shared", "local"):
                        tag = f"head_diag_{mode}_{head}_local_head"
                        args = SimpleNamespace(device="cpu", dataset="Synthetic", num_classes=100, global_rounds=30,
                            local_epochs=5, batch_size=16, local_learning_rate=.005, num_clients=20, join_ratio=1.,
                            random_join_ratio=False, few_shot=0, algorithm="FedTargetProj", time_select=False,
                            goal="smoke", time_threthold=1e9, top_cnt=10, auto_break=False, eval_gap=1,
                            client_drop_rate=0., train_slow_rate=0., send_slow_rate=0., resume=False,
                            model_family="Decom_CNN-5-512", models_folder_name="", niid=1, partition="pat",
                            class_per_client=20, exp_name=tag, save_file_paths=[],
                            models=[factory.format(rank=r) for r in (.9, .37, .35, .25, .15)],
                            global_model=factory.format(rank=.15), target_client_id=0, seed=0,
                            is_regular=1, regular_lamda=.001, target_proj_mode=mode, target_prox_mu=0.,
                            projection_self_weight=.125 if mode == "projection_softmax" else None,
                            target_head_mode=head, head_diag_collect=True, head_diag_rounds="10,20,30",
                            head_diag_dir=str(root / "snap"), save_folder_name=str(root / "c" / tag),
                            training_log_root=str(root / "l" / tag), training_log_dir=str(root / "l" / tag))
                        with redirect_stdout(io.StringIO()), patch("flcore.clients.clientbase.read_client_data", return_value=data), \
                                patch("flcore.servers.serverbase.read_client_data", return_value=data):
                            server = FedTargetProj(args, 0)
                            hashes.append(server.head_diag_collector.public_head_sha256)
                            self.assertEqual([c.train_time_cost["num_rounds"] for c in server.clients], [0] * 20)
                            self.assertEqual(server.head_diag_collector.rounds, (10, 20, 30))
                self.assertEqual(len(set(hashes)), 1)
            finally:
                os.chdir(before)


if __name__ == "__main__":
    unittest.main()

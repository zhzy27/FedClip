"""Run separately in the training environment: actual CNN, disk I/O and H5 export."""

from contextlib import redirect_stdout
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
from flcore.clients.clientbase import load_item
from flcore.servers.serverTargetProj import FedTargetProj, PRE_LOCAL_MODES, LAYER_GROUP_MODES, LAYER_MODES
from utils.projection_variants import LAYER_PROJECTION_MODES, PROJECTION_WEIGHTING_MODES
from utils.dwa_aggregation import DWA_MODES
from utils.dwa_adaptive_aggregation import DWA_ADAPTIVE_MODES

DWA_TEST_MODES = (*DWA_MODES, *DWA_ADAPTIVE_MODES)


class LayerMaskCNNRuntimeTests(unittest.TestCase):
    mode = "layer_mask"

    def test_heterogeneous_cnn_training_snapshots_and_export(self):
        torch.set_num_threads(2)
        generator = torch.Generator().manual_seed(29)
        data = [(torch.randn(3, 32, 32, generator=generator), torch.tensor(cid % 2))
                for cid in range(2)]
        factory = "Hyper_CNN_512(in_features=3, num_classes=2, n_kernels=16, ratio_LR={rank}, input_size=32)"
        previous_cwd = os.getcwd()
        with tempfile.TemporaryDirectory() as directory:
            args = SimpleNamespace(
                device="cpu", dataset="Synthetic", num_classes=2, global_rounds=1,
                local_epochs=1, batch_size=2, local_learning_rate=0.005, num_clients=2,
                join_ratio=1.0, random_join_ratio=False, few_shot=0, algorithm="FedTargetProj",
                time_select=False, goal="smoke", time_threthold=1e9, top_cnt=10, auto_break=False,
                save_folder_name=str(Path(directory) / "checkpoints"), eval_gap=1, client_drop_rate=0.0,
                train_slow_rate=0.0, send_slow_rate=0.0, resume=False,
                final_model_root=str(Path(directory) / "final"), h5_result_root=str(Path(directory) / "h5"),
                model_family="Decom_CNN-5-512", models_folder_name="", niid=1, partition="pat",
                class_per_client=2, exp_name="layer_mask_smoke", save_file_paths=[],
                models=[factory.format(rank=0.9), factory.format(rank=0.15)],
                global_model=factory.format(rank=0.15), target_client_id=0,
                target_proj_mode=self.mode, seed=0, is_regular=1, regular_lamda=1e-3,
            )
            output = io.StringIO()
            if self.mode == "apa_logit" or self.mode in DWA_TEST_MODES:
                args.num_clients = 3
                args.models.append(factory.format(rank=0.5))
            if self.mode in DWA_TEST_MODES:
                args.local_epochs = 5
            try:
                os.chdir(directory)
                with redirect_stdout(output), patch("flcore.servers.serverbase.read_client_data", return_value=data), \
                        patch("flcore.clients.clientbase.read_client_data", return_value=data):
                    server = FedTargetProj(args, 0)
                    if self.mode in LAYER_GROUP_MODES:
                        self.assertEqual(list(server.layer_groups), ["conv1", "conv2", "fc1", "fc2", "fc3"])
                        for layer, names in server.layer_groups.items():
                            self.assertEqual(names, [f"{layer}.weight", f"{layer}.bias"])
                    else:
                        self.assertFalse(hasattr(server, "layer_groups"))
                    # Check snapshots immediately after download, before any local training.
                    original_capture = server._capture_pre_local_parameters
                    captures = []

                    def capture_and_check():
                        original_capture()
                        shapes = []
                        for client in server.clients:
                            pre = server._load_pre_local_parameters(client.id)
                            actual = server._load_full_parameters(client.id)
                            shapes.append({name: p.shape for name, p in actual.items()})
                            for name in actual:
                                torch.testing.assert_close(pre[name], actual[name], rtol=0, atol=0)
                        self.assertEqual(shapes[0], shapes[1])
                        captures.append(server.cur_ground)

                    server._capture_pre_local_parameters = capture_and_check
                    target = server.clients[0]
                    test_post_local = target.test_post_local
                    observed = []

                    def observe_read_only():
                        checkpoint = Path(target.save_folder_name) / f"{target.role}_model.pt"
                        before = hashlib.sha256(checkpoint.read_bytes()).digest()
                        result = test_post_local()
                        self.assertEqual(hashlib.sha256(checkpoint.read_bytes()).digest(), before)
                        observed.append(result[0] / result[1])
                        return result

                    target.test_post_local = observe_read_only
                    guidance_observations = []
                    if self.mode in DWA_TEST_MODES:
                        original_guidance = target.build_dwa_guidance
                        def observe_guidance(current_round, post_local_round):
                            checkpoint = Path(target.save_folder_name) / f"{target.role}_model.pt"
                            before = hashlib.sha256(checkpoint.read_bytes()).digest()
                            rng = random.getstate(), np.random.get_state(), torch.get_rng_state()
                            payload = original_guidance(current_round, post_local_round)
                            self.assertEqual(hashlib.sha256(checkpoint.read_bytes()).digest(), before)
                            self.assertEqual(random.getstate(), rng[0])
                            np.testing.assert_array_equal(np.random.get_state()[1], rng[1][1])
                            torch.testing.assert_close(torch.get_rng_state(), rng[2], rtol=0, atol=0)
                            self.assertEqual(payload["loop_round"], current_round)
                            guidance_observations.append(payload["metadata"])
                            return payload
                        target.build_dwa_guidance = observe_guidance
                    server.train()
                    self.assertEqual(captures, [0, 1] if self.mode in PRE_LOCAL_MODES else [])
                    self.assertEqual(len(observed), 2)
                    self.assertEqual([row["target_post_local_acc"] for row in server.target_proj_history], observed)
                    for client in server.clients:
                        self.assertEqual(client.train_time_cost["num_rounds"], 2)
                    local = [load_item(client.role, "model", server.save_folder_name) for client in server.clients]
                    self.assertNotEqual(local[0].fc1.weight_u.shape, local[1].fc1.weight_u.shape)
                    history_file = f"{self.mode}_history.json" if self.mode in ("apa", "apa_logit", *DWA_TEST_MODES) else f"{self.mode}_matrices.json"
                    with open(Path(server.final_model_dir()) / history_file) as stream:
                        history = json.load(stream)["history"]
                    self.assertEqual(len(history), 2)
                    if self.mode in LAYER_MODES:
                        self.assertEqual(history[1]["mask_matrix"][0], [1, 1, 1, 1, 1])
                    if self.mode in DWA_TEST_MODES:
                        self.assertEqual(len(guidance_observations), 2)
                        for entry in history:
                            if self.mode in DWA_ADAPTIVE_MODES:
                                self.assertAlmostEqual(sum(client["aggregation_weight"] for client in entry["clients"]), 1.)
                                self.assertGreater(entry["helper_total_weight"], 0.)
                                self.assertEqual(entry["target_weight"], entry["clients"][0]["aggregation_weight"])
                                self.assertGreaterEqual(entry["effective_all_client_count"], 1.)
                                self.assertGreaterEqual(entry["effective_helper_count"], 1.)
                            else:
                                self.assertEqual(entry["target_weight"], .05)
                                self.assertAlmostEqual(entry["helper_total_weight"], .95)
                            self.assertEqual(entry["guidance_epochs"], 1)
                            self.assertEqual(entry["guidance_lr"], .005)
                            self.assertEqual(entry["guidance_source_post_local_round"], entry["loop_round"])
                            self.assertEqual(len(entry["clients"]), 3)
                            self.assertGreater(entry["guidance_extra_upload_bytes"], 0)
                            self.assertGreaterEqual(entry["guidance_extra_train_seconds"], 0)
                            self.assertNotIn("layer_groups", entry)
                        self.assertAlmostEqual(history[-1]["last10_target_local_acc"], sum(observed) / 2)
                    elif self.mode == "apa_logit":
                        self.assertEqual([entry["apa_logit_update_enabled"] for entry in history], [0, 1])
                        self.assertEqual(server.apa_logits.dtype, torch.float64)
                        self.assertEqual(server.apa_logits.shape, (2,))
                        self.assertTrue(all(client["apa_logit_grad"] is None for client in history[0]["clients"]))
                        for entry in history:
                            self.assertEqual(entry["target_weight"], .05)
                            self.assertAlmostEqual(entry["helper_total_weight"], .95)
                            self.assertGreater(entry["min_helper_weight"], 0.)
                            self.assertNotIn("cosine_matrix", entry)
                        for cid in range(3):
                            basis = torch.load(server._apa_logit_basis_path(0, cid), weights_only=True)
                            full = server._load_full_parameters(cid)
                            self.assertEqual(basis["loop_round"], 0)
                            self.assertEqual({key: value.shape for key, value in basis["parameters"].items()},
                                             {key: value.shape for key, value in full.items()})
                    elif self.mode == "apa":
                        self.assertEqual([entry["apa_weight_update_enabled"] for entry in history], [0, 1])
                        self.assertEqual([client["apa_weight"] for client in history[0]["clients"]], [.5, .5])
                        self.assertTrue(all(client["apa_grad"] is None for client in history[0]["clients"]))
                        for entry in history:
                            weights = torch.tensor([client["apa_weight"] for client in entry["clients"]])
                            self.assertTrue(torch.isfinite(weights).all())
                            self.assertTrue(((weights >= 0) & (weights <= 1)).all())
                            self.assertAlmostEqual(weights.sum().item(), 1.)
                            self.assertNotIn("cosine_matrix", entry)
                        for cid in range(2):
                            basis = torch.load(server._apa_basis_path(0, cid), weights_only=True)
                            full = server._load_full_parameters(cid)
                            self.assertEqual(basis["loop_round"], 0)
                            self.assertEqual({key: value.shape for key, value in basis["parameters"].items()},
                                             {key: value.shape for key, value in full.items()})
                    elif self.mode in (*PROJECTION_WEIGHTING_MODES, "softmax_only"):
                        for entry in history:
                            self.assertEqual(entry["projection_scope"], "none" if self.mode == "softmax_only" else "full_model")
                            self.assertEqual(entry["delta_scope"], "global")
                            self.assertEqual(len(entry["clients"]), 2)
                            self.assertEqual(entry["target_weight"], .5)
                            self.assertAlmostEqual(entry["helper_total_weight"], .5)
                            self.assertNotIn("layer_groups", entry)
                            if self.mode == "softmax_only":
                                self.assertIn("mean_helper_weight", entry)
                                self.assertIn("std_helper_weight", entry)
                                self.assertEqual(entry["removed_update_ratio"], 0.)
                                self.assertTrue(all(client["projection_coefficient"] == 0. for client in entry["clients"]))
                        if self.mode in ("projection_softmax", "softmax_only"):
                            self.assertEqual(history[1]["temperature"], .2)
                        else:
                            self.assertIn("relu_fallback_used", history[1])
                    else:
                        self.assertEqual(len(history[1]["cosine_matrix"][1]), 1 if self.mode == "projection_local" else 5)
                    if self.mode in LAYER_PROJECTION_MODES:
                        self.assertNotIn("mask_matrix", history[1])
                        self.assertEqual(history[1]["delta_scope"], "local" if self.mode.endswith("local") else "global")
                    if self.mode in ("layer_softmax", "layer_relu"):
                        for entry in history:
                            self.assertEqual(entry["final_weight_matrix"][0], ["anchor"] * 5)
                            self.assertNotIn("budget_layers", entry)
                            self.assertIn("target_client_test_acc", entry)
                            for column, mass in enumerate(entry["original_weight_mass"]):
                                self.assertAlmostEqual(sum(row[column] for row in entry["helper_weight_matrix"]), mass)
                            for row in entry["weight_summary"]:
                                self.assertAlmostEqual(row["final_weight_mass"], row["original_weight_mass"])
                                self.assertIn(row["effective_helper_count"], (0.0, 1.0))
                        for suffix in ("weights.csv", "layers.csv"):
                            self.assertTrue((Path(server.final_model_dir()) / f"{self.mode}_{suffix}").is_file())
                        self.assertFalse((Path(server.final_model_dir()) / "layer_mask_matrices.json").exists())
                    if self.mode == "layer_mask_budget":
                        for entry in history:
                            self.assertEqual(entry["budget_beta"], 1.0)
                            self.assertEqual(len(entry["budget_layers"]), 5)
                            for layer in entry["budget_layers"]:
                                self.assertLessEqual(layer["clipped_helper_norm"], layer["target_local_norm"] + 1e-7)
                                self.assertGreaterEqual(layer["budget_scale"], 0.0)
                                self.assertLessEqual(layer["budget_scale"], 1.0)
                        self.assertTrue((Path(server.final_model_dir()) / "layer_mask_budget_layers.csv").is_file())
                        self.assertFalse((Path(server.final_model_dir()) / "layer_mask_matrices.json").exists())
                    with h5py.File(args.save_file_paths[0]) as result:
                        if self.mode in DWA_TEST_MODES:
                            group = result["target_projection"]
                            for name in ("guidance_squared_distance", "dwa_q", "aggregation_weight"):
                                self.assertEqual(group[name].shape, (2, 3))
                            self.assertTrue(torch.isnan(torch.tensor(group["dwa_q"][:, 0])).all())
                            self.assertTrue(torch.isfinite(torch.tensor(group["dwa_q"][:, 1:])).all())
                            if self.mode in DWA_ADAPTIVE_MODES:
                                self.assertEqual(group["all_client_q"].shape, (2, 3))
                                self.assertTrue(torch.isfinite(torch.tensor(group["guidance_squared_distance"][:, 0])).all())
                                self.assertEqual(group.attrs["target_weight_count"], 2)
                                self.assertIn("target_weight_gt_0_95_count", group.attrs)
                            self.assertAlmostEqual(group.attrs["last10_target_local_acc"], sum(observed) / 2)
                            self.assertEqual(group["guidance_loss_rule"].asstr()[0], "CrossEntropy + regular_lamda * frobenius_decay")
                            if self.mode in ("dwa_soft_projection", "dwa_adaptive_self_projection"):
                                self.assertIn("removed_update_ratio", group)
                        elif self.mode == "apa_logit":
                            group = result["target_projection"]
                            for name in ("apa_logit", "apa_q", "apa_weight", "apa_raw_grad", "apa_centered_grad", "apa_logit_grad"):
                                self.assertEqual(group[name].shape, (2, 3))
                            self.assertTrue(torch.isnan(torch.tensor(group["apa_logit_grad"][0])).all())
                            self.assertTrue(torch.isfinite(torch.tensor(group["apa_logit_grad"][1, 1:])).all())
                            self.assertTrue(torch.isnan(torch.tensor(group["apa_logit_grad"][:, 0])).all())
                            self.assertEqual(group.attrs["apa_gradient_scope"], "helper_logits_before_update")
                        elif self.mode == "apa":
                            group = result["target_projection"]
                            self.assertIn("apa_proxy_loss", group)
                            self.assertEqual(group.attrs["apa_basis_scope"], "previous_completed_aggregation_uploads")
                            for name in ("apa_weight", "apa_grad", "apa_velocity"):
                                self.assertEqual(group[name].shape, (2, 2))
                            self.assertTrue(torch.isnan(torch.tensor(group["apa_grad"][0])).all())
                            self.assertTrue(torch.isfinite(torch.tensor(group["apa_grad"][1])).all())
                        elif self.mode in LAYER_MODES:
                            self.assertIn("masked_update_ratio", result["target_projection"])
                        else:
                            self.assertIn("removed_update_ratio", result["target_projection"])
                        self.assertEqual(list(result["target_projection"]["target_post_local_acc"][:]), observed)
                        self.assertEqual(result["target_projection"].attrs["final_target_local_acc"], observed[-1])
                        self.assertEqual(result["target_projection"].attrs["best_target_local_acc"], max(observed))
                        self.assertEqual(result["target_projection"].attrs["mode"], self.mode)
                        if self.mode == "softmax_only":
                            self.assertEqual(result["target_projection"].attrs["projection_enabled"], 0)
                        if self.mode in (*PROJECTION_WEIGHTING_MODES, "softmax_only"):
                            self.assertEqual(result["target_projection"].attrs["weighting_scope"],
                                             "helper_similarity_only_before_projection")
                            if self.mode in ("projection_softmax", "softmax_only"):
                                self.assertEqual(result["target_projection"].attrs["temperature"], .2)
                            else:
                                self.assertEqual(len(result["target_projection"]["relu_fallback_used"]), 2)
                        if self.mode in ("layer_softmax", "layer_relu"):
                            self.assertNotIn("budget_beta", result["target_projection"].attrs)
                            self.assertIn("mean_effective_helper_count", result["target_projection"])
                            if self.mode == "layer_softmax":
                                self.assertEqual(result["target_projection"].attrs["softmax_tau"], 0.2)
                        if self.mode == "layer_mask_budget":
                            self.assertEqual(result["target_projection"].attrs["budget_beta"], 1.0)
                            self.assertIn("clipped_helper_norm", result["target_projection"])
                    suffixes = ["metrics.csv", "weights.csv" if self.mode in ("apa", "apa_logit", *DWA_TEST_MODES) else "clients.csv"]
                    suffixes += ["cosines.csv"] if self.mode in LAYER_MODES else []
                    suffixes += ["layers.csv", "layer_summary.csv"] if self.mode in LAYER_PROJECTION_MODES else []
                    for filename in (f"{self.mode}_{suffix}" for suffix in suffixes):
                        self.assertTrue((Path(server.final_model_dir()) / filename).is_file())
                if self.mode in LAYER_GROUP_MODES:
                    self.assertIn("Recovered full-W layer groups", output.getvalue())
                if self.mode in DWA_TEST_MODES:
                    self.assertIn(f"[{self.mode}][Round 2]", output.getvalue())
                    self.assertIn("post-local mean accuracy:", output.getvalue())
                elif self.mode == "apa_logit":
                    self.assertIn("[APA-Logit][Round 2]", output.getvalue())
                elif self.mode == "apa":
                    self.assertIn("[APA][Round 2]", output.getvalue())
                elif self.mode in LAYER_MODES:
                    self.assertIn("[LayerMask][Round 2] Mask matrix", output.getvalue())
                elif self.mode in (*PROJECTION_WEIGHTING_MODES, "softmax_only"):
                    self.assertIn(f"[{self.mode}][Round 2] target_weight=", output.getvalue())
                    self.assertIn("aggregation_weight=", output.getvalue())
                else:
                    self.assertIn(f"[{self.mode}][Round 2] conflict matrix", output.getvalue())
                if self.mode == "layer_mask_budget":
                    self.assertIn("[LayerBudget][Round 2]", output.getvalue())
                if self.mode in ("layer_softmax", "layer_relu"):
                    label = "LayerSoftmax" if self.mode == "layer_softmax" else "LayerReLU"
                    self.assertIn(f"[{label}][Round 2] Aggregation weight matrix", output.getvalue())
            finally:
                os.chdir(previous_cwd)


class LayerMaskBudgetCNNRuntimeTests(LayerMaskCNNRuntimeTests):
    mode = "layer_mask_budget"


class LayerSoftmaxCNNRuntimeTests(LayerMaskCNNRuntimeTests):
    mode = "layer_softmax"


class LayerReLUCNNRuntimeTests(LayerMaskCNNRuntimeTests):
    mode = "layer_relu"


class ProjectionLocalCNNRuntimeTests(LayerMaskCNNRuntimeTests):
    mode = "projection_local"


class LayerProjectionGlobalCNNRuntimeTests(LayerMaskCNNRuntimeTests):
    mode = "layer_projection_global"


class LayerProjectionLocalCNNRuntimeTests(LayerMaskCNNRuntimeTests):
    mode = "layer_projection_local"


class ProjectionSoftmaxCNNRuntimeTests(LayerMaskCNNRuntimeTests):
    mode = "projection_softmax"


class ProjectionReLUCNNRuntimeTests(LayerMaskCNNRuntimeTests):
    mode = "projection_relu"


class SoftmaxOnlyCNNRuntimeTests(LayerMaskCNNRuntimeTests):
    mode = "softmax_only"


class APACNNRuntimeTests(LayerMaskCNNRuntimeTests):
    mode = "apa"


class APALogitCNNRuntimeTests(LayerMaskCNNRuntimeTests):
    mode = "apa_logit"


class DWASoftCNNRuntimeTests(LayerMaskCNNRuntimeTests):
    mode = "dwa_soft"


class DWASoftProjectionCNNRuntimeTests(LayerMaskCNNRuntimeTests):
    mode = "dwa_soft_projection"


class DWAAdaptiveSelfCNNRuntimeTests(LayerMaskCNNRuntimeTests):
    mode = "dwa_adaptive_self"


class DWAAdaptiveSelfProjectionCNNRuntimeTests(LayerMaskCNNRuntimeTests):
    mode = "dwa_adaptive_self_projection"


if __name__ == "__main__":
    unittest.main()

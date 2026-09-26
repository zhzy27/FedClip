"""Target-centric aggregation with standalone low-rank local training."""

import csv
import json
import os
import random
import shutil
import time

import numpy as np
import torch

from flcore.clients.clientbase import load_item, save_item
from flcore.clients.clientTargetProj import clientTargetProj
from flcore.servers.serverbase import Server
from flcore.trainmodel.models import Model_Distribe
from utils.target_projection import MODES, aggregate_target_updates


class FedTargetProj(Server):
    def __init__(self, args, times):
        self.target_client_id = int(args.target_client_id)
        self.target_proj_mode = args.target_proj_mode
        if self.target_proj_mode not in MODES:
            raise ValueError(f"Unknown target projection mode: {self.target_proj_mode}")
        if not 0 <= self.target_client_id < args.num_clients:
            raise ValueError("target_client_id must be in [0, num_clients).")
        if args.join_ratio != 1.0 or args.random_join_ratio or args.client_drop_rate != 0:
            raise ValueError("FedTargetProj v1 requires full participation and no client drops.")
        if args.time_select:
            raise ValueError("FedTargetProj v1 does not support time-based client selection.")
        if getattr(args, "resume", False):
            raise ValueError("FedTargetProj v1 requires a fresh run for matched controls.")
        if args.global_rounds < 1 or not 1 <= args.eval_gap <= args.global_rounds:
            raise ValueError("Require global_rounds >= 1 and 1 <= eval_gap <= global_rounds.")
        if getattr(args, "use_asymmetric_lr", 0) or getattr(args, "use_loss_specific_u_scaling", 0):
            raise ValueError("FedTargetProj uses a single learning rate and no U-specific gradient scaling.")
        self.target_proj_history = []
        # Baseline constructors intentionally seed model initialization at zero.
        # Seed all streams before construction, and reset training streams after it.
        self.seed = int(getattr(args, "seed", 0))
        self._seed_streams()
        super().__init__(args, times)
        self.set_slow_clients()
        self.set_clients(clientTargetProj)
        global_model = Model_Distribe(args, -1, is_global=True).to(self.device)
        global_model = self._recover_if_needed(global_model).to(self.device)
        save_item(global_model, self.role, "model", self.save_folder_name)
        self.Budget = []
        self._seed_streams()
        print(
            f"[FedTargetProj] mode={self.target_proj_mode} "
            f"target_client_id={self.target_client_id} seed={self.seed} "
            f"metrics_dir={self.save_folder_name}"
        )

    def _seed_streams(self):
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)

    def train(self):
        # Preserve the repository's zero-based inclusive round convention.
        for loop_round in range(self.global_rounds + 1):
            self.cur_ground = loop_round
            start = time.perf_counter()
            self.selected_clients = self.select_clients()
            if loop_round > 0 and loop_round % self.eval_gap == 0:
                print(f"\n-------------Round number: {loop_round}-------------")
                self.evaluate(epoch=loop_round)
            self.send_parameters()
            for client in self.selected_clients:
                client.train(current_round=loop_round)
            self.receive_ids()
            self.aggregate_parameters_avg()
            self.Budget.append(time.perf_counter() - start)
            print(f"[Round {loop_round}] time cost: {self.Budget[-1]:.3f}s")
            if self.auto_break and self.check_done(acc_lss=[self.rs_test_acc], top_cnt=self.top_cnt):
                break
        print(f"Best local-model accuracy: {max(self.rs_test_acc, default=float('nan')):.6f}")
        print(f"Average time cost per round: {np.mean(self.Budget):.3f}s")
        self.save_results()
        self.save_json_file()

    @staticmethod
    def _recover_if_needed(model):
        if any(name.endswith(("conv_v", "weight_v")) for name, _ in model.named_parameters()):
            model.recover_larger_model()
        return model

    def _load_full_parameters(self, client_id):
        client = self.clients[client_id]
        model = load_item(client.role, "model", client.save_folder_name)
        if model is None:
            raise RuntimeError(f"Client_{client_id} uploaded model is missing.")
        # load_item deserializes a separate model; recovery cannot alter the upload.
        model = self._recover_if_needed(model.to(self.device)).to(self.device)
        return dict(model.named_parameters())

    @torch.no_grad()
    def aggregate_parameters_avg(self):
        if set(self.uploaded_ids) != set(range(self.num_clients)):
            raise RuntimeError("FedTargetProj requires an upload from every client.")
        if len(self.uploaded_weights) != len(self.uploaded_ids):
            raise RuntimeError("Uploaded IDs and sample weights have different lengths.")
        global_model = load_item(self.role, "model", self.save_folder_name)
        if global_model is None:
            raise RuntimeError("Server model is missing before projection aggregation.")
        global_model = self._recover_if_needed(global_model.to(self.device)).to(self.device)
        global_params = dict(global_model.named_parameters())
        target_params = self._load_full_parameters(self.target_client_id)

        def uploads():
            for client_id, weight in zip(self.uploaded_ids, self.uploaded_weights):
                parameters = (target_params if client_id == self.target_client_id
                              else self._load_full_parameters(client_id))
                yield client_id, weight, parameters

        parameters, metrics, client_rows = aggregate_target_updates(
            global_params, target_params, uploads(), self.target_client_id, self.target_proj_mode
        )
        for name, value in global_params.items():
            value.copy_(parameters[name])
        # Keep server buffers exactly as baseline Avg does: no buffer aggregation.
        save_item(global_model, self.role, "model", self.save_folder_name)
        # Release aggregation tensors before the extra target inference pass.
        del global_model, global_params, target_params, parameters

        completed_round = self.cur_ground + 1
        row = {
            "round": completed_round,
            "loop_round": self.cur_ground,
            "mode": self.target_proj_mode,
            "target_client_id": self.target_client_id,
            "seed": self.seed,
            "target_client_test_acc": None,
            **metrics,
        }
        if completed_round % self.eval_gap == 0 or self.cur_ground == self.global_rounds:
            correct, samples, _ = self.clients[self.target_client_id].test_downloaded_global()
            if samples <= 0:
                raise RuntimeError("Target client has no test samples.")
            row["target_client_test_acc"] = float(correct) / samples
        self.target_proj_history.append(row)
        self._append_csv("target_proj_metrics.csv", [row])
        self._append_csv("target_proj_clients.csv", [
            {"round": completed_round, "mode": self.target_proj_mode, **item}
            for item in client_rows
        ])
        self._save_target_metrics()
        print("[FedTargetProj] " + " ".join(
            f"{key}={value:.8g}" if isinstance(value, float) else f"{key}={value}"
            for key, value in row.items() if value is not None
        ))

    def _append_csv(self, filename, rows):
        path = os.path.join(self.save_folder_name, filename)
        write_header = not os.path.exists(path)
        with open(path, "a", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            if write_header:
                writer.writeheader()
            writer.writerows(rows)

    def _save_target_metrics(self):
        path = os.path.join(self.save_folder_name, "target_proj_metrics.json")
        with open(path, "w", encoding="utf-8") as stream:
            json.dump({
                "args": self._to_json_serializable(vars(self.args)),
                "accuracy_scope": "post_aggregation_target_download_before_local_training",
                "projection_diagnostics": "hypothetical_projection_in_all_modes",
                "round_semantics": "completed aggregations; loop_round is the zero-based index",
                "history": self.target_proj_history,
            }, stream, ensure_ascii=False, indent=2, allow_nan=False)

    def export_final_models(self):
        super().export_final_models()
        for filename in ("target_proj_metrics.csv", "target_proj_clients.csv", "target_proj_metrics.json"):
            shutil.copy2(os.path.join(self.save_folder_name, filename), self.final_model_dir())

    def save_results(self):
        previous_paths = len(getattr(self.args, "save_file_paths", []))
        super().save_results()
        import h5py

        for path in getattr(self.args, "save_file_paths", [])[previous_paths:]:
            with h5py.File(path, "a") as result:
                group = result.create_group("target_projection")
                group.attrs["mode"] = self.target_proj_mode
                group.attrs["target_client_id"] = self.target_client_id
                group.attrs["seed"] = self.seed
                group.attrs["accuracy_scope"] = "post_aggregation_target_download_before_local_training"
                group.attrs["projection_diagnostics"] = "hypothetical_projection_in_all_modes"
                for name in self.target_proj_history[0]:
                    if name == "mode":
                        continue
                    group.create_dataset(name, data=[
                        np.nan if row[name] is None else row[name]
                        for row in self.target_proj_history
                    ])

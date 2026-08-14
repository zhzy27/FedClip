import csv
import math
import os
import random
import time
from pathlib import Path

import torch

from flcore.clients.clientCLIP import clientCLIP
from flcore.clients.clientbase import load_item, save_item
from flcore.servers.serverbase import Server
from flcore.trainmodel.models import Model_Distribe


class FedCLIP(Server):
    """FedCLIP server using sample-weighted averaging in full-weight space."""

    def __init__(self, args, times):
        super().__init__(args, times)
        self.u_subspace_diag_output_dir = (
            self._resolve_u_subspace_diag_output_dir()
        )
        args.u_subspace_diag_dir_full = str(
            self.u_subspace_diag_output_dir
        )
        if bool(getattr(args, "u_subspace_diag", 0)):
            self.u_subspace_diag_output_dir.mkdir(
                parents=True, exist_ok=True
            )
            print(
                "U-subspace diagnostic directory: "
                f"{self.u_subspace_diag_output_dir}"
            )
        self.set_slow_clients()
        self.set_clients(clientCLIP)
        print(
            f"\nJoin ratio / total clients: "
            f"{self.join_ratio} / {self.num_clients}"
        )
        print("Finished creating server and clients.")

        self.Budget = []
        global_model = Model_Distribe(args, -1, is_global=True).to(self.device)
        global_model.recover_larger_model()
        save_item(global_model, self.role, "model", self.save_folder_name)

    def train(self):
        for current_round in range(self.global_rounds + 1):
            self.cur_ground = current_round
            round_start = time.time()
            self.selected_clients = self.select_clients()

            if current_round > 0 and current_round % self.eval_gap == 0:
                print(f"\n-------------Round number: {current_round}-------------")
                print("\nEvaluate heterogeneous models")
                self.evaluate(epoch=current_round)

            self.send_parameters()
            self._train_selected_clients(current_round)
            self.receive_ids()

            self._synchronize_cuda()
            aggregation_start = time.time()
            self.aggregate_parameters_avg()
            self._synchronize_cuda()
            print(
                f"[Round {current_round:03d}] aggregation time: "
                f"{time.time() - aggregation_start:.3f}s"
            )

            self.Budget.append(time.time() - round_start)
            print(
                "-" * 25,
                "time cost",
                "-" * 25,
                self.Budget[-1],
            )
            if self.auto_break and self.check_done(
                acc_lss=[self.rs_test_acc], top_cnt=self.top_cnt
            ):
                break

        if self.rs_test_acc:
            print("\nBest accuracy.")
            print(max(self.rs_test_acc))
        measured_rounds = self.Budget[1:] or self.Budget
        if measured_rounds:
            print("\nAverage time cost per round.")
            print(sum(measured_rounds) / len(measured_rounds))

        self.save_results()
        self.save_json_file()

    def _train_selected_clients(self, current_round):
        self._synchronize_cuda()
        wall_start = time.time()
        client_times = []
        subspace_stats = []
        subspace_layer_stats = []
        for client in self.selected_clients:
            train_time = client.train(current_round=current_round)
            client_times.append(float(train_time or 0.0))
            if client.last_u_subspace_stats is not None:
                subspace_stats.append(client.last_u_subspace_stats)
            subspace_layer_stats.extend(
                client.last_u_subspace_layer_stats
            )
        self._synchronize_cuda()
        print(
            f"[Round {current_round:03d}] local training: "
            f"sum={sum(client_times):.3f}s | "
            f"wall={time.time() - wall_start:.3f}s | "
            f"clients={len(client_times)}"
        )
        diagnostic_stats = [
            item
            for item in subspace_stats
            if item.get("diag_enabled", False)
        ]
        if diagnostic_stats:
            self._write_u_subspace_diagnostics(
                diagnostic_stats, subspace_layer_stats
            )
            print(
                f"[USubspaceSummary] round={current_round} "
                f"lambda={self._finite_mean(diagnostic_stats, 'lambda_sub'):.6g} "
                f"drift_mean={self._finite_mean(diagnostic_stats, 'mean_subspace_drift_norm'):.6e} "
                f"drift_max={self._finite_max(diagnostic_stats, 'max_subspace_drift_norm'):.6e} "
                f"angle_mean={self._finite_mean(diagnostic_stats, 'mean_principal_angle_deg'):.3f}deg "
                f"angle_max={self._finite_max(diagnostic_stats, 'max_principal_angle_deg'):.3f}deg "
                f"grad_ratio={self._finite_mean(diagnostic_stats, 'u_sub_to_base_grad_ratio'):.6e} "
                f"grad_cos={self._finite_mean(diagnostic_stats, 'u_base_sub_grad_cos'):.6f} "
                f"R_U={self._finite_mean(diagnostic_stats, 'mean_R_U'):.6e} "
                f"R_V={self._finite_mean(diagnostic_stats, 'mean_R_V'):.6e} "
                f"clients={len(diagnostic_stats)}"
            )
        elif subspace_stats:
            mean_loss = sum(
                item["mean_loss"] for item in subspace_stats
            ) / len(subspace_stats)
            mean_drift = sum(
                item["drift_norm"] for item in subspace_stats
            ) / len(subspace_stats)
            print(
                f"[USubspaceRegSummary] round={current_round} "
                f"mean_loss={mean_loss:.6e} "
                f"mean_drift_norm={mean_drift:.6e} "
                f"clients={len(subspace_stats)}"
            )

    @staticmethod
    def _finite_values(rows, key):
        values = []
        for row in rows:
            try:
                value = float(row[key])
            except (KeyError, TypeError, ValueError):
                continue
            if math.isfinite(value):
                values.append(value)
        return values

    @classmethod
    def _finite_mean(cls, rows, key):
        values = cls._finite_values(rows, key)
        return sum(values) / len(values) if values else float("nan")

    @classmethod
    def _finite_max(cls, rows, key):
        values = cls._finite_values(rows, key)
        return max(values) if values else float("nan")

    @staticmethod
    def _append_csv_rows(path, fieldnames, rows):
        if not rows:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        needs_header = not path.exists() or path.stat().st_size == 0
        with path.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle, fieldnames=fieldnames, extrasaction="ignore"
            )
            if needs_header:
                writer.writeheader()
            for row in rows:
                writer.writerow(
                    {field: row.get(field, "") for field in fieldnames}
                )

    def _resolve_u_subspace_diag_output_dir(self):
        args = getattr(self, "args", None)
        configured_root = str(
            getattr(args, "u_subspace_diag_dir", "") or ""
        ).strip()
        if configured_root:
            dataset = str(
                getattr(self, "dataset", "unknown_dataset")
            )
            algorithm = str(
                getattr(self, "algorithm", "unknown_algorithm")
            )
            run_id = str(
                getattr(
                    self,
                    "run_id",
                    Path(self.save_folder_name).name,
                )
            )
            return (
                Path(configured_root).expanduser()
                / dataset
                / algorithm
                / run_id
            )

        launcher_log_dir = os.environ.get(
            "FEDCLIP_TRAIN_LOG_DIR", ""
        ).strip()
        if launcher_log_dir:
            return Path(launcher_log_dir).expanduser()
        return Path(self.save_folder_name)

    def _write_u_subspace_diagnostics(self, summaries, layer_stats):
        output_dir = getattr(
            self,
            "u_subspace_diag_output_dir",
            self._resolve_u_subspace_diag_output_dir(),
        )
        summary_fields = [
            "round",
            "client_id",
            "u_lr",
            "v_lr",
            "lambda_sub",
            "mean_subspace_drift_norm",
            "max_subspace_drift_norm",
            "mean_principal_angle_deg",
            "max_principal_angle_deg",
            "mean_R_U",
            "mean_R_V",
            "u_base_grad_norm",
            "u_sub_grad_norm",
            "u_sub_weighted_grad_norm",
            "u_sub_to_base_grad_ratio",
            "u_base_sub_grad_cos",
            "mean_u_sigma_min",
            "mean_u_condition_number",
        ]
        layer_fields = [
            "round",
            "client_id",
            "layer_name",
            "rank",
            "u_lr",
            "lambda_sub",
            "subspace_drift_sq",
            "subspace_drift_norm",
            "principal_angle_mean_deg",
            "principal_angle_max_deg",
            "principal_angle_median_deg",
            "R_U",
            "R_V",
            "u_sigma_min",
            "u_sigma_max",
            "u_condition_number",
        ]
        self._append_csv_rows(
            output_dir / "u_subspace_round_summary.csv",
            summary_fields,
            summaries,
        )
        self._append_csv_rows(
            output_dir / "u_subspace_layer_stats.csv",
            layer_fields,
            layer_stats,
        )

    def _synchronize_cuda(self):
        if torch.cuda.is_available() and str(self.device).startswith("cuda"):
            torch.cuda.synchronize(self.device)

    def receive_ids(self):
        if not self.selected_clients:
            raise RuntimeError("No clients were selected for aggregation.")
        num_active = int(
            (1 - self.client_drop_rate) * self.current_num_join_clients
        )
        if num_active <= 0:
            raise RuntimeError("No active clients remain after client dropout.")
        active_clients = random.sample(self.selected_clients, num_active)

        total_samples = sum(client.train_samples for client in active_clients)
        if total_samples <= 0:
            raise RuntimeError("Active clients contain no training samples.")
        self.uploaded_ids = [client.id for client in active_clients]
        self.uploaded_weights = [
            client.train_samples / total_samples for client in active_clients
        ]

    def send_parameters(self):
        if not self.selected_clients:
            raise RuntimeError("No clients were selected for model dispatch.")
        for client in self.selected_clients:
            start_time = time.time()
            client.set_parameters()
            client.send_time_cost["num_rounds"] += 1
            client.send_time_cost["total_cost"] += 2 * (
                time.time() - start_time
            )

    @staticmethod
    def _has_low_rank_params(model):
        return any(
            name.endswith("conv_v") or name.endswith("weight_v")
            for name, _ in model.named_parameters()
        )

    def _recover_if_needed(self, model):
        if self._has_low_rank_params(model):
            model.recover_larger_model()
        # Some recovery implementations create replacement layers on CPU.
        # Move the reconstructed model again before full-weight aggregation.
        return model.to(self.device)

    def aggregate_parameters_avg(self):
        if not self.uploaded_ids:
            raise RuntimeError("No client model was uploaded for Avg.")

        global_model = load_item(self.role, "model", self.save_folder_name)
        if global_model is None:
            raise RuntimeError("Server global model is missing before Avg.")
        global_model = self._recover_if_needed(global_model.to(self.device))
        global_params = dict(global_model.named_parameters())
        expected_names = list(global_params)

        with torch.no_grad():
            for global_param in global_params.values():
                global_param.zero_()

        for client_id, weight in zip(
            self.uploaded_ids, self.uploaded_weights
        ):
            client = self.clients[client_id]
            client_model = load_item(
                client.role, "model", client.save_folder_name
            )
            if client_model is None:
                raise RuntimeError(
                    f"Client_{client_id} uploaded model is missing."
                )
            full_model = self._recover_if_needed(
                client_model.to(self.device)
            )
            source_params = dict(full_model.named_parameters())
            if list(source_params) != expected_names:
                raise RuntimeError(
                    f"Client_{client_id} full model is "
                    "incompatible with the server model."
                )
            with torch.no_grad():
                for name, global_param in global_params.items():
                    source_param = source_params[name]
                    if source_param.shape != global_param.shape:
                        raise RuntimeError(
                            f"Avg shape mismatch for {name}: "
                            f"server={tuple(global_param.shape)}, "
                            f"client={tuple(source_param.shape)}"
                        )
                    source_param = source_param.to(
                        device=global_param.device,
                        dtype=global_param.dtype,
                    )
                    global_param.add_(source_param, alpha=weight)

        save_item(global_model, self.role, "model", self.save_folder_name)
        print(f"Avg weights: {self.uploaded_weights}")

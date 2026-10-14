"""Independent two-stage C0 training and full-parameter server feedback."""

import copy
import json
import math
import os
from pathlib import Path
import shutil
import time

import h5py
import torch

from flcore.clients.clientbase import load_item, save_item
from flcore.servers.serverbase import Server
from utils.feedback_aggregation import aggregate_feedback_updates
from utils.feedback_options import FEEDBACK_SELF_WEIGHT
from utils.group_projection import preserved_group_rng
from utils.target_projection import _delta, _dot
from utils.training_logs import mirror_result


def feedback_accuracy_summary(history):
    if not history:
        return {}
    result = {}
    for label, key in (("stage1", "target_stage1_post_local_acc"),
                       ("intermediate_download", "target_intermediate_download_acc"),
                       ("stage2", "target_stage2_post_local_acc")):
        best = max(history, key=lambda row: row[key])  # Earliest equal best.
        result[label] = dict(final=history[-1][key], best=best[key], best_round=best["round"],
            last10=math.fsum(row[key] for row in history[-10:]) / len(history[-10:]),
            last10_count=min(10, len(history)),
            last20=math.fsum(row[key] for row in history[-20:]) / len(history[-20:]),
            last20_count=min(20, len(history)))
    return result


class TargetFeedbackRuntime:
    def _feedback_sync(self):
        if str(self.device).startswith("cuda"):
            torch.cuda.synchronize(self.device)

    def _feedback_slot(self):
        return Path(self.save_folder_name) / "feedback_stage1" / f"slot_{self.cur_ground % 2}"

    def _feedback_train_client(self, client, stage):
        self._feedback_sync()
        start = time.perf_counter()
        client.train(current_round=self.cur_ground)
        self._feedback_sync()
        wall = time.perf_counter() - start
        stats = copy.deepcopy(client.last_normal_train_stats)
        if stats["local_epochs"] != self.local_epochs:
            raise RuntimeError("Feedback must run all five ordinary epochs in each stage.")
        row = dict(round=self.cur_ground + 1, loop_round=self.cur_ground, mode=self.target_proj_mode,
            seed=self.seed, stage=stage, client_id=client.id, local_epochs=stats["local_epochs"],
            train_batches=stats["train_batches"], processed_train_samples=stats["processed_train_samples"],
            train_samples=client.train_samples, ce_loss=stats["ce_loss"],
            weighted_frobenius_loss=stats["regularization_loss"], regular_lamda=self.args.regular_lamda,
            frobenius_penalty=stats["regularization_loss"] / self.args.regular_lamda
                if self.args.is_regular == 1 and self.args.regular_lamda else 0.,
            optimizer_train_seconds=stats["train_seconds"], train_call_seconds=wall,
            learning_rate=client.learning_rate, proximal_loss=0., grad_clip_norm=10.)
        print("[FeedbackTrain] " + " ".join(f"{key}={value}" for key, value in row.items()))
        return row

    def _feedback_accuracy(self):
        # Observe the *actual* current checkpoint; no second hypothetical download.
        correct, samples, _ = self.clients[0].test_post_local()
        if samples <= 0 or not 0 <= correct <= samples:
            raise ValueError("Feedback target needs valid test counts.")
        return float(correct) / samples

    @torch.no_grad()
    def _feedback_cache_first_stage(self, training):
        self._feedback_stage1_round = None
        if len(self.uploaded_ids) != self.num_clients or set(self.uploaded_ids) != set(range(self.num_clients)):
            raise ValueError("Feedback requires every ordinary first-stage upload exactly once.")
        if len(self.uploaded_weights) != self.num_clients:
            raise ValueError("Feedback upload IDs and sample fractions differ.")
        self._feedback_sync()
        start = time.perf_counter()
        slot = self._feedback_slot()
        slot.mkdir(parents=True, exist_ok=True)
        with preserved_group_rng():
            # Recovery is an extra observation; only real download/train consumes RNG.
            for cid in self.uploaded_ids:
                full = self._load_full_parameters(cid)
                _dot(full, full)  # Reject nonfinite ordinary parameters before persisting.
                torch.save(dict(client_id=cid, loop_round=self.cur_ground, stage="ordinary_stage1",
                    parameters={name: p.detach().cpu().clone() for name, p in full.items()}),
                    slot / f"Client_{cid}_full.pt")
                del full
            shutil.copy2(Path(self.save_folder_name) / "Client_0_model.pt", slot / "Client_0_low_rank.pt")
        metadata = dict(loop_round=self.cur_ground, round=self.cur_ground + 1,
            upload_ids=list(self.uploaded_ids), sample_weights=list(self.uploaded_weights),
            training=copy.deepcopy(training), retention="two_round_slots_not_overwritten_by_stage2")
        (slot / "metadata.json").write_text(json.dumps(metadata, indent=2, allow_nan=False), encoding="utf-8")
        self._feedback_stage1_round = self.cur_ground
        self._feedback_sync()
        return time.perf_counter() - start

    def _feedback_load_first_stage(self, cid):
        if self._feedback_stage1_round != self.cur_ground:
            raise RuntimeError("Feedback needs this round's completed first-stage cache.")
        saved = torch.load(self._feedback_slot() / f"Client_{cid}_full.pt", map_location="cpu", weights_only=True)
        if (saved["loop_round"] != self.cur_ground or saved["client_id"] != cid
                or saved["stage"] != "ordinary_stage1"):
            raise RuntimeError("Stale or incompatible first-stage feedback upload.")
        return saved["parameters"]

    @torch.no_grad()
    def _feedback_aggregate_first_stage(self):
        self._feedback_sync()
        start = time.perf_counter()
        with preserved_group_rng():
            model = load_item(self.role, "model", self.save_folder_name)
            if model is None:
                raise RuntimeError("Missing feedback round's shared starting server.")
            model = self._recover_if_needed(model.to(self.device)).to(self.device)
            initial = dict(model.named_parameters())
            torch.save(dict(loop_round=self.cur_ground,
                parameters={n: p.detach().cpu().clone() for n, p in initial.items()}),
                self._feedback_slot() / "server_start.pt")
            def uploads():
                for cid, weight in zip(self.uploaded_ids, self.uploaded_weights):
                    yield cid, weight, self._feedback_load_first_stage(cid)
            result, metrics, clients = aggregate_feedback_updates(initial, self._feedback_load_first_stage(0),
                uploads(), dict(zip(self.uploaded_ids, self.uploaded_weights)), 0, self.target_proj_mode)
            for name, parameter in initial.items():
                parameter.copy_(result[name])
            save_item(model, self.role, "model", self.save_folder_name)  # H_t, including unchanged server buffers.
            torch.save(model, self._feedback_slot() / "intermediate_server.pt")
        self._feedback_sync()
        return metrics, clients, time.perf_counter() - start

    @torch.no_grad()
    def _feedback_commit_target(self):
        """Server <- recovered C0 stage2 parameters; never mix H or old uploads."""
        self._feedback_sync()
        start = time.perf_counter()
        with preserved_group_rng():
            model = load_item(self.role, "model", self.save_folder_name)
            if model is None:
                raise RuntimeError("Missing intermediate feedback server.")
            model = self._recover_if_needed(model.to(self.device)).to(self.device)
            intermediate = dict(model.named_parameters())
            full = self._load_full_parameters(0)
            change = _delta(full, intermediate)  # Names, shape and full-W scope check.
            norm = math.sqrt(_dot(change, change))
            for name, parameter in intermediate.items():
                parameter.copy_(full[name])
            # No C0/client buffer feedback. Keep the shared server's original buffers.
            save_item(model, self.role, "model", self.save_folder_name)
            torch.save(model, self._feedback_slot() / "feedback_server.pt")
        self._feedback_sync()
        return norm, time.perf_counter() - start

    def _train_target_feedback(self):
        self.feedback_history = []
        for loop_round in range(self.global_rounds + 1):
            self.cur_ground = loop_round
            self._feedback_sync()
            start = time.perf_counter()
            print(f"\n{'=' * 72}\n[Feedback][Round {loop_round + 1}][loop_round {loop_round}] {self.target_proj_mode}")
            print("[FeedbackStage1] Shared S_t download; ALL clients ordinary local training")
            self.selected_clients = self.select_clients()
            self.send_parameters()
            training = [self._feedback_train_client(client, "ordinary_stage1") for client in self.selected_clients]
            stage1_acc = self._feedback_accuracy()
            self.receive_ids()  # Preserve normal sample weights and normal upload summation order.
            capture_seconds = self._feedback_cache_first_stage(training)
            print("[FeedbackAggregation] Immutable stage1 full-W uploads -> intermediate H_t")
            metrics, clients, aggregation_seconds = self._feedback_aggregate_first_stage()
            print("[FeedbackStage2Download] ONLY C0 downloads H_t using its original SVD capacity")
            self._feedback_sync()
            download_start = time.perf_counter()
            self.clients[0].set_parameters()  # Real download: its parameter/RNG changes are retained.
            self._feedback_sync()
            download_seconds = time.perf_counter() - download_start
            intermediate_acc = self._feedback_accuracy()
            print("[FeedbackStage2Train] C0 ordinary five-epoch training FROM actual H_t download")
            stage2 = self._feedback_train_client(self.clients[0], "target_feedback_stage2")
            stage2_acc = self._feedback_accuracy()
            print("[FeedbackServer] Recovered C0 stage2 C_t -> next shared server S_(t+1); parameters only")
            stage2_delta_norm, feedback_seconds = self._feedback_commit_target()
            self._feedback_sync()
            elapsed = time.perf_counter() - start
            self.Budget.append(elapsed)
            target_stage1 = next(row for row in training if row["client_id"] == 0)
            row = dict(round=loop_round + 1, loop_round=loop_round, mode=self.target_proj_mode, seed=self.seed,
                target_client_id=0, target_stage1_post_local_acc=stage1_acc,
                target_intermediate_download_acc=intermediate_acc, target_stage2_post_local_acc=stage2_acc,
                target_stage2_finetune_gain=stage2_acc - intermediate_acc,
                target_stage1_global_delta_norm=metrics["target_delta_norm"],
                target_stage2_vs_intermediate_full_delta_norm=stage2_delta_norm,
                target_total_local_epochs=target_stage1["local_epochs"] + stage2["local_epochs"],
                helper_local_epochs=self.local_epochs, stage1_train_batches=sum(r["train_batches"] for r in training),
                stage1_sample_visits=sum(r["processed_train_samples"] for r in training),
                stage1_train_call_seconds=math.fsum(r["train_call_seconds"] for r in training),
                first_stage_capture_seconds=capture_seconds, aggregation_seconds=aggregation_seconds,
                target_stage2_download_svd_seconds=download_seconds,
                target_stage2_train_call_seconds=stage2["train_call_seconds"],
                target_recovery_feedback_seconds=feedback_seconds, round_seconds=elapsed, **metrics)
            for label, stats in (("target_stage1", target_stage1), ("target_stage2", stage2)):
                for key in ("ce_loss", "weighted_frobenius_loss", "train_batches", "processed_train_samples"):
                    row[label + "_" + key] = stats[key]
            self.feedback_history.append(dict(**row, clients=clients, training=training + [stage2],
                aggregation_order=list(self.uploaded_ids), first_stage_cache=str(self._feedback_slot())))
            self.rs_test_acc.append(stage2_acc)
            common = dict(round=loop_round + 1, loop_round=loop_round, mode=self.target_proj_mode, seed=self.seed)
            self._append_csv("feedback_metrics.csv", [row])
            self._append_csv("feedback_clients.csv", [{**common, **client} for client in clients])
            self._append_csv("feedback_training.csv", training + [stage2])
            self._save_feedback_json()
            print("[FeedbackMetrics] " + " ".join(f"{key}={value}" for key, value in row.items()))
            print(f"[FeedbackAccuracy] stage1={100 * stage1_acc:.4f}% H_download={100 * intermediate_acc:.4f}% "
                  f"stage2={100 * stage2_acc:.4f}% same-download gain={100 * (stage2_acc - intermediate_acc):+.4f}pp")
        summary = feedback_accuracy_summary(self.feedback_history)
        for stage, values in summary.items():
            print(f"[FeedbackSummary][{stage}] final={100 * values['final']:.4f}% "
                  f"best={100 * values['best']:.4f}% earliest_best_round={values['best_round']} "
                  f"last10={100 * values['last10']:.4f}% last20={100 * values['last20']:.4f}%")
        self.save_results()
        self.save_json_file()

    def _feedback_document(self):
        return dict(schema=1, mode=self.target_proj_mode, seed=self.seed, target_client_id=0,
            args=self._to_json_serializable(vars(self.args)), primary_accuracy_metric="target_stage2_post_local_acc",
            accuracy_storage_unit="fraction_0_to_1_matching_existing_metrics", console_accuracy_unit="percent",
            local_epochs_per_stage=self.local_epochs, target_epochs_per_round=2 * self.local_epochs,
            helper_epochs_per_round=self.local_epochs, fixed_self_weight=FEEDBACK_SELF_WEIGHT,
            fixed_self_weight_applies="first_stage_intermediate_aggregation_only; Avg uses sample fraction",
            feedback_rule="next_server_named_parameters = recovered_C0_stage2_named_parameters",
            buffers_rule="shared_server_buffers_preserved; each_client_keeps_its_own_buffers_on_download",
            stage2_rng_rule="real_download_and_training_advance_rng; extra_evaluation_and_recovery_preserve_rng",
            upload_retention="two_round_slots; first_stage_full_uploads_and_C0_low_rank_not_overwritten_by_stage2",
            round_semantics="round=loop_round+1; range(global_rounds+1)",
            summary=feedback_accuracy_summary(self.feedback_history), history=self.feedback_history)

    def _save_feedback_json(self):
        with open(Path(self.report_dir()) / "feedback_metrics.json", "w", encoding="utf-8") as stream:
            json.dump(self._feedback_document(), stream, indent=2, allow_nan=False)

    def _save_feedback_json_file(self):
        with open(Path(self.report_dir()) / f"{self.args.exp_name}-{self.run_id}.json", "w", encoding="utf-8") as stream:
            json.dump(self._feedback_document(), stream, indent=2, allow_nan=False)

    def _save_feedback_results(self):
        history = self.feedback_history
        if not history:
            raise ValueError("Cannot export empty feedback history.")
        folder = Path(self.h5_result_dir())
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / f"{self.target_proj_mode}_{self.run_id}.h5"
        if not hasattr(self.args, "save_file_paths"):
            self.args.save_file_paths = []
        self.args.save_file_paths.append(str(path))
        with h5py.File(path, "w") as file:
            group = file.create_group("target_feedback")
            document = self._feedback_document()
            for key in ("mode", "seed", "target_client_id", "primary_accuracy_metric", "accuracy_storage_unit",
                        "local_epochs_per_stage", "target_epochs_per_round", "helper_epochs_per_round", "buffers_rule",
                        "feedback_rule", "stage2_rng_rule", "upload_retention", "round_semantics"):
                group.attrs[key] = document[key]
            group.attrs["summary_json"] = json.dumps(document["summary"])
            group.attrs["config_json"] = json.dumps(document["args"])
            for stage, values in document["summary"].items():
                for key, value in values.items():
                    group.attrs[stage + "_" + key] = value
            for key, value in history[0].items():
                if isinstance(value, (int, float)):
                    group.create_dataset(key, data=[row[key] for row in history])
            group.create_dataset("client_ids", data=[row["client_id"] for row in history[0]["clients"]])
            for key in history[0]["clients"][0]:
                if key != "client_id":
                    group.create_dataset("client_" + key, data=[[c[key] for c in row["clients"]] for row in history])
            group.create_dataset("aggregation_order", data=[row["aggregation_order"] for row in history])
            group.create_dataset("training_client_ids", data=[
                [r["client_id"] for r in row["training"]] for row in history])
            group.create_dataset("training_stage", data=[
                [r["stage"] for r in row["training"]] for row in history], dtype=h5py.string_dtype("utf-8"))
            for key in ("ce_loss", "weighted_frobenius_loss", "frobenius_penalty", "local_epochs", "train_batches",
                        "processed_train_samples", "train_call_seconds", "optimizer_train_seconds"):
                group.create_dataset("training_" + key, data=[[r[key] for r in row["training"]] for row in history])
        mirror_result(path, self.report_dir())
        print(f"[FeedbackResults] H5={path}; metrics={self.report_dir()}")
        self.export_final_models()

    def _export_feedback_final(self):
        Server.export_final_models(self)
        path = Path(self.final_model_dir()) / "manifest.json"
        with path.open(encoding="utf-8") as stream:
            manifest = json.load(stream)
        manifest.update(mode=self.target_proj_mode, primary_accuracy_metric="target_stage2_post_local_acc",
            target_feedback_summary=feedback_accuracy_summary(self.feedback_history),
            feedback_rule="Server_model.pt = recovered_C0_stage2_parameters; original_server_buffers",
            client_checkpoint_stages="C0=stage2; helpers=stage1", target_epochs_per_round=2 * self.local_epochs,
            helper_epochs_per_round=self.local_epochs)
        path.write_text(json.dumps(self._to_json_serializable(manifest), indent=2, allow_nan=False), encoding="utf-8")


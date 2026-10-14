"""Helpers train first; C0 trains once after aggregation and becomes the next server."""

import copy
import json
import math
from pathlib import Path
import time

import h5py
import torch

from flcore.clients.clientbase import load_item, save_item
from flcore.servers.serverbase import Server
from utils.feedback_aggregation import aggregate_feedback_updates
from utils.feedback_options import feedback_anchor_weight
from utils.group_projection import preserved_group_rng
from utils.target_projection import _delta, _dot
from utils.training_logs import mirror_result


def feedback_accuracy_summary(history):
    if not history:
        return {}
    result = {}
    for label, key in (("intermediate_download", "target_intermediate_download_acc"),
                       ("target_local", "target_post_local_acc")):
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
        return Path(self.save_folder_name) / "feedback_helper_uploads" / f"slot_{self.cur_ground % 2}"

    @torch.no_grad()
    def _feedback_begin_round(self):
        """Freeze S_t before any client download, decomposition or train call."""
        self._feedback_helper_round = None
        slot = self._feedback_slot()
        slot.mkdir(parents=True, exist_ok=True)
        with preserved_group_rng():
            model = load_item(self.role, "model", self.save_folder_name)
            if model is None:
                raise RuntimeError("Missing starting feedback server.")
            model = self._recover_if_needed(model.to(self.device)).to(self.device)
            _dot(dict(model.named_parameters()), dict(model.named_parameters()))
            torch.save(model, slot / "starting_server_model.pt")
            torch.save(dict(loop_round=self.cur_ground,
                parameters={n: p.detach().cpu().clone() for n, p in model.named_parameters()}), slot / "server_start.pt")
        self._feedback_anchor_round = self.cur_ground

    def _feedback_send_helpers(self):
        """Normal parameter download, but never to C0 before helper aggregation."""
        for client in self.clients:
            if client.id == 0:
                continue
            start = time.perf_counter()
            client.set_parameters()
            client.send_time_cost["num_rounds"] += 1
            client.send_time_cost["total_cost"] += 2 * (time.perf_counter() - start)

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
    def _feedback_cache_helpers(self, training):
        self._feedback_helper_round = None
        helpers = set(range(self.num_clients)) - {0}
        if len(self.uploaded_ids) != len(helpers) or set(self.uploaded_ids) != helpers:
            raise ValueError("Feedback requires exactly the 19 helper uploads, with no C0 upload.")
        if len(self.uploaded_weights) != len(helpers):
            raise ValueError("Feedback upload IDs and sample fractions differ.")
        if len(training) != len(helpers) or {r["client_id"] for r in training} != helpers:
            raise ValueError("Helper training records must exclude C0 and cover every helper.")
        self._feedback_sync()
        start = time.perf_counter()
        slot = self._feedback_slot()
        slot.mkdir(parents=True, exist_ok=True)
        with preserved_group_rng():
            # Recovery is an extra observation; only real download/train consumes RNG.
            for cid in self.uploaded_ids:
                full = self._load_full_parameters(cid)
                _dot(full, full)  # Reject nonfinite ordinary parameters before persisting.
                torch.save(dict(client_id=cid, loop_round=self.cur_ground, stage="helper_local",
                    parameters={name: p.detach().cpu().clone() for name, p in full.items()}),
                    slot / f"Client_{cid}_full.pt")
                del full
        self._feedback_helper_counts = {cid: self.clients[cid].train_samples for cid in self.uploaded_ids}
        metadata = dict(loop_round=self.cur_ground, round=self.cur_ground + 1,
            upload_ids=list(self.uploaded_ids), helper_train_counts=self._feedback_helper_counts,
            helper_sample_weights=list(self.uploaded_weights),
            training=copy.deepcopy(training), retention="two_round_helper_slots; no_C0_first_stage_upload")
        (slot / "metadata.json").write_text(json.dumps(metadata, indent=2, allow_nan=False), encoding="utf-8")
        self._feedback_helper_round = self.cur_ground
        self._feedback_sync()
        return time.perf_counter() - start

    def _feedback_load_helper(self, cid):
        if cid == 0:
            raise ValueError("C0 has no helper-stage upload; its old checkpoint is not a current-round upload.")
        if self._feedback_helper_round != self.cur_ground:
            raise RuntimeError("Feedback needs this round's completed helper cache.")
        saved = torch.load(self._feedback_slot() / f"Client_{cid}_full.pt", map_location="cpu", weights_only=True)
        if (saved["loop_round"] != self.cur_ground or saved["client_id"] != cid
                or saved["stage"] != "helper_local"):
            raise RuntimeError("Stale or incompatible helper feedback upload.")
        return saved["parameters"]

    @torch.no_grad()
    def _feedback_aggregate_first_stage(self):
        self._feedback_sync()
        start = time.perf_counter()
        with preserved_group_rng():
            if self._feedback_anchor_round != self.cur_ground:
                raise RuntimeError("Missing this round's immutable server anchor.")
            frozen = torch.load(self._feedback_slot() / "server_start.pt", map_location="cpu", weights_only=True)
            if frozen["loop_round"] != self.cur_ground:
                raise RuntimeError("Stale server anchor.")
            model = torch.load(self._feedback_slot() / "starting_server_model.pt", map_location=self.device, weights_only=False)
            initial = dict(model.named_parameters())
            current = load_item(self.role, "model", self.save_folder_name)
            if current is None:
                raise RuntimeError("Missing shared server before helper aggregation.")
            actual = dict(current.named_parameters())
            if actual.keys() != initial.keys() or any(not torch.equal(actual[n].cpu(), frozen["parameters"][n]) for n in initial):
                raise RuntimeError("Helper operations changed S_t before aggregation.")
            def uploads():
                for cid in self.uploaded_ids:
                    yield cid, self._feedback_helper_counts[cid], self._feedback_load_helper(cid)
            result, metrics, clients = aggregate_feedback_updates(initial, uploads(), self._feedback_helper_counts,
                0, self.target_proj_mode, getattr(self.args, "feedback_anchor_weight", None))
            for name, parameter in initial.items():
                parameter.copy_(result[name])
            save_item(model, self.role, "model", self.save_folder_name)  # H_t, including unchanged server buffers.
            torch.save(model, self._feedback_slot() / "intermediate_server.pt")
        self._feedback_sync()
        return metrics, clients, time.perf_counter() - start

    @torch.no_grad()
    def _feedback_commit_target(self):
        """Server <- recovered C0's only training result; never mix H or old uploads."""
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
        feedback_anchor_weight(self.target_proj_mode, getattr(self.args, "feedback_anchor_weight", None))
        self.feedback_history = []
        for loop_round in range(self.global_rounds + 1):
            self.cur_ground = loop_round
            self._feedback_sync()
            start = time.perf_counter()
            print(f"\n{'=' * 72}\n[Feedback][Round {loop_round + 1}][loop_round {loop_round}] {self.target_proj_mode}")
            self._feedback_begin_round()
            print("[FeedbackHelpers] ONLY helpers download frozen S_t and train ordinary local5")
            self.selected_clients = [client for client in self.select_clients() if client.id != 0]
            if len(self.selected_clients) != self.num_clients - 1:
                raise RuntimeError("Feedback needs every helper exactly once.")
            self.current_num_join_clients = len(self.selected_clients)
            download_start = time.perf_counter()
            self._feedback_send_helpers()
            self._feedback_sync()
            helper_download_seconds = time.perf_counter() - download_start
            training = [self._feedback_train_client(client, "helper_local") for client in self.selected_clients]
            self.receive_ids()  # Normal helper sample weights and random upload order, with k=19.
            capture_seconds = self._feedback_cache_helpers(training)
            print("[FeedbackAggregation] ONLY immutable helper full-W uploads, optional S_t retention -> H_t")
            metrics, clients, aggregation_seconds = self._feedback_aggregate_first_stage()
            print("[FeedbackTargetDownload] C0 downloads H_t using its original SVD capacity")
            self._feedback_sync()
            download_start = time.perf_counter()
            self.clients[0].set_parameters()  # Real download: its parameter/RNG changes are retained.
            self._feedback_sync()
            download_seconds = time.perf_counter() - download_start
            self.clients[0].send_time_cost["num_rounds"] += 1
            self.clients[0].send_time_cost["total_cost"] += 2 * download_seconds
            intermediate_acc = self._feedback_accuracy()
            print("[FeedbackTargetTrain] C0's ONLY ordinary local5 training, FROM actual H_t download")
            target_training = self._feedback_train_client(self.clients[0], "target_post_aggregation")
            target_acc = self._feedback_accuracy()
            print("[FeedbackServer] Recovered C0 C_t -> next shared server S_(t+1); parameters only")
            target_delta_norm, feedback_seconds = self._feedback_commit_target()
            self._feedback_sync()
            elapsed = time.perf_counter() - start
            self.Budget.append(elapsed)
            row = dict(round=loop_round + 1, loop_round=loop_round, mode=self.target_proj_mode, seed=self.seed,
                target_client_id=0, target_intermediate_download_acc=intermediate_acc, target_post_local_acc=target_acc,
                target_finetune_gain=target_acc - intermediate_acc,
                target_vs_intermediate_full_delta_norm=target_delta_norm,
                target_total_local_epochs=target_training["local_epochs"],
                helper_local_epochs=self.local_epochs, helper_train_batches=sum(r["train_batches"] for r in training),
                helper_sample_visits=sum(r["processed_train_samples"] for r in training),
                helper_train_call_seconds=math.fsum(r["train_call_seconds"] for r in training),
                helper_download_seconds=helper_download_seconds,
                helper_capture_seconds=capture_seconds, aggregation_seconds=aggregation_seconds,
                target_download_svd_seconds=download_seconds,
                target_train_call_seconds=target_training["train_call_seconds"],
                target_recovery_feedback_seconds=feedback_seconds, round_seconds=elapsed, **metrics)
            for key in ("ce_loss", "weighted_frobenius_loss", "train_batches", "processed_train_samples"):
                row["target_" + key] = target_training[key]
            self.feedback_history.append(dict(**row, clients=clients, training=training + [target_training],
                aggregation_order=list(self.uploaded_ids), helper_cache=str(self._feedback_slot())))
            self.rs_test_acc.append(target_acc)
            common = dict(round=loop_round + 1, loop_round=loop_round, mode=self.target_proj_mode, seed=self.seed)
            self._append_csv("feedback_metrics.csv", [row])
            self._append_csv("feedback_clients.csv", [{**common, **client} for client in clients])
            self._append_csv("feedback_training.csv", training + [target_training])
            self._save_feedback_json()
            print("[FeedbackMetrics] " + " ".join(f"{key}={value}" for key, value in row.items()))
            print(f"[FeedbackAccuracy] H_download={100 * intermediate_acc:.4f}% "
                  f"target_local5={100 * target_acc:.4f}% same-download gain={100 * (target_acc - intermediate_acc):+.4f}pp")
        summary = feedback_accuracy_summary(self.feedback_history)
        for stage, values in summary.items():
            print(f"[FeedbackSummary][{stage}] final={100 * values['final']:.4f}% "
                  f"best={100 * values['best']:.4f}% earliest_best_round={values['best_round']} "
                  f"last10={100 * values['last10']:.4f}% last20={100 * values['last20']:.4f}%")
        self.save_results()
        self.save_json_file()

    def _feedback_document(self):
        return dict(schema=2, mode=self.target_proj_mode, seed=self.seed, target_client_id=0,
            args=self._to_json_serializable(vars(self.args)), primary_accuracy_metric="target_post_local_acc",
            accuracy_storage_unit="fraction_0_to_1_matching_existing_metrics", console_accuracy_unit="percent",
            local_epochs_per_stage=self.local_epochs, target_epochs_per_round=self.local_epochs,
            helper_epochs_per_round=self.local_epochs,
            server_anchor_weight=feedback_anchor_weight(self.target_proj_mode, getattr(self.args, "feedback_anchor_weight", None)),
            server_anchor_weight_meaning="retained_fraction_of_immutable_round_start_S_t; not_current_C0_upload",
            first_round_anchor="common_initialized_server; no_target_prewarm",
            feedback_rule="next_server_named_parameters = recovered_post_aggregation_C0_named_parameters",
            buffers_rule="shared_server_buffers_preserved; each_client_keeps_its_own_buffers_on_download",
            target_rng_rule="real_download_and_training_advance_rng; extra_evaluation_and_recovery_preserve_rng",
            upload_retention="two_round_helper_slots; no_first_stage_C0_upload",
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
                        "feedback_rule", "target_rng_rule", "upload_retention", "round_semantics", "server_anchor_weight_meaning"):
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
        manifest.update(mode=self.target_proj_mode, primary_accuracy_metric="target_post_local_acc",
            target_feedback_summary=feedback_accuracy_summary(self.feedback_history),
            feedback_rule="Server_model.pt = recovered_post_aggregation_C0_parameters; original_server_buffers",
            client_checkpoint_stages="C0=post_aggregation_local5; helpers=pre_aggregation_local5", target_epochs_per_round=self.local_epochs,
            helper_epochs_per_round=self.local_epochs)
        path.write_text(json.dumps(self._to_json_serializable(manifest), indent=2, allow_nan=False), encoding="utf-8")


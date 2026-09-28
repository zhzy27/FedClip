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
from utils.layer_mask import (
    aggregate_layer_mask, logical_layer_groups, print_layer_mask_diagnostics,
)
from utils.layer_mask_budget import (
    BUDGET_BETA, aggregate_layer_mask_budget, print_layer_budget_diagnostics,
)
from utils.layer_weighting import (
    WEIGHTING_MODES, SOFTMAX_TAU, aggregate_layer_weighting, print_layer_weighting_diagnostics,
)
from utils.projection_variants import (
    LOCAL_PROJECTION_MODES, LAYER_PROJECTION_MODES, SOURCE_MODES, PROJECTION_VARIANT_MODES,
    validate_source_config, aggregate_source_projection, aggregate_projection_variant,
    print_projection_variant,
    PROJECTION_WEIGHTING_MODES, PROJECTION_SOFTMAX_TAU, aggregate_projection_weighting,
)
from utils.apa_aggregation import (
    APA_SERVER_LR, APA_MOMENTUM, APA_SELF_WEIGHT, aggregate_apa, validate_apa_options,
)
from utils.apa_logit_aggregation import APA_LOGIT_LR, aggregate_apa_logit, validate_apa_logit_lr


LAYER_MODES = ("layer_mask", "layer_mask_budget", *WEIGHTING_MODES)
PRE_LOCAL_MODES = (*LAYER_MODES, *LOCAL_PROJECTION_MODES)
LAYER_GROUP_MODES = (*LAYER_MODES, *LAYER_PROJECTION_MODES)


class FedTargetProj(Server):
    def __init__(self, args, times):
        self.target_client_id = int(args.target_client_id)
        self.target_proj_mode = args.target_proj_mode
        if self.target_proj_mode not in (*MODES, *LAYER_MODES, *PROJECTION_VARIANT_MODES, "apa", "apa_logit"):
            raise ValueError(f"Unknown target projection mode: {self.target_proj_mode}")
        validate_source_config(args)
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
        self.layer_mask_history = []
        self.projection_variant_history = []
        self._pre_local_round = None
        if self.target_proj_mode == "apa":
            validate_apa_options(getattr(args, "apa_server_lr", APA_SERVER_LR),
                                 getattr(args, "apa_momentum", APA_MOMENTUM),
                                 getattr(args, "apa_self_weight", APA_SELF_WEIGHT))
            self.apa_weights, self.apa_velocity, self.apa_basis_round = None, None, None
            self.apa_history = []
        if self.target_proj_mode == "apa_logit":
            validate_apa_logit_lr(getattr(args, "apa_logit_lr", APA_LOGIT_LR))
            self.apa_logits, self.apa_logit_basis_round = None, None
            self.apa_logit_history = []
        # Baseline constructors intentionally seed model initialization at zero.
        # Seed all streams before construction, and reset training streams after it.
        self.seed = int(getattr(args, "seed", 0))
        self._seed_streams()
        super().__init__(args, times)
        self.set_slow_clients()
        self.set_clients(clientTargetProj)
        global_model = Model_Distribe(args, -1, is_global=True).to(self.device)
        global_model = self._recover_if_needed(global_model).to(self.device)
        if self.target_proj_mode in LAYER_GROUP_MODES:
            self.layer_groups = logical_layer_groups(dict(global_model.named_parameters()))
            print("[LayerMask] Recovered full-W layer groups:")
            for layer, names in self.layer_groups.items():
                print(f"  {layer}: {', '.join(names)}")
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
            if self.target_proj_mode in PRE_LOCAL_MODES:
                self._capture_pre_local_parameters()
            for client in self.selected_clients:
                client.train(current_round=loop_round)
            correct, samples, _ = self.clients[self.target_client_id].test_post_local()
            if samples <= 0:
                raise RuntimeError("Target client has no post-local test samples.")
            self._post_local_acc = float(correct) / samples
            self._post_local_round = loop_round
            self.receive_ids()
            self.aggregate_parameters_avg()
            self.Budget.append(time.perf_counter() - start)
            print(f"[Round {loop_round}] time cost: {self.Budget[-1]:.3f}s")
            if self.auto_break and self.check_done(acc_lss=[self.rs_test_acc], top_cnt=self.top_cnt):
                break
        summary = self._local_accuracy_summary()
        print(f"Final Client {self.target_client_id} post-local accuracy: {summary['final_target_local_acc']:.6f}")
        print(f"Best Client {self.target_client_id} post-local accuracy: {summary['best_target_local_acc']:.6f}")
        print(f"Best Client {self.target_client_id} post-local round: {summary['best_target_local_round']}")
        print(f"Diagnostic all-client best mean accuracy: {max(self.rs_test_acc, default=float('nan')):.6f}")
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

    def _pre_local_path(self, client_id):
        return os.path.join(self.save_folder_name, "layer_mask_pre", f"Client_{client_id}.pt")

    @torch.no_grad()
    def _capture_pre_local_parameters(self):
        """Snapshot actual downloaded models before training, one client at a time.

        Recovery may allocate randomly initialized modules before filling weights.
        Restore RNG streams so this extra server observation cannot alter training.
        CPU snapshots on disk avoid retaining all clients' full models in GPU RAM.
        """
        self._pre_local_round = None
        os.makedirs(os.path.dirname(self._pre_local_path(0)), exist_ok=True)
        python_rng, numpy_rng = random.getstate(), np.random.get_state()
        try:
            with torch.random.fork_rng():
                for client in self.selected_clients:
                    parameters = self._load_full_parameters(client.id)
                    torch.save({name: value.detach().cpu().clone() for name, value in parameters.items()},
                               self._pre_local_path(client.id))
        finally:
            random.setstate(python_rng)
            np.random.set_state(numpy_rng)
        self._pre_local_round = self.cur_ground

    def _load_pre_local_parameters(self, client_id):
        if self._pre_local_round != self.cur_ground:
            raise RuntimeError("LayerMask requires this round's pre-local snapshots before training.")
        return torch.load(self._pre_local_path(client_id), map_location=self.device, weights_only=True)

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

        if self.target_proj_mode == "apa":
            parameters, metrics, client_rows = self._aggregate_apa(global_params, target_params, uploads())
        elif self.target_proj_mode == "apa_logit":
            parameters, metrics, client_rows = self._aggregate_apa_logit(global_params, target_params, uploads())
        elif self.target_proj_mode in LAYER_MODES:
            target_pre = self._load_pre_local_parameters(self.target_client_id)

            def masked_uploads():
                for cid, weight, post in uploads():
                    yield cid, weight, post, self._load_pre_local_parameters(cid)

            if self.target_proj_mode in WEIGHTING_MODES:
                parameters, metrics, client_rows, layer_rows, matrices = aggregate_layer_weighting(
                    global_params, target_params, target_pre, masked_uploads,
                    self.target_client_id, self.layer_groups, self.target_proj_mode,
                )
            else:
                aggregate_mask = (aggregate_layer_mask_budget if self.target_proj_mode == "layer_mask_budget"
                                  else aggregate_layer_mask)
                parameters, metrics, client_rows, layer_rows, matrices = aggregate_mask(
                    global_params, target_params, target_pre, masked_uploads(),
                    self.target_client_id, self.layer_groups,
                )
            del target_pre
        elif self.target_proj_mode in PROJECTION_VARIANT_MODES:
            if self.target_proj_mode in (*PROJECTION_WEIGHTING_MODES, "softmax_only"):
                parameters, metrics, client_rows, layer_rows, matrices = aggregate_projection_weighting(
                    global_params, target_params, uploads, self.target_client_id, self.target_proj_mode,
                )
            elif self.target_proj_mode in SOURCE_MODES:
                parameters, metrics, client_rows, layer_rows, matrices = aggregate_source_projection(
                    global_params, target_params, uploads(), self.target_client_id,
                    dict(zip(self.uploaded_ids, self.uploaded_weights)), self.target_proj_mode,
                )
            else:
                use_pre = self.target_proj_mode in LOCAL_PROJECTION_MODES
                target_pre = self._load_pre_local_parameters(self.target_client_id) if use_pre else None
                variant_uploads = ((cid, weight, post, self._load_pre_local_parameters(cid) if use_pre else None)
                                   for cid, weight, post in uploads())
                parameters, metrics, client_rows, layer_rows, matrices = aggregate_projection_variant(
                    global_params, target_params, variant_uploads, self.target_client_id, self.target_proj_mode,
                    target_pre=target_pre, groups=getattr(self, "layer_groups", None),
                )
                del target_pre
        else:
            parameters, metrics, client_rows = aggregate_target_updates(
                global_params, target_params, uploads(), self.target_client_id, self.target_proj_mode
            )
        for name, value in global_params.items():
            value.copy_(parameters[name])
        # Keep server buffers exactly as baseline Avg does: no buffer aggregation.
        save_item(global_model, self.role, "model", self.save_folder_name)
        if self.target_proj_mode == "apa":
            self.apa_weights, self.apa_velocity = self._apa_pending_state
            self.apa_basis_round = self.cur_ground
            del self._apa_pending_state
        if self.target_proj_mode == "apa_logit":
            self.apa_logits = self._apa_logit_pending_state
            self.apa_logit_basis_round = self.cur_ground
            del self._apa_logit_pending_state
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
            "target_post_local_acc": (getattr(self, "_post_local_acc", None)
                                      if getattr(self, "_post_local_round", None) == self.cur_ground else None),
            **metrics,
        }
        row.update(self._local_accuracy_summary(row))
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
        if self.target_proj_mode in LAYER_MODES:
            self._record_layer_mask(row, client_rows, layer_rows, matrices)
        elif self.target_proj_mode in PROJECTION_VARIANT_MODES:
            self._record_projection_variant(row, client_rows, layer_rows, matrices)
        elif self.target_proj_mode == "apa":
            self._record_apa(row, client_rows)
        elif self.target_proj_mode == "apa_logit":
            self._record_apa_logit(row, client_rows)
        print("[FedTargetProj] " + " ".join(
            f"{key}={value:.8g}" if isinstance(value, float) else f"{key}={value}"
            for key, value in row.items() if value is not None
        ))

    def _apa_basis_path(self, loop_round, client_id):
        # Two slots preserve the previous complete basis while the next is written.
        return os.path.join(self.save_folder_name, "apa_basis", f"slot_{loop_round % 2}", f"Client_{client_id}.pt")

    def _load_apa_bases(self):
        previous = self.cur_ground - 1
        if previous < 0 or getattr(self, "apa_basis_round", None) != previous:
            raise RuntimeError("APA requires the immediately previous aggregation's basis.")
        for cid in range(self.num_clients):
            saved = torch.load(self._apa_basis_path(previous, cid), map_location=self.device, weights_only=True)
            if saved["loop_round"] != previous or saved["client_id"] != cid:
                raise RuntimeError("APA basis checkpoint is stale or has the wrong client ID.")
            yield cid, saved["parameters"]

    def _aggregate_apa(self, global_params, target_params, uploads):
        def cache_basis(cid, parameters):
            path = self._apa_basis_path(self.cur_ground, cid)
            os.makedirs(os.path.dirname(path), exist_ok=True)
            torch.save(dict(loop_round=self.cur_ground, client_id=cid,
                            parameters={name: value.detach().cpu().clone() for name, value in parameters.items()}), path)

        parameters, metrics, rows, weights, velocity = aggregate_apa(
            global_params, target_params, uploads, self.target_client_id,
            dict(zip(self.uploaded_ids, self.uploaded_weights)), self.cur_ground,
            weights=getattr(self, "apa_weights", None), velocity=getattr(self, "apa_velocity", None),
            bases=self._load_apa_bases() if self.cur_ground > 0 else None, cache_basis=cache_basis,
            server_lr=getattr(self.args, "apa_server_lr", APA_SERVER_LR),
            momentum=getattr(self.args, "apa_momentum", APA_MOMENTUM),
            self_weight=getattr(self.args, "apa_self_weight", APA_SELF_WEIGHT))
        self._apa_pending_state = weights, velocity
        return parameters, metrics, rows

    def _record_apa(self, row, clients):
        self._append_csv("apa_metrics.csv", [row])
        self._append_csv("apa_weights.csv", [
            {"round": row["round"], "loop_round": row["loop_round"], **client} for client in clients])
        if not hasattr(self, "apa_history"):
            self.apa_history = []
        self.apa_history.append({**row, "clients": clients})
        with open(os.path.join(self.save_folder_name, "apa_history.json"), "w", encoding="utf-8") as stream:
            json.dump(dict(proxy_scope="pre_decomposition_full_W_server_vs_target_post",
                           basis_scope="previous_completed_aggregation_uploads",
                           self_weight_scope="after_clip_before_normalization",
                           summary=self._local_accuracy_summary(), history=self.apa_history),
                      stream, indent=2, allow_nan=False)
        for client in clients:
            print(f"[APA][Round {row['round']}] " + " ".join(f"{key}={value}" for key, value in client.items()))

    def _apa_logit_basis_path(self, loop_round, client_id):
        return os.path.join(self.save_folder_name, "apa_logit_basis", f"slot_{loop_round % 2}", f"Client_{client_id}.pt")

    def _load_apa_logit_bases(self):
        previous = self.cur_ground - 1
        if previous < 0 or getattr(self, "apa_logit_basis_round", None) != previous:
            raise RuntimeError("APA-Logit requires the immediately previous aggregation's basis.")
        for cid in range(self.num_clients):
            saved = torch.load(self._apa_logit_basis_path(previous, cid), map_location=self.device, weights_only=True)
            if saved["loop_round"] != previous or saved["client_id"] != cid:
                raise RuntimeError("APA-Logit basis checkpoint is stale or has the wrong client ID.")
            yield cid, saved["parameters"]

    def _aggregate_apa_logit(self, global_params, target_params, uploads):
        def cache_basis(cid, parameters):
            path = self._apa_logit_basis_path(self.cur_ground, cid)
            os.makedirs(os.path.dirname(path), exist_ok=True)
            torch.save(dict(loop_round=self.cur_ground, client_id=cid,
                            parameters={name: value.detach().cpu().clone() for name, value in parameters.items()}), path)

        parameters, metrics, rows, logits = aggregate_apa_logit(
            global_params, target_params, uploads, self.target_client_id,
            dict(zip(self.uploaded_ids, self.uploaded_weights)), self.cur_ground,
            logits=getattr(self, "apa_logits", None),
            bases=self._load_apa_logit_bases() if self.cur_ground > 0 else None, cache_basis=cache_basis,
            lr=getattr(self.args, "apa_logit_lr", APA_LOGIT_LR))
        self._apa_logit_pending_state = logits
        return parameters, metrics, rows

    def _record_apa_logit(self, row, clients):
        self._append_csv("apa_logit_metrics.csv", [row])
        self._append_csv("apa_logit_weights.csv", [
            {"round": row["round"], "loop_round": row["loop_round"], **client} for client in clients])
        if not hasattr(self, "apa_logit_history"):
            self.apa_logit_history = []
        self.apa_logit_history.append({**row, "clients": clients})
        with open(os.path.join(self.save_folder_name, "apa_logit_history.json"), "w", encoding="utf-8") as stream:
            json.dump(dict(proxy_scope="pre_decomposition_full_W_server_vs_target_post",
                           basis_scope="previous_completed_aggregation_uploads",
                           gradient_scope="helper_logits_before_update", weight_scope="after_logit_update",
                           summary=self._local_accuracy_summary(), history=self.apa_logit_history),
                      stream, indent=2, allow_nan=False)
        for client in clients:
            print(f"[APA-Logit][Round {row['round']}] " + " ".join(f"{key}={value}" for key, value in client.items()))

    def _local_accuracy_summary(self, current=None):
        history = self.target_proj_history + ([current] if current is not None else [])
        measured = [row for row in history if row.get("target_post_local_acc") is not None]
        if not measured:
            return dict(final_target_local_acc=None, best_target_local_acc=None, best_target_local_round=None)
        best = max(measured, key=lambda row: row["target_post_local_acc"])
        return dict(final_target_local_acc=measured[-1]["target_post_local_acc"],
                    best_target_local_acc=best["target_post_local_acc"], best_target_local_round=best["round"])

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
                "primary_accuracy_metric": "target_post_local_acc",
                "target_post_local_accuracy_scope": "saved_target_checkpoint_after_local_training_before_aggregation",
                "summary": self._local_accuracy_summary(),
                "projection_diagnostics": self._diagnostic_scope(),
                "round_semantics": "completed aggregations; loop_round is the zero-based index",
                "history": self.target_proj_history,
            }, stream, ensure_ascii=False, indent=2, allow_nan=False)

    def _diagnostic_scope(self):
        if self.target_proj_mode == "apa_logit":
            return "apa_logit_previous_basis_full_W_proxy"
        if self.target_proj_mode == "apa":
            return "apa_previous_basis_full_W_proxy"
        if self.target_proj_mode in PROJECTION_VARIANT_MODES:
            return self.target_proj_mode
        if self.target_proj_mode in WEIGHTING_MODES:
            return f"local_delta_{self.target_proj_mode}_mass_preserving"
        if self.target_proj_mode == "layer_mask_budget":
            return "local_delta_layer_mask_budget"
        return ("local_delta_layer_mask" if self.target_proj_mode == "layer_mask"
                else "hypothetical_projection_in_all_modes")

    def _record_projection_variant(self, row, clients, layers, matrices):
        prefix = self.target_proj_mode
        print_projection_variant(row["round"], prefix, matrices, clients)
        self._append_csv(f"{prefix}_metrics.csv", [row])
        self._append_csv(f"{prefix}_clients.csv", [{"round": row["round"], **item} for item in clients])
        if prefix in LAYER_PROJECTION_MODES:
            self._append_csv(f"{prefix}_layers.csv", [{"round": row["round"], **item} for item in layers])
            self._append_csv(f"{prefix}_layer_summary.csv", [
                {"round": row["round"], **item} for item in matrices["layer_summary"]])
        if not hasattr(self, "projection_variant_history"):
            self.projection_variant_history = []
        self.projection_variant_history.append({**row, **matrices})
        with open(os.path.join(self.save_folder_name, f"{prefix}_matrices.json"), "w", encoding="utf-8") as stream:
            json.dump({"summary": self._local_accuracy_summary(), "history": self.projection_variant_history},
                      stream, indent=2, allow_nan=False)

    def _record_layer_mask(self, row, client_rows, layer_rows, matrices):
        round_number = row["round"]
        print_layer_mask_diagnostics(round_number, matrices, client_rows, row)
        prefix = self.target_proj_mode
        if prefix == "layer_mask_budget":
            print_layer_budget_diagnostics(round_number, matrices)
            self._append_csv("layer_mask_budget_layers.csv", [
                {"round": round_number, "target_client_test_acc": row["target_client_test_acc"], **item}
                for item in matrices["budget_layers"]
            ])
        elif prefix in WEIGHTING_MODES:
            print_layer_weighting_diagnostics(round_number, matrices)
            self._append_csv(f"{prefix}_weights.csv", [
                {"round": round_number, **item} for item in layer_rows
            ])
            self._append_csv(f"{prefix}_layers.csv", [
                {"round": round_number, **item} for item in matrices["weight_summary"]
            ])
        self._append_csv(f"{prefix}_metrics.csv", [row])
        self._append_csv(f"{prefix}_clients.csv", [
            {"round": round_number, **item} for item in client_rows
        ])
        self._append_csv(f"{prefix}_cosines.csv", [
            {"round": round_number, **item} for item in layer_rows
        ])
        self.layer_mask_history.append({
            "round": round_number, "loop_round": self.cur_ground,
            "target_client_test_acc": row["target_client_test_acc"], **matrices,
            "target_post_local_acc": row["target_post_local_acc"],
            **self._local_accuracy_summary(),
        })
        with open(os.path.join(self.save_folder_name, f"{prefix}_matrices.json"),
                  "w", encoding="utf-8") as stream:
            json.dump({"history": self.layer_mask_history}, stream, indent=2, allow_nan=False)

    def export_final_models(self):
        super().export_final_models()
        filenames = ["target_proj_metrics.csv", "target_proj_clients.csv", "target_proj_metrics.json"]
        if self.target_proj_mode == "layer_mask":
            filenames += ["layer_mask_metrics.csv", "layer_mask_clients.csv",
                          "layer_mask_cosines.csv", "layer_mask_matrices.json"]
        elif self.target_proj_mode == "layer_mask_budget":
            filenames += ["layer_mask_budget_metrics.csv", "layer_mask_budget_clients.csv",
                          "layer_mask_budget_cosines.csv", "layer_mask_budget_matrices.json",
                          "layer_mask_budget_layers.csv"]
        elif self.target_proj_mode in WEIGHTING_MODES:
            filenames += [f"{self.target_proj_mode}_{suffix}" for suffix in (
                "metrics.csv", "clients.csv", "cosines.csv", "matrices.json", "weights.csv", "layers.csv",
            )]
        elif self.target_proj_mode in PROJECTION_VARIANT_MODES:
            suffixes = ["metrics.csv", "clients.csv", "matrices.json"]
            if self.target_proj_mode in LAYER_PROJECTION_MODES:
                suffixes += ["layers.csv", "layer_summary.csv"]
            filenames += [f"{self.target_proj_mode}_{suffix}" for suffix in suffixes]
        elif self.target_proj_mode == "apa":
            filenames += ["apa_metrics.csv", "apa_weights.csv", "apa_history.json"]
        elif self.target_proj_mode == "apa_logit":
            filenames += ["apa_logit_metrics.csv", "apa_logit_weights.csv", "apa_logit_history.json"]
        for filename in filenames:
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
                group.attrs["primary_accuracy_metric"] = "target_post_local_acc"
                group.attrs["target_post_local_accuracy_scope"] = "saved_target_checkpoint_after_local_training_before_aggregation"
                for key, value in self._local_accuracy_summary().items():
                    group.attrs[key] = np.nan if value is None else value
                group.attrs["projection_diagnostics"] = self._diagnostic_scope()
                if self.target_proj_mode == "apa_logit":
                    group.attrs["apa_proxy_scope"] = "pre_decomposition_full_W_server_vs_target_post"
                    group.attrs["apa_basis_scope"] = "previous_completed_aggregation_uploads"
                    group.attrs["apa_gradient_scope"] = "helper_logits_before_update"
                    group.attrs["apa_weight_scope"] = "after_logit_update"
                    for name in ("apa_logit", "apa_q", "apa_weight", "apa_raw_grad", "apa_centered_grad",
                                 "apa_logit_grad", "apa_logit_before", "apa_q_before"):
                        group.create_dataset(name, data=[
                            [np.nan if client[name] is None else client[name] for client in row["clients"]]
                            for row in self.apa_logit_history])
                if self.target_proj_mode == "apa":
                    group.attrs["apa_proxy_scope"] = "pre_decomposition_full_W_server_vs_target_post"
                    group.attrs["apa_basis_scope"] = "previous_completed_aggregation_uploads"
                    group.attrs["apa_self_weight_scope"] = "after_clip_before_normalization"
                    for name in ("apa_weight", "apa_grad", "apa_velocity"):
                        group.create_dataset(name, data=[
                            [np.nan if client[name] is None else client[name] for client in row["clients"]]
                            for row in self.apa_history])
                if self.target_proj_mode in (*PROJECTION_WEIGHTING_MODES, "softmax_only"):
                    group.attrs["weighting_scope"] = "helper_similarity_only_before_projection"
                    if self.target_proj_mode in ("projection_softmax", "softmax_only"):
                        group.attrs["temperature"] = PROJECTION_SOFTMAX_TAU
                    if self.target_proj_mode == "softmax_only":
                        group.attrs["projection_enabled"] = 0
                if self.target_proj_mode == "layer_mask_budget":
                    group.attrs["budget_beta"] = BUDGET_BETA
                elif self.target_proj_mode in WEIGHTING_MODES:
                    group.attrs["weighting_normalization"] = "preserve_positive_helper_sample_weight_mass"
                    if self.target_proj_mode == "layer_softmax":
                        group.attrs["softmax_tau"] = SOFTMAX_TAU
                for name in self.target_proj_history[0]:
                    if name == "mode":
                        continue
                    group.create_dataset(name, data=[
                        np.nan if row[name] is None else row[name]
                        for row in self.target_proj_history
                    ])

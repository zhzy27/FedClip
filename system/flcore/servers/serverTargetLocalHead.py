"""Local-classifier metadata and exports; synchronous training stays in FedTargetProj."""

import copy
import json
from pathlib import Path
import shutil

import torch

from flcore.clients.clientbase import load_item
from utils.target_head import backbone_parameters, classifier_names, preserved_head_rng
from utils.target_local_schedule import fedrep_schedule


class TargetLocalHeadRuntime:
    def _initialize_local_heads(self, global_model):
        excluded = classifier_names(global_model)
        backbone = backbone_parameters(global_model)
        self.local_head_clients_history = []
        self.local_head_policy = dict(target_head_mode="local", classifier_module="fc3",
            classifier_parameter_names=list(excluded),
            classifier_aliases=[name for name, p in global_model.named_parameters(remove_duplicate=False)
                if id(p) in {id(v) for v in global_model.fc3.parameters()}],
            classifier_shape={n: list(dict(global_model.named_parameters())[n].shape) for n in excluded},
            backbone_parameter_names=list(backbone),
            backbone_parameter_count=sum(p.numel() for p in backbone.values()),
            classifier_parameter_count=sum(p.numel() for p in global_model.fc3.parameters()),
            aggregation_parameter_scope="recovered_full_W_backbone_excluding_fc3",
            projection_parameter_scope="backbone_only_for_delta_dot_norm_cosine_softmax_projection_and_aggregation",
            classifier_initialization="one_copy_of_original_public_server_fc3_before_round_zero",
            classifier_download_rule="retain_each_client_post_local_fc3_after_initialization",
            buffers_rule="keep_local_buffers_on_download; never_aggregate_buffers; keep_original_server_buffers",
            primary_accuracy_metric="target_post_local_acc",
            primary_model="ordinary_post_local_C0_backbone_and_C0_local_fc3",
            aggregated_evaluation_model="aggregated_backbone_decomposed_at_C0_capacity_plus_current_C0_local_fc3",
            server_checkpoint_scope="backbone_only; internal_Server_model_pt_contains_unused_initial_fc3_placeholder",
            server_checkpoint_direct_inference_valid=False)
        if fedrep_schedule(self.args):
            self.fedrep_local_history = []
            self.local_head_policy.update(target_local_schedule="fedrep", head_epochs=5, backbone_epochs=5, total_data_epochs=10,
                schedule_semantics="local_head_CE5_then_backbone_CE_plus_F5_no_download_SVD_or_head_reset_between_stages",
                primary_model="C0_after_head5_then_backbone5; no_final_head_refit",
                endpoint_metric_scope="C0_observations_only; primary_is_after_backbone",
                implementation_scope="FedRep_style_training_ablation_in_heterogeneous_low_rank_framework_not_full_paper_reproduction")
            print("[FedRepSchedule] head_epochs=5 backbone_epochs=5 total_data_epochs=10 for ALL clients; -ls is backbone epochs; no final head refit")
        for client in self.clients:
            client.initialize_local_head(global_model)
        global_model._target_server_backbone_only = True
        path = Path(self.report_dir()) / "local_head_policy.json"
        path.write_text(json.dumps(self.local_head_policy, indent=2), encoding="utf-8")
        print("[LocalHead] excluded parameters=" + ", ".join(excluded))
        print("[LocalHead] aliases=" + ", ".join(self.local_head_policy["classifier_aliases"]))
        print("[LocalHead] shared backbone parameters=" + ", ".join(backbone))
        print("[LocalHead] server fc3 is an unused initialization placeholder; inference requires a client's local fc3.")

    def _record_local_head_clients(self, row, clients):
        self.local_head_clients_history.append(dict(round=row["round"], clients=clients))
        path = Path(self.report_dir()) / "local_head_clients.json"
        path.write_text(json.dumps(dict(policy=self.local_head_policy, history=self.local_head_clients_history),
                                   indent=2, allow_nan=False), encoding="utf-8")

    def _record_fedrep_local(self, client):
        stats = copy.deepcopy(client.last_fedrep_train_stats)
        if stats["loop_round"] != self.cur_ground or stats["client_id"] != client.id:
            raise RuntimeError("FedRep local statistics are stale or belong to a different client.")
        stats.update(mode=self.target_proj_mode, seed=self.seed)
        self.fedrep_local_history.append(stats)
        common = {k: stats[k] for k in ("round", "loop_round", "mode", "seed", "client_id",
                  "target_local_schedule", "head_epochs", "backbone_epochs", "total_data_epochs")}
        self._append_csv("fedrep_clients.csv", [{k: v for k, v in stats.items() if k not in ("phases", "endpoints")}])
        self._append_csv("fedrep_training.csv", [dict(**common, **stage) for stage in stats["phases"]])
        if stats["endpoints"]:
            self._append_csv("fedrep_target_stages.csv", [dict(**common, **stage) for stage in stats["endpoints"]])

    def _fedrep_target_metrics(self):
        stats = next(row for row in reversed(self.fedrep_local_history)
                     if row["client_id"] == 0 and row["loop_round"] == self.cur_ground)
        if [r["stage"] for r in stats["endpoints"]] != ["downloaded_before_head", "after_head", "after_backbone_primary"]:
            raise RuntimeError("FedRep C0 endpoint sequence is incomplete.")
        before, head, backbone = stats["endpoints"]
        if backbone["endpoint_test_accuracy"] != self._post_local_acc:
            raise RuntimeError("FedRep primary accuracy must match the ordinary saved post-local C0 checkpoint.")
        return dict(target_local_schedule="fedrep", head_epochs=5, backbone_epochs=5, total_data_epochs=10,
            target_downloaded_before_head_acc=before["endpoint_test_accuracy"],
            target_after_head_acc=head["endpoint_test_accuracy"], target_after_backbone_acc=backbone["endpoint_test_accuracy"])

    def _save_fedrep_metrics(self):
        document = dict(policy=self.local_head_policy, summary=self._local_accuracy_summary(),
            primary_accuracy_metric="target_post_local_acc_after_both_stages",
            loss_semantics="train_mean_* = sample_visit_weighted_training_means; endpoint_* = separate_CE_only_observations",
            timing_semantics="stage_train_seconds_exclude_endpoint_observations; total_local_call_seconds_includes_setup_observation_and_checkpoint_IO",
            history=self.fedrep_local_history)
        (Path(self.report_dir()) / "fedrep_training.json").write_text(json.dumps(document, indent=2, allow_nan=False), encoding="utf-8")

    def _save_local_head_h5(self, group):
        for key, value in self.local_head_policy.items():
            group.attrs[key] = json.dumps(value) if isinstance(value, (list, dict)) else value
        history = self.local_head_clients_history
        ids = sorted(c["client_id"] for c in history[0]["clients"])
        group.create_dataset("client_ids", data=ids)
        for key in ("aggregation_weight", "sample_weight", "cosine_before_projection", "conflict",
                    "projection_coefficient", "removed_component_norm", "projection_enabled"):
            group.create_dataset("client_" + key, data=[
                [next(c[key] for c in row["clients"] if c["client_id"] == cid) for cid in ids] for row in history])
        if fedrep_schedule(self.args):
            training = group.create_group("fedrep_training")
            lookup = {(r["round"], r["client_id"]): r for r in self.fedrep_local_history}
            for key in ("head_epochs", "backbone_epochs", "total_data_epochs", "head_train_batches", "backbone_train_batches",
                        "head_train_sample_visits", "backbone_train_sample_visits", "head_train_mean_ce", "backbone_train_mean_ce",
                        "backbone_train_mean_weighted_frobenius", "head_train_seconds", "backbone_train_seconds",
                        "optimizer_train_seconds", "endpoint_observation_seconds", "total_local_call_seconds"):
                training.create_dataset(key, data=[[lookup[row["round"], cid][key] for cid in ids] for row in history])
            training.attrs["target_endpoint_stages"] = json.dumps(["downloaded_before_head", "after_head", "after_backbone_primary"])
            for key in ("endpoint_train_ce", "endpoint_train_accuracy", "endpoint_test_ce", "endpoint_test_accuracy"):
                training.create_dataset("target_" + key, data=[
                    [r[key] for r in lookup[row["round"], 0]["endpoints"]] for row in history])

    @torch.no_grad()
    def _export_local_head_models(self):
        folder = Path(self.final_model_dir())
        folder.mkdir(parents=True, exist_ok=False)
        with preserved_head_rng():
            server = load_item(self.role, "model", self.save_folder_name)
            # The exported server artifact deliberately has no inference classifier.
            parameters = {n: p.detach().cpu().clone() for n, p in backbone_parameters(server).items()}
            torch.save(dict(schema=1, kind="shared_backbone_parameters_only", parameters=parameters,
                            excluded_parameter_names=list(classifier_names(server)),
                            direct_inference_valid=False), folder / "Server_backbone_only.pt")
            for client in self.clients:
                shutil.copy2(Path(client.save_folder_name) / f"{client.role}_model.pt", folder)
            target = self.clients[self.target_client_id]
            model = target._download_local_backbone(target._load_model(), server)
            torch.save(model, folder / "C0_aggregated_backbone_local_head.pt")
        manifest = dict(schema=1, mode=self.target_proj_mode, run_id=self.run_id,
            args=self._to_json_serializable(vars(self.args)), training_log_dir=self.report_dir(),
            summary=self._local_accuracy_summary(), **self.local_head_policy,
            exported_files=sorted(p.name for p in folder.glob("*.pt")),
            model_roles=dict(Server_backbone_only="recovered shared backbone parameters; no classifier; not directly inferable",
                Client_i_model="client i ordinary final post-local backbone plus its own local fc3; Client_0_model.pt is the primary endpoint",
                C0_aggregated_backbone_local_head="final aggregate downloaded at C0 capacity with final C0 post-local fc3; aggregation evaluation endpoint"))
        if fedrep_schedule(self.args):
            manifest["final_round_training"] = [r for r in self.fedrep_local_history if r["round"] == self.target_proj_history[-1]["round"]]
            manifest["model_roles"]["Client_i_model"] = "client i after head5 then backbone5; Client_0_model.pt is the primary endpoint; no subsequent head refit"
        (folder / "manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False), encoding="utf-8")
        print(f"[LocalHeadExport] backbone-only server, ordinary client models and C0 aggregate evaluation model: {folder}")

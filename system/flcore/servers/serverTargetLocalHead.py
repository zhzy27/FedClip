"""Local-classifier metadata and exports; synchronous training stays in FedTargetProj."""

import json
from pathlib import Path
import shutil

import torch

from flcore.clients.clientbase import load_item
from utils.target_head import backbone_parameters, classifier_names, preserved_head_rng


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
        (folder / "manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False), encoding="utf-8")
        print(f"[LocalHeadExport] backbone-only server, ordinary client models and C0 aggregate evaluation model: {folder}")

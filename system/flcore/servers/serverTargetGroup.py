"""Separate group observation/export flow; normal single-target flow is untouched."""

import copy
import json
import os
import shutil
import time

import h5py
import torch

from flcore.clients.clientbase import load_item, save_item
from flcore.servers.serverbase import Server
from utils.group_projection import (aggregate_group_updates, accuracy_statistics, group_accuracy_summary,
                                    preserved_group_rng)
from utils.training_logs import mirror_result


@torch.no_grad()
def downloaded_group_accuracy(client):
    """Normal SVD download into a private model; never write a local checkpoint."""
    with preserved_group_rng():
        model = copy.deepcopy(client._load_model())
        server_model = copy.deepcopy(client._load_model("Server"))
        server_model.decom_larger_model(model.ratio_LR)
        source, destination = dict(server_model.named_parameters()), dict(model.named_parameters())
        if source.keys() != destination.keys() or any(source[n].shape != p.shape for n, p in destination.items()):
            raise ValueError("Group evaluation download has incompatible low-rank parameters.")
        for name, parameter in destination.items():
            parameter.copy_(source[name])
        model.eval()
        correct, count = 0, 0
        for x, y in client.load_test_data():
            x, y = client._move_batch(x, y)
            correct += int((model(x).argmax(1) == y).sum())
            count += len(y)
        if count <= 0:
            raise ValueError("Target group member has no test data.")
        return correct, count


class TargetGroupRuntime:
    def _group_primary_key(self):
        return ("group_post_local_sample_weighted_acc" if self.target_proj_mode == "group_softmax_only"
                else "group_post_local_acc")

    def _train_target_group(self):
        self.group_history = []
        with preserved_group_rng():
            capacities = {}
            for cid in self.target_group_ids:
                model = self.clients[cid]._load_model()
                capacities[cid] = dict(low_rank_ratio=float(model.ratio_LR),
                    parameter_count=sum(p.numel() for p in model.parameters()),
                    train_samples=self.clients[cid].train_samples)
                del model
        for loop_round in range(self.global_rounds + 1):
            self.cur_ground = loop_round
            start = time.perf_counter()
            self.selected_clients = self.select_clients()
            self.send_parameters()
            for client in self.selected_clients:
                client.train(current_round=loop_round)
            members = []
            # Every mode performs the same observations, independently of scores.
            with preserved_group_rng():
                for cid in self.target_group_ids:
                    correct, samples, _ = self.clients[cid].test_post_local()
                    if samples <= 0:
                        raise ValueError("Target group member has no ordinary post-local test samples.")
                    members.append(dict(client_id=cid, post_local_acc=float(correct) / samples,
                                        test_samples=samples, **capacities[cid]))
                    if self.target_proj_mode == "group_softmax_only":
                        members[-1]["post_local_correct"] = int(correct)
            self._group_local_members = members
            self._group_local_round = loop_round
            self.receive_ids()
            self.aggregate_parameters_avg()
            self.Budget.append(time.perf_counter() - start)
            print(f"[TargetGroup][Round {loop_round + 1}] time={self.Budget[-1]:.3f}s")
        summary = group_accuracy_summary(self.group_history)
        print(f"[TargetGroup] ids={self.target_group_ids} ordinary post-local summary={summary['group_post_local']}")
        for cid, row in summary["members"].items():
            print(f"[TargetGroupMember] client={cid} ordinary post-local summary={row}")
        self.save_results()
        self.save_json_file()

    @torch.no_grad()
    def _aggregate_target_group(self):
        if set(self.uploaded_ids) != set(range(self.num_clients)) or len(self.uploaded_ids) != self.num_clients:
            raise ValueError("All clients must upload once, including in group_avg_only.")
        if len(self.uploaded_weights) != self.num_clients:
            raise ValueError("Group upload IDs and sample weights differ.")
        if getattr(self, "_group_local_round", None) != self.cur_ground:
            raise ValueError("This round's ordinary group evaluation must precede aggregation.")
        with preserved_group_rng():
            model = load_item(self.role, "model", self.save_folder_name)
            if model is None:
                raise ValueError("Missing shared group server model.")
            model = self._recover_if_needed(model.to(self.device)).to(self.device)
            initial = dict(model.named_parameters())
            def uploads():
                for cid, weight in zip(self.uploaded_ids, self.uploaded_weights):
                    yield cid, weight, self._load_full_parameters(cid)
            result, metrics, clients, matrices = aggregate_group_updates(initial, uploads,
                dict(zip(self.uploaded_ids, self.uploaded_weights)), self.target_group_ids,
                self.target_proj_mode, getattr(self.args, "target_group_weight", None))
            for name, parameter in initial.items():
                parameter.copy_(result[name])
            # The single server's buffers remain unchanged; only named parameters aggregate.
            save_item(model, self.role, "model", self.save_folder_name)
            del model, initial, result
            members = copy.deepcopy(self._group_local_members)
            for member in members:
                correct, samples = downloaded_group_accuracy(self.clients[member["client_id"]])
                if samples != member["test_samples"]:
                    raise ValueError("Group evaluation sample count changed within the round.")
                member["post_aggregation_acc"] = float(correct) / samples
                if self.target_proj_mode == "group_softmax_only":
                    member["post_aggregation_correct"] = int(correct)
        local = accuracy_statistics(members, "post_local_acc")
        aggregate = accuracy_statistics(members, "post_aggregation_acc")
        row = dict(round=self.cur_ground + 1, loop_round=self.cur_ground, seed=self.seed,
            mode=self.target_proj_mode, target_group_ids=",".join(map(str, self.target_group_ids)),
            group_post_local_acc=local["macro"], group_agg_acc=aggregate["macro"],
            group_post_local_sample_weighted_acc=local["sample_weighted"],
            group_agg_sample_weighted_acc=aggregate["sample_weighted"],
            group_worst_post_local_acc=local["worst"], group_post_local_std=local["std"],
            group_worst_agg_acc=aggregate["worst"], group_agg_std=aggregate["std"], **metrics)
        self.group_history.append({**row, "members": members, "clients": clients, **matrices})
        # Internal generic manifest compatibility; exported group datasets use explicit names.
        self.rs_test_acc.append(row[self._group_primary_key()])
        self._append_csv("target_group_metrics.csv", [row])
        common = dict(round=row["round"], loop_round=row["loop_round"], seed=self.seed, mode=self.target_proj_mode)
        self._append_csv("target_group_members.csv", [{**common, **m} for m in members])
        self._append_csv("target_group_clients.csv", [{**common, **c} for c in clients])
        self._save_group_json()
        print("[TargetGroup] " + " ".join(f"{k}={v}" for k, v in row.items()))
        for client in clients:
            print("[TargetGroupWeight] " + " ".join(f"{k}={v}" for k, v in client.items()))

    def _group_document(self):
        args = dict(vars(self.args))
        args.pop("target_client_id", None)  # This group's primary metric is not C0.
        return dict(schema=1, target_group_ids=self.target_group_ids, mode=self.target_proj_mode,
            seed=self.seed, args=self._to_json_serializable(args),
            primary_accuracy_metric=self._group_primary_key(),
            primary_average=("test_sample_weighted_average" if self.target_proj_mode == "group_softmax_only"
                             else "unweighted_client_macro_average"),
            local_scope="saved_ordinary_5epoch_checkpoints_before_aggregation",
            aggregate_scope="normal_SVD_download_of_current_aggregate_before_next_local_training",
            temporal_analysis="group_local(t) -> group_agg(t) -> group_local(t+1)",
            round_semantics="round=loop_round+1; range(global_rounds+1)",
            membership_scope="predefined_diagnostic_group_not_test_selected",
            summary=group_accuracy_summary(self.group_history), history=self.group_history)

    def _save_group_json(self):
        for name in ("target_group_metrics.json", "target_group_matrices.json"):
            with open(os.path.join(self.report_dir(), name), "w", encoding="utf-8") as stream:
                json.dump(self._group_document(), stream, indent=2, allow_nan=False)

    def _save_group_results(self):
        if not self.group_history:
            raise ValueError("Cannot export empty group history.")
        folder = self.h5_result_dir()
        os.makedirs(folder, exist_ok=True)
        path = os.path.join(folder, f"{self.target_proj_mode}_{self.run_id}.h5")
        if not hasattr(self.args, "save_file_paths"):
            self.args.save_file_paths = []
        self.args.save_file_paths.append(path)
        history = self.group_history
        with h5py.File(path, "w") as file:
            group = file.create_group("target_group")
            group.attrs["target_group_ids"] = self.target_group_ids
            group.attrs["mode"] = self.target_proj_mode
            group.attrs["seed"] = self.seed
            group.attrs["primary_accuracy_metric"] = self._group_primary_key()
            if self.target_proj_mode == "group_softmax_only":
                group.attrs["projection_enabled"] = 0
                group.attrs["primary_average"] = "test_sample_weighted_average"
            group.attrs["temporal_analysis"] = "group_local(t) -> group_agg(t) -> group_local(t+1)"
            group.attrs["zero_mass_convention"] = "effective role count=0; min/max role weights=0"
            group.attrs["summary_json"] = json.dumps(group_accuracy_summary(history))
            for key, value in group_accuracy_summary(history)["group_post_local"].items():
                group.attrs["group_post_local_" + key] = value
            group.attrs["config_json"] = json.dumps(self._group_document()["args"])
            for key, value in history[0].items():
                if isinstance(value, (int, float)):
                    group.create_dataset(key, data=[row[key] for row in history])
            group.create_dataset("member_ids", data=self.target_group_ids)
            for key in ("post_local_acc", "post_aggregation_acc", "test_samples", "low_rank_ratio", "parameter_count"):
                group.create_dataset("member_" + key, data=[[m[key] for m in row["members"]] for row in history])
            if self.target_proj_mode == "group_softmax_only":
                for key in ("post_local_correct", "post_aggregation_correct"):
                    group.create_dataset("member_" + key, data=[[m[key] for m in row["members"]] for row in history])
            for key in ("original_sample_weight", "aggregation_weight", "in_target_group", "outside_q",
                        "raw_softmax_score", "cosine_before_projection", "conflict", "raw_conflict_with_reference",
                        "projection_coefficient", "delta_norm", "removed_component_norm"):
                group.create_dataset("client_" + key, data=[[c[key] for c in row["clients"]] for row in history])
            group.create_dataset("client_ids", data=[c["client_id"] for c in history[0]["clients"]])
            group.create_dataset("group_pair_cosine_matrix", data=[row["group_pair_cosine_matrix"] for row in history])
            group.create_dataset("group_pair_conflict_matrix", data=[row["group_pair_conflict_matrix"] for row in history])
        print(f"[TargetGroup] H5={path}")
        mirror_result(path, self.report_dir())
        self.export_final_models()

    def _export_group_final(self):
        Server.export_final_models(self)
        target = self.final_model_dir()
        if os.path.abspath(self.report_dir()) == os.path.abspath(self.save_folder_name):
            for name in ("target_group_metrics.csv", "target_group_members.csv", "target_group_clients.csv",
                         "target_group_metrics.json", "target_group_matrices.json"):
                shutil.copy2(os.path.join(self.report_dir(), name), target)
        path = os.path.join(target, "manifest.json")
        with open(path, encoding="utf-8") as stream:
            manifest = json.load(stream)
        manifest.update(target_group_ids=self.target_group_ids, primary_accuracy_metric=self._group_primary_key(),
                        target_group_summary=group_accuracy_summary(self.group_history))
        manifest["args"].pop("target_client_id", None)
        with open(path, "w", encoding="utf-8") as stream:
            json.dump(self._to_json_serializable(manifest), stream, indent=2, allow_nan=False)

    def _save_group_json_file(self):
        folder = self.report_dir()
        os.makedirs(folder, exist_ok=True)
        with open(os.path.join(folder, f"{self.args.exp_name}-{self.run_id}.json"), "w", encoding="utf-8") as stream:
            json.dump(self._group_document(), stream, indent=2, allow_nan=False)

import csv
import copy
import random
import time
from flcore.clients.clientCLIP import clientCLIP
from flcore.servers.serverbase import Server
from flcore.clients.clientbase import load_item, save_item
from flcore.trainmodel.models import  Model_Distribe
import torch
from utils.get_clip_text_encoder import get_clip_class_embeddings
from utils.factor_loss_diagnostics import DIAGNOSTIC_FIELDS
from utils.agg_path_diagnostics import (
    AGG_PATH_FIELDS,
    GLOBAL_TRUNCATION_FIELDS,
    PRELOCAL_DOWNLOAD_FIELDS,
    aggregation_human_round,
    aggregation_path_consistency_rows,
    append_csv_rows,
    diagnostic_round_selected,
    global_truncation_rows,
    prelocal_source_aggregation_round,
    resolve_diagnostic_output_dir,
)
from utils.anchor_mechanism_diagnostics import (
    FEATURE_SCALE_SUMMARY_FIELDS,
    PROTOTYPE_LOCAL_DRIFT_FIELDS,
    SEMANTIC_PROTOTYPE_CLASS_SUMMARY_FIELDS,
    SEMANTIC_PROTOTYPE_CLIENT_FIELDS,
    SEMANTIC_PROTOTYPE_SUMMARY_FIELDS,
    feature_scale_summary_rows,
    prototype_class_summary_rows,
    prototype_client_rows,
    prototype_human_round,
    prototype_local_drift_rows,
    prototype_round_selected,
    prototype_summary_row,
)
from utils.feature_auxiliary_diagnostics import AUX_GRADIENT_SCALE_FIELDS
from utils.factor_continuation import (
    factor_rank_signature,
    sample_weighted_factor_average,
    validate_factor_continuation_mode,
)
import numpy as np
import os
import math

class FedCLIP(Server):
    def __init__(self, args, times):
        super().__init__(args, times)

        self.homogeneous_capacity = bool(
            getattr(args, "homogeneous_capacity", 0)
        )
        self.homogeneous_ratio = float(
            getattr(args, "homogeneous_ratio", 0.35)
        )
        self.factor_continuation = bool(
            getattr(args, "factor_continuation", 0)
        )
        validate_factor_continuation_mode(
            self.factor_continuation,
            self.homogeneous_capacity,
        )

        self.enable_agg_path_diagnostics = bool(
            getattr(args, "enable_agg_path_diagnostics", 0)
        )
        self.agg_diagnostic_output_dir = None
        if self.enable_agg_path_diagnostics:
            self.agg_diagnostic_output_dir = resolve_diagnostic_output_dir(
                getattr(args, "agg_diagnostic_output_dir", ""),
                self.save_folder_name,
            )
            os.makedirs(self.agg_diagnostic_output_dir, exist_ok=True)
            args.agg_diagnostic_output_dir_resolved = (
                self.agg_diagnostic_output_dir
            )
            print(
                "[AggPathDiag] CSV output directory: "
                f"{self.agg_diagnostic_output_dir}"
            )

        self.enable_semantic_prototype_diagnostics = bool(
            getattr(args, "enable_semantic_prototype_diagnostics", 0)
        )
        self.prototype_diagnostic_stage = getattr(
            args, "prototype_diagnostic_stage", "both"
        )
        self.prototype_diagnostic_output_dir = None
        self._prelocal_prototype_cache = {}
        if self.enable_semantic_prototype_diagnostics:
            self.prototype_diagnostic_output_dir = resolve_diagnostic_output_dir(
                getattr(args, "prototype_diagnostic_output_dir", ""),
                self.save_folder_name,
            )
            os.makedirs(self.prototype_diagnostic_output_dir, exist_ok=True)
            args.prototype_diagnostic_output_dir_resolved = (
                self.prototype_diagnostic_output_dir
            )
            print(
                "[SemanticPrototype] CSV output directory: "
                f"{self.prototype_diagnostic_output_dir}"
            )

        self.enable_aux_gradient_scale_diagnostics = bool(
            getattr(args, "enable_aux_gradient_scale_diagnostics", 0)
        )
        self.aux_gradient_diagnostic_output_dir = None
        if self.enable_aux_gradient_scale_diagnostics:
            self.aux_gradient_diagnostic_output_dir = resolve_diagnostic_output_dir(
                getattr(args, "aux_gradient_diagnostic_output_dir", ""),
                self.save_folder_name,
            )
            os.makedirs(self.aux_gradient_diagnostic_output_dir, exist_ok=True)
            args.aux_gradient_diagnostic_output_dir_resolved = (
                self.aux_gradient_diagnostic_output_dir
            )
            print(
                "[AuxGradientScale] CSV output directory: "
                f"{self.aux_gradient_diagnostic_output_dir}"
            )

        # select slow clients
        self.set_slow_clients()
        self.set_clients(clientCLIP)
        if self.homogeneous_capacity or self.factor_continuation:
            self._validate_mechanism_configuration()

        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print("Finished creating server and clients.")

        # self.load_model()
        self.Budget = []
        #创建全局base用于之后聚合
        global_model = Model_Distribe(args, -1,is_global=True).to(self.device)
        global_model.recover_larger_model()
        self.global_acc=[]
        save_item(global_model, self.role, 'model', self.save_folder_name)
        clip_text_features,clip_text_features_norm = get_clip_class_embeddings(self.dataset,model_name= "ViT-B/32",prompt_template= "a photo of {}",device = self.device)
        self.clip_text_features,self.clip_text_features_norm = clip_text_features.float(),clip_text_features_norm.float()
        

    def train(self):
        for i in range(self.global_rounds+1):
            self.cur_ground = i
            s_t = time.time()
            self.selected_clients = self.select_clients()
            # 下发就测试
            # self.send_parameters()
            if i > 0 and i % self.eval_gap == 0: 
                print(f"\n-------------Round number: {i} 聚合前-------------")
                print("\nEvaluate heterogeneous models")
                self.evaluate(epoch=i)
            self.send_parameters()
            if self._semantic_prototype_diagnostic_target(i, "prelocal"):
                prototype_round = prototype_human_round(i)
                prelocal_prototypes = self._collect_semantic_prototypes(
                    prototype_round, "prelocal"
                )
                self._record_semantic_prototypes(
                    prototype_round, "prelocal", prelocal_prototypes
                )
                if self.prototype_diagnostic_stage == "both":
                    self._prelocal_prototype_cache[prototype_round] = (
                        prelocal_prototypes
                    )
            if self._should_record_prelocal_download(i):
                self._record_prelocal_download_accuracy(i)
            # if i%self.eval_gap == 0: # 再测一次看看到底那一次又问题
            #     print(f"\n-------------Round number: {i} 聚合后-------------")
            #     print("\nEvaluate heterogeneous models")
            #     self.evaluate(epoch=i)
                # self.
            if torch.cuda.is_available() and str(self.device).startswith("cuda"):
                torch.cuda.synchronize(self.device)
            local_train_wall_start = time.time()
            client_train_times = []
            factor_update_stats = []
            ce_anchor_diagnostics = []
            aux_gradient_scale_diagnostics = []
            for client in self.selected_clients:
                client_train_time = client.train(current_round=i)
                if client_train_time is None:
                    client_train_time = getattr(client, "last_train_time_cost", 0.0)
                client_train_times.append((client.id, float(client_train_time)))
                client_factor_stats = getattr(
                    client, "last_factor_update_stats", None
                )
                if client_factor_stats is not None:
                    factor_update_stats.append(dict(client_factor_stats))
                client_diagnostics = getattr(
                    client, "last_ce_anchor_diagnostics", None
                )
                if client_diagnostics:
                    ce_anchor_diagnostics.extend(
                        dict(row) for row in client_diagnostics
                    )
                client_aux_diagnostics = getattr(
                    client, "last_aux_gradient_scale_diagnostics", None
                )
                if client_aux_diagnostics:
                    aux_gradient_scale_diagnostics.extend(
                        dict(row) for row in client_aux_diagnostics
                    )
            if torch.cuda.is_available() and str(self.device).startswith("cuda"):
                torch.cuda.synchronize(self.device)
            local_train_wall_time = time.time() - local_train_wall_start
            local_train_sum_time = sum(train_time for _, train_time in client_train_times)
            print(
                f"⏱️ [Round {i:03d}] 本地训练总耗时: "
                f"sum_client={local_train_sum_time:.3f}s | wall={local_train_wall_time:.3f}s | "
                f"clients={len(client_train_times)}"
            )
            print(
                "⏱️ [Round {:03d}] 客户端训练耗时明细: {}".format(
                    i,
                    ", ".join(
                        f"Client_{client_id}:{train_time:.3f}s"
                        for client_id, train_time in client_train_times
                    )
                )
            )
            self._record_factor_update_stats(i, factor_update_stats)
            self._record_ce_anchor_diagnostics(ce_anchor_diagnostics)
            self._record_aux_gradient_scale_diagnostics(
                aux_gradient_scale_diagnostics
            )
            if self._semantic_prototype_diagnostic_target(i, "postlocal"):
                prototype_round = prototype_human_round(i)
                postlocal_prototypes = self._collect_semantic_prototypes(
                    prototype_round, "postlocal"
                )
                self._record_semantic_prototypes(
                    prototype_round, "postlocal", postlocal_prototypes
                )
                if self.prototype_diagnostic_stage == "both":
                    prelocal_prototypes = self._prelocal_prototype_cache.pop(
                        prototype_round, None
                    )
                    if prelocal_prototypes is None:
                        raise RuntimeError(
                            "Missing matching pre-local prototypes for human "
                            f"round {prototype_round}."
                        )
                    self._record_prototype_local_drift(
                        prototype_round,
                        prelocal_prototypes,
                        postlocal_prototypes,
                    )
            

            self.receive_ids()
            rank_metadata = None
            aggregation_round = None
            if self._agg_path_diagnostic_round(i):
                aggregation_round = aggregation_human_round(i)
                rank_metadata = self._record_aggregation_path_consistency(
                    aggregation_round
                )
            if torch.cuda.is_available() and str(self.device).startswith("cuda"):
                torch.cuda.synchronize(self.device)
            aggregation_wall_start = time.time()
            if self.factor_continuation:
                self.aggregate_parameters_factor_continuation()
            else:
                self.aggregate_parameters_avg()
            if rank_metadata is not None and not self.factor_continuation:
                self._record_global_truncation_stats(
                    aggregation_round, rank_metadata
                )
            elif rank_metadata is not None:
                print(
                    "[SVDMechanism] global truncation diagnostic=N/A "
                    "for factor_continuation (clients receive averaged U/V "
                    "directly; no download-time re-SVD is performed)."
                )
            if torch.cuda.is_available() and str(self.device).startswith("cuda"):
                torch.cuda.synchronize(self.device)
            aggregation_wall_time = time.time() - aggregation_wall_start
            print(f"⏱️ [Round {i:03d}] 聚合总墙钟耗时: {aggregation_wall_time:.3f}s")
            self.Budget.append(time.time() - s_t)
            print('-'*25, 'time cost', '-'*25, self.Budget[-1])

            if self.auto_break and self.check_done(acc_lss=[self.rs_test_acc], top_cnt=self.top_cnt):
                break

        print("\nBest accuracy.")
        # self.print_(max(self.rs_test_acc), max(
        #     self.rs_train_acc), min(self.rs_train_loss))
        print(max(self.rs_test_acc))
        
        print("\nBest Global accuracy.")
        # self.print_(max(self.rs_test_acc), max(
        #     self.rs_train_acc), min(self.rs_train_loss))
        if len(self.global_acc) > 0:
            print(max(self.global_acc))
        else:
            print("未记录 Global accuracy")
        print("\nAverage time cost per round.")
        print(sum(self.Budget[1:])/len(self.Budget[1:]))

        self.save_results()
        self.save_json_file()

    def _semantic_prototype_diagnostic_target(self, loop_round, stage):
        if not self.enable_semantic_prototype_diagnostics:
            return False
        configured_stage = self.prototype_diagnostic_stage
        if configured_stage != "both" and configured_stage != stage:
            return False
        return prototype_round_selected(
            loop_round,
            getattr(
                self.args,
                "prototype_diagnostic_rounds",
                "1,5,10,20,30,40,50,60,70,80,90,100",
            ),
        )

    def _prototype_diagnostic_csv_path(self, filename):
        if not self.prototype_diagnostic_output_dir:
            raise RuntimeError(
                "Semantic prototype diagnostic output directory is unset."
            )
        return os.path.join(self.prototype_diagnostic_output_dir, filename)

    def _collect_semantic_prototypes(self, round_number, stage):
        results = {}
        for client in self.selected_clients:
            results[int(client.id)] = client.collect_semantic_prototypes(
                round_number, stage
            )
        return results

    def _record_semantic_prototypes(
        self, round_number, stage, client_results
    ):
        client_rows = prototype_client_rows(
            round_number, stage, client_results
        )
        summary = prototype_summary_row(
            round_number, stage, client_results, client_rows
        )
        class_rows = prototype_class_summary_rows(
            round_number, stage, client_results
        )
        feature_scale_rows = feature_scale_summary_rows(
            round_number, stage, client_results
        )
        append_csv_rows(
            self._prototype_diagnostic_csv_path(
                "semantic_prototype_client.csv"
            ),
            SEMANTIC_PROTOTYPE_CLIENT_FIELDS,
            client_rows,
        )
        append_csv_rows(
            self._prototype_diagnostic_csv_path(
                "semantic_prototype_summary.csv"
            ),
            SEMANTIC_PROTOTYPE_SUMMARY_FIELDS,
            [summary],
        )
        append_csv_rows(
            self._prototype_diagnostic_csv_path(
                "semantic_prototype_class_summary.csv"
            ),
            SEMANTIC_PROTOTYPE_CLASS_SUMMARY_FIELDS,
            class_rows,
        )
        append_csv_rows(
            self._prototype_diagnostic_csv_path("feature_scale_summary.csv"),
            FEATURE_SCALE_SUMMARY_FIELDS,
            feature_scale_rows,
        )
        print(
            f"[SemanticPrototype] round={round_number} stage={stage} "
            f"classes={len(client_rows)} "
            f"cross_client_cos={summary['overall_same_class_cos']:.6f} "
            f"train_anchor_cos={summary['mean_train_anchor_cos']:.6f} "
            f"true_clip_cos={summary['mean_true_clip_anchor_cos']:.6f}"
        )

    def _record_aux_gradient_scale_diagnostics(self, rows):
        if not rows:
            return
        rows.sort(key=lambda row: (int(row["round"]), int(row["client_id"])))
        csv_path = os.path.join(
            self.aux_gradient_diagnostic_output_dir,
            "aux_gradient_scale_diagnostics.csv",
        )
        append_csv_rows(csv_path, AUX_GRADIENT_SCALE_FIELDS, rows)

        def finite_mean(field):
            values = [
                float(row[field])
                for row in rows
                if math.isfinite(float(row[field]))
            ]
            return float(np.mean(values)) if values else float("nan")

        print(
            f"[AuxGradientScale] round={int(rows[0]['round'])} "
            f"mode={rows[0]['aux_loss_mode']} clients={len(rows)} "
            f"CE={finite_mean('ce_grad_norm'):.6e} "
            f"aux={finite_mean('aux_grad_norm'):.6e} "
            f"aux/CE={finite_mean('aux_to_ce_grad_ratio'):.6e} "
            f"feature_norm={finite_mean('feature_norm_mean'):.6e}+-"
            f"{finite_mean('feature_norm_std'):.6e} "
            f"target_norm={finite_mean('target_feature_norm'):.6e} "
            f"global_anchor_norm={finite_mean('global_anchor_norm'):.6e}"
        )

    def _record_prototype_local_drift(
        self, round_number, prelocal, postlocal
    ):
        rows = prototype_local_drift_rows(
            round_number, prelocal, postlocal
        )
        append_csv_rows(
            self._prototype_diagnostic_csv_path(
                "prototype_local_drift.csv"
            ),
            PROTOTYPE_LOCAL_DRIFT_FIELDS,
            rows,
        )
        overall = next(
            (
                row
                for row in rows
                if row["record_type"] == "overall_summary"
            ),
            None,
        )
        if overall is not None:
            print(
                f"[PrototypeLocalDrift] round={round_number} "
                f"drift={overall['local_proto_drift']:.6f} "
                f"samples={overall['sample_count']}"
            )

    def _record_factor_update_stats(self, current_round, client_stats):
        if not client_stats:
            return

        csv_fields = [
            'round',
            'client_id',
            'capacity',
            'factor_ranks',
            'homogeneous_capacity',
            'homogeneous_ratio',
            'factor_continuation',
            'feature_aux_loss',
            'u_lr_ratio',
            'v_lr_ratio',
            'u_lr',
            'v_lr',
            'R_U',
            'R_V',
            'R_U_over_R_V',
            'D_U',
            'D_V',
            'U_subspace_drift',
            'U_subspace_drift_norm',
            'C_U',
            'C_V',
            'C_UV',
            'C_U_over_C_V',
            'D_W',
        ]
        csv_path = os.path.join(
            self.save_folder_name, 'factor_update_stats.csv'
        )
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        write_header = not os.path.exists(csv_path)
        with open(csv_path, 'a', newline='', encoding='utf-8') as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=csv_fields)
            if write_header:
                writer.writeheader()
            writer.writerows(client_stats)

        def mean_std(field_name):
            values = np.asarray(
                [stats[field_name] for stats in client_stats],
                dtype=np.float64,
            )
            return float(values.mean()), float(values.std())

        mean_r_u, std_r_u = mean_std('R_U')
        mean_r_v, std_r_v = mean_std('R_V')
        mean_ratio, _ = mean_std('R_U_over_R_V')
        ratio_of_means = mean_r_u / (mean_r_v + 1e-12)
        mean_d_u, _ = mean_std('D_U')
        mean_d_v, _ = mean_std('D_V')
        mean_subspace, std_subspace = mean_std('U_subspace_drift')
        mean_subspace_norm, std_subspace_norm = mean_std(
            'U_subspace_drift_norm'
        )
        mean_c_u, std_c_u = mean_std('C_U')
        mean_c_v, std_c_v = mean_std('C_V')
        mean_c_ratio, _ = mean_std('C_U_over_C_V')
        ratio_mean_c_u_mean_c_v = mean_c_u / (mean_c_v + 1e-12)
        mean_c_uv, std_c_uv = mean_std('C_UV')
        mean_d_w, std_d_w = mean_std('D_W')

        print(f"[FactorUpdate] round={current_round}")
        print(
            f"lr_U={client_stats[0]['u_lr']:.6f} "
            f"lr_V={client_stats[0]['v_lr']:.6f}"
        )
        print(
            f"R_U={mean_r_u:.6e}±{std_r_u:.6e} "
            f"R_V={mean_r_v:.6e}±{std_r_v:.6e} "
            f"mean(R_U/R_V)={mean_ratio:.6e} "
            f"mean(R_U)/mean(R_V)={ratio_of_means:.6e}"
        )
        print(
            f"SubspaceDrift={mean_subspace:.6e}±{std_subspace:.6e}"
        )
        print(
            f"mean_U_subspace_drift_norm={mean_subspace_norm:.6e} "
            f"std_U_subspace_drift_norm={std_subspace_norm:.6e}"
        )
        print(
            f"C_U={mean_c_u:.6e}±{std_c_u:.6e} "
            f"C_V={mean_c_v:.6e}±{std_c_v:.6e} "
            f"mean_C_U_over_C_V={mean_c_ratio:.6e} "
            f"ratio_mean_C_U_mean_C_V={ratio_mean_c_u_mean_c_v:.6e} "
            f"C_UV={mean_c_uv:.6e}±{std_c_uv:.6e}"
        )
        print(
            f"D_U={mean_d_u:.6e} D_V={mean_d_v:.6e} "
            f"mean_D_W={mean_d_w:.6e} std_D_W={std_d_w:.6e}"
        )

    def _record_ce_anchor_diagnostics(self, diagnostic_rows):
        if not diagnostic_rows:
            return

        diagnostic_rows.sort(
            key=lambda row: (
                int(row['round']),
                int(row['client_id']),
                str(row['layer']),
            )
        )
        csv_path = os.path.join(
            self.save_folder_name, 'ce_anchor_diagnostics.csv'
        )
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        write_header = not os.path.exists(csv_path)
        with open(csv_path, 'a', newline='', encoding='utf-8') as csv_file:
            writer = csv.DictWriter(
                csv_file,
                fieldnames=DIAGNOSTIC_FIELDS,
                extrasaction='ignore',
            )
            if write_header:
                writer.writeheader()
            writer.writerows(diagnostic_rows)

        overall_rows = [
            row for row in diagnostic_rows if row['layer'] == '__overall__'
        ]
        if not overall_rows:
            return

        def finite_mean(field_name):
            values = []
            for row in overall_rows:
                try:
                    value = float(row[field_name])
                except (KeyError, TypeError, ValueError):
                    continue
                if math.isfinite(value):
                    values.append(value)
            return float(np.mean(values)) if values else float('nan')

        current_round = int(overall_rows[0]['round'])
        print(
            f"[CEAnchorDiag] round={current_round} "
            f"clients={len(overall_rows)} "
            f"u_cos={finite_mean('u_ce_anchor_cos'):.6f} "
            f"v_cos={finite_mean('v_ce_anchor_cos'):.6f} "
            f"u_ratio={finite_mean('u_weighted_anchor_to_ce_ratio'):.6e} "
            f"v_ratio={finite_mean('v_weighted_anchor_to_ce_ratio'):.6e} "
            f"wpath_u_cos={finite_mean('wpath_u_ce_anchor_cos'):.6f} "
            f"wpath_v_cos={finite_mean('wpath_v_ce_anchor_cos'):.6f}"
        )
        if bool(getattr(self.args, 'enable_virtual_step_diagnostics', 0)):
            print(
                f"[VirtualStepDiag] round={current_round} "
                f"CE->U ΔA={finite_mean('virtual_ce_to_u_delta_anchor'):.6e} "
                f"CE->V ΔA={finite_mean('virtual_ce_to_v_delta_anchor'):.6e} "
                f"A->U ΔCE={finite_mean('virtual_anchor_to_u_delta_ce'):.6e} "
                f"A->V ΔCE={finite_mean('virtual_anchor_to_v_delta_ce'):.6e}"
            )
            print(
                f"[VirtualCommonStepDiag] round={current_round} "
                f"CE->U ΔA={finite_mean('virtual_common_ce_to_u_delta_anchor'):.6e} "
                f"CE->V ΔA={finite_mean('virtual_common_ce_to_v_delta_anchor'):.6e} "
                f"A->U ΔCE={finite_mean('virtual_common_anchor_to_u_delta_ce'):.6e} "
                f"A->V ΔCE={finite_mean('virtual_common_anchor_to_v_delta_ce'):.6e}"
            )
        print(
            f"[GradClipDiag] round={current_round} "
            f"pre={finite_mean('pre_clip_total_grad_norm'):.6e} "
            f"active={finite_mean('clip_was_active'):.3f} "
            f"post={finite_mean('post_clip_total_grad_norm'):.6e} "
            f"coef={finite_mean('clip_coef'):.6f}"
        )

    def _agg_path_diagnostic_round(self, current_round):
        if not self.enable_agg_path_diagnostics:
            return False
        return diagnostic_round_selected(
            aggregation_human_round(current_round),
            getattr(
                self.args,
                "agg_diagnostic_rounds",
                "1,5,10,20,30,40,50,60,70,80,90,100",
            ),
        )

    def _should_record_prelocal_download(self, send_round):
        if not self.enable_agg_path_diagnostics:
            return False
        if int(send_round) == 0:
            return True
        return diagnostic_round_selected(
            prelocal_source_aggregation_round(send_round),
            getattr(
                self.args,
                "agg_diagnostic_rounds",
                "1,5,10,20,30,40,50,60,70,80,90,100",
            ),
        )

    def _agg_diagnostic_csv_path(self, filename):
        if not self.agg_diagnostic_output_dir:
            raise RuntimeError("Aggregation diagnostic output directory is unset.")
        return os.path.join(self.agg_diagnostic_output_dir, filename)

    def _record_prelocal_download_accuracy(self, send_round):
        rows = []
        capacity_totals = {}
        total_correct = 0
        total_samples = 0
        source_round = prelocal_source_aggregation_round(send_round)

        for client in self.selected_clients:
            correct, test_samples, _ = client.test_metrics()
            model = load_item(client.role, "model", client.save_folder_name)
            if model is None:
                raise RuntimeError(
                    f"Missing downloaded model for Client_{client.id}."
                )
            capacity = float(getattr(model, "ratio_LR", float("nan")))
            accuracy = float(correct) / max(int(test_samples), 1)
            rows.append(
                {
                    "round": int(send_round),
                    "send_round": int(send_round),
                    "source_aggregation_round": source_round,
                    "client_id": int(client.id),
                    "capacity": capacity,
                    "test_samples": int(test_samples),
                    "download_acc": accuracy,
                }
            )
            total_correct += int(correct)
            total_samples += int(test_samples)
            group = capacity_totals.setdefault(capacity, [0, 0])
            group[0] += int(correct)
            group[1] += int(test_samples)

        append_csv_rows(
            self._agg_diagnostic_csv_path("prelocal_download_acc.csv"),
            PRELOCAL_DOWNLOAD_FIELDS,
            rows,
        )
        overall = total_correct / max(total_samples, 1)
        grouped = ", ".join(
            f"capacity={capacity:g}:{correct / max(samples, 1):.6f}"
            for capacity, (correct, samples) in sorted(capacity_totals.items())
        )
        print(
            f"[PreLocalDownload] send_round={send_round} "
            f"source_aggregation_round={source_round} "
            f"sample_weighted_acc={overall:.6f} | {grouped}"
        )

    def _record_aggregation_path_consistency(self, current_round):
        client_updates = {}
        try:
            for client_id in self.uploaded_ids:
                updates = getattr(
                    self.clients[client_id], "last_agg_path_updates", None
                )
                if not updates:
                    raise RuntimeError(
                        f"Client_{client_id} did not provide aggregation-path "
                        f"updates for round {current_round}."
                    )
                client_updates[int(client_id)] = updates

            rows, rank_metadata = aggregation_path_consistency_rows(
                current_round,
                client_updates,
                self.uploaded_ids,
                self.uploaded_weights,
            )
            append_csv_rows(
                self._agg_diagnostic_csv_path("agg_path_consistency.csv"),
                AGG_PATH_FIELDS,
                rows,
            )
            overall = next(row for row in rows if row["layer"] == "__overall__")
            print(
                f"[AggPathDiag] round={current_round} "
                f"S_U={overall['S_U']:.6f} S_V={overall['S_V']:.6f} "
                f"same_rank_u_cos={overall['same_rank_u_cos']:.6f} "
                f"cross_rank_u_cos={overall['cross_rank_u_cos']:.6f} "
                f"same_rank_v_cos={overall['same_rank_v_cos']:.6f} "
                f"cross_rank_v_cos={overall['cross_rank_v_cos']:.6f}"
            )
            return rank_metadata
        finally:
            for client in self.selected_clients:
                client.last_agg_path_updates = {}

    def _record_global_truncation_stats(self, current_round, rank_metadata):
        global_model = load_item(self.role, "model", self.save_folder_name)
        if global_model is None:
            raise RuntimeError(
                "Server global model is missing after Avg aggregation."
            )
        global_model = global_model.to(self.device)
        rows = global_truncation_rows(
            current_round,
            dict(global_model.named_parameters()),
            rank_metadata,
        )
        append_csv_rows(
            self._agg_diagnostic_csv_path("global_truncation_stats.csv"),
            GLOBAL_TRUNCATION_FIELDS,
            rows,
        )
        mean_retained = float(
            np.mean([row["retained_energy"] for row in rows])
        )
        mean_error = float(
            np.mean([row["relative_truncation_error"] for row in rows])
        )
        print(
            f"[GlobalTruncation] round={current_round} "
            f"entries={len(rows)} mean_retained_energy={mean_retained:.6f} "
            f"mean_relative_error={mean_error:.6f}"
        )


    #从客户顿接受id信息和样本数信息
    def receive_ids(self):
        assert (len(self.selected_clients) > 0)

        active_clients = random.sample(
            self.selected_clients, int((1-self.client_drop_rate) * self.current_num_join_clients))

        self.uploaded_ids = []
        self.uploaded_weights = []
        tot_samples = 0
        for client in active_clients:
            tot_samples += client.train_samples
            self.uploaded_ids.append(client.id)
            self.uploaded_weights.append(client.train_samples)
        for i, w in enumerate(self.uploaded_weights):
            self.uploaded_weights[i] = w / tot_samples

    def _validate_mechanism_configuration(self):
        capacities = []
        signatures = []
        for client in self.clients:
            model = load_item(client.role, "model", client.save_folder_name)
            if model is None:
                raise RuntimeError(
                    f"Missing initialized model for Client_{client.id}."
                )
            capacities.append(float(getattr(model, "ratio_LR", float("nan"))))
            signatures.append(factor_rank_signature(model))

        if self.homogeneous_capacity:
            if any(
                not math.isclose(
                    capacity,
                    self.homogeneous_ratio,
                    rel_tol=1e-8,
                    abs_tol=1e-8,
                )
                for capacity in capacities
            ):
                raise RuntimeError(
                    "Homogeneous client construction failed: expected every "
                    f"ratio_LR={self.homogeneous_ratio}, got {capacities}."
                )

        if self.factor_continuation:
            if not signatures or not signatures[0]:
                raise RuntimeError(
                    "factor_continuation requires low-rank U/V parameters."
                )
            for client_id, signature in enumerate(signatures[1:], start=1):
                if signature != signatures[0]:
                    raise RuntimeError(
                        "factor_continuation requires identical client factor "
                        f"ranks; Client_0={signatures[0]}, "
                        f"Client_{client_id}={signature}."
                    )

        rank_summary = [
            f"{name}:{rank}" for name, rank in (signatures[0] if signatures else ())
        ]
        print(
            "[SVDMechanism] "
            f"homogeneous_capacity={int(self.homogeneous_capacity)} "
            f"homogeneous_ratio={self.homogeneous_ratio:g} "
            f"factor_continuation={int(self.factor_continuation)} "
            f"feature_aux_loss={getattr(self.args, 'feature_aux_loss', 'mse')} "
            f"u_lr_ratio={float(getattr(self.args, 'u_lr_ratio', 0.1)):g} "
            f"v_lr_ratio={float(getattr(self.args, 'v_lr_ratio', 1.0)):g} "
            f"client_capacities={capacities} "
            f"actual_ranks={rank_summary if self.homogeneous_capacity else 'heterogeneous'}"
        )
    #发送模型参数（之后可能会修改，因为测试方法要保持一致，训练完后测试个性化性能）
    def send_parameters(self):
        assert (len(self.selected_clients) > 0)

        for client in self.selected_clients:
            start_time = time.time()
            #有的客户端会实现
            client.set_parameters(current_round=self.cur_ground)

            client.send_time_cost['num_rounds'] += 1
            client.send_time_cost['total_cost'] += 2 * (time.time() - start_time)

    def _has_low_rank_params(self, model):
        # 用参数名判断模型当前是否处在低秩形态；ResNet 的低秩卷积参数名是 conv_u/conv_v。
        return any(
            # 只要存在 V 矩阵，就说明这个模型还没有完全恢复成全秩卷积。
            name.endswith('conv_v') or name.endswith('weight_v')
            # 遍历模型的所有命名参数；这里不需要参数值，只需要名字。
            for name, _ in model.named_parameters()
        )

    def _recover_if_needed(self, model):
        # 聚合最终在全秩空间执行，所以只有低秩模型需要先恢复。
        if self._has_low_rank_params(model):
            # 调用模型自己的恢复接口；低秩 ResNet 和低秩 CNN 都使用这个接口名。
            model.recover_larger_model()
        # 返回同一个 model，方便调用处链式理解。
        return model

    # FedCLIP uses one aggregation rule: recover each client model to full rank,
    # then apply sample-weighted FedAvg to every model parameter.
    def aggregate_parameters_avg(self):
        """Sample-weighted averaging of complete recovered models."""
        assert len(self.uploaded_ids) > 0
        print("🚀 开始 Avg 聚合：恢复满秩后按客户端样本量聚合完整模型")

        uploaded_full_param_dicts = []
        for cid in self.uploaded_ids:
            client = self.clients[cid]
            client_model = load_item(client.role, 'model', client.save_folder_name)
            if client_model is None:
                raise RuntimeError(f"Client_{cid} uploaded model is missing.")
            full_model = copy.deepcopy(client_model).to(self.device)
            self._recover_if_needed(full_model)
            full_model = full_model.to(self.device)
            uploaded_full_param_dicts.append(dict(full_model.named_parameters()))

        self._save_sample_weighted_global(uploaded_full_param_dicts)
        print(f"✅ Avg 聚合完成，样本量权重: {self.uploaded_weights}")

    def aggregate_parameters_factor_continuation(self):
        """Sample-weighted U/V continuation with no full-W re-SVD reset."""
        assert len(self.uploaded_ids) > 0
        if not self.homogeneous_capacity:
            raise RuntimeError(
                "factor_continuation requires homogeneous client capacity."
            )

        client_models = []
        for cid in self.uploaded_ids:
            client = self.clients[cid]
            model = load_item(client.role, "model", client.save_folder_name)
            if model is None:
                raise RuntimeError(f"Client_{cid} uploaded model is missing.")
            model = copy.deepcopy(model).to(self.device)
            if not self._has_low_rank_params(model):
                raise RuntimeError(
                    f"Client_{cid} has no low-rank factors to continue."
                )
            client_models.append(model)

        factor_model = sample_weighted_factor_average(
            client_models, self.uploaded_weights
        )
        save_item(
            factor_model,
            self.role,
            "factor_model",
            self.save_folder_name,
        )

        full_model = copy.deepcopy(factor_model).to(self.device)
        self._recover_if_needed(full_model)
        save_item(full_model, self.role, "model", self.save_folder_name)
        print(
            "✅ FactorContinuation 聚合完成：U/V 与普通参数按样本量分别 "
            f"FedAvg，权重: {self.uploaded_weights}"
        )

    def _save_sample_weighted_global(self, uploaded_full_param_dicts):
        """Save one full-rank, sample-weighted global model from recovered clients."""

        global_model = load_item(self.role, 'model', self.save_folder_name)
        if global_model is None:
            raise RuntimeError("Server global model is missing before Avg aggregation.")
        global_model = global_model.to(self.device)
        self._recover_if_needed(global_model)
        global_model = global_model.to(self.device)
        global_params = dict(global_model.named_parameters())

        reference_names = global_params.keys()
        for source_idx, source_params in enumerate(uploaded_full_param_dicts):
            if source_params.keys() != reference_names:
                raise RuntimeError(
                    f"Client_{self.uploaded_ids[source_idx]} full model is incompatible with the Avg global model."
                )

        for global_param in global_params.values():
            global_param.data.zero_()
        for source_idx, weight in enumerate(self.uploaded_weights):
            for name, global_param in global_params.items():
                source_param = uploaded_full_param_dicts[source_idx][name]
                if source_param.shape != global_param.shape:
                    raise RuntimeError(
                        f"Avg shape mismatch for {name}: global={tuple(global_param.shape)}, "
                        f"client={tuple(source_param.shape)}"
                    )
                global_param.data += source_param.data * weight

        save_item(global_model, self.role, 'model', self.save_folder_name)

import csv
import copy
import math
import os
import random
import time
from datetime import datetime

import torch

from flcore.clients.clientCLIP import clientCLIP
from flcore.clients.clientbase import load_item, save_item
from flcore.servers.serverbase import Server
from flcore.trainmodel.models import Model_Distribe
from utils.get_clip_text_encoder import get_clip_class_embeddings


class FedCLIP(Server):
    def __init__(self, args, times):
        super().__init__(args, times)

        self.set_slow_clients()
        self.set_clients(clientCLIP)

        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print("Finished creating server and clients.")

        self.Budget = []
        self.global_acc = []
        self.personal_residuals = {cid: {} for cid in range(self.num_clients)}
        self.client_start_full_weights = {}
        self._projection_diagnostic_paths_printed = False
        projection_diagnostic_timestamp = datetime.now().strftime(
            "%Y%m%d_%H%M%S_%f"
        )
        repository_root = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "..", "..")
        )
        self.projection_diagnostic_folder = os.path.join(
            repository_root,
            "projection_csv_logs",
        )
        os.makedirs(self.projection_diagnostic_folder, exist_ok=True)
        self.projection_client_diagnostic_csv = os.path.join(
            self.projection_diagnostic_folder,
            f"{projection_diagnostic_timestamp}_projection_client_diagnostics.csv",
        )
        self.projection_direction_diagnostic_csv = os.path.join(
            self.projection_diagnostic_folder,
            f"{projection_diagnostic_timestamp}_projection_direction_diagnostics.csv",
        )

        global_model = Model_Distribe(args, -1, is_global=True).to(self.device)
        self._recover_if_needed(global_model)
        save_item(global_model, self.role, "model", self.save_folder_name)

        clip_text_features, clip_text_features_norm = get_clip_class_embeddings(
            self.dataset,
            model_name="ViT-B/32",
            prompt_template="a photo of {}",
            device=self.device,
        )
        self.clip_text_features = clip_text_features.float()
        self.clip_text_features_norm = clip_text_features_norm.float()

    def train(self):
        for i in range(self.global_rounds + 1):
            self.cur_ground = i
            round_start = time.time()
            self.selected_clients = self.select_clients()

            if i > 0 and i % self.eval_gap == 0:
                print(f"\n-------------Round number: {i} 聚合前-------------")
                print("\nEvaluate heterogeneous models")
                self.evaluate(epoch=i)

            self.send_parameters()

            client_train_times = []
            local_train_wall_start = time.time()
            for client in self.selected_clients:
                train_time = client.train(current_round=i)
                if train_time is None:
                    train_time = getattr(client, "last_train_time_cost", 0.0)
                client_train_times.append((client.id, float(train_time)))
            local_train_wall_time = time.time() - local_train_wall_start
            local_train_sum_time = sum(train_time for _, train_time in client_train_times)
            print(
                f"⏱️ [Round {i:03d}] 本地训练总耗时: "
                f"sum_client={local_train_sum_time:.3f}s | wall={local_train_wall_time:.3f}s | "
                f"clients={len(client_train_times)}"
            )

            self.receive_ids()

            aggregation_start = time.time()
            aggregation_mode = self._aggregation_mode()
            if self._is_resnet_model() and aggregation_mode != "avg":
                print(
                    f"当前 {aggregation_mode} 聚合只实现 CNN 路径，"
                    "ResNet 暂时使用 FedAvg 聚合。"
                )
                self.aggregate_avg(save_personalized=True)
            elif aggregation_mode != "avg" and self._is_projection_warmup_round():
                print(
                    f"{aggregation_mode} warm-up: round={self.cur_ground}, "
                    f"warmup_rounds={self._projection_warmup_rounds()}，使用 FedAvg。"
                )
                self.aggregate_avg(save_personalized=True)
            elif aggregation_mode == "projection":
                self.aggregate_common_residual_projection_cnn()
            elif aggregation_mode == "consensus_projection":
                self.aggregate_consensus_projection()
            elif aggregation_mode == "sign_personalized_projection":
                self.aggregate_sign_personalized_projection()
            elif aggregation_mode == "sign_projection_norm_restore":
                self.aggregate_sign_projection_norm_restore()
            elif aggregation_mode == "sign_projection_no_group_renorm":
                self.aggregate_sign_projection_no_group_renorm()
            elif aggregation_mode == "sign_projection_weight":
                self.aggregate_sign_projection_weight()
            elif aggregation_mode == "delta_avg":
                self.aggregate_delta_avg()
            elif aggregation_mode == "avg":
                self.aggregate_avg(save_personalized=True)
            else:
                raise ValueError(f"Unsupported aggregation_mode: {aggregation_mode}")
            aggregation_time = time.time() - aggregation_start
            print(f"⏱️ [Round {i:03d}] 聚合总墙钟耗时: {aggregation_time:.3f}s")

            self.Budget.append(time.time() - round_start)
            print("-" * 25, "time cost", "-" * 25, self.Budget[-1])

            if self.auto_break and self.check_done(acc_lss=[self.rs_test_acc], top_cnt=self.top_cnt):
                break

        print("\nBest accuracy.")
        print(max(self.rs_test_acc))

        print("\nBest Global accuracy.")
        if len(self.global_acc) > 0:
            print(max(self.global_acc))
        else:
            print("未记录 Global accuracy")

        print("\nAverage time cost per round.")
        print(sum(self.Budget[1:]) / len(self.Budget[1:]))

        self.save_results()
        self.save_json_file()

    def send_parameters(self):
        assert len(self.selected_clients) > 0
        capture_delta_start = (
            self._aggregation_mode() in {
                "delta_avg",
                "sign_personalized_projection",
                "sign_projection_norm_restore",
                "sign_projection_no_group_renorm",
            }
            and not self._is_resnet_model()
            and not self._is_projection_warmup_round()
        )
        if capture_delta_start:
            self.client_start_full_weights = {}
        for client in self.selected_clients:
            start_time = time.time()
            client.set_parameters()
            client.send_time_cost["num_rounds"] += 1
            client.send_time_cost["total_cost"] += 2 * (time.time() - start_time)
            if capture_delta_start:
                self._snapshot_client_start_full_weights(client)

    def receive_ids(self):
        assert len(self.selected_clients) > 0

        active_clients = random.sample(
            self.selected_clients,
            int((1 - self.client_drop_rate) * self.current_num_join_clients),
        )

        self.uploaded_ids = []
        self.uploaded_weights = []
        total_samples = 0
        for client in active_clients:
            total_samples += client.train_samples
            self.uploaded_ids.append(client.id)
            self.uploaded_weights.append(client.train_samples)

        self.uploaded_weights = [
            weight / total_samples for weight in self.uploaded_weights
        ]

    def _is_resnet_model(self):
        return "resnet" in getattr(self.args, "model_family", "").lower()

    def _aggregation_mode(self):
        mode = getattr(self.args, "aggregation_mode", None)
        if mode is not None:
            return mode
        return (
            "projection"
            if bool(getattr(self.args, "use_common_residual_projection", 1))
            else "avg"
        )

    def _projection_warmup_rounds(self):
        ratio = float(getattr(self.args, "projection_warmup_ratio", 0.2))
        ratio = min(1.0, max(0.0, ratio))
        return int(round(self.global_rounds * ratio))

    def _is_projection_warmup_round(self):
        return self.cur_ground <= self._projection_warmup_rounds()

    def _projection_layer_scope(self):
        scope = str(getattr(self.args, "projection_layer_scope", "low_rank"))
        valid_scopes = {
            "low_rank",
            "low_rank_plus_classifier",
            "all_weight",
        }
        if scope not in valid_scopes:
            raise ValueError(
                "projection_layer_scope must be 'low_rank', "
                "'low_rank_plus_classifier', or 'all_weight'."
            )
        return scope

    def _projection_layer_scope_for_mode(self, mode_name, input_kind):
        if (
            mode_name == "sign_projection_no_group_renorm"
            and input_kind == "delta"
        ):
            return self._projection_layer_scope()
        return "low_rank"

    def _personalized_rank_selection_enabled(self, mode_name=None):
        if mode_name is None:
            mode_name = self._aggregation_mode()
        raw_flag = int(getattr(self.args, "personalized_rank_selection", 0))
        if raw_flag not in (0, 1):
            raise ValueError("personalized_rank_selection must be 0 or 1.")
        if raw_flag == 1 and mode_name not in {
            "sign_projection_no_group_renorm",
            "sign_projection_weight",
        }:
            raise ValueError(
                "personalized_rank_selection is only supported by "
                "sign_projection_no_group_renorm and sign_projection_weight."
            )
        return raw_flag == 1

    def _personalized_rank_force_u1(self):
        raw_flag = int(getattr(self.args, "personalized_rank_force_u1", 1))
        if raw_flag not in (0, 1):
            raise ValueError("personalized_rank_force_u1 must be 0 or 1.")
        return raw_flag == 1

    def _personalized_rank_mode(self):
        mode = str(getattr(self.args, "personalized_rank_mode", "fixed"))
        if mode not in {"fixed", "energy"}:
            raise ValueError("personalized_rank_mode must be 'fixed' or 'energy'.")
        return mode

    def _personalized_rank_energy(self):
        threshold = float(getattr(self.args, "personalized_rank_energy", 0.8))
        if not math.isfinite(threshold) or not 0.0 < threshold <= 1.0:
            raise ValueError("personalized_rank_energy must be in the interval (0, 1].")
        return threshold

    def _personalized_g_scale_enabled(self):
        raw_flag = int(getattr(self.args, "personalized_g_scale", 1))
        if raw_flag not in (0, 1):
            raise ValueError("personalized_g_scale must be 0 or 1.")
        return raw_flag == 1

    def _local_update_views(self):
        views = int(getattr(self.args, "local_update_views", 1))
        if views not in (1, 2):
            raise ValueError("local_update_views must be 1 or 2.")
        return views

    def _personalized_repeatability_threshold(self):
        threshold = float(
            getattr(self.args, "personalized_repeatability_threshold", -1.0)
        )
        if not math.isfinite(threshold) or not -1.0 <= threshold <= 1.0:
            raise ValueError(
                "personalized_repeatability_threshold must be in [-1, 1]."
            )
        return threshold

    def _personalized_coeff_mode(self):
        mode = str(getattr(self.args, "personalized_coeff_mode", "same_sign"))
        if mode not in {"same_sign", "self", "avg"}:
            raise ValueError(
                "personalized_coeff_mode must be 'same_sign', 'self', or 'avg'."
            )
        return mode

    def _personalized_m_filter_mode(self):
        mode = str(getattr(self.args, "personalized_m_filter_mode", "none"))
        if mode not in {"none", "dominant_side"}:
            raise ValueError(
                "personalized_m_filter_mode must be 'none' or "
                "'dominant_side'."
            )
        return mode

    def _personalized_dominance_threshold(self):
        threshold = float(
            getattr(self.args, "personalized_dominance_threshold", 0.7)
        )
        if not math.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
            raise ValueError(
                "personalized_dominance_threshold must be in [0, 1]."
            )
        return threshold

    def _personalized_tail_scale(self):
        scale = float(getattr(self.args, "personalized_tail_scale", 1.0))
        if not math.isfinite(scale) or scale < 0.0:
            raise ValueError(
                "personalized_tail_scale must be finite and non-negative."
            )
        return scale

    def _is_sign_projection_diagnostic_round(self):
        first_projection_round = self._projection_warmup_rounds() + 1
        if self._personalized_rank_selection_enabled():
            return (
                self.cur_ground == first_projection_round
                or (
                    self.cur_ground >= first_projection_round
                    and self.cur_ground in {1, 20, 21, 50, 100}
                )
            )
        return (
            self.cur_ground == first_projection_round
            or (
                self.cur_ground > first_projection_round
                and self.cur_ground % 10 == 0
            )
        )

    @staticmethod
    def _is_sign_projection_console_layer(name):
        return name == "conv2.weight" or name == "fc2.weight"

    def _should_use_projection(self):
        return (
            self._aggregation_mode() == "projection"
            and not self._is_resnet_model()
            and not self._is_projection_warmup_round()
        )

    def _has_low_rank_params(self, model):
        return any(
            name.endswith("conv_v")
            or name.endswith("conv_u")
            or name.endswith("weight_v")
            or name.endswith("weight_u")
            for name, _ in model.named_parameters()
        )

    def _recover_if_needed(self, model):
        if hasattr(model, "recover_larger_model") and self._has_low_rank_params(model):
            model.recover_larger_model()
        return model

    def _decompose_if_needed(self, model, rank_rate):
        if hasattr(model, "decom_larger_model") and rank_rate < 1.0 and not self._has_low_rank_params(model):
            model.decom_larger_model(rank_rate)
        return model

    def _low_rank_start_folder(self):
        return os.path.join(self.save_folder_name, "low_rank_start")

    def _projectable_weight_names_from_low_rank_model(self, model):
        names = set()
        for name, _ in model.named_parameters():
            if name.endswith(".conv_u") or name.endswith(".conv_v"):
                names.add(name.rsplit(".", 1)[0] + ".weight")
            elif name.endswith(".weight_u") or name.endswith(".weight_v"):
                names.add(name.rsplit(".", 1)[0] + ".weight")
        return names

    def _get_projectable_weight_names(
        self,
        full_model,
        low_rank_model,
        scope,
    ):
        low_rank_names = self._projectable_weight_names_from_low_rank_model(
            low_rank_model
        )
        if scope == "low_rank":
            return low_rank_names

        matrix_weight_names = [
            name
            for name, param in full_model.named_parameters()
            if name.endswith(".weight") and param.ndim >= 2
        ]
        if scope == "all_weight":
            return set(matrix_weight_names)

        classifier_candidates = [
            name
            for name, param in full_model.named_parameters()
            if name.endswith(".weight") and param.ndim == 2
        ]
        if classifier_candidates:
            low_rank_names.add(classifier_candidates[-1])
        return low_rank_names

    def _snapshot_client_start_full_weights(self, client):
        low_rank_start = load_item(client.role, "model", client.save_folder_name)
        if low_rank_start is None:
            raise RuntimeError(
                f"DeltaAvg 无法保存 Client_{client.id} 的训练起点：客户端模型不存在。"
            )

        start_full_model = copy.deepcopy(low_rank_start).to("cpu")
        self._recover_if_needed(start_full_model)
        start_full_model = start_full_model.to("cpu")
        self.client_start_full_weights[client.id] = {
            name: param.data.detach().cpu().clone()
            for name, param in start_full_model.named_parameters()
        }

    def _load_old_start_model(self, cid, rank_rate):
        old_model = load_item(self.role, f"model_{cid}", self._low_rank_start_folder())
        if old_model is None:
            print(
                f"⚠️ Client_{cid} 缺少 low_rank_start 起点模型，"
                f"路径: {self._low_rank_start_folder()}。将回退到服务器保存的个性化模型计算 delta，"
                f"请确认这不是异常断点或缓存丢失导致的。"
            )
            old_model = load_item(self.role, f"model_{cid}", self.save_folder_name)
        if old_model is None:
            print(
                f"⚠️ Client_{cid} 也缺少服务器个性化旧模型，"
                f"将回退到通用 Server_model 作为 delta 起点。"
            )
            old_model = load_item(self.role, "model", self.save_folder_name)
        old_model = copy.deepcopy(old_model).to(self.device)
        if not self._has_low_rank_params(old_model):
            self._decompose_if_needed(old_model, rank_rate)
        self._recover_if_needed(old_model)
        return old_model.to(self.device)

    def _client_full_models_and_deltas(self):
        uploaded_full_param_dicts = []
        full_delta_param_dicts = []
        projectable_weight_names = set()
        rank_rates = {}

        for cid in self.uploaded_ids:
            client = self.clients[cid]
            low_rank_model = load_item(client.role, "model", client.save_folder_name)
            rank_rate = float(getattr(low_rank_model, "ratio_LR", 1.0))
            rank_rates[cid] = rank_rate
            projectable_weight_names.update(
                self._projectable_weight_names_from_low_rank_model(low_rank_model)
            )

            current_full = copy.deepcopy(low_rank_model).to(self.device)
            self._recover_if_needed(current_full)
            current_full = current_full.to(self.device)
            current_dict = dict(current_full.named_parameters())

            old_full = self._load_old_start_model(cid, rank_rate)
            old_dict = dict(old_full.named_parameters())

            delta_dict = {}
            for name, current_param in current_dict.items():
                if name not in old_dict:
                    continue
                old_data = old_dict[name].data.to(current_param.device)
                delta_dict[name] = current_param.data.detach().clone() - old_data.detach().clone()

            uploaded_full_param_dicts.append(current_dict)
            full_delta_param_dicts.append(delta_dict)

        return (
            uploaded_full_param_dicts,
            full_delta_param_dicts,
            projectable_weight_names,
            rank_rates,
        )

    def aggregate_avg(self, save_personalized=True):
        assert len(self.uploaded_ids) > 0

        global_model = load_item(self.role, "model", self.save_folder_name).to(self.device)
        self._recover_if_needed(global_model)
        global_model = global_model.to(self.device)
        global_param_dict = dict(global_model.named_parameters())

        for param in global_model.parameters():
            param.data.zero_()

        for weight, cid in zip(self.uploaded_weights, self.uploaded_ids):
            client = self.clients[cid]
            client_model = load_item(client.role, "model", client.save_folder_name)
            full_model = copy.deepcopy(client_model).to(self.device)
            self._recover_if_needed(full_model)
            full_model = full_model.to(self.device)
            client_param_dict = dict(full_model.named_parameters())
            for name, global_param in global_param_dict.items():
                if name in client_param_dict:
                    global_param.data += client_param_dict[name].data * weight

        save_item(global_model, self.role, "model", self.save_folder_name)

        if save_personalized:
            self.personal_residuals = {cid: {} for cid in range(self.num_clients)}
            for cid in range(self.num_clients):
                save_item(copy.deepcopy(global_model), self.role, f"model_{cid}", self.save_folder_name)

        print(f"执行 FedAvg 聚合，聚合权重为 {self.uploaded_weights}")

    def aggregate_delta_avg(self):
        assert len(self.uploaded_ids) > 0

        print("🚀 执行 CNN DeltaAvg 聚合")
        uploaded_full_param_dicts = []
        projectable_weight_names = set()

        for cid in self.uploaded_ids:
            if cid not in self.client_start_full_weights:
                raise RuntimeError(
                    f"DeltaAvg 缺少 Client_{cid} 本轮实际训练起点 S_i^t。"
                    "为避免 delta 错位，已停止聚合，不会回退到服务器当前模型。"
                )

            client = self.clients[cid]
            uploaded_low_rank_model = load_item(
                client.role,
                "model",
                client.save_folder_name,
            )
            if uploaded_low_rank_model is None:
                raise RuntimeError(f"DeltaAvg 无法加载 Client_{cid} 的上传模型。")

            projectable_weight_names.update(
                self._projectable_weight_names_from_low_rank_model(uploaded_low_rank_model)
            )
            uploaded_full_model = copy.deepcopy(uploaded_low_rank_model).to(self.device)
            self._recover_if_needed(uploaded_full_model)
            uploaded_full_model = uploaded_full_model.to(self.device)
            uploaded_full_param_dicts.append(dict(uploaded_full_model.named_parameters()))

        global_model = load_item(self.role, "model", self.save_folder_name).to(self.device)
        self._recover_if_needed(global_model)
        global_model = global_model.to(self.device)
        global_param_dict = dict(global_model.named_parameters())
        alpha = [float(weight) for weight in self.uploaded_weights]

        for name, global_param in global_param_dict.items():
            can_delta_update = (
                name in projectable_weight_names
                and all(name in params for params in uploaded_full_param_dicts)
                and all(
                    name in self.client_start_full_weights[cid]
                    for cid in self.uploaded_ids
                )
            )

            if can_delta_update:
                averaged_delta = torch.zeros_like(global_param.data)
                for weight, cid, uploaded_params in zip(
                    alpha,
                    self.uploaded_ids,
                    uploaded_full_param_dicts,
                ):
                    uploaded_weight = uploaded_params[name].data
                    start_weight = self.client_start_full_weights[cid][name].to(
                        uploaded_weight.device
                    )
                    averaged_delta += weight * (uploaded_weight - start_weight)
                global_param.data += averaged_delta
            else:
                global_param.data.zero_()
                for weight, uploaded_params in zip(alpha, uploaded_full_param_dicts):
                    if name in uploaded_params:
                        global_param.data += weight * uploaded_params[name].data

        save_item(global_model, self.role, "model", self.save_folder_name)
        self.personal_residuals = {cid: {} for cid in range(self.num_clients)}
        self._save_personalized_models_from_global(global_model, False)
        self.client_start_full_weights = {}
        print(f"DeltaAvg 聚合完成，样本量权重为 {self.uploaded_weights}")

    def aggregate_consensus_projection(self):
        assert len(self.uploaded_ids) > 0

        print("🚀 执行 CNN 方向一致性加权 Projection 聚合")
        (
            uploaded_full_param_dicts,
            delta_param_dicts,
            projectable_weight_names,
            _rank_rates,
        ) = self._client_full_models_and_deltas()

        global_model = load_item(self.role, "model", self.save_folder_name).to(self.device)
        self._recover_if_needed(global_model)
        global_model = global_model.to(self.device)
        global_param_dict = dict(global_model.named_parameters())
        alpha = [float(weight) for weight in self.uploaded_weights]

        for name, global_param in global_param_dict.items():
            can_project = (
                name in projectable_weight_names
                and all(name in delta_dict for delta_dict in delta_param_dicts)
            )
            if can_project:
                consensus_delta = self._consensus_projected_update_for_layer(
                    name,
                    delta_param_dicts,
                    alpha,
                    global_param.data.shape,
                )
                global_param.data += consensus_delta.to(global_param.device)
            else:
                global_param.data.zero_()
                for weight, uploaded_params in zip(alpha, uploaded_full_param_dicts):
                    if name in uploaded_params:
                        global_param.data += weight * uploaded_params[name].data

        save_item(global_model, self.role, "model", self.save_folder_name)
        self.personal_residuals = {cid: {} for cid in range(self.num_clients)}
        self._save_personalized_models_from_global(global_model, False)
        print(f"方向一致性 Projection 聚合完成，样本量权重为 {self.uploaded_weights}")

    def aggregate_sign_personalized_projection(self):
        self._aggregate_sign_projection_variant(
            mode_name="sign_personalized_projection",
            group_renorm=True,
            norm_restore=False,
        )

    def aggregate_sign_projection_norm_restore(self):
        self._aggregate_sign_projection_variant(
            mode_name="sign_projection_norm_restore",
            group_renorm=True,
            norm_restore=True,
        )

    def aggregate_sign_projection_no_group_renorm(self):
        self._aggregate_sign_projection_variant(
            mode_name="sign_projection_no_group_renorm",
            group_renorm=False,
            norm_restore=True,
        )

    def aggregate_sign_projection_weight(self):
        self._aggregate_sign_projection_variant(
            mode_name="sign_projection_weight",
            group_renorm=False,
            norm_restore=True,
            input_kind="weight",
        )

    def _aggregate_sign_projection_variant(
        self,
        mode_name,
        group_renorm,
        norm_restore,
        input_kind="delta",
    ):
        assert len(self.uploaded_ids) > 0
        if input_kind not in {"delta", "weight"}:
            raise ValueError("input_kind must be 'delta' or 'weight'.")
        uses_full_weights = input_kind == "weight"
        expected_input_kind = (
            "weight" if mode_name == "sign_projection_weight" else "delta"
        )
        if input_kind != expected_input_kind:
            raise ValueError(
                f"{mode_name} requires input_kind='{expected_input_kind}'."
            )
        projection_layer_scope = self._projection_layer_scope_for_mode(
            mode_name,
            input_kind,
        )

        mode_labels = {
            "sign_personalized_projection": "符号一致性个性化 Projection",
            "sign_projection_norm_restore": "符号 Projection + 整体范数恢复",
            "sign_projection_no_group_renorm": "符号 Projection（无组内归一化）+ 整体范数恢复",
            "sign_projection_weight": "完整权重符号 Projection（无组内归一化）+ 整体范数恢复",
        }
        print(f"🚀 执行 CNN {mode_labels[mode_name]} 聚合")
        personalized_rank_selection = self._personalized_rank_selection_enabled(
            mode_name
        )
        local_update_views = self._local_update_views()
        repeatability_threshold = (
            self._personalized_repeatability_threshold()
        )
        personalized_coeff_mode = self._personalized_coeff_mode()
        personalized_m_filter_mode = self._personalized_m_filter_mode()
        personalized_dominance_threshold = (
            self._personalized_dominance_threshold()
        )
        personalized_tail_scale = self._personalized_tail_scale()
        if repeatability_threshold > -1.0 and (
            local_update_views != 2
            or not personalized_rank_selection
            or self._personalized_rank_mode() != "energy"
        ):
            raise ValueError(
                "Repeatability filtering requires local_update_views=2, "
                "personalized_rank_selection=1, and rank_mode=energy."
            )
        if personalized_coeff_mode != "same_sign" and not personalized_rank_selection:
            raise ValueError(
                "Coefficient-mode self/avg ablations require personalized "
                "rank selection."
            )
        if personalized_m_filter_mode == "dominant_side" and (
            mode_name != "sign_projection_no_group_renorm"
            or uses_full_weights
            or not personalized_rank_selection
            or self._personalized_rank_mode() != "energy"
            or personalized_coeff_mode != "same_sign"
        ):
            raise ValueError(
                "dominant_side M filtering requires delta-based "
                "sign_projection_no_group_renorm, personalized energy "
                "rank selection, and same_sign coefficients."
            )
        if personalized_tail_scale != 1.0 and (
            not personalized_rank_selection
            or not self._personalized_rank_force_u1()
        ):
            raise ValueError(
                "Tail-scale ablation requires personalized rank selection "
                "with force_u1=1."
            )
        if (
            personalized_tail_scale != 1.0
            and repeatability_threshold > -1.0
        ):
            raise ValueError(
                "Tail-scale and repeatability-filter ablations must be run "
                "separately because filtering can remove the K=1 base."
            )
        if personalized_rank_selection:
            personalized_rank_mode = self._personalized_rank_mode()
            personalized_rank_num = int(
                getattr(self.args, "personalized_rank_num", 5)
            )
            if personalized_rank_mode == "fixed" and personalized_rank_num < 1:
                raise ValueError("personalized_rank_num must be at least 1.")
            personalized_rank_force_u1 = self._personalized_rank_force_u1()
            personalized_g_scale = self._personalized_g_scale_enabled()
            if personalized_rank_mode == "fixed":
                if personalized_rank_force_u1:
                    selection_description = "固定保留方向 0，再选择其余方向"
                else:
                    selection_description = "从全部有效方向自由选择 Top-M"
                rank_description = f"mode=fixed, M={personalized_rank_num}"
            else:
                personalized_rank_energy = self._personalized_rank_energy()
                if personalized_rank_force_u1:
                    selection_description = "先计入方向 0，再累计其余方向能量"
                else:
                    selection_description = "按全部有效方向自由累计能量"
                rank_description = (
                    f"mode=energy, tau={personalized_rank_energy:.6g}"
                )
            print(
                "客户端自适应方向选择已启用: "
                f"{rank_description}, "
                f"force_u1={int(personalized_rank_force_u1)}，"
                f"personalized_g_scale={int(personalized_g_scale)}，"
                f"{selection_description}。"
            )
        if (
            local_update_views != 1
            or repeatability_threshold > -1.0
            or personalized_coeff_mode != "same_sign"
            or personalized_m_filter_mode != "none"
            or personalized_tail_scale != 1.0
        ):
            print(
                "个性化方向诊断设置: "
                f"local_update_views={local_update_views}, "
                f"repeatability_threshold={repeatability_threshold:.6g}, "
                f"coeff_mode={personalized_coeff_mode}, "
                f"m_filter_mode={personalized_m_filter_mode}, "
                f"dominance_threshold="
                f"{personalized_dominance_threshold:.6g}, "
                f"tail_scale={personalized_tail_scale:.6g}。"
            )
        uploaded_full_param_dicts = []
        projection_param_dicts = []
        projection_param_dicts_b = [] if local_update_views == 2 else None
        projectable_weight_names = set()

        for cid in self.uploaded_ids:
            if not uses_full_weights and cid not in self.client_start_full_weights:
                raise RuntimeError(
                    f"Sign Personalized Projection 缺少 Client_{cid} 本轮训练起点 S_i^t。"
                )

            client = self.clients[cid]
            uploaded_low_rank_model = load_item(client.role, "model", client.save_folder_name)
            if uploaded_low_rank_model is None:
                raise RuntimeError(f"无法加载 Client_{cid} 的上传模型。")

            uploaded_full_model = copy.deepcopy(uploaded_low_rank_model).to(self.device)
            self._recover_if_needed(uploaded_full_model)
            uploaded_full_model = uploaded_full_model.to(self.device)
            projectable_weight_names.update(
                self._get_projectable_weight_names(
                    uploaded_full_model,
                    uploaded_low_rank_model,
                    projection_layer_scope,
                )
            )
            uploaded_params = dict(uploaded_full_model.named_parameters())

            if uses_full_weights:
                projection_params = uploaded_params
                start_params = None
            else:
                start_params = self.client_start_full_weights[cid]
                projection_params = {}
                for name, uploaded_param in uploaded_params.items():
                    if name in start_params:
                        projection_params[name] = (
                            uploaded_param.data.detach().clone()
                            - start_params[name].to(uploaded_param.device)
                        )

            uploaded_full_param_dicts.append(uploaded_params)
            projection_param_dicts.append(projection_params)

            if local_update_views == 2:
                view_b_round = getattr(
                    client,
                    "local_update_view_b_round",
                    None,
                )
                if view_b_round != self.cur_ground:
                    raise RuntimeError(
                        f"Client_{cid} B-view round mismatch: "
                        f"expected {self.cur_ground}, got {view_b_round}."
                    )
                uploaded_low_rank_model_b = load_item(
                    client.role,
                    "model_view_b",
                    client.save_folder_name,
                )
                if uploaded_low_rank_model_b is None:
                    raise RuntimeError(
                        f"无法加载 Client_{cid} 的 B 视图上传模型。"
                    )
                uploaded_full_model_b = copy.deepcopy(
                    uploaded_low_rank_model_b
                ).to(self.device)
                self._recover_if_needed(uploaded_full_model_b)
                uploaded_params_b = dict(
                    uploaded_full_model_b.to(self.device).named_parameters()
                )
                projection_params_b = {}
                for name, uploaded_param_b in uploaded_params_b.items():
                    if uses_full_weights:
                        projection_params_b[name] = (
                            uploaded_param_b.data.detach().cpu().clone()
                        )
                    elif name in start_params:
                        projection_params_b[name] = (
                            uploaded_param_b.data.detach().clone()
                            - start_params[name].to(uploaded_param_b.device)
                        ).detach().cpu()
                projection_param_dicts_b.append(projection_params_b)

        global_model = load_item(self.role, "model", self.save_folder_name).to(self.device)
        self._recover_if_needed(global_model)
        global_model = global_model.to(self.device)
        global_param_dict = dict(global_model.named_parameters())
        alpha = [float(weight) for weight in self.uploaded_weights]
        personalized_updates = {cid: {} for cid in self.uploaded_ids}

        actual_projected_names = [
            name
            for name in global_param_dict
            if (
                name in projectable_weight_names
                and all(
                    name in projection_params
                    for projection_params in projection_param_dicts
                )
                and (
                    uses_full_weights
                    or all(
                        name in self.client_start_full_weights[cid]
                        for cid in self.uploaded_ids
                    )
                )
            )
        ]
        actual_projected_name_set = set(actual_projected_names)
        diagnostic_round = (
            bool(actual_projected_names)
            and self._is_sign_projection_diagnostic_round()
        )
        if (
            diagnostic_round
            and mode_name == "sign_projection_no_group_renorm"
        ):
            major_weight_names = [
                name
                for name, param in global_param_dict.items()
                if name.endswith(".weight") and param.ndim >= 2
            ]
            averaged_major_weight_names = [
                name
                for name in major_weight_names
                if name not in actual_projected_name_set
            ]
            print(
                f"[ProjectionLayerScope] round={self.cur_ground} "
                f"scope={projection_layer_scope}"
            )
            print(f"  projected_layers={actual_projected_names}")
            print(
                "  averaged_major_weight_layers="
                f"{averaged_major_weight_names}"
            )

        for name, global_param in global_param_dict.items():
            can_project = name in actual_projected_name_set
            if (
                can_project
                and projection_param_dicts_b is not None
                and not all(
                    name in projection_params_b
                    for projection_params_b in projection_param_dicts_b
                )
            ):
                raise RuntimeError(
                    f"B 视图缺少 A 视图可投影层 {name}，拒绝改变 A 的聚合路径。"
                )
            if can_project:
                personalized_values, average_value = (
                    self._sign_personalized_update_for_layer(
                        name,
                        projection_param_dicts,
                        alpha,
                        global_param.data.shape,
                        delta_param_dicts_b=projection_param_dicts_b,
                        log_diagnostics=diagnostic_round,
                        console_diagnostics=(
                            diagnostic_round
                            and (
                                self._is_sign_projection_console_layer(name)
                                or (
                                    mode_name
                                    == "sign_projection_no_group_renorm"
                                    and projection_layer_scope != "low_rank"
                                )
                            )
                        ),
                        group_renorm=group_renorm,
                        norm_restore=norm_restore,
                        mode_name=mode_name,
                        input_kind=input_kind,
                        projection_layer_scope=projection_layer_scope,
                    )
                )
                if uses_full_weights:
                    global_param.data.copy_(
                        average_value.to(global_param.device)
                    )
                else:
                    global_param.data += average_value.to(global_param.device)
                for cid, personalized_value in zip(
                    self.uploaded_ids,
                    personalized_values,
                ):
                    personalized_updates[cid][name] = (
                        personalized_value.detach().cpu()
                    )
            else:
                global_param.data.zero_()
                for weight, uploaded_params in zip(alpha, uploaded_full_param_dicts):
                    if name in uploaded_params:
                        global_param.data += weight * uploaded_params[name].data

        save_item(global_model, self.role, "model", self.save_folder_name)
        self.personal_residuals = {cid: {} for cid in range(self.num_clients)}
        if uses_full_weights:
            self._save_sign_personalized_weight_models(
                global_model,
                personalized_updates,
            )
        else:
            self._save_sign_personalized_models(
                global_model,
                personalized_updates,
            )
        self.client_start_full_weights = {}
        print(f"{mode_labels[mode_name]} 聚合完成，样本量权重为 {self.uploaded_weights}")

    def aggregate_common_residual_projection_cnn(self):
        assert len(self.uploaded_ids) > 0

        print("🚀 执行 CNN 公共-残差投影聚合")
        (
            uploaded_full_param_dicts,
            delta_param_dicts,
            projectable_weight_names,
            _rank_rates,
        ) = self._client_full_models_and_deltas()

        global_model = load_item(self.role, "model", self.save_folder_name).to(self.device)
        self._recover_if_needed(global_model)
        global_model = global_model.to(self.device)
        global_param_dict = dict(global_model.named_parameters())

        alpha = [float(w) for w in self.uploaded_weights]
        projection_use_residual = bool(getattr(self.args, "projection_use_residual", 1))
        use_residual_ema = bool(getattr(self.args, "projection_residual_ema", 0))
        current_round_residuals = {cid: {} for cid in self.uploaded_ids}
        if projection_use_residual:
            if use_residual_ema:
                print(
                    f"残差模式: EMA | mu={getattr(self.args, 'personal_residual_mu', 0.9)} "
                    f"gamma={getattr(self.args, 'personal_residual_gamma', 0.5)}"
                )
            else:
                print(f"残差模式: current-round only | beta={getattr(self.args, 'personal_residual_beta', 0.1)}")
        else:
            print("残差模式: disabled")

        for name, global_param in global_param_dict.items():
            if name in projectable_weight_names and all(name in delta_dict for delta_dict in delta_param_dicts):
                common_delta, residuals = self._common_projected_update_for_layer(
                    name,
                    delta_param_dicts,
                    alpha,
                    global_param.data.shape,
                )
                global_param.data += common_delta.to(global_param.device)
                if projection_use_residual:
                    if use_residual_ema:
                        self._update_personal_residuals(name, residuals, global_param.data)
                    else:
                        beta = float(getattr(self.args, "personal_residual_beta", 0.1))
                        for cid, residual in zip(self.uploaded_ids, residuals):
                            current_round_residuals[cid][name] = (beta * residual).detach().cpu()
            else:
                global_param.data.zero_()
                for weight, client_param_dict in zip(alpha, uploaded_full_param_dicts):
                    if name in client_param_dict:
                        global_param.data += client_param_dict[name].data * weight

        save_item(global_model, self.role, "model", self.save_folder_name)
        if projection_use_residual and use_residual_ema:
            self._save_personalized_models_from_global(global_model, True)
        else:
            self._save_personalized_models_from_global(
                global_model,
                False,
                current_round_residuals if projection_use_residual else None,
            )

    @staticmethod
    def _projection_rank_from_energy(
        eigvals,
        positive,
        energy_threshold,
        k_max,
    ):
        if not math.isfinite(energy_threshold) or not (
            0.0 < energy_threshold <= 1.0
        ):
            raise ValueError("projection_energy must be in the interval (0, 1].")
        if not bool(torch.isfinite(eigvals).all()):
            raise FloatingPointError(
                "Projection eigenvalues contain NaN or Inf."
            )

        rank_r = int(positive.sum().item())
        if rank_r < 1:
            raise ValueError("Projection rank selection requires positive energy.")

        positive_eigvals = eigvals[positive]
        total_energy = positive_eigvals.sum()
        if not bool(torch.isfinite(total_energy)) or total_energy <= 0:
            raise ValueError("Projection energy must be finite and positive.")
        # The caller has already established positive energy, so adding eps to
        # this denominator is both unnecessary and incorrect for tiny updates:
        # it can make even the final cumulative ratio smaller than the target.
        cumulative = torch.cumsum(positive_eigvals, dim=0) / total_energy
        threshold_hits = torch.nonzero(
            cumulative >= energy_threshold,
            as_tuple=False,
        )
        # For energy_threshold=1, reduction-order rounding can leave the last
        # cumulative value infinitesimally below 1. Treat that as selecting R.
        k_energy = (
            int(threshold_hits[0].item()) + 1
            if threshold_hits.numel() > 0
            else rank_r
        )
        return max(1, min(k_energy, k_max, rank_r))

    def _common_projected_update_for_layer(self, name, delta_param_dicts, alpha, target_shape):
        eps = 1e-12
        device = self.device
        raw_vecs = [
            delta_dict[name].detach().to(device).reshape(-1).float()
            for delta_dict in delta_param_dicts
        ]
        weighted_unit_vecs = []
        for weight, vec in zip(alpha, raw_vecs):
            norm = torch.norm(vec)
            unit_vec = vec / (norm + eps)
            weighted_unit_vecs.append(unit_vec * math.sqrt(max(weight, eps)))

        num_clients = len(raw_vecs)
        gram = torch.zeros((num_clients, num_clients), device=device)
        for i in range(num_clients):
            for j in range(i, num_clients):
                value = torch.dot(weighted_unit_vecs[i], weighted_unit_vecs[j])
                gram[i, j] = value
                gram[j, i] = value

        try:
            eigvals, eigvecs = torch.linalg.eigh(gram)
        except RuntimeError:
            print(f"⚠️ 层 {name} Gram 分解失败，退回该层 FedAvg delta。")
            avg_delta = self._weighted_average_vectors(raw_vecs, alpha).reshape(target_shape)
            return avg_delta, [torch.zeros_like(avg_delta) for _ in raw_vecs]

        order = torch.argsort(eigvals, descending=True)
        eigvals = eigvals[order].clamp_min(0)
        eigvecs = eigvecs[:, order]
        positive = eigvals > eps
        total_energy = eigvals[positive].sum()
        if total_energy <= eps:
            avg_delta = self._weighted_average_vectors(raw_vecs, alpha).reshape(target_shape)
            return avg_delta, [torch.zeros_like(avg_delta) for _ in raw_vecs]

        energy_threshold = float(getattr(self.args, "projection_energy", 0.8))
        k_max = int(getattr(self.args, "projection_k_max", 5))
        k_max = max(1, min(k_max, num_clients))
        k = self._projection_rank_from_energy(
            eigvals,
            positive,
            energy_threshold,
            k_max,
        )

        h = eigvecs[:, :k]
        lam = eigvals[:k].clamp_min(eps)

        common_vecs = []
        residual_vecs = []
        for vec in raw_vecs:
            q = torch.stack([torch.dot(col, vec) for col in weighted_unit_vecs])
            beta = h @ ((h.t() @ q) / lam)
            common_vec = torch.zeros_like(vec)
            for coeff, col in zip(beta, weighted_unit_vecs):
                common_vec += coeff * col
            common_vecs.append(common_vec)
            residual_vecs.append(vec - common_vec)

        common_avg = self._weighted_average_vectors(common_vecs, alpha).reshape(target_shape)
        residuals = [residual.reshape(target_shape) for residual in residual_vecs]
        return common_avg, residuals

    def _consensus_projected_update_for_layer(self, name, delta_param_dicts, alpha, target_shape):
        eps = 1e-12
        device = self.device
        raw_vecs = [
            delta_dict[name].detach().to(device).reshape(-1).float()
            for delta_dict in delta_param_dicts
        ]

        unit_vecs = []
        weighted_unit_vecs = []
        for weight, vec in zip(alpha, raw_vecs):
            unit_vec = vec / (torch.norm(vec) + eps)
            unit_vecs.append(unit_vec)
            weighted_unit_vecs.append(unit_vec * math.sqrt(max(weight, eps)))

        num_clients = len(raw_vecs)
        gram = torch.zeros((num_clients, num_clients), device=device)
        for i in range(num_clients):
            for j in range(i, num_clients):
                value = torch.dot(weighted_unit_vecs[i], weighted_unit_vecs[j])
                gram[i, j] = value
                gram[j, i] = value

        try:
            eigvals, eigvecs = torch.linalg.eigh(gram)
        except RuntimeError:
            print(f"⚠️ 层 {name} Gram 分解失败，退回该层 FedAvg delta。")
            return self._weighted_average_vectors(raw_vecs, alpha).reshape(target_shape)

        order = torch.argsort(eigvals, descending=True)
        eigvals = eigvals[order].clamp_min(0)
        eigvecs = eigvecs[:, order]
        positive = eigvals > eps
        total_energy = eigvals[positive].sum()
        if total_energy <= eps:
            return self._weighted_average_vectors(raw_vecs, alpha).reshape(target_shape)

        energy_threshold = float(getattr(self.args, "projection_energy", 0.8))
        k_max = int(getattr(self.args, "projection_k_max", 5))
        k_max = max(1, min(k_max, num_clients))
        k = self._projection_rank_from_energy(
            eigvals,
            positive,
            energy_threshold,
            k_max,
        )

        h = eigvecs[:, :k]
        sigma = torch.sqrt(eigvals[:k].clamp_min(eps))
        average_update = self._weighted_average_vectors(raw_vecs, alpha)
        consensus_update = torch.zeros_like(average_update)
        alpha_tensor = torch.tensor(alpha, device=device, dtype=average_update.dtype)

        for direction_idx in range(k):
            direction = torch.zeros_like(average_update)
            for client_idx, weighted_unit_vec in enumerate(weighted_unit_vecs):
                direction += h[client_idx, direction_idx] * weighted_unit_vec
            direction = direction / sigma[direction_idx]

            client_projections = torch.stack([
                torch.dot(unit_vec, direction) for unit_vec in unit_vecs
            ])
            consensus = torch.abs(torch.sum(alpha_tensor * client_projections)) / (
                torch.sum(alpha_tensor * torch.abs(client_projections)) + eps
            )
            consensus = consensus.clamp(0.0, 1.0)
            average_projection = torch.dot(average_update, direction)
            consensus_update += consensus * average_projection * direction

        return consensus_update.reshape(target_shape)

    @staticmethod
    def _direction_repeatability(projections_a, projections_b, eps=1e-12):
        if projections_a.shape != projections_b.shape:
            raise ValueError("A/B projection tensors must have the same shape.")
        if eps <= 0.0:
            raise ValueError("Repeatability epsilon must be positive.")
        if not bool(torch.isfinite(projections_a).all()) or not bool(
            torch.isfinite(projections_b).all()
        ):
            raise FloatingPointError("A/B projections contain NaN or Inf.")

        # Divide by a per-entry scale before squaring. This is algebraically
        # equivalent to 2ab / (a^2 + b^2 + eps), while finite float32 raw
        # projections cannot overflow during the diagnostic calculation.
        calculation_dtype = (
            torch.float64
            if projections_a.dtype in {torch.float16, torch.bfloat16, torch.float32}
            else projections_a.dtype
        )
        a = projections_a.to(calculation_dtype)
        b = projections_b.to(calculation_dtype)
        sqrt_eps = math.sqrt(eps)
        scale = torch.maximum(torch.abs(a), torch.abs(b)).clamp_min(sqrt_eps)
        scaled_a = a / scale
        scaled_b = b / scale
        denominator = (
            scaled_a.square()
            + scaled_b.square()
            + (sqrt_eps / scale).square()
        )
        repeatability = (
            2.0 * scaled_a * scaled_b / denominator
        ).clamp(-1.0, 1.0)
        repeatability = repeatability.to(projections_a.dtype)
        if not bool(torch.isfinite(repeatability).all()):
            raise FloatingPointError("Direction repeatability contains NaN or Inf.")
        return repeatability

    @staticmethod
    def _dominant_side_filter(
        direction_projections,
        alpha_tensor,
        selected_direction_mask_raw,
        dominance_threshold,
        sign_epsilon=1e-8,
        denominator_epsilon=1e-12,
    ):
        if direction_projections.ndim != 2:
            raise ValueError("Direction projections must be a 2-D tensor.")
        if selected_direction_mask_raw.shape != direction_projections.shape:
            raise ValueError(
                "Raw selection mask must match direction projection shape."
            )
        if alpha_tensor.ndim != 1 or alpha_tensor.shape[0] != (
            direction_projections.shape[0]
        ):
            raise ValueError("Alpha weights must match the client dimension.")
        if not 0.0 <= dominance_threshold <= 1.0:
            raise ValueError("Dominance threshold must be in [0, 1].")
        if sign_epsilon <= 0.0 or denominator_epsilon <= 0.0:
            raise ValueError("Dominance-filter epsilons must be positive.")
        if not bool(torch.isfinite(direction_projections).all()) or not bool(
            torch.isfinite(alpha_tensor).all()
        ):
            raise FloatingPointError(
                "Dominance-filter projections or alpha weights are non-finite."
            )

        positive_side = direction_projections > sign_epsilon
        negative_side = direction_projections < -sign_epsilon
        weighted_squared = (
            alpha_tensor.unsqueeze(1) * direction_projections.square()
        )
        positive_energy = torch.where(
            positive_side,
            weighted_squared,
            torch.zeros_like(weighted_squared),
        ).sum(dim=0)
        negative_energy = torch.where(
            negative_side,
            weighted_squared,
            torch.zeros_like(weighted_squared),
        ).sum(dim=0)
        total_energy = positive_energy + negative_energy
        energy_difference = positive_energy - negative_energy
        dominance_ratio = torch.where(
            total_energy > denominator_epsilon,
            torch.maximum(positive_energy, negative_energy)
            / (total_energy + denominator_epsilon),
            torch.zeros_like(total_energy),
        )
        dominant_sign = torch.zeros_like(total_energy, dtype=torch.int8)
        dominant_sign[energy_difference > denominator_epsilon] = 1
        dominant_sign[energy_difference < -denominator_epsilon] = -1
        direction_has_dominant_side = (
            (dominance_ratio >= dominance_threshold)
            & (dominant_sign != 0)
            & (total_energy > denominator_epsilon)
        )
        client_matches_dominant_side = (
            (positive_side & (dominant_sign.unsqueeze(0) > 0))
            | (negative_side & (dominant_sign.unsqueeze(0) < 0))
        )
        keep_mask = (
            selected_direction_mask_raw
            & direction_has_dominant_side.unsqueeze(0)
            & client_matches_dominant_side
        )
        balanced_filter_mask = (
            selected_direction_mask_raw
            & ~direction_has_dominant_side.unsqueeze(0)
        )
        weak_side_filter_mask = (
            selected_direction_mask_raw
            & direction_has_dominant_side.unsqueeze(0)
            & ~client_matches_dominant_side
        )
        return (
            positive_energy,
            negative_energy,
            dominance_ratio,
            dominant_sign,
            keep_mask,
            balanced_filter_mask,
            weak_side_filter_mask,
        )

    @staticmethod
    def _select_personalized_directions(
        eigvals,
        eigvecs,
        rank_r,
        personalized_rank_num,
        force_u1=True,
        rank_mode="fixed",
        energy_threshold=0.8,
        eps=1e-12,
    ):
        """Select fixed Top-M or a minimal per-client energy prefix.

        ``eigvals`` are sigma squared for the weighted unit-update matrix used by
        this aggregation path, so each entry below is sigma_k^2 * v_{i,k}^2.
        """
        if rank_r < 1:
            raise ValueError("rank_r must be at least 1.")
        if rank_mode not in {"fixed", "energy"}:
            raise ValueError("rank_mode must be 'fixed' or 'energy'.")
        if rank_mode == "fixed" and personalized_rank_num < 1:
            raise ValueError("personalized_rank_num must be at least 1.")
        if rank_mode == "energy" and (
            not math.isfinite(energy_threshold)
            or not 0.0 < energy_threshold <= 1.0
        ):
            raise ValueError("energy_threshold must be in the interval (0, 1].")

        direction_scores = (
            eigvals[:rank_r].unsqueeze(0) * eigvecs[:, :rank_r].square()
        )
        if not bool(torch.isfinite(direction_scores).all()):
            raise FloatingPointError("Personalized direction scores contain NaN or Inf.")
        selected_direction_mask = torch.zeros_like(
            direction_scores,
            dtype=torch.bool,
        )
        selected_counts = torch.zeros(
            direction_scores.shape[0],
            device=direction_scores.device,
            dtype=torch.long,
        )
        zero_energy_fallback = torch.zeros_like(
            selected_counts,
            dtype=torch.bool,
        )

        if rank_mode == "fixed":
            effective_rank_num = min(personalized_rank_num, rank_r)
            if force_u1:
                selected_direction_mask[:, 0] = True
                if effective_rank_num > 1:
                    client_top_directions = torch.argsort(
                        direction_scores[:, 1:],
                        dim=1,
                        descending=True,
                        stable=True,
                    )[:, :effective_rank_num - 1] + 1
                    selected_direction_mask.scatter_(
                        1,
                        client_top_directions,
                        True,
                    )
            else:
                client_top_directions = torch.argsort(
                    direction_scores,
                    dim=1,
                    descending=True,
                    stable=True,
                )[:, :effective_rank_num]
                selected_direction_mask.scatter_(1, client_top_directions, True)
            selected_counts.fill_(effective_rank_num)
        else:
            total_scores = direction_scores.sum(dim=1)
            zero_energy_fallback = total_scores <= eps
            for client_idx in range(direction_scores.shape[0]):
                if bool(zero_energy_fallback[client_idx]):
                    continue

                if force_u1:
                    remaining_order = torch.argsort(
                        direction_scores[client_idx, 1:],
                        descending=True,
                        stable=True,
                    ) + 1
                    selection_order = torch.cat((
                        torch.zeros(
                            1,
                            device=direction_scores.device,
                            dtype=torch.long,
                        ),
                        remaining_order,
                    ))
                else:
                    selection_order = torch.argsort(
                        direction_scores[client_idx],
                        descending=True,
                        stable=True,
                    )

                ordered_scores = direction_scores[
                    client_idx,
                    selection_order,
                ]
                cumulative_ratios = torch.cumsum(ordered_scores, dim=0) / (
                    total_scores[client_idx]
                )
                positive_positions = torch.nonzero(
                    ordered_scores > 0,
                    as_tuple=False,
                ).flatten()
                last_positive_position = int(positive_positions[-1].item())
                cumulative_ratios[last_positive_position:] = 1.0
                if energy_threshold == 1.0:
                    selected_count = last_positive_position + 1
                else:
                    threshold_hits = torch.nonzero(
                        cumulative_ratios >= energy_threshold,
                        as_tuple=False,
                    ).flatten()
                    selected_count = (
                        int(threshold_hits[0].item()) + 1
                        if threshold_hits.numel() > 0
                        else last_positive_position + 1
                    )
                selected_ids = selection_order[:selected_count]
                selected_direction_mask[client_idx, selected_ids] = True
                selected_counts[client_idx] = selected_count

        if selected_direction_mask.dtype != torch.bool:
            raise AssertionError("Personalized direction mask must be boolean.")
        if not torch.equal(selected_direction_mask.sum(dim=1), selected_counts):
            raise AssertionError(
                "Selected direction counts must match the boolean mask."
            )
        selected_coordinates = torch.nonzero(
            selected_direction_mask,
            as_tuple=False,
        )
        if selected_coordinates.numel() > 0:
            selected_direction_ids = selected_coordinates[:, 1]
        else:
            selected_direction_ids = torch.empty(
                0,
                device=direction_scores.device,
                dtype=torch.long,
            )
        if selected_direction_ids.numel() > 0 and (
            int(selected_direction_ids.min().item()) < 0
            or int(selected_direction_ids.max().item()) >= rank_r
        ):
            raise AssertionError(
                "Selected direction indices must be within [0, rank_r)."
            )
        non_fallback = ~zero_energy_fallback
        if bool(non_fallback.any()) and not bool(
            ((selected_counts[non_fallback] >= 1)
             & (selected_counts[non_fallback] <= rank_r)).all()
        ):
            raise AssertionError(
                "Non-fallback clients must select between 1 and rank_r directions."
            )
        if not bool((selected_counts[zero_energy_fallback] == 0).all()):
            raise AssertionError("Zero-energy fallback clients must select no directions.")
        if force_u1 and bool(non_fallback.any()) and not bool(
            selected_direction_mask[non_fallback, 0].all()
        ):
            raise AssertionError(
                "force_u1 requires every non-fallback client to select direction 0."
            )

        return (
            selected_direction_mask,
            direction_scores,
            selected_counts,
            zero_energy_fallback,
        )

    def _sign_personalized_update_for_layer(
        self,
        name,
        delta_param_dicts,
        alpha,
        target_shape,
        delta_param_dicts_b=None,
        log_diagnostics=False,
        console_diagnostics=False,
        group_renorm=True,
        norm_restore=False,
        mode_name="sign_personalized_projection",
        input_kind="delta",
        projection_layer_scope="low_rank",
    ):
        if input_kind not in {"delta", "weight"}:
            raise ValueError("input_kind must be 'delta' or 'weight'.")
        expected_input_kind = (
            "weight" if mode_name == "sign_projection_weight" else "delta"
        )
        if input_kind != expected_input_kind:
            raise ValueError(
                f"{mode_name} requires input_kind='{expected_input_kind}'."
            )
        input_label = "完整权重" if input_kind == "weight" else "delta"
        fallback_label = "平均完整权重" if input_kind == "weight" else "DeltaAvg"
        eps = 1e-12
        log_zero = 1e-8
        device = self.device
        raw_vecs = [
            delta_dict[name].detach().to(device).reshape(-1).float()
            for delta_dict in delta_param_dicts
        ]
        if not all(bool(torch.isfinite(vec).all()) for vec in raw_vecs):
            raise FloatingPointError(
                f"层 {name} 的客户端{input_label}出现 NaN 或 Inf。"
            )
        raw_vecs_b = None
        if delta_param_dicts_b is not None:
            if len(delta_param_dicts_b) != len(delta_param_dicts):
                raise ValueError("A/B local-update view counts must match.")
            raw_vecs_b = [
                delta_dict[name].detach().to(device).reshape(-1).float()
                for delta_dict in delta_param_dicts_b
            ]
            if not all(
                vec.shape == raw_vecs[client_idx].shape
                and bool(torch.isfinite(vec).all())
                for client_idx, vec in enumerate(raw_vecs_b)
            ):
                raise FloatingPointError(
                    f"层 {name} 的 B 视图客户端{input_label}非有限或形状不匹配。"
                )
        average_delta = self._weighted_average_vectors(raw_vecs, alpha)

        unit_vecs = []
        weighted_unit_vecs = []
        for weight, vec in zip(alpha, raw_vecs):
            unit_vec = vec / (torch.norm(vec) + eps)
            unit_vecs.append(unit_vec)
            weighted_unit_vecs.append(unit_vec * math.sqrt(max(weight, eps)))

        num_clients = len(raw_vecs)
        gram = torch.zeros((num_clients, num_clients), device=device)
        for i in range(num_clients):
            for j in range(i, num_clients):
                value = torch.dot(weighted_unit_vecs[i], weighted_unit_vecs[j])
                gram[i, j] = value
                gram[j, i] = value

        try:
            eigvals, eigvecs = torch.linalg.eigh(gram)
        except RuntimeError:
            print(
                f"⚠️ 层 {name} Gram 分解失败，"
                f"退回该层个性化{fallback_label}。"
            )
            fallback = average_delta.reshape(target_shape)
            return [fallback.clone() for _ in raw_vecs], fallback

        order = torch.argsort(eigvals, descending=True)
        eigvals = eigvals[order].clamp_min(0)
        eigvecs = eigvecs[:, order]
        positive = eigvals > eps
        total_energy = eigvals[positive].sum()
        if total_energy <= eps:
            fallback = average_delta.reshape(target_shape)
            return [fallback.clone() for _ in raw_vecs], fallback

        energy_threshold = float(getattr(self.args, "projection_energy", 0.8))
        k_max = int(getattr(self.args, "projection_k_max", 5))
        k_max = max(1, min(k_max, num_clients))
        k = self._projection_rank_from_energy(
            eigvals,
            positive,
            energy_threshold,
            k_max,
        )
        rank_r = int(positive.sum().item())
        personalized_rank_selection = self._personalized_rank_selection_enabled(
            mode_name
        )
        personalized_rank_num_requested = int(
            getattr(self.args, "personalized_rank_num", 5)
        )
        personalized_rank_force_u1 = self._personalized_rank_force_u1()
        personalized_rank_mode = self._personalized_rank_mode()
        personalized_rank_energy = (
            self._personalized_rank_energy()
            if personalized_rank_mode == "energy"
            else None
        )
        personalized_g_scale = self._personalized_g_scale_enabled()
        local_update_views = self._local_update_views()
        repeatability_threshold = (
            self._personalized_repeatability_threshold()
        )
        personalized_coeff_mode = self._personalized_coeff_mode()
        personalized_m_filter_mode = self._personalized_m_filter_mode()
        personalized_dominance_threshold = (
            self._personalized_dominance_threshold()
        )
        personalized_tail_scale = self._personalized_tail_scale()
        if local_update_views == 2 and raw_vecs_b is None:
            raise ValueError(
                "local_update_views=2 requires B-view layer "
                f"{input_kind} inputs."
            )
        if repeatability_threshold > -1.0 and (
            local_update_views != 2
            or raw_vecs_b is None
            or not personalized_rank_selection
            or personalized_rank_mode != "energy"
        ):
            raise ValueError(
                "Repeatability filtering requires two A/B views and "
                "personalized energy rank selection."
            )
        if personalized_coeff_mode != "same_sign" and not personalized_rank_selection:
            raise ValueError(
                "Coefficient-mode self/avg ablations require personalized "
                "rank selection."
            )
        if personalized_m_filter_mode == "dominant_side" and (
            mode_name != "sign_projection_no_group_renorm"
            or input_kind != "delta"
            or not personalized_rank_selection
            or personalized_rank_mode != "energy"
            or personalized_coeff_mode != "same_sign"
        ):
            raise ValueError(
                "dominant_side M filtering requires delta-based "
                "sign_projection_no_group_renorm, personalized energy "
                "rank selection, and same_sign coefficients."
            )
        if (
            personalized_tail_scale != 1.0
            and repeatability_threshold > -1.0
        ):
            raise ValueError(
                "Tail-scale and repeatability-filter ablations must be run "
                "separately because filtering can remove the K=1 base."
            )
        if personalized_rank_selection:
            (
                selected_direction_mask,
                direction_scores,
                selected_direction_counts,
                zero_energy_fallback,
            ) = self._select_personalized_directions(
                eigvals,
                eigvecs,
                rank_r,
                personalized_rank_num_requested,
                personalized_rank_force_u1,
                personalized_rank_mode,
                personalized_rank_energy,
                eps,
            )
            working_direction_count = rank_r
            personalized_rank_num_effective = (
                min(personalized_rank_num_requested, rank_r)
                if personalized_rank_mode == "fixed"
                else None
            )
        else:
            working_direction_count = k
            selected_direction_mask = torch.ones(
                (num_clients, working_direction_count),
                device=device,
                dtype=torch.bool,
            )
            direction_scores = (
                eigvals[:working_direction_count].unsqueeze(0)
                * eigvecs[:, :working_direction_count].square()
            )
            selected_direction_counts = torch.full(
                (num_clients,),
                working_direction_count,
                device=device,
                dtype=torch.long,
            )
            zero_energy_fallback = torch.zeros(
                num_clients,
                device=device,
                dtype=torch.bool,
            )
            personalized_rank_num_effective = working_direction_count

        selected_direction_mask_raw = selected_direction_mask.clone()
        selected_direction_counts_raw = selected_direction_counts.clone()
        selected_direction_mask_before_repeatability = (
            selected_direction_mask_raw
        )
        selected_direction_counts_before_repeatability = (
            selected_direction_counts_raw
        )

        h = eigvecs[:, :working_direction_count]
        sigma = torch.sqrt(eigvals[:working_direction_count].clamp_min(eps))
        alpha_tensor = torch.tensor(alpha, device=device, dtype=average_delta.dtype)

        # Recover the selected left singular directions so target participation
        # strengths can be checked directly in the original update space.
        left_directions = []
        for direction_idx in range(working_direction_count):
            direction = torch.zeros_like(average_delta)
            for client_idx, weighted_unit_vec in enumerate(weighted_unit_vecs):
                direction += h[client_idx, direction_idx] * weighted_unit_vec
            left_directions.append(direction / sigma[direction_idx])
        left_directions = torch.stack(left_directions)

        direction_projections = []
        for vec in raw_vecs:
            direction_projections.append(
                torch.stack([
                    torch.dot(vec, direction)
                    for direction in left_directions
                ])
            )
        direction_projections = torch.stack(direction_projections)

        normalized_direction_projections = torch.stack([
            torch.stack([
                torch.dot(unit_vec, direction)
                for direction in left_directions
            ])
            for unit_vec in unit_vecs
        ])
        direction_projections_b = None
        normalized_direction_projections_b = None
        repeatability_raw = None
        repeatability_normalized = None
        if raw_vecs_b is not None:
            unit_vecs_b = [
                vec / (torch.norm(vec) + eps) for vec in raw_vecs_b
            ]
            direction_projections_b = torch.stack([
                torch.stack([
                    torch.dot(vec, direction)
                    for direction in left_directions
                ])
                for vec in raw_vecs_b
            ])
            normalized_direction_projections_b = torch.stack([
                torch.stack([
                    torch.dot(unit_vec, direction)
                    for direction in left_directions
                ])
                for unit_vec in unit_vecs_b
            ])
            repeatability_raw = self._direction_repeatability(
                direction_projections,
                direction_projections_b,
                eps,
            )
            repeatability_normalized = self._direction_repeatability(
                normalized_direction_projections,
                normalized_direction_projections_b,
                eps,
            )

        dominance_filter_enabled = (
            personalized_m_filter_mode == "dominant_side"
        )
        dominance_positive_energy = torch.zeros(
            working_direction_count,
            device=device,
            dtype=average_delta.dtype,
        )
        dominance_negative_energy = torch.zeros_like(
            dominance_positive_energy
        )
        dominance_ratio = torch.zeros_like(dominance_positive_energy)
        dominant_sign = torch.zeros(
            working_direction_count,
            device=device,
            dtype=torch.int8,
        )
        dominance_keep_mask = torch.ones_like(
            selected_direction_mask_raw,
            dtype=torch.bool,
        )
        dominance_balanced_filter_mask = torch.zeros_like(
            selected_direction_mask_raw,
            dtype=torch.bool,
        )
        dominance_weak_side_filter_mask = torch.zeros_like(
            selected_direction_mask_raw,
            dtype=torch.bool,
        )
        if dominance_filter_enabled:
            (
                dominance_positive_energy,
                dominance_negative_energy,
                dominance_ratio,
                dominant_sign,
                dominance_keep_mask,
                dominance_balanced_filter_mask,
                dominance_weak_side_filter_mask,
            ) = self._dominant_side_filter(
                direction_projections,
                alpha_tensor,
                selected_direction_mask_raw,
                personalized_dominance_threshold,
                sign_epsilon=log_zero,
                denominator_epsilon=eps,
            )

        repeatability_filter_enabled = repeatability_threshold > -1.0
        if repeatability_filter_enabled:
            selected_direction_mask = (
                selected_direction_mask_before_repeatability
                & (repeatability_normalized >= repeatability_threshold)
            )
            selected_direction_counts = selected_direction_mask.sum(dim=1)
            repeatability_empty_fallback = (
                (selected_direction_counts_before_repeatability > 0)
                & (selected_direction_counts == 0)
            )
        else:
            selected_direction_mask = (
                selected_direction_mask_before_repeatability
            )
            selected_direction_counts = (
                selected_direction_counts_before_repeatability
            )
            repeatability_empty_fallback = torch.zeros(
                num_clients,
                device=device,
                dtype=torch.bool,
            )

        selected_direction_mask_before_dominance = (
            selected_direction_mask.clone()
        )
        if dominance_filter_enabled:
            dominance_balanced_filter_mask = (
                dominance_balanced_filter_mask
                & selected_direction_mask_before_dominance
            )
            dominance_weak_side_filter_mask = (
                dominance_weak_side_filter_mask
                & selected_direction_mask_before_dominance
            )
            selected_direction_mask = (
                selected_direction_mask & dominance_keep_mask
            )
            selected_direction_counts = selected_direction_mask.sum(dim=1)
        dominance_empty_after_filter = torch.zeros(
            num_clients,
            device=device,
            dtype=torch.bool,
        )
        if dominance_filter_enabled:
            dominance_empty_after_filter = (
                (selected_direction_mask_before_dominance.sum(dim=1) > 0)
                & (selected_direction_counts == 0)
            )
        dominance_balanced_filtered_direction_count = int(
            dominance_balanced_filter_mask.any(dim=0).sum().item()
        )
        dominance_weak_side_filtered_client_direction_count = int(
            dominance_weak_side_filter_mask.sum().item()
        )

        tail_missing_u1_fallback = torch.zeros(
            num_clients,
            device=device,
            dtype=torch.bool,
        )
        if personalized_rank_selection and personalized_tail_scale != 1.0:
            tail_missing_u1_fallback = ~selected_direction_mask[:, 0]
        fallback_used = (
            zero_energy_fallback
            | repeatability_empty_fallback
            | tail_missing_u1_fallback
        )
        # An empty dominance-filtered set intentionally reconstructs a zero
        # update. It is not a DeltaAvg fallback and directions are not refilled.

        direct_strengths_unclamped = torch.abs(
            normalized_direction_projections
        )
        target_strengths = torch.clamp(direct_strengths_unclamped, 0.0, 1.0)
        svd_strengths = torch.abs(
            sigma.unsqueeze(0)
            * h
            / (torch.sqrt(alpha_tensor.clamp_min(eps)).unsqueeze(1) + eps)
        )
        strength_formula_max_error = torch.max(
            torch.abs(direct_strengths_unclamped - svd_strengths)
        )

        masks = []
        same_sign_weight_masses = torch.zeros(
            (num_clients, working_direction_count),
            device=device,
            dtype=average_delta.dtype,
        )
        group_coefficients_with_renorm = torch.zeros(
            (num_clients, working_direction_count),
            device=device,
            dtype=average_delta.dtype,
        )
        group_coefficients_without_renorm = torch.zeros(
            (num_clients, working_direction_count),
            device=device,
            dtype=average_delta.dtype,
        )
        personalized_coefficients = torch.zeros(
            (num_clients, working_direction_count),
            device=device,
            dtype=average_delta.dtype,
        )
        mask_symmetric = True
        self_mask_valid = True
        for direction_idx in range(working_direction_count):
            v = h[:, direction_idx]
            mask = (v.unsqueeze(1) * v.unsqueeze(0)) > 0
            masks.append(mask)
            mask_symmetric = mask_symmetric and torch.equal(mask, mask.t())

            significant = torch.abs(v) > log_zero
            if torch.any(significant):
                self_mask_valid = self_mask_valid and bool(
                    torch.all(torch.diag(mask)[significant]).item()
                )

            for target_idx in range(num_clients):
                active = mask[target_idx].to(average_delta.dtype)
                denominator = torch.sum(alpha_tensor * active)
                numerator = torch.sum(
                    alpha_tensor
                    * active
                    * direction_projections[:, direction_idx]
                )
                coefficient_with_renorm = numerator / (denominator + eps)
                same_sign_weight_masses[target_idx, direction_idx] = denominator
                group_coefficients_with_renorm[
                    target_idx, direction_idx
                ] = coefficient_with_renorm
                group_coefficients_without_renorm[
                    target_idx, direction_idx
                ] = numerator
                personalized_coefficients[target_idx, direction_idx] = (
                    coefficient_with_renorm if group_renorm else numerator
                )

        coefficients_same_sign = group_coefficients_without_renorm
        coefficients_self = direction_projections
        average_direction_coefficients = torch.stack([
            torch.dot(average_delta, direction)
            for direction in left_directions
        ])
        coefficients_avg = average_direction_coefficients.unsqueeze(0).expand(
            num_clients,
            -1,
        )
        if personalized_coeff_mode == "same_sign":
            coefficient_mode_values = personalized_coefficients
        elif personalized_coeff_mode == "self":
            coefficient_mode_values = coefficients_self
        else:
            coefficient_mode_values = coefficients_avg

        scaled_same_sign_coefficients = (
            target_strengths * personalized_coefficients
        )
        scaled_personalized_coefficients = (
            target_strengths * coefficient_mode_values
        )
        output_personalized_coefficients = (
            scaled_personalized_coefficients
            if (not personalized_rank_selection or personalized_g_scale)
            else coefficient_mode_values
        )
        if personalized_rank_selection:
            selected_mask_float = selected_direction_mask.to(
                coefficient_mode_values.dtype
            )
            selected_personalized_coefficients = (
                coefficient_mode_values * selected_mask_float
            )
            selected_output_personalized_coefficients = (
                output_personalized_coefficients * selected_mask_float
            )
            if personalized_tail_scale != 1.0:
                tail_multipliers = torch.full(
                    (working_direction_count,),
                    personalized_tail_scale,
                    device=device,
                    dtype=coefficient_mode_values.dtype,
                )
                tail_multipliers[0] = 1.0
                selected_personalized_coefficients = (
                    selected_personalized_coefficients * tail_multipliers
                )
                selected_output_personalized_coefficients = (
                    selected_output_personalized_coefficients
                    * tail_multipliers
                )
        else:
            selected_personalized_coefficients = coefficient_mode_values
            selected_output_personalized_coefficients = (
                output_personalized_coefficients
            )
        unscaled_personalized_vecs = []
        personalized_vecs_before_restore = []
        personalized_vecs = []
        gamma_raw_values = []
        gamma_used_values = []
        gamma_max = float(getattr(self.args, "projection_norm_scale_max", 2.0))
        if norm_restore and (not math.isfinite(gamma_max) or gamma_max <= 0):
            raise ValueError(
                "projection_norm_scale_max must be finite and greater than 0."
            )
        average_delta_norm = torch.norm(average_delta)
        for target_idx in range(num_clients):
            if bool(fallback_used[target_idx]):
                fallback_vec = average_delta.clone()
                gamma_raw = torch.ones_like(average_delta_norm)
                gamma_used = torch.ones_like(average_delta_norm)
                unscaled_personalized_vecs.append(fallback_vec.clone())
                personalized_vecs_before_restore.append(fallback_vec.clone())
                personalized_vecs.append(fallback_vec)
                gamma_raw_values.append(gamma_raw)
                gamma_used_values.append(gamma_used)
                continue

            if personalized_rank_selection and personalized_tail_scale == 0.0:
                unscaled_source_coefficients = h[:, :1] @ (
                    selected_personalized_coefficients[target_idx, :1]
                    / sigma[:1]
                )
                scaled_source_coefficients = h[:, :1] @ (
                    selected_output_personalized_coefficients[target_idx, :1]
                    / sigma[:1]
                )
            else:
                unscaled_source_coefficients = h @ (
                    selected_personalized_coefficients[target_idx] / sigma
                )
                scaled_source_coefficients = h @ (
                    selected_output_personalized_coefficients[target_idx] / sigma
                )
            unscaled_personalized_vec = torch.zeros_like(average_delta)
            personalized_vec = torch.zeros_like(average_delta)
            for unscaled_coefficient, scaled_coefficient, weighted_unit_vec in zip(
                unscaled_source_coefficients,
                scaled_source_coefficients,
                weighted_unit_vecs,
            ):
                unscaled_personalized_vec += (
                    unscaled_coefficient * weighted_unit_vec
                )
                personalized_vec += scaled_coefficient * weighted_unit_vec

            if norm_restore:
                gamma_raw = average_delta_norm / (
                    torch.norm(personalized_vec) + eps
                )
                gamma_used = torch.clamp(gamma_raw, max=gamma_max)
                final_personalized_vec = gamma_used * personalized_vec
            else:
                gamma_raw = torch.ones_like(average_delta_norm)
                gamma_used = torch.ones_like(gamma_raw)
                final_personalized_vec = personalized_vec

            unscaled_personalized_vecs.append(unscaled_personalized_vec)
            personalized_vecs_before_restore.append(personalized_vec)
            personalized_vecs.append(final_personalized_vec)
            gamma_raw_values.append(gamma_raw)
            gamma_used_values.append(gamma_used)

        gamma_raw_values = torch.stack(gamma_raw_values)
        gamma_used_values = torch.stack(gamma_used_values)

        finite_tensors = [
            eigvals,
            left_directions,
            direction_projections,
            normalized_direction_projections,
            direct_strengths_unclamped,
            target_strengths,
            svd_strengths,
            strength_formula_max_error,
            same_sign_weight_masses,
            group_coefficients_with_renorm,
            group_coefficients_without_renorm,
            personalized_coefficients,
            coefficients_same_sign,
            coefficients_self,
            coefficients_avg,
            coefficient_mode_values,
            scaled_same_sign_coefficients,
            scaled_personalized_coefficients,
            output_personalized_coefficients,
            selected_personalized_coefficients,
            selected_output_personalized_coefficients,
            direction_scores,
            selected_direction_counts,
            dominance_positive_energy,
            dominance_negative_energy,
            dominance_ratio,
            gamma_raw_values,
            gamma_used_values,
            *unscaled_personalized_vecs,
            *personalized_vecs_before_restore,
            *personalized_vecs,
        ]
        if raw_vecs_b is not None:
            finite_tensors.extend([
                *raw_vecs_b,
                direction_projections_b,
                normalized_direction_projections_b,
                repeatability_raw,
                repeatability_normalized,
            ])
        finite_ok = all(
            bool(torch.isfinite(tensor).all()) for tensor in finite_tensors
        )
        strength_in_range = bool(
            torch.all((target_strengths >= 0.0) & (target_strengths <= 1.0))
        )
        if not finite_ok:
            raise FloatingPointError(f"层 {name} 的符号一致性聚合出现 NaN 或 Inf。")
        if not strength_in_range:
            raise AssertionError(f"层 {name} 的目标方向参与强度不在 [0, 1] 内。")
        if not mask_symmetric:
            raise AssertionError(f"层 {name} 的符号掩码不满足 m_ij,k = m_ji,k。")
        if not self_mask_valid:
            raise AssertionError(f"层 {name} 的显著非零方向不满足 m_ii,k = 1。")

        input_energy = sum(torch.dot(vec, vec) for vec in weighted_unit_vecs)
        reconstructed_energy = eigvals[positive].sum()
        reconstruction_error = torch.sqrt(
            torch.clamp(input_energy - reconstructed_energy, min=0.0)
        ) / (torch.sqrt(input_energy) + eps)
        if not bool(torch.isfinite(reconstruction_error)):
            raise FloatingPointError(f"层 {name} 的 SVD 重构误差为 NaN 或 Inf。")
        if reconstruction_error.item() > 1e-3:
            print(
                f"⚠️ 层 {name} 的 SVD 相对重构误差偏大: "
                f"{reconstruction_error.item():.3e}"
            )

        if log_diagnostics:
            self._print_sign_projection_diagnostics(
                name=name,
                raw_vecs=raw_vecs,
                raw_vecs_b=raw_vecs_b,
                unit_vecs=unit_vecs,
                average_delta=average_delta,
                unscaled_personalized_vecs=unscaled_personalized_vecs,
                personalized_vecs_before_restore=personalized_vecs_before_restore,
                personalized_vecs=personalized_vecs,
                alpha_tensor=alpha_tensor,
                eigvals=eigvals,
                eigvecs=eigvecs,
                positive=positive,
                h=h,
                sigma=sigma,
                masks=masks,
                direction_projections=direction_projections,
                direction_projections_b=direction_projections_b,
                normalized_direction_projections=(
                    normalized_direction_projections
                ),
                normalized_direction_projections_b=(
                    normalized_direction_projections_b
                ),
                repeatability_raw=repeatability_raw,
                repeatability_normalized=repeatability_normalized,
                same_sign_weight_masses=same_sign_weight_masses,
                group_coefficients_with_renorm=group_coefficients_with_renorm,
                group_coefficients_without_renorm=group_coefficients_without_renorm,
                personalized_coefficients=personalized_coefficients,
                coefficients_same_sign=coefficients_same_sign,
                coefficients_self=coefficients_self,
                coefficients_avg=coefficients_avg,
                coefficient_mode_values=coefficient_mode_values,
                target_strengths=target_strengths,
                scaled_same_sign_coefficients=(
                    scaled_same_sign_coefficients
                ),
                scaled_personalized_coefficients=scaled_personalized_coefficients,
                output_personalized_coefficients=output_personalized_coefficients,
                selected_output_personalized_coefficients=(
                    selected_output_personalized_coefficients
                ),
                selected_direction_mask_before_repeatability=(
                    selected_direction_mask_before_repeatability
                ),
                selected_direction_counts_before_repeatability=(
                    selected_direction_counts_before_repeatability
                ),
                selected_direction_mask_before_dominance=(
                    selected_direction_mask_before_dominance
                ),
                selected_direction_mask=selected_direction_mask,
                selected_direction_counts=selected_direction_counts,
                dominance_filter_enabled=dominance_filter_enabled,
                dominance_positive_energy=dominance_positive_energy,
                dominance_negative_energy=dominance_negative_energy,
                dominance_ratio=dominance_ratio,
                dominant_sign=dominant_sign,
                dominance_keep_mask=dominance_keep_mask,
                dominance_balanced_filter_mask=(
                    dominance_balanced_filter_mask
                ),
                dominance_weak_side_filter_mask=(
                    dominance_weak_side_filter_mask
                ),
                dominance_empty_after_filter=(
                    dominance_empty_after_filter
                ),
                dominance_balanced_filtered_direction_count=(
                    dominance_balanced_filtered_direction_count
                ),
                dominance_weak_side_filtered_client_direction_count=(
                    dominance_weak_side_filtered_client_direction_count
                ),
                zero_energy_fallback=zero_energy_fallback,
                repeatability_empty_fallback=repeatability_empty_fallback,
                tail_missing_u1_fallback=tail_missing_u1_fallback,
                fallback_used=fallback_used,
                direction_scores=direction_scores,
                uniform_selected_k=k,
                personalized_rank_selection=personalized_rank_selection,
                personalized_rank_num_requested=personalized_rank_num_requested,
                personalized_rank_num_effective=personalized_rank_num_effective,
                personalized_rank_force_u1=personalized_rank_force_u1,
                personalized_rank_mode=personalized_rank_mode,
                personalized_rank_energy=personalized_rank_energy,
                personalized_g_scale=personalized_g_scale,
                local_update_views=local_update_views,
                repeatability_threshold=repeatability_threshold,
                personalized_coeff_mode=personalized_coeff_mode,
                personalized_m_filter_mode=personalized_m_filter_mode,
                personalized_dominance_threshold=(
                    personalized_dominance_threshold
                ),
                personalized_tail_scale=personalized_tail_scale,
                gamma_raw_values=gamma_raw_values,
                gamma_used_values=gamma_used_values,
                group_renorm=group_renorm,
                norm_restore=norm_restore,
                mode_name=mode_name,
                input_kind=input_kind,
                projection_layer_scope=projection_layer_scope,
                strength_formula_max_error=strength_formula_max_error,
                strength_in_range=strength_in_range,
                reconstruction_error=reconstruction_error,
                mask_symmetric=mask_symmetric,
                self_mask_valid=self_mask_valid,
                finite_ok=finite_ok,
                log_zero=log_zero,
                console_diagnostics=console_diagnostics,
            )

        personalized = [vec.reshape(target_shape) for vec in personalized_vecs]
        return personalized, average_delta.reshape(target_shape)

    def _print_sign_projection_diagnostics(
        self,
        name,
        raw_vecs,
        raw_vecs_b,
        unit_vecs,
        average_delta,
        unscaled_personalized_vecs,
        personalized_vecs_before_restore,
        personalized_vecs,
        alpha_tensor,
        eigvals,
        eigvecs,
        positive,
        h,
        sigma,
        masks,
        direction_projections,
        direction_projections_b,
        normalized_direction_projections,
        normalized_direction_projections_b,
        repeatability_raw,
        repeatability_normalized,
        same_sign_weight_masses,
        group_coefficients_with_renorm,
        group_coefficients_without_renorm,
        personalized_coefficients,
        coefficients_same_sign,
        coefficients_self,
        coefficients_avg,
        coefficient_mode_values,
        target_strengths,
        scaled_same_sign_coefficients,
        scaled_personalized_coefficients,
        output_personalized_coefficients,
        selected_output_personalized_coefficients,
        selected_direction_mask_before_repeatability,
        selected_direction_counts_before_repeatability,
        selected_direction_mask_before_dominance,
        selected_direction_mask,
        selected_direction_counts,
        dominance_filter_enabled,
        dominance_positive_energy,
        dominance_negative_energy,
        dominance_ratio,
        dominant_sign,
        dominance_keep_mask,
        dominance_balanced_filter_mask,
        dominance_weak_side_filter_mask,
        dominance_empty_after_filter,
        dominance_balanced_filtered_direction_count,
        dominance_weak_side_filtered_client_direction_count,
        zero_energy_fallback,
        repeatability_empty_fallback,
        tail_missing_u1_fallback,
        fallback_used,
        direction_scores,
        uniform_selected_k,
        personalized_rank_selection,
        personalized_rank_num_requested,
        personalized_rank_num_effective,
        personalized_rank_force_u1,
        personalized_rank_mode,
        personalized_rank_energy,
        personalized_g_scale,
        local_update_views,
        repeatability_threshold,
        personalized_coeff_mode,
        personalized_m_filter_mode,
        personalized_dominance_threshold,
        personalized_tail_scale,
        gamma_raw_values,
        gamma_used_values,
        group_renorm,
        norm_restore,
        mode_name,
        input_kind,
        projection_layer_scope,
        strength_formula_max_error,
        strength_in_range,
        reconstruction_error,
        mask_symmetric,
        self_mask_valid,
        finite_ok,
        log_zero,
        console_diagnostics,
    ):
        del unit_vecs  # Selected directions were already checked directly above.
        del masks
        del same_sign_weight_masses

        uses_full_weights = input_kind == "weight"
        average_reference_semantics = (
            "avg_weight" if uses_full_weights else "delta_avg"
        )
        personalized_output_semantics = (
            "aggregated_weight" if uses_full_weights else "aggregated_update"
        )
        personalized_writeback_semantics = (
            "copy_absolute" if uses_full_weights else "add_to_client_start"
        )
        global_writeback_semantics = (
            "copy_average_weight" if uses_full_weights else "add_average_delta"
        )
        eps = 1e-12
        working_direction_count = h.shape[1]
        k = uniform_selected_k
        rank_r = int(positive.sum().item())
        num_clients = len(raw_vecs)
        full_eigvals = eigvals[:rank_r]
        full_sigma = torch.sqrt(full_eigvals.clamp_min(eps))
        full_h = eigvecs[:, :rank_r]
        full_energy = full_eigvals / (full_eigvals.sum() + eps)
        full_cumulative = torch.cumsum(full_energy, dim=0)
        singular_values_csv = self._csv_sequence(
            [float(value) for value in full_sigma.detach().cpu().tolist()]
        )
        singular_energy_ratios_csv = self._csv_sequence(
            [float(value) for value in full_energy.detach().cpu().tolist()]
        )
        cumulative_energy_csv = self._csv_sequence(
            [float(value) for value in full_cumulative.detach().cpu().tolist()]
        )

        sqrt_alpha = torch.sqrt(alpha_tensor.clamp_min(eps)).unsqueeze(1)
        signed_unit_coefficients = (
            full_sigma.unsqueeze(0) * full_h / (sqrt_alpha + eps)
        )
        full_g = torch.clamp(torch.abs(signed_unit_coefficients), 0.0, 1.0)
        full_g[:, :working_direction_count] = target_strengths

        raw_norms = torch.stack([torch.norm(vec) for vec in raw_vecs])
        full_a = signed_unit_coefficients * (raw_norms + eps).unsqueeze(1)
        full_a[:, :working_direction_count] = direction_projections
        full_normalized_a = signed_unit_coefficients.clone()
        full_normalized_a[:, :working_direction_count] = (
            normalized_direction_projections
        )
        full_a_b = None
        full_normalized_a_b = None
        full_repeatability_raw = None
        full_repeatability_normalized = None
        if raw_vecs_b is not None:
            full_a_b = torch.full_like(full_a, float("nan"))
            full_normalized_a_b = torch.full_like(full_a, float("nan"))
            full_repeatability_raw = torch.full_like(full_a, float("nan"))
            full_repeatability_normalized = torch.full_like(
                full_a,
                float("nan"),
            )
            full_a_b[:, :working_direction_count] = direction_projections_b
            full_normalized_a_b[:, :working_direction_count] = (
                normalized_direction_projections_b
            )
            full_repeatability_raw[:, :working_direction_count] = (
                repeatability_raw
            )
            full_repeatability_normalized[:, :working_direction_count] = (
                repeatability_normalized
            )
        average_a = alpha_tensor @ full_a

        full_direction_scores = full_eigvals.unsqueeze(0) * full_h.square()
        full_direction_scores[:, :working_direction_count] = direction_scores
        full_energy_order = torch.argsort(
            full_direction_scores,
            dim=1,
            descending=True,
            stable=True,
        )
        full_energy_ranks = torch.empty_like(full_energy_order)
        full_energy_ranks.scatter_(
            1,
            full_energy_order,
            torch.arange(
                1,
                rank_r + 1,
                device=self.device,
                dtype=full_energy_order.dtype,
            ).unsqueeze(0).expand(num_clients, -1),
        )
        full_selected_direction_mask_before_repeatability = torch.zeros(
            (num_clients, rank_r),
            device=self.device,
            dtype=torch.bool,
        )
        full_selected_direction_mask_before_repeatability[
            :, :working_direction_count
        ] = selected_direction_mask_before_repeatability
        full_selected_direction_mask_before_dominance = torch.zeros(
            (num_clients, rank_r),
            device=self.device,
            dtype=torch.bool,
        )
        full_selected_direction_mask_before_dominance[
            :, :working_direction_count
        ] = selected_direction_mask_before_dominance
        full_selected_direction_mask = torch.zeros(
            (num_clients, rank_r),
            device=self.device,
            dtype=torch.bool,
        )
        full_selected_direction_mask[:, :working_direction_count] = (
            selected_direction_mask
        )
        full_dominance_weighted_squared = (
            alpha_tensor.unsqueeze(1) * full_a.square()
        )
        full_dominance_positive_energy = torch.where(
            full_a > log_zero,
            full_dominance_weighted_squared,
            torch.zeros_like(full_dominance_weighted_squared),
        ).sum(dim=0)
        full_dominance_negative_energy = torch.where(
            full_a < -log_zero,
            full_dominance_weighted_squared,
            torch.zeros_like(full_dominance_weighted_squared),
        ).sum(dim=0)
        full_dominance_total_energy = (
            full_dominance_positive_energy
            + full_dominance_negative_energy
        )
        full_dominance_energy_difference = (
            full_dominance_positive_energy
            - full_dominance_negative_energy
        )
        full_dominance_ratio = torch.where(
            full_dominance_total_energy > eps,
            torch.maximum(
                full_dominance_positive_energy,
                full_dominance_negative_energy,
            ) / (full_dominance_total_energy + eps),
            torch.zeros_like(full_dominance_total_energy),
        )
        full_dominant_sign = torch.zeros(
            rank_r,
            device=self.device,
            dtype=torch.int8,
        )
        full_dominant_sign[full_dominance_energy_difference > eps] = 1
        full_dominant_sign[full_dominance_energy_difference < -eps] = -1
        full_dominance_keep_mask = torch.zeros(
            (num_clients, rank_r),
            device=self.device,
            dtype=torch.bool,
        )
        full_dominance_balanced_filter_mask = torch.zeros_like(
            full_dominance_keep_mask
        )
        full_dominance_weak_side_filter_mask = torch.zeros_like(
            full_dominance_keep_mask
        )
        full_dominance_positive_energy[:working_direction_count] = (
            dominance_positive_energy
        )
        full_dominance_negative_energy[:working_direction_count] = (
            dominance_negative_energy
        )
        full_dominance_ratio[:working_direction_count] = dominance_ratio
        full_dominant_sign[:working_direction_count] = dominant_sign
        full_dominance_keep_mask[:, :working_direction_count] = (
            dominance_keep_mask
        )
        full_dominance_balanced_filter_mask[
            :, :working_direction_count
        ] = dominance_balanced_filter_mask
        full_dominance_weak_side_filter_mask[
            :, :working_direction_count
        ] = dominance_weak_side_filter_mask
        if personalized_rank_selection and personalized_rank_mode == "energy":
            uniform_reference_kind = "M_i"
            uniform_reference_size = None
        elif personalized_rank_selection:
            uniform_reference_kind = "M"
            uniform_reference_size = personalized_rank_num_effective
        else:
            uniform_reference_kind = "K"
            uniform_reference_size = k
        actual_selection_counts = full_selected_direction_mask.sum(dim=1)
        if not torch.equal(actual_selection_counts, selected_direction_counts):
            raise AssertionError(
                "Diagnostic selection counts must match the production mask."
            )
        actual_selection_counts_before = (
            full_selected_direction_mask_before_repeatability.sum(dim=1)
        )
        if not torch.equal(
            actual_selection_counts_before,
            selected_direction_counts_before_repeatability,
        ):
            raise AssertionError(
                "Diagnostic pre-filter counts must match the production mask."
            )
        non_fallback = ~zero_energy_fallback
        if (
            personalized_rank_selection
            and personalized_rank_force_u1
            and bool(non_fallback.any())
            and not bool(
                full_selected_direction_mask_before_repeatability[
                    non_fallback,
                    0,
                ].all()
            )
        ):
            raise AssertionError(
                "force_u1 requires every non-fallback client to select direction 0."
            )
        u1_selection_rate = float(
            full_selected_direction_mask[:, 0].float().mean().item()
        )
        u1_selection_rate_before = float(
            full_selected_direction_mask_before_repeatability[:, 0]
            .float()
            .mean()
            .item()
        )

        full_masks = []
        full_same_sign_count = torch.zeros(
            (num_clients, rank_r), device=self.device, dtype=torch.long
        )
        full_same_sign_mass = torch.zeros_like(full_a)
        full_b_with_renorm = torch.zeros_like(full_a)
        full_b_without_renorm = torch.zeros_like(full_a)
        for direction_idx in range(rank_r):
            v = full_h[:, direction_idx]
            mask = (v.unsqueeze(1) * v.unsqueeze(0)) > 0
            full_masks.append(mask)
            active = mask.to(alpha_tensor.dtype)
            denominator = active @ alpha_tensor
            numerator = active @ (alpha_tensor * full_a[:, direction_idx])
            full_same_sign_count[:, direction_idx] = mask.sum(dim=1)
            full_same_sign_mass[:, direction_idx] = denominator
            full_b_with_renorm[:, direction_idx] = numerator / (
                denominator + eps
            )
            full_b_without_renorm[:, direction_idx] = numerator

        full_b = (
            full_b_with_renorm.clone()
            if group_renorm
            else full_b_without_renorm.clone()
        )
        full_b_with_renorm[:, :working_direction_count] = (
            group_coefficients_with_renorm
        )
        full_b_without_renorm[:, :working_direction_count] = (
            group_coefficients_without_renorm
        )
        full_b[:, :working_direction_count] = personalized_coefficients
        full_coefficients_same_sign = full_b_without_renorm.clone()
        full_coefficients_same_sign[:, :working_direction_count] = (
            coefficients_same_sign
        )
        full_coefficients_self = full_a.clone()
        full_coefficients_self[:, :working_direction_count] = coefficients_self
        full_coefficients_avg = average_a.unsqueeze(0).expand(
            num_clients,
            -1,
        ).clone()
        full_coefficients_avg[:, :working_direction_count] = coefficients_avg
        if personalized_coeff_mode == "same_sign":
            full_coefficient_mode_values = full_b.clone()
        elif personalized_coeff_mode == "self":
            full_coefficient_mode_values = full_coefficients_self.clone()
        else:
            full_coefficient_mode_values = full_coefficients_avg.clone()
        full_coefficient_mode_values[:, :working_direction_count] = (
            coefficient_mode_values
        )
        # Keep the historical g_times_b diagnostics tied to the same-sign
        # candidate even when the active ablation uses self or DeltaAvg.
        full_gb = full_g * full_b
        full_gb[:, :working_direction_count] = scaled_same_sign_coefficients
        full_selected_gb = full_gb * full_selected_direction_mask.to(full_gb.dtype)
        full_restored_gb = full_gb * gamma_used_values.unsqueeze(1)
        full_restored_selected_gb = (
            full_selected_gb * gamma_used_values.unsqueeze(1)
        )
        full_g_active_coefficients = full_g * full_coefficient_mode_values
        full_g_active_coefficients[:, :working_direction_count] = (
            scaled_personalized_coefficients
        )
        full_output_coefficients = (
            full_g_active_coefficients.clone()
            if (not personalized_rank_selection or personalized_g_scale)
            else full_coefficient_mode_values.clone()
        )
        full_output_coefficients[:, :working_direction_count] = (
            output_personalized_coefficients
        )
        full_selected_output_coefficients = (
            full_output_coefficients
            * full_selected_direction_mask.to(full_output_coefficients.dtype)
        )
        full_selected_output_coefficients[:, :working_direction_count] = (
            selected_output_personalized_coefficients
        )
        full_selected_output_coefficients_before_repeatability = (
            full_output_coefficients
            * full_selected_direction_mask_before_repeatability.to(
                full_output_coefficients.dtype
            )
        )
        full_selected_output_coefficients_before_dominance = (
            full_output_coefficients
            * full_selected_direction_mask_before_dominance.to(
                full_output_coefficients.dtype
            )
        )
        if personalized_rank_selection and personalized_tail_scale != 1.0:
            full_selected_output_coefficients_before_repeatability[:, 1:] *= (
                personalized_tail_scale
            )
            full_selected_output_coefficients_before_dominance[:, 1:] *= (
                personalized_tail_scale
            )
        if bool(fallback_used.any()):
            # These clients bypass the selected-direction reconstruction and
            # return DeltaAvg exactly, so the "actual output" diagnostic must
            # expose DeltaAvg's coefficients rather than an all-zero mask.
            full_selected_output_coefficients[fallback_used] = (
                full_coefficients_avg[fallback_used]
            )
        full_restored_selected_output_coefficients = (
            full_selected_output_coefficients
            * gamma_used_values.unsqueeze(1)
        )

        positive_a = full_a > 0
        negative_a = full_a < 0
        positive_client_count = positive_a.sum(dim=0)
        negative_client_count = negative_a.sum(dim=0)
        positive_weight_mass = (
            positive_a.to(alpha_tensor.dtype) * alpha_tensor.unsqueeze(1)
        ).sum(dim=0)
        negative_weight_mass = (
            negative_a.to(alpha_tensor.dtype) * alpha_tensor.unsqueeze(1)
        ).sum(dim=0)
        positive_weighted_sum = (
            torch.where(positive_a, full_a, torch.zeros_like(full_a))
            * alpha_tensor.unsqueeze(1)
        ).sum(dim=0)
        negative_weighted_sum = (
            torch.where(negative_a, full_a, torch.zeros_like(full_a))
            * alpha_tensor.unsqueeze(1)
        ).sum(dim=0)
        absolute_weighted_sum = (
            torch.abs(full_a) * alpha_tensor.unsqueeze(1)
        ).sum(dim=0)
        cancellation_ratio = torch.abs(average_a) / (
            absolute_weighted_sum + eps
        )

        k5 = min(5, rank_r)
        k10 = min(10, rank_r)
        avg_norm = torch.norm(average_delta)
        client_rows = []
        client_metrics = []
        for target_idx, client_id in enumerate(self.uploaded_ids):
            target_g = full_g[target_idx]
            target_a = full_a[target_idx]
            target_b = full_b[target_idx]
            target_gb = full_gb[target_idx]
            self_norm = raw_norms[target_idx]
            selected_indices_before_tensor = torch.nonzero(
                full_selected_direction_mask_before_repeatability[target_idx],
                as_tuple=False,
            ).flatten()
            selected_indices_before_dominance_tensor = torch.nonzero(
                full_selected_direction_mask_before_dominance[target_idx],
                as_tuple=False,
            ).flatten()
            selected_indices_tensor = torch.nonzero(
                full_selected_direction_mask[target_idx],
                as_tuple=False,
            ).flatten()
            selected_direction_ids_before = [
                int(index.item()) for index in selected_indices_before_tensor
            ]
            selected_direction_ids_before_dominance = [
                int(index.item())
                for index in selected_indices_before_dominance_tensor
            ]
            selected_direction_ids = [
                int(index.item()) for index in selected_indices_tensor
            ]
            selected_scores_tensor = full_direction_scores[
                target_idx, selected_indices_tensor
            ]
            selected_g_tensor = full_g[target_idx, selected_indices_tensor]
            selected_support_count_tensor = full_same_sign_count[
                target_idx, selected_indices_tensor
            ]
            selected_support_mass_tensor = full_same_sign_mass[
                target_idx, selected_indices_tensor
            ]
            selected_scores = [
                float(value.item()) for value in selected_scores_tensor
            ]
            selected_g_values = [
                float(value.item()) for value in selected_g_tensor
            ]
            selected_support_counts = [
                int(value.item()) for value in selected_support_count_tensor
            ]
            selected_support_masses = [
                float(value.item()) for value in selected_support_mass_tensor
            ]
            selected_score_sum = torch.sum(selected_scores_tensor)
            selected_score_sum_before = torch.sum(
                full_direction_scores[
                    target_idx,
                    selected_indices_before_tensor,
                ]
            )
            total_score_sum = torch.sum(full_direction_scores[target_idx])
            zero_energy_fallback_value = bool(
                zero_energy_fallback[target_idx].item()
            )
            repeatability_empty_fallback_value = bool(
                repeatability_empty_fallback[target_idx].item()
            )
            tail_missing_u1_fallback_value = bool(
                tail_missing_u1_fallback[target_idx].item()
            )
            fallback_used_value = bool(fallback_used[target_idx].item())
            if total_score_sum > eps:
                selected_score_ratio = torch.clamp(
                    selected_score_sum / total_score_sum,
                    min=0.0,
                    max=1.0,
                )
                selected_score_ratio_before = torch.clamp(
                    selected_score_sum_before / total_score_sum,
                    min=0.0,
                    max=1.0,
                )
            else:
                selected_score_ratio = torch.zeros_like(total_score_sum)
                selected_score_ratio_before = torch.zeros_like(
                    total_score_sum
                )
            selected_direction_count_before = int(
                selected_direction_counts_before_repeatability[
                    target_idx
                ].item()
            )
            selected_direction_count = int(
                selected_direction_counts[target_idx].item()
            )
            selected_direction_count_before_dominance = int(
                full_selected_direction_mask_before_dominance[
                    target_idx
                ].sum().item()
            )
            dominance_balanced_filtered_count = int(
                full_dominance_balanced_filter_mask[target_idx].sum().item()
            )
            dominance_weak_side_filtered_count = int(
                full_dominance_weak_side_filter_mask[target_idx].sum().item()
            )
            dominance_empty_after_filter_value = bool(
                dominance_empty_after_filter[target_idx].item()
            )
            if selected_direction_count != len(selected_direction_ids):
                raise AssertionError(
                    "Per-client selected_count must match selected direction ids."
                )
            if personalized_rank_selection and personalized_rank_mode == "energy":
                score_tolerance = (
                    torch.finfo(full_direction_scores.dtype).eps
                    * max(rank_r, 1)
                    * torch.abs(total_score_sum)
                )
                energy_threshold_met_before = bool(
                    not zero_energy_fallback_value
                    and selected_score_sum_before + score_tolerance
                    >= personalized_rank_energy * total_score_sum
                )
                energy_threshold_met = bool(
                    not fallback_used_value
                    and selected_score_sum + score_tolerance
                    >= personalized_rank_energy * total_score_sum
                )
            else:
                energy_threshold_met_before = None
                energy_threshold_met = None
            u1_selected_before = bool(
                full_selected_direction_mask_before_repeatability[
                    target_idx,
                    0,
                ].item()
            )
            u1_selected = bool(
                full_selected_direction_mask[target_idx, 0].item()
            )
            if zero_energy_fallback_value:
                u1_score_rank_1based = None
            else:
                score_order = torch.argsort(
                    full_direction_scores[target_idx],
                    descending=True,
                    stable=True,
                )
                u1_score_rank_1based = int(
                    torch.nonzero(score_order == 0, as_tuple=False)[0].item()
                ) + 1
            client_uniform_reference_size = (
                selected_direction_count
                if uniform_reference_size is None
                else uniform_reference_size
            )
            uniform_overlap_count = int(
                torch.sum(
                    selected_indices_tensor < client_uniform_reference_size
                ).item()
            )
            uniform_overlap_ratio = uniform_overlap_count / max(
                selected_direction_count,
                1,
            )
            uniform_k_coverage_ratio = uniform_overlap_count / max(
                client_uniform_reference_size,
                1,
            )
            overlap_union_count = (
                selected_direction_count
                + client_uniform_reference_size
                - uniform_overlap_count
            )
            uniform_k_jaccard = uniform_overlap_count / max(
                overlap_union_count,
                1,
            )
            norm_before_g = torch.norm(
                unscaled_personalized_vecs[target_idx]
            )
            norm_after_g_before_restore = torch.norm(
                personalized_vecs_before_restore[target_idx]
            )
            norm_after_restore = torch.norm(personalized_vecs[target_idx])
            final_to_avg_norm_ratio = norm_after_restore / (avg_norm + eps)
            gamma_capped = bool(
                gamma_raw_values[target_idx] > gamma_used_values[target_idx]
            )
            cos_after_restore_with_avg = self._safe_cosine(
                personalized_vecs[target_idx],
                average_delta,
            )
            cos_after_restore_with_self = self._safe_cosine(
                personalized_vecs[target_idx],
                raw_vecs[target_idx],
            )
            cosine_before_repeatability_with_avg = (
                self._projection_coefficient_cosine(
                    full_selected_output_coefficients_before_repeatability[
                        target_idx
                    ],
                    average_a,
                    avg_norm,
                )
            )
            norm_before_dominance_filter = torch.norm(
                full_selected_output_coefficients_before_dominance[target_idx]
            )
            cosine_before_dominance_with_avg = (
                self._projection_coefficient_cosine(
                    full_selected_output_coefficients_before_dominance[
                        target_idx
                    ],
                    average_a,
                    avg_norm,
                )
            )
            filtered_selected_energy_fraction = float(
                (
                    (selected_score_sum_before - selected_score_sum).clamp_min(0)
                    / (selected_score_sum_before + eps)
                ).item()
            )
            retained_raw_selected_energy_fraction = float(
                (
                    selected_score_sum
                    / (selected_score_sum_before + eps)
                ).clamp(min=0.0, max=1.0).item()
            )

            metrics_k5 = self._projection_prefix_metrics(
                target_a,
                target_b,
                target_gb,
                average_a,
                self_norm,
                avg_norm,
                k5,
            )
            metrics_k10 = self._projection_prefix_metrics(
                target_a,
                target_b,
                target_gb,
                average_a,
                self_norm,
                avg_norm,
                k10,
            )
            metrics_kr = self._projection_prefix_metrics(
                target_a,
                target_b,
                target_gb,
                average_a,
                self_norm,
                avg_norm,
                rank_r,
            )

            increment = target_gb[5:k10]
            increment_norm = torch.norm(increment)
            increment_self_cos = self._projection_coefficient_cosine(
                increment,
                target_a[5:k10],
                self_norm,
            )
            increment_avg_cos = self._projection_coefficient_cosine(
                increment,
                average_a[5:k10],
                avg_norm,
            )
            sign_k5 = torch.zeros(rank_r, device=self.device)
            sign_k5[:k5] = target_gb[:k5]
            increment_full = torch.zeros(rank_r, device=self.device)
            increment_full[5:k10] = increment
            increment_sign_k5_cos = self._safe_cosine(
                increment_full,
                sign_k5,
            )

            coverage_k5 = torch.sum(target_g[:k5].square()).item()
            coverage_k10 = torch.sum(target_g[:k10].square()).item()
            coverage_kr = torch.sum(target_g.square()).item()
            top_count = min(5, rank_r)
            top_indices = torch.topk(target_g, k=top_count).indices
            top_ranks = [int(index.item()) + 1 for index in top_indices]
            top_values = [float(target_g[index].item()) for index in top_indices]
            missed_top_ranks = [rank for rank in top_ranks if rank > k]
            max_g_rank = int(torch.argmax(target_g).item()) + 1
            max_g_after_k = (
                float(torch.max(target_g[k:]).item()) if k < rank_r else 0.0
            )

            metrics = {
                "client_id": client_id,
                "target_g": target_g,
                "target_a": target_a,
                "target_b": target_b,
                "target_gb": target_gb,
                "selected_direction_ids_before": selected_direction_ids_before,
                "selected_direction_ids_before_dominance": (
                    selected_direction_ids_before_dominance
                ),
                "selected_direction_ids": selected_direction_ids,
                "selected_scores": selected_scores,
                "selected_g_values": selected_g_values,
                "selected_support_counts": selected_support_counts,
                "selected_support_masses": selected_support_masses,
                "selected_score_sum": float(selected_score_sum.item()),
                "selected_score_sum_before": float(
                    selected_score_sum_before.item()
                ),
                "total_score_sum": float(total_score_sum.item()),
                "selected_score_ratio": float(selected_score_ratio.item()),
                "selected_score_ratio_before": float(
                    selected_score_ratio_before.item()
                ),
                "selected_count": selected_direction_count,
                "selected_count_before": selected_direction_count_before,
                "selected_count_before_dominance": (
                    selected_direction_count_before_dominance
                ),
                "dominance_balanced_filtered_count": (
                    dominance_balanced_filtered_count
                ),
                "dominance_weak_side_filtered_count": (
                    dominance_weak_side_filtered_count
                ),
                "dominance_empty_after_filter": (
                    dominance_empty_after_filter_value
                ),
                "energy_threshold_met": energy_threshold_met,
                "energy_threshold_met_before": energy_threshold_met_before,
                "zero_energy_fallback": zero_energy_fallback_value,
                "repeatability_empty_fallback": (
                    repeatability_empty_fallback_value
                ),
                "tail_missing_u1_fallback": (
                    tail_missing_u1_fallback_value
                ),
                "fallback_used": fallback_used_value,
                "u1_selected": u1_selected,
                "u1_selected_before": u1_selected_before,
                "u1_score_rank_1based": u1_score_rank_1based,
                "uniform_reference_kind": uniform_reference_kind,
                "uniform_reference_size": client_uniform_reference_size,
                "uniform_overlap_count": uniform_overlap_count,
                "uniform_overlap_ratio": uniform_overlap_ratio,
                "uniform_k_coverage_ratio": uniform_k_coverage_ratio,
                "uniform_k_jaccard": uniform_k_jaccard,
                "coverage_k5": coverage_k5,
                "coverage_k10": coverage_k10,
                "coverage_kr": coverage_kr,
                "top_ranks": top_ranks,
                "top_values": top_values,
                "missed_top_ranks": missed_top_ranks,
                "max_g_after_k": max_g_after_k,
                "metrics_k5": metrics_k5,
                "metrics_k10": metrics_k10,
                "metrics_kr": metrics_kr,
                "increment_norm": float(increment_norm.item()),
                "increment_self_cos": increment_self_cos,
                "increment_avg_cos": increment_avg_cos,
                "increment_sign_k5_cos": increment_sign_k5_cos,
                "norm_before_g": float(norm_before_g.item()),
                "norm_after_g_before_restore": float(
                    norm_after_g_before_restore.item()
                ),
                "gamma_raw": float(gamma_raw_values[target_idx].item()),
                "gamma_used": float(gamma_used_values[target_idx].item()),
                "gamma_capped": gamma_capped,
                "norm_after_restore": float(norm_after_restore.item()),
                "final_to_avg_norm_ratio": float(
                    final_to_avg_norm_ratio.item()
                ),
                "cos_after_restore_with_avg": cos_after_restore_with_avg,
                "cos_after_restore_with_self": cos_after_restore_with_self,
                "cosine_before_repeatability_with_avg": (
                    cosine_before_repeatability_with_avg
                ),
                "norm_before_dominance_filter": float(
                    norm_before_dominance_filter.item()
                ),
                "cosine_before_dominance_with_avg": (
                    cosine_before_dominance_with_avg
                ),
                "filtered_selected_energy_fraction": (
                    filtered_selected_energy_fraction
                ),
                "retained_raw_selected_energy_fraction": (
                    retained_raw_selected_energy_fraction
                ),
            }
            client_metrics.append(metrics)
            diagnostic_rank_num_requested = (
                None
                if (
                    personalized_rank_selection
                    and personalized_rank_mode == "energy"
                )
                else personalized_rank_num_requested
            )
            diagnostic_rank_num_effective = (
                selected_direction_count
                if (
                    personalized_rank_selection
                    and personalized_rank_mode == "energy"
                )
                else personalized_rank_num_effective
            )
            client_rows.append({
                "round": self.cur_ground,
                "layer": name,
                "layer_name": name,
                "client_id": client_id,
                "aggregation_mode": mode_name,
                "projection_layer_scope": projection_layer_scope,
                "projection_input_kind": input_kind,
                "average_reference_semantics": average_reference_semantics,
                "personalized_output_semantics": (
                    personalized_output_semantics
                ),
                "norm_restore_reference": average_reference_semantics,
                "personalized_writeback_semantics": (
                    personalized_writeback_semantics
                ),
                "global_writeback_semantics": global_writeback_semantics,
                "start_weight_added": int(not uses_full_weights),
                "group_renorm": int(group_renorm),
                "norm_restore": int(norm_restore),
                "personalized_rank_selection": int(personalized_rank_selection),
                "personalized_rank_mode": personalized_rank_mode,
                "personalized_rank_energy": personalized_rank_energy,
                "personalized_g_scale": int(personalized_g_scale),
                "local_update_views": local_update_views,
                "personalized_repeatability_threshold": (
                    repeatability_threshold
                ),
                "personalized_coeff_mode": personalized_coeff_mode,
                "personalized_tail_scale": personalized_tail_scale,
                "personalized_m_filter_mode": personalized_m_filter_mode,
                "personalized_dominance_threshold": (
                    personalized_dominance_threshold
                ),
                "personalized_rank_num_requested": diagnostic_rank_num_requested,
                "personalized_rank_num_effective": diagnostic_rank_num_effective,
                "personalized_rank_force_u1": int(personalized_rank_force_u1),
                "rank_R": rank_r,
                "selected_K": k,
                "selected_count": selected_direction_count,
                "selected_count_before": selected_direction_count_before,
                "selected_count_after": selected_direction_count,
                "selected_count_raw": selected_direction_count_before,
                "selected_count_before_dominance": (
                    selected_direction_count_before_dominance
                ),
                "selected_count_after_m_filter": selected_direction_count,
                "dominance_balanced_filtered_count": (
                    dominance_balanced_filtered_count
                ),
                "dominance_weak_side_filtered_count": (
                    dominance_weak_side_filtered_count
                ),
                "dominance_empty_after_filter": int(
                    dominance_empty_after_filter_value
                ),
                "layer_dominance_balanced_filtered_direction_count": (
                    dominance_balanced_filtered_direction_count
                ),
                "layer_dominance_weak_side_filtered_client_direction_count": (
                    dominance_weak_side_filtered_client_direction_count
                ),
                "energy_ratio_before": float(
                    selected_score_ratio_before.item()
                ),
                "energy_ratio_after": float(selected_score_ratio.item()),
                "energy_threshold_met": (
                    int(energy_threshold_met)
                    if energy_threshold_met is not None
                    else None
                ),
                "energy_threshold_met_before": (
                    int(energy_threshold_met_before)
                    if energy_threshold_met_before is not None
                    else None
                ),
                "zero_energy_fallback": int(zero_energy_fallback_value),
                "repeatability_empty_fallback": int(
                    repeatability_empty_fallback_value
                ),
                "tail_missing_u1_fallback": int(
                    tail_missing_u1_fallback_value
                ),
                "fallback_used": int(fallback_used_value),
                "u1_selected": int(u1_selected),
                "u1_selected_before_repeatability": int(
                    u1_selected_before
                ),
                "u1_score_rank_1based": u1_score_rank_1based,
                "selected_direction_ids_0based": self._csv_sequence(
                    selected_direction_ids
                ),
                "selected_direction_ids_before_repeatability": self._csv_sequence(
                    selected_direction_ids_before
                ),
                "selected_direction_ids_before_dominance": self._csv_sequence(
                    selected_direction_ids_before_dominance
                ),
                "selected_direction_ids": self._csv_sequence(
                    selected_direction_ids
                ),
                "selected_direction_scores": self._csv_sequence(selected_scores),
                "selected_direction_g": self._csv_sequence(selected_g_values),
                "selected_direction_support_counts": self._csv_sequence(
                    selected_support_counts
                ),
                "selected_direction_support_masses": self._csv_sequence(
                    selected_support_masses
                ),
                "selected_score_sum": float(selected_score_sum.item()),
                "selected_score_sum_before": float(
                    selected_score_sum_before.item()
                ),
                "total_score_sum": float(total_score_sum.item()),
                "selected_score_ratio": float(selected_score_ratio.item()),
                "selected_energy_ratio": float(selected_score_ratio.item()),
                "singular_values": singular_values_csv,
                "singular_energy_ratios": singular_energy_ratios_csv,
                "cumulative_energy": cumulative_energy_csv,
                "filtered_selected_energy_fraction": (
                    filtered_selected_energy_fraction
                ),
                "retained_raw_local_energy_ratio": float(
                    selected_score_ratio.item()
                ),
                "retained_raw_selected_energy_fraction": (
                    retained_raw_selected_energy_fraction
                ),
                "uniform_reference_kind": uniform_reference_kind,
                "uniform_reference_size": client_uniform_reference_size,
                "uniform_K_overlap_count": uniform_overlap_count,
                "uniform_K_overlap_ratio": uniform_overlap_ratio,
                "uniform_K_coverage_ratio": uniform_k_coverage_ratio,
                "uniform_K_jaccard": uniform_k_jaccard,
                "uniform_M_overlap_count": (
                    uniform_overlap_count if personalized_rank_selection else None
                ),
                "uniform_M_overlap_ratio": (
                    uniform_overlap_ratio if personalized_rank_selection else None
                ),
                "uniform_M_coverage_ratio": (
                    uniform_k_coverage_ratio
                    if personalized_rank_selection
                    else None
                ),
                "uniform_M_jaccard": (
                    uniform_k_jaccard if personalized_rank_selection else None
                ),
                "coverage_K5": coverage_k5,
                "coverage_K10": coverage_k10,
                "coverage_KR": coverage_kr,
                "residual_K5": 1.0 - coverage_k5,
                "residual_K10": 1.0 - coverage_k10,
                "residual_KR": 1.0 - coverage_kr,
                "top5_g_direction_indices": self._csv_sequence(top_ranks),
                "top5_g_values": self._csv_sequence(top_values),
                "top5_g_not_selected": self._csv_sequence(missed_top_ranks),
                "max_g": float(torch.max(target_g).item()),
                "max_g_rank": max_g_rank,
                "max_g_selected": int(max_g_rank <= k),
                "max_g_selected_by_client": int(
                    full_selected_direction_mask[
                        target_idx,
                        max_g_rank - 1,
                    ].item()
                ),
                "max_g_after_K": max_g_after_k,
                "mean_g": float(torch.mean(target_g).item()),
                "min_g": float(torch.min(target_g).item()),
                "norm_self": float(self_norm.item()),
                "norm_avg": float(avg_norm.item()),
                "norm_delta_avg": (
                    None if uses_full_weights else float(avg_norm.item())
                ),
                "weight_norm": (
                    float(self_norm.item()) if uses_full_weights else None
                ),
                "avg_weight_norm": (
                    float(avg_norm.item()) if uses_full_weights else None
                ),
                "norm_before_g": float(norm_before_g.item()),
                "update_norm_before_m_filter": (
                    None
                    if uses_full_weights
                    else float(norm_before_dominance_filter.item())
                ),
                "update_norm_after_m_filter_before_restore": (
                    None
                    if uses_full_weights
                    else float(norm_after_g_before_restore.item())
                ),
                "norm_after_g_before_restore": float(
                    norm_after_g_before_restore.item()
                ),
                "update_norm_before_restore": (
                    None
                    if uses_full_weights
                    else float(norm_after_g_before_restore.item())
                ),
                "projected_norm_before_restore": (
                    None
                    if uses_full_weights
                    else float(norm_after_g_before_restore.item())
                ),
                "projected_weight_norm_before_restore": (
                    float(norm_after_g_before_restore.item())
                    if uses_full_weights
                    else None
                ),
                "gamma_raw": float(gamma_raw_values[target_idx].item()),
                "gamma_used": float(gamma_used_values[target_idx].item()),
                "gamma": float(gamma_used_values[target_idx].item()),
                "gamma_capped": int(gamma_capped),
                "norm_after_restore": float(norm_after_restore.item()),
                "update_norm_after_restore": (
                    None
                    if uses_full_weights
                    else float(norm_after_restore.item())
                ),
                "projected_norm_after_restore": (
                    None
                    if uses_full_weights
                    else float(norm_after_restore.item())
                ),
                "projected_weight_norm_after_restore": (
                    float(norm_after_restore.item())
                    if uses_full_weights
                    else None
                ),
                "aggregated_weight_norm": (
                    float(norm_after_restore.item())
                    if uses_full_weights
                    else None
                ),
                "final_to_avg_weight_norm_ratio": (
                    float(final_to_avg_norm_ratio.item())
                    if uses_full_weights
                    else None
                ),
                "final_to_delta_avg_norm_ratio": (
                    None
                    if uses_full_weights
                    else float(final_to_avg_norm_ratio.item())
                ),
                "cos_after_restore_with_avg": cos_after_restore_with_avg,
                "cos_after_restore_with_self": cos_after_restore_with_self,
                "cosine_with_delta_avg": (
                    None if uses_full_weights else cos_after_restore_with_avg
                ),
                "cos_final_delta_avg": (
                    None if uses_full_weights else cos_after_restore_with_avg
                ),
                "cos_final_avg_weight": (
                    cos_after_restore_with_avg if uses_full_weights else None
                ),
                "cosine_with_client_A": cos_after_restore_with_self,
                "cosine_before_m_filter_with_delta_avg": (
                    None
                    if uses_full_weights
                    else cosine_before_dominance_with_avg
                ),
                "cosine_after_m_filter_with_delta_avg": (
                    None if uses_full_weights else cos_after_restore_with_avg
                ),
                "cosine_before_repeatability_with_delta_avg": (
                    None
                    if uses_full_weights
                    else cosine_before_repeatability_with_avg
                ),
                "cosine_after_repeatability_with_delta_avg": (
                    None if uses_full_weights else cos_after_restore_with_avg
                ),
                "cosine_before_repeatability_with_avg_weight": (
                    cosine_before_repeatability_with_avg
                    if uses_full_weights
                    else None
                ),
                "cosine_after_repeatability_with_avg_weight": (
                    cos_after_restore_with_avg if uses_full_weights else None
                ),
                "norm_selfproj_K5": metrics_k5["norm_self_projection"],
                "norm_selfproj_K10": metrics_k10["norm_self_projection"],
                "norm_selfproj_KR": metrics_kr["norm_self_projection"],
                "norm_selfresidual_K5": metrics_k5["norm_self_residual"],
                "norm_selfresidual_K10": metrics_k10["norm_self_residual"],
                "norm_selfresidual_KR": metrics_kr["norm_self_residual"],
                "norm_sign_K5": metrics_k5["norm_sign"],
                "norm_sign_K10": metrics_k10["norm_sign"],
                "norm_sign_KR": metrics_kr["norm_sign"],
                "norm_gsign_K5": metrics_k5["norm_gsign"],
                "norm_gsign_K10": metrics_k10["norm_gsign"],
                "norm_gsign_KR": metrics_kr["norm_gsign"],
                "cos_selfproj_self_K5": metrics_k5["cos_selfproj_self"],
                "cos_selfproj_self_K10": metrics_k10["cos_selfproj_self"],
                "cos_selfproj_self_KR": metrics_kr["cos_selfproj_self"],
                "cos_sign_self_K5": metrics_k5["cos_sign_self"],
                "cos_sign_self_K10": metrics_k10["cos_sign_self"],
                "cos_sign_self_KR": metrics_kr["cos_sign_self"],
                "cos_gsign_self_K5": metrics_k5["cos_gsign_self"],
                "cos_gsign_self_K10": metrics_k10["cos_gsign_self"],
                "cos_gsign_self_KR": metrics_kr["cos_gsign_self"],
                "cos_selfproj_avg_K5": metrics_k5["cos_selfproj_avg"],
                "cos_selfproj_avg_K10": metrics_k10["cos_selfproj_avg"],
                "cos_selfproj_avg_KR": metrics_kr["cos_selfproj_avg"],
                "cos_sign_avg_K5": metrics_k5["cos_sign_avg"],
                "cos_sign_avg_K10": metrics_k10["cos_sign_avg"],
                "cos_sign_avg_KR": metrics_kr["cos_sign_avg"],
                "cos_gsign_avg_K5": metrics_k5["cos_gsign_avg"],
                "cos_gsign_avg_K10": metrics_k10["cos_gsign_avg"],
                "cos_gsign_avg_KR": metrics_kr["cos_gsign_avg"],
                "norm_increment_6_10": float(increment_norm.item()),
                "cos_increment_self": increment_self_cos,
                "cos_increment_avg": increment_avg_cos,
                "cos_increment_sign_K5": increment_sign_k5_cos,
            })

        direction_rows = []
        for target_idx, client_id in enumerate(self.uploaded_ids):
            selection_metrics = client_metrics[target_idx]
            for direction_idx in range(rank_r):
                same_sign_mass = full_same_sign_mass[
                    target_idx, direction_idx
                ]
                weight_sum_before_g = same_sign_mass / (same_sign_mass + eps)
                weight_sum_after_g = (
                    full_g[target_idx, direction_idx] * weight_sum_before_g
                )
                avg_coefficient = average_a[direction_idx]
                sign_coefficient = full_b[target_idx, direction_idx]
                final_coefficient = full_gb[target_idx, direction_idx]
                active_g_coefficient = full_g_active_coefficients[
                    target_idx,
                    direction_idx,
                ]
                selected_final_coefficient = full_selected_gb[
                    target_idx, direction_idx
                ]
                output_coefficient = full_output_coefficients[
                    target_idx, direction_idx
                ]
                selected_output_coefficient = full_selected_output_coefficients[
                    target_idx, direction_idx
                ]
                restored_coefficient = full_restored_gb[
                    target_idx, direction_idx
                ]
                restored_selected_coefficient = full_restored_selected_gb[
                    target_idx, direction_idx
                ]
                restored_selected_output_coefficient = (
                    full_restored_selected_output_coefficients[
                        target_idx,
                        direction_idx,
                    ]
                )
                ratio_denominator = torch.abs(avg_coefficient) + eps
                all_clients_same_sign = bool(
                    full_same_sign_count[target_idx, direction_idx].item()
                    == num_clients
                )
                dominant_sign_value = int(
                    full_dominant_sign[direction_idx].item()
                )
                dominant_sign_label = {
                    -1: "negative",
                    0: "none",
                    1: "positive",
                }[dominant_sign_value]
                direction_rows.append({
                    "round": self.cur_ground,
                    "layer": name,
                    "layer_name": name,
                    "client_id": client_id,
                    "aggregation_mode": mode_name,
                    "projection_layer_scope": projection_layer_scope,
                    "projection_input_kind": input_kind,
                    "average_reference_semantics": (
                        average_reference_semantics
                    ),
                    "personalized_output_semantics": (
                        personalized_output_semantics
                    ),
                    "norm_restore_reference": average_reference_semantics,
                    "personalized_writeback_semantics": (
                        personalized_writeback_semantics
                    ),
                    "global_writeback_semantics": global_writeback_semantics,
                    "start_weight_added": int(not uses_full_weights),
                    "group_renorm": int(group_renorm),
                    "norm_restore": int(norm_restore),
                    "personalized_rank_selection": int(
                        personalized_rank_selection
                    ),
                    "personalized_rank_mode": personalized_rank_mode,
                    "personalized_rank_energy": personalized_rank_energy,
                    "personalized_g_scale": int(personalized_g_scale),
                    "local_update_views": local_update_views,
                    "personalized_repeatability_threshold": (
                        repeatability_threshold
                    ),
                    "personalized_coeff_mode": personalized_coeff_mode,
                    "personalized_tail_scale": personalized_tail_scale,
                    "personalized_m_filter_mode": personalized_m_filter_mode,
                    "personalized_dominance_threshold": (
                        personalized_dominance_threshold
                    ),
                    "personalized_rank_num_requested": (
                        None
                        if (
                            personalized_rank_selection
                            and personalized_rank_mode == "energy"
                        )
                        else personalized_rank_num_requested
                    ),
                    "personalized_rank_num_effective": (
                        selection_metrics["selected_count"]
                        if (
                            personalized_rank_selection
                            and personalized_rank_mode == "energy"
                        )
                        else personalized_rank_num_effective
                    ),
                    "personalized_rank_force_u1": int(
                        personalized_rank_force_u1
                    ),
                    "direction_id_0based": direction_idx,
                    "direction_index": direction_idx,
                    "k": direction_idx + 1,
                    "rank_R": rank_r,
                    "selected_K": k,
                    "selected_count": selection_metrics["selected_count"],
                    "selected_count_before": selection_metrics[
                        "selected_count_before"
                    ],
                    "selected_count_after": selection_metrics[
                        "selected_count"
                    ],
                    "selected_count_raw": selection_metrics[
                        "selected_count_before"
                    ],
                    "selected_count_before_dominance": selection_metrics[
                        "selected_count_before_dominance"
                    ],
                    "selected_count_after_m_filter": selection_metrics[
                        "selected_count"
                    ],
                    "dominance_balanced_filtered_count": selection_metrics[
                        "dominance_balanced_filtered_count"
                    ],
                    "dominance_weak_side_filtered_count": selection_metrics[
                        "dominance_weak_side_filtered_count"
                    ],
                    "dominance_empty_after_filter": int(
                        selection_metrics["dominance_empty_after_filter"]
                    ),
                    "layer_dominance_balanced_filtered_direction_count": (
                        dominance_balanced_filtered_direction_count
                    ),
                    "layer_dominance_weak_side_filtered_client_direction_count": (
                        dominance_weak_side_filtered_client_direction_count
                    ),
                    "selected_before_repeatability": int(
                        full_selected_direction_mask_before_repeatability[
                            target_idx,
                            direction_idx,
                        ].item()
                    ),
                    "selected_after_repeatability": int(
                        full_selected_direction_mask_before_dominance[
                            target_idx,
                            direction_idx,
                        ].item()
                    ),
                    "selected_in_raw_m": int(
                        full_selected_direction_mask_before_repeatability[
                            target_idx,
                            direction_idx,
                        ].item()
                    ),
                    "selected_before_dominance": int(
                        full_selected_direction_mask_before_dominance[
                            target_idx,
                            direction_idx,
                        ].item()
                    ),
                    "selected_after_m_filter": int(
                        full_selected_direction_mask[
                            target_idx,
                            direction_idx,
                        ].item()
                    ),
                    "dominance_positive_energy": (
                        float(full_dominance_positive_energy[direction_idx].item())
                        if dominance_filter_enabled
                        else None
                    ),
                    "dominance_negative_energy": (
                        float(full_dominance_negative_energy[direction_idx].item())
                        if dominance_filter_enabled
                        else None
                    ),
                    "dominance_ratio": (
                        float(full_dominance_ratio[direction_idx].item())
                        if dominance_filter_enabled
                        else None
                    ),
                    "dominant_sign": (
                        dominant_sign_label if dominance_filter_enabled else None
                    ),
                    "dominant_sign_numeric": (
                        dominant_sign_value if dominance_filter_enabled else None
                    ),
                    "dominance_keep_for_client": (
                        int(
                            full_dominance_keep_mask[
                                target_idx,
                                direction_idx,
                            ].item()
                        )
                        if dominance_filter_enabled
                        else None
                    ),
                    "dominance_filtered_balanced": int(
                        full_dominance_balanced_filter_mask[
                            target_idx,
                            direction_idx,
                        ].item()
                    ),
                    "dominance_filtered_weak_side": int(
                        full_dominance_weak_side_filter_mask[
                            target_idx,
                            direction_idx,
                        ].item()
                    ),
                    "selected_direction_ids": self._csv_sequence(
                        selection_metrics["selected_direction_ids"]
                    ),
                    "selected_score_sum": selection_metrics[
                        "selected_score_sum"
                    ],
                    "total_score_sum": selection_metrics[
                        "total_score_sum"
                    ],
                    "energy_threshold_met": (
                        int(selection_metrics["energy_threshold_met"])
                        if selection_metrics["energy_threshold_met"] is not None
                        else None
                    ),
                    "energy_threshold_met_before": (
                        int(selection_metrics["energy_threshold_met_before"])
                        if selection_metrics["energy_threshold_met_before"]
                        is not None
                        else None
                    ),
                    "energy_ratio_before": selection_metrics[
                        "selected_score_ratio_before"
                    ],
                    "energy_ratio_after": selection_metrics[
                        "selected_score_ratio"
                    ],
                    "filtered_selected_energy_fraction": selection_metrics[
                        "filtered_selected_energy_fraction"
                    ],
                    "retained_raw_local_energy_ratio": selection_metrics[
                        "selected_score_ratio"
                    ],
                    "retained_raw_selected_energy_fraction": selection_metrics[
                        "retained_raw_selected_energy_fraction"
                    ],
                    "update_norm_before_m_filter": (
                        selection_metrics["norm_before_dominance_filter"]
                        if not uses_full_weights
                        else None
                    ),
                    "update_norm_after_m_filter_before_restore": (
                        selection_metrics["norm_after_g_before_restore"]
                        if not uses_full_weights
                        else None
                    ),
                    "cosine_before_m_filter_with_delta_avg": (
                        selection_metrics["cosine_before_dominance_with_avg"]
                        if not uses_full_weights
                        else None
                    ),
                    "cosine_after_m_filter_with_delta_avg": (
                        selection_metrics["cos_after_restore_with_avg"]
                        if not uses_full_weights
                        else None
                    ),
                    "zero_energy_fallback": int(
                        selection_metrics["zero_energy_fallback"]
                    ),
                    "repeatability_empty_fallback": int(
                        selection_metrics["repeatability_empty_fallback"]
                    ),
                    "tail_missing_u1_fallback": int(
                        selection_metrics["tail_missing_u1_fallback"]
                    ),
                    "fallback_used": int(
                        selection_metrics["fallback_used"]
                    ),
                    "u1_selected": int(selection_metrics["u1_selected"]),
                    "selected_by_current_K": int(direction_idx < k),
                    "uniform_reference_kind": selection_metrics[
                        "uniform_reference_kind"
                    ],
                    "uniform_reference_size": selection_metrics[
                        "uniform_reference_size"
                    ],
                    "selected_by_uniform_reference": int(
                        direction_idx
                        < selection_metrics["uniform_reference_size"]
                    ),
                    "selected_by_client": int(
                        full_selected_direction_mask[
                            target_idx, direction_idx
                        ].item()
                    ),
                    "direction_score": float(
                        full_direction_scores[target_idx, direction_idx].item()
                    ),
                    "client_energy_score": float(
                        full_direction_scores[target_idx, direction_idx].item()
                    ),
                    "energy_rank": int(
                        full_energy_ranks[target_idx, direction_idx].item()
                    ),
                    "selected_score_ratio": selection_metrics[
                        "selected_score_ratio"
                    ],
                    "selected_energy_ratio": selection_metrics[
                        "selected_score_ratio"
                    ],
                    "uniform_K_overlap_ratio": selection_metrics[
                        "uniform_overlap_ratio"
                    ],
                    "uniform_K_overlap_count": selection_metrics[
                        "uniform_overlap_count"
                    ],
                    "uniform_K_coverage_ratio": selection_metrics[
                        "uniform_k_coverage_ratio"
                    ],
                    "uniform_K_jaccard": selection_metrics[
                        "uniform_k_jaccard"
                    ],
                    "uniform_M_overlap_ratio": (
                        selection_metrics["uniform_overlap_ratio"]
                        if personalized_rank_selection
                        else None
                    ),
                    "uniform_M_overlap_count": (
                        selection_metrics["uniform_overlap_count"]
                        if personalized_rank_selection
                        else None
                    ),
                    "uniform_M_coverage_ratio": (
                        selection_metrics["uniform_k_coverage_ratio"]
                        if personalized_rank_selection
                        else None
                    ),
                    "uniform_M_jaccard": (
                        selection_metrics["uniform_k_jaccard"]
                        if personalized_rank_selection
                        else None
                    ),
                    "sigma": float(full_sigma[direction_idx].item()),
                    "singular_value": float(
                        full_sigma[direction_idx].item()
                    ),
                    "energy": float(full_energy[direction_idx].item()),
                    "cumulative_energy": float(full_cumulative[direction_idx].item()),
                    "g": float(full_g[target_idx, direction_idx].item()),
                    "a_self": float(full_a[target_idx, direction_idx].item()),
                    "a_A_raw": float(
                        full_a[target_idx, direction_idx].item()
                    ),
                    "a_A_weight": (
                        float(full_a[target_idx, direction_idx].item())
                        if uses_full_weights
                        else None
                    ),
                    "a_B_raw": (
                        float(full_a_b[target_idx, direction_idx].item())
                        if raw_vecs_b is not None
                        and direction_idx < working_direction_count
                        else None
                    ),
                    "a_B_weight": (
                        float(full_a_b[target_idx, direction_idx].item())
                        if uses_full_weights
                        and raw_vecs_b is not None
                        and direction_idx < working_direction_count
                        else None
                    ),
                    "a_A_normalized": float(
                        full_normalized_a[target_idx, direction_idx].item()
                    ),
                    "a_B_normalized": (
                        float(
                            full_normalized_a_b[
                                target_idx,
                                direction_idx,
                            ].item()
                        )
                        if raw_vecs_b is not None
                        and direction_idx < working_direction_count
                        else None
                    ),
                    "repeatability_raw": (
                        float(
                            full_repeatability_raw[
                                target_idx,
                                direction_idx,
                            ].item()
                        )
                        if raw_vecs_b is not None
                        and direction_idx < working_direction_count
                        else None
                    ),
                    "repeatability_normalized": (
                        float(
                            full_repeatability_normalized[
                                target_idx,
                                direction_idx,
                            ].item()
                        )
                        if raw_vecs_b is not None
                        and direction_idx < working_direction_count
                        else None
                    ),
                    "a_avg": float(avg_coefficient.item()),
                    "a_avg_weight": (
                        float(avg_coefficient.item())
                        if uses_full_weights
                        else None
                    ),
                    "b_sign": float(sign_coefficient.item()),
                    "coeff_same_sign": float(
                        full_coefficients_same_sign[
                            target_idx,
                            direction_idx,
                        ].item()
                    ),
                    "coeff_self": float(
                        full_coefficients_self[
                            target_idx,
                            direction_idx,
                        ].item()
                    ),
                    "coeff_avg": float(
                        full_coefficients_avg[
                            target_idx,
                            direction_idx,
                        ].item()
                    ),
                    "coeff_after_mode": float(
                        full_coefficient_mode_values[
                            target_idx,
                            direction_idx,
                        ].item()
                    ),
                    "same_sign_over_self_abs": float(
                        (
                            torch.abs(
                                full_coefficients_same_sign[
                                    target_idx,
                                    direction_idx,
                                ]
                            )
                            / (
                                torch.abs(
                                    full_coefficients_self[
                                        target_idx,
                                        direction_idx,
                                    ]
                                )
                                + eps
                            )
                        ).item()
                    ),
                    "same_sign_over_avg_abs": float(
                        (
                            torch.abs(
                                full_coefficients_same_sign[
                                    target_idx,
                                    direction_idx,
                                ]
                            )
                            / (
                                torch.abs(
                                    full_coefficients_avg[
                                        target_idx,
                                        direction_idx,
                                    ]
                                )
                                + eps
                            )
                        ).item()
                    ),
                    "group_coeff_with_renorm": float(
                        full_b_with_renorm[target_idx, direction_idx].item()
                    ),
                    "group_coeff_without_renorm": float(
                        full_b_without_renorm[target_idx, direction_idx].item()
                    ),
                    "g_times_b": float(final_coefficient.item()),
                    "g_times_active_coeff": float(
                        active_g_coefficient.item()
                    ),
                    "g_times_b_after_selection": float(
                        selected_final_coefficient.item()
                    ),
                    "output_coefficient_before_selection": float(
                        output_coefficient.item()
                    ),
                    "output_coefficient_after_selection": float(
                        selected_output_coefficient.item()
                    ),
                    "final_coeff_before_restore": float(
                        selected_output_coefficient.item()
                    ),
                    "final_coeff": float(
                        restored_selected_output_coefficient.item()
                    ),
                    "tail_scale": personalized_tail_scale,
                    "gamma_used": float(gamma_used_values[target_idx].item()),
                    "gamma_raw": float(gamma_raw_values[target_idx].item()),
                    "gamma": float(gamma_used_values[target_idx].item()),
                    "gamma_capped": int(
                        gamma_raw_values[target_idx]
                        > gamma_used_values[target_idx]
                    ),
                    "weight_norm": (
                        float(raw_norms[target_idx].item())
                        if uses_full_weights
                        else None
                    ),
                    "avg_weight_norm": (
                        float(avg_norm.item()) if uses_full_weights else None
                    ),
                    "projected_weight_norm_before_restore": (
                        float(
                            torch.norm(
                                personalized_vecs_before_restore[target_idx]
                            ).item()
                        )
                        if uses_full_weights
                        else None
                    ),
                    "projected_weight_norm_after_restore": (
                        float(torch.norm(personalized_vecs[target_idx]).item())
                        if uses_full_weights
                        else None
                    ),
                    "final_to_avg_weight_norm_ratio": (
                        float(
                            (
                                torch.norm(personalized_vecs[target_idx])
                                / (avg_norm + eps)
                            ).item()
                        )
                        if uses_full_weights
                        else None
                    ),
                    "cos_final_avg_weight": (
                        self._safe_cosine(
                            personalized_vecs[target_idx],
                            average_delta,
                        )
                        if uses_full_weights
                        else None
                    ),
                    "coefficient_after_restore": float(
                        restored_coefficient.item()
                    ),
                    "coefficient_after_selection_and_restore": float(
                        restored_selected_output_coefficient.item()
                    ),
                    "counterfactual_gb_after_selection_and_restore": float(
                        restored_selected_coefficient.item()
                    ),
                    "counterfactual_same_sign_gb_after_selection_and_restore": float(
                        restored_selected_coefficient.item()
                    ),
                    "same_sign_count": int(
                        full_same_sign_count[target_idx, direction_idx].item()
                    ),
                    "same_sign_mass": float(same_sign_mass.item()),
                    "support_count": int(
                        full_same_sign_count[target_idx, direction_idx].item()
                    ),
                    "support_mass": float(same_sign_mass.item()),
                    "positive_client_count": int(
                        positive_client_count[direction_idx].item()
                    ),
                    "negative_client_count": int(
                        negative_client_count[direction_idx].item()
                    ),
                    "positive_weight_mass": float(
                        positive_weight_mass[direction_idx].item()
                    ),
                    "negative_weight_mass": float(
                        negative_weight_mass[direction_idx].item()
                    ),
                    "positive_weighted_sum": float(
                        positive_weighted_sum[direction_idx].item()
                    ),
                    "negative_weighted_sum": float(
                        negative_weighted_sum[direction_idx].item()
                    ),
                    "cancellation_ratio": float(
                        cancellation_ratio[direction_idx].item()
                    ),
                    "sign_amplification": float(
                        (torch.abs(sign_coefficient) / ratio_denominator).item()
                    ),
                    "final_ratio": float(
                        (torch.abs(final_coefficient) / ratio_denominator).item()
                    ),
                    "weight_sum_before_g": float(weight_sum_before_g.item()),
                    "weight_sum_after_g": float(weight_sum_after_g.item()),
                    "all_clients_same_sign": int(all_clients_same_sign),
                    "abs_b_minus_avg": float(
                        torch.abs(sign_coefficient - avg_coefficient).item()
                    ),
                    "abs_gb_minus_avg": float(
                        torch.abs(final_coefficient - avg_coefficient).item()
                    ),
                })

        client_csv = self.projection_client_diagnostic_csv
        direction_csv = self.projection_direction_diagnostic_csv
        self._append_projection_diagnostic_rows(client_csv, client_rows)
        self._append_projection_diagnostic_rows(direction_csv, direction_rows)
        if not self._projection_diagnostic_paths_printed:
            print(f"Projection 客户端诊断 CSV: {client_csv}")
            print(f"Projection 方向诊断 CSV: {direction_csv}")
            self._projection_diagnostic_paths_printed = True

        if console_diagnostics:
            self._print_projection_console_diagnostics(
                name=name,
                rank_r=rank_r,
                selected_k=k,
                full_sigma=full_sigma,
                full_energy=full_energy,
                full_cumulative=full_cumulative,
                full_h=full_h,
                full_a=full_a,
                average_a=average_a,
                full_g=full_g,
                full_b=full_b,
                full_b_with_renorm=full_b_with_renorm,
                full_b_without_renorm=full_b_without_renorm,
                full_gb=full_gb,
                full_output_coefficients=full_output_coefficients,
                full_masks=full_masks,
                full_same_sign_count=full_same_sign_count,
                full_same_sign_mass=full_same_sign_mass,
                positive_client_count=positive_client_count,
                negative_client_count=negative_client_count,
                positive_weight_mass=positive_weight_mass,
                negative_weight_mass=negative_weight_mass,
                positive_weighted_sum=positive_weighted_sum,
                negative_weighted_sum=negative_weighted_sum,
                cancellation_ratio=cancellation_ratio,
                client_metrics=client_metrics,
                raw_norms=raw_norms,
                avg_norm=avg_norm,
                k5=k5,
                k10=k10,
                mode_name=mode_name,
                input_kind=input_kind,
                projection_layer_scope=projection_layer_scope,
                group_renorm=group_renorm,
                norm_restore=norm_restore,
                personalized_rank_selection=personalized_rank_selection,
                personalized_rank_num_requested=personalized_rank_num_requested,
                personalized_rank_num_effective=personalized_rank_num_effective,
                personalized_rank_force_u1=personalized_rank_force_u1,
                personalized_rank_mode=personalized_rank_mode,
                personalized_rank_energy=personalized_rank_energy,
                personalized_g_scale=personalized_g_scale,
                local_update_views=local_update_views,
                repeatability_threshold=repeatability_threshold,
                personalized_coeff_mode=personalized_coeff_mode,
                personalized_tail_scale=personalized_tail_scale,
                personalized_m_filter_mode=personalized_m_filter_mode,
                personalized_dominance_threshold=(
                    personalized_dominance_threshold
                ),
                dominance_filter_enabled=dominance_filter_enabled,
                full_dominance_positive_energy=full_dominance_positive_energy,
                full_dominance_negative_energy=full_dominance_negative_energy,
                full_dominance_ratio=full_dominance_ratio,
                full_dominant_sign=full_dominant_sign,
                dominance_balanced_filtered_direction_count=(
                    dominance_balanced_filtered_direction_count
                ),
                dominance_weak_side_filtered_client_direction_count=(
                    dominance_weak_side_filtered_client_direction_count
                ),
                uniform_reference_kind=uniform_reference_kind,
                uniform_reference_size=uniform_reference_size,
                u1_selection_rate=u1_selection_rate,
                u1_selection_rate_before=u1_selection_rate_before,
                strength_formula_max_error=strength_formula_max_error,
                strength_in_range=strength_in_range,
                reconstruction_error=reconstruction_error,
                mask_symmetric=mask_symmetric,
                self_mask_valid=self_mask_valid,
                finite_ok=finite_ok,
                log_zero=log_zero,
            )

    @staticmethod
    def _projection_coefficient_cosine(coefficients, reference, reference_norm):
        coefficient_norm = torch.norm(coefficients)
        denominator = coefficient_norm * reference_norm
        if denominator <= 1e-12:
            return 0.0
        return float((torch.dot(coefficients, reference) / denominator).item())

    def _projection_prefix_metrics(
        self,
        self_coefficients,
        sign_coefficients,
        gsign_coefficients,
        average_coefficients,
        self_norm,
        average_norm,
        prefix,
    ):
        self_projection = self_coefficients[:prefix]
        sign_projection = sign_coefficients[:prefix]
        gsign_projection = gsign_coefficients[:prefix]
        self_projection_norm = torch.norm(self_projection)
        residual_squared = torch.clamp(
            self_norm.square() - self_projection_norm.square(),
            min=0.0,
        )
        return {
            "norm_self_projection": float(self_projection_norm.item()),
            "norm_self_residual": float(torch.sqrt(residual_squared).item()),
            "norm_sign": float(torch.norm(sign_projection).item()),
            "norm_gsign": float(torch.norm(gsign_projection).item()),
            "cos_selfproj_self": self._projection_coefficient_cosine(
                self_projection,
                self_coefficients[:prefix],
                self_norm,
            ),
            "cos_sign_self": self._projection_coefficient_cosine(
                sign_projection,
                self_coefficients[:prefix],
                self_norm,
            ),
            "cos_gsign_self": self._projection_coefficient_cosine(
                gsign_projection,
                self_coefficients[:prefix],
                self_norm,
            ),
            "cos_selfproj_avg": self._projection_coefficient_cosine(
                self_projection,
                average_coefficients[:prefix],
                average_norm,
            ),
            "cos_sign_avg": self._projection_coefficient_cosine(
                sign_projection,
                average_coefficients[:prefix],
                average_norm,
            ),
            "cos_gsign_avg": self._projection_coefficient_cosine(
                gsign_projection,
                average_coefficients[:prefix],
                average_norm,
            ),
        }

    @staticmethod
    def _csv_sequence(values):
        return ";".join(f"{value:.12g}" if isinstance(value, float) else str(value) for value in values)

    @staticmethod
    def _append_projection_diagnostic_rows(path, rows):
        if not rows:
            return
        os.makedirs(os.path.dirname(path), exist_ok=True)
        write_header = not os.path.exists(path) or os.path.getsize(path) == 0
        with open(path, "a", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
            if write_header:
                writer.writeheader()
            writer.writerows(rows)

    def _print_projection_console_diagnostics(
        self,
        name,
        rank_r,
        selected_k,
        full_sigma,
        full_energy,
        full_cumulative,
        full_h,
        full_a,
        average_a,
        full_g,
        full_b,
        full_b_with_renorm,
        full_b_without_renorm,
        full_gb,
        full_output_coefficients,
        full_masks,
        full_same_sign_count,
        full_same_sign_mass,
        positive_client_count,
        negative_client_count,
        positive_weight_mass,
        negative_weight_mass,
        positive_weighted_sum,
        negative_weighted_sum,
        cancellation_ratio,
        client_metrics,
        raw_norms,
        avg_norm,
        k5,
        k10,
        mode_name,
        input_kind,
        projection_layer_scope,
        group_renorm,
        norm_restore,
        personalized_rank_selection,
        personalized_rank_num_requested,
        personalized_rank_num_effective,
        personalized_rank_force_u1,
        personalized_rank_mode,
        personalized_rank_energy,
        personalized_g_scale,
        local_update_views,
        repeatability_threshold,
        personalized_coeff_mode,
        personalized_tail_scale,
        personalized_m_filter_mode,
        personalized_dominance_threshold,
        dominance_filter_enabled,
        full_dominance_positive_energy,
        full_dominance_negative_energy,
        full_dominance_ratio,
        full_dominant_sign,
        dominance_balanced_filtered_direction_count,
        dominance_weak_side_filtered_client_direction_count,
        uniform_reference_kind,
        uniform_reference_size,
        u1_selection_rate,
        u1_selection_rate_before,
        strength_formula_max_error,
        strength_in_range,
        reconstruction_error,
        mask_symmetric,
        self_mask_valid,
        finite_ok,
        log_zero,
    ):
        average_norm_label = (
            "avg_weight_norm" if input_kind == "weight" else "norm_delta_avg"
        )
        self_norm_label = (
            "weight_norm" if input_kind == "weight" else "norm_self_original"
        )
        if personalized_rank_selection and personalized_rank_mode == "energy":
            rank_selection_description = (
                f"rank_mode=energy tau={personalized_rank_energy:.6g} "
                "selected_M=per_client"
            )
            overlap_reference_description = "uniform_top_M_i(per_client)"
        else:
            rank_selection_description = (
                f"rank_mode={personalized_rank_mode} "
                f"requested_M={personalized_rank_num_requested} "
                f"effective_M={personalized_rank_num_effective}"
            )
            overlap_reference_description = (
                f"uniform_top_{uniform_reference_kind}"
                f"({uniform_reference_size})"
            )
        selected_count_values = [
            metrics["selected_count"] for metrics in client_metrics
        ]
        selected_count_values_before = [
            metrics["selected_count_before"] for metrics in client_metrics
        ]
        selected_count_summary = (
            f"selected_count(min/mean/max)="
            f"{min(selected_count_values)}/"
            f"{sum(selected_count_values) / len(selected_count_values):.3f}/"
            f"{max(selected_count_values)}"
        )
        zero_energy_fallback_count = sum(
            int(metrics["zero_energy_fallback"])
            for metrics in client_metrics
        )
        fallback_used_count = sum(
            int(metrics["fallback_used"]) for metrics in client_metrics
        )
        dominance_empty_after_filter_count = sum(
            int(metrics["dominance_empty_after_filter"])
            for metrics in client_metrics
        )
        print(
            f"[SignProjection诊断] mode={mode_name} "
            f"round={self.cur_ground} layer={name} "
            f"input_kind={input_kind} "
            f"projection_layer_scope={projection_layer_scope} "
            f"clients={len(self.uploaded_ids)} rank_R={rank_r} "
            f"selected_K={selected_k} group_renorm={group_renorm} "
            f"norm_restore={norm_restore} "
            f"personalized_rank_selection={personalized_rank_selection} "
            f"{rank_selection_description} "
            f"force_u1={bool(personalized_rank_force_u1)} "
            f"personalized_g_scale={int(personalized_g_scale)} "
            f"local_update_views={local_update_views} "
            f"repeatability_threshold={repeatability_threshold:.6g} "
            f"coeff_mode={personalized_coeff_mode} "
            f"tail_scale={personalized_tail_scale:.6g} "
            f"m_filter_mode={personalized_m_filter_mode} "
            f"dominance_threshold={personalized_dominance_threshold:.6g} "
            f"u1_selection_rate_before={u1_selection_rate_before:.6f} "
            f"u1_selection_rate={u1_selection_rate:.6f} "
            f"selected_count_before(min/mean/max)="
            f"{min(selected_count_values_before)}/"
            f"{sum(selected_count_values_before) / len(selected_count_values_before):.3f}/"
            f"{max(selected_count_values_before)} "
            f"{selected_count_summary} "
            f"zero_energy_fallback_count={zero_energy_fallback_count} "
            f"fallback_used_count={fallback_used_count} "
            f"dominance_balanced_filtered_directions="
            f"{dominance_balanced_filtered_direction_count} "
            f"dominance_weak_side_filtered_client_directions="
            f"{dominance_weak_side_filtered_client_direction_count} "
            f"dominance_empty_after_filter_count="
            f"{dominance_empty_after_filter_count} "
            f"overlap_reference={overlap_reference_description}"
        )
        print(
            "  singular_values="
            + str([round(value, 6) for value in full_sigma.detach().cpu().tolist()])
        )
        print(
            "  energy_per_direction="
            + str([round(value, 6) for value in full_energy.detach().cpu().tolist()])
        )
        print(
            "  cumulative_energy="
            + str([round(value, 6) for value in full_cumulative.detach().cpu().tolist()])
        )
        print(
            f"  diagnostic_prefixes: K=5(effective={k5}), "
            f"K=10(effective={k10}), K=R(effective={rank_r})"
        )

        for direction_idx in range(selected_k):
            v = full_h[:, direction_idx]
            top_count = min(3, len(self.uploaded_ids))
            top_indices = torch.topk(v.square(), k=top_count).indices.tolist()
            top_clients = [self.uploaded_ids[index] for index in top_indices]
            top_v2 = [round(float(v[index].square().item()), 6) for index in top_indices]
            dominance_description = ""
            if dominance_filter_enabled:
                dominant_sign_label = {
                    -1: "negative",
                    0: "none",
                    1: "positive",
                }[int(full_dominant_sign[direction_idx].item())]
                dominance_description = (
                    f" E_pos="
                    f"{full_dominance_positive_energy[direction_idx].item():.6f}"
                    f" E_neg="
                    f"{full_dominance_negative_energy[direction_idx].item():.6f}"
                    f" dominance_ratio="
                    f"{full_dominance_ratio[direction_idx].item():.6f}"
                    f" dominant_sign={dominant_sign_label}"
                )
            print(
                f"  k={direction_idx + 1} sigma={full_sigma[direction_idx].item():.6f} "
                f"energy={full_energy[direction_idx].item():.6f} "
                f"positive_count={int(positive_client_count[direction_idx].item())} "
                f"negative_count={int(negative_client_count[direction_idx].item())} "
                f"positive_mass={positive_weight_mass[direction_idx].item():.6f} "
                f"negative_mass={negative_weight_mass[direction_idx].item():.6f} "
                f"positive_sum={positive_weighted_sum[direction_idx].item():.6f} "
                f"negative_sum={negative_weighted_sum[direction_idx].item():.6f} "
                f"a_avg={average_a[direction_idx].item():.6f} "
                f"cancellation={cancellation_ratio[direction_idx].item():.6f} "
                f"top_clients={top_clients} v2={top_v2}"
                f"{dominance_description}"
            )

        sampled_targets = min(3, len(self.uploaded_ids))
        for target_idx in range(sampled_targets):
            metrics = client_metrics[target_idx]
            target_cid = metrics["client_id"]
            target_g = metrics["target_g"]
            print(
                f"    target={target_cid} personalized_selection: "
                f"ids_0based={metrics['selected_direction_ids']} "
                f"scores={[round(value, 8) for value in metrics['selected_scores']]} "
                f"g={[round(value, 6) for value in metrics['selected_g_values']]} "
                f"support_count={metrics['selected_support_counts']} "
                f"support_mass="
                f"{[round(value, 6) for value in metrics['selected_support_masses']]} "
                f"selected_count(before/after)="
                f"{metrics['selected_count_before']}/{metrics['selected_count']} "
                f"selected_count(before_dominance)="
                f"{metrics['selected_count_before_dominance']} "
                f"dominance_filtered(balanced/weak_side)="
                f"{metrics['dominance_balanced_filtered_count']}/"
                f"{metrics['dominance_weak_side_filtered_count']} "
                f"dominance_empty={metrics['dominance_empty_after_filter']} "
                f"score_ratio(before/after)="
                f"{metrics['selected_score_ratio_before']:.6f}/"
                f"{metrics['selected_score_ratio']:.6f} "
                f"retained_raw_selected_energy_fraction="
                f"{metrics['retained_raw_selected_energy_fraction']:.6f} "
                f"energy_threshold_met={metrics['energy_threshold_met']} "
                f"zero_energy_fallback={metrics['zero_energy_fallback']} "
                f"repeatability_empty_fallback="
                f"{metrics['repeatability_empty_fallback']} "
                f"fallback_used={metrics['fallback_used']} "
                f"u1_selected={metrics['u1_selected']} "
                f"u1_score_rank_1based={metrics['u1_score_rank_1based']} "
                f"uniform_{uniform_reference_kind}_overlap="
                f"{metrics['uniform_overlap_count']}/"
                f"{len(metrics['selected_direction_ids'])} "
                f"ratio={metrics['uniform_overlap_ratio']:.6f}"
            )
            print(
                f"    target={target_cid} coverage(K5/K10/KR)="
                f"{metrics['coverage_k5']:.6f}/"
                f"{metrics['coverage_k10']:.6f}/"
                f"{metrics['coverage_kr']:.6f} residual(K5/K10)="
                f"{1.0 - metrics['coverage_k5']:.6f}/"
                f"{1.0 - metrics['coverage_k10']:.6f} "
                f"top5_rank={metrics['top_ranks']} "
                f"top5_g={[round(value, 6) for value in metrics['top_values']]} "
                f"not_selected={metrics['missed_top_ranks']} "
                f"max_g_after_K={metrics['max_g_after_k']:.6f}"
            )
            if personalized_rank_selection:
                detailed_direction_indices = metrics["selected_direction_ids"]
            else:
                detailed_direction_indices = range(selected_k)
            for direction_idx in detailed_direction_indices:
                active_indices = torch.nonzero(
                    full_masks[direction_idx][target_idx],
                    as_tuple=False,
                ).flatten().tolist()
                active_ids = [self.uploaded_ids[index] for index in active_indices]
                same_mass = full_same_sign_mass[target_idx, direction_idx]
                weight_before = same_mass / (same_mass + 1e-12)
                weight_after = target_g[direction_idx] * weight_before
                avg_value = average_a[direction_idx]
                b_value = full_b[target_idx, direction_idx]
                gb_value = full_gb[target_idx, direction_idx]
                output_value = full_output_coefficients[
                    target_idx,
                    direction_idx,
                ]
                ratio_denominator = torch.abs(avg_value) + 1e-12
                all_same = int(
                    full_same_sign_count[target_idx, direction_idx].item()
                    == len(self.uploaded_ids)
                )
                print(
                    f"      k={direction_idx + 1} g={target_g[direction_idx].item():.6f} "
                    f"a_self={full_a[target_idx, direction_idx].item():.6f} "
                    f"a_avg={avg_value.item():.6f} b={b_value.item():.6f} "
                    f"g*b={gb_value.item():.6f} output={output_value.item():.6f} "
                    f"same_sign={len(active_ids)} ids={active_ids} "
                    f"mass={same_mass.item():.6f} "
                    f"sign_amp={(torch.abs(b_value) / ratio_denominator).item():.6f} "
                    f"final_ratio={(torch.abs(gb_value) / ratio_denominator).item():.6f} "
                    f"weight_sum(before/after_g)={weight_before.item():.6f}/"
                    f"{weight_after.item():.6f} all_same={all_same} "
                    f"abs(b-avg)={torch.abs(b_value - avg_value).item():.6f} "
                    f"abs(gb-avg)={torch.abs(gb_value - avg_value).item():.6f}"
                )
                if not group_renorm:
                    print(
                        f"        no_group_renorm: "
                        f"same_sign_weight_mass={same_mass.item():.6f} "
                        f"group_coeff_with_renorm="
                        f"{full_b_with_renorm[target_idx, direction_idx].item():.6f} "
                        f"group_coeff_without_renorm="
                        f"{full_b_without_renorm[target_idx, direction_idx].item():.6f}"
                    )

            for prefix_name, prefix_metrics in (
                ("K5", metrics["metrics_k5"]),
                ("K10", metrics["metrics_k10"]),
                ("KR", metrics["metrics_kr"]),
            ):
                print(
                    f"      {prefix_name}: {self_norm_label}="
                    f"{raw_norms[target_idx].item():.6f} "
                    f"{average_norm_label}={avg_norm.item():.6f} "
                    f"norm_self_projection={prefix_metrics['norm_self_projection']:.6f} "
                    f"norm_self_residual={prefix_metrics['norm_self_residual']:.6f} "
                    f"norm_sign_before_g={prefix_metrics['norm_sign']:.6f} "
                    f"norm_sign_after_g={prefix_metrics['norm_gsign']:.6f} "
                    f"cos(selfproj,self)={prefix_metrics['cos_selfproj_self']:.6f} "
                    f"cos(sign,self)={prefix_metrics['cos_sign_self']:.6f} "
                    f"cos(gsign,self)={prefix_metrics['cos_gsign_self']:.6f} "
                    f"cos(selfproj,avg)={prefix_metrics['cos_selfproj_avg']:.6f} "
                    f"cos(sign,avg)={prefix_metrics['cos_sign_avg']:.6f} "
                    f"cos(gsign,avg)={prefix_metrics['cos_gsign_avg']:.6f}"
                )
            print(
                f"      increment_6_10: norm={metrics['increment_norm']:.6f} "
                f"cos(self)={metrics['increment_self_cos']:.6f} "
                f"cos(avg)={metrics['increment_avg_cos']:.6f} "
                f"cos(sign_K5)={metrics['increment_sign_k5_cos']:.6f}"
            )
            print(
                f"      g(mean/min/max)={target_g.mean().item():.6f}/"
                f"{target_g.min().item():.6f}/{target_g.max().item():.6f}"
            )
            print(
                f"      norm_restore: {average_norm_label}={avg_norm.item():.6f} "
                f"norm_before_g={metrics['norm_before_g']:.6f} "
                f"norm_after_g_before_restore="
                f"{metrics['norm_after_g_before_restore']:.6f} "
                f"gamma_raw={metrics['gamma_raw']:.6f} "
                f"gamma_used={metrics['gamma_used']:.6f} "
                f"norm_after_restore={metrics['norm_after_restore']:.6f} "
                f"cos_after_restore_with_avg="
                f"{metrics['cos_after_restore_with_avg']:.6f} "
                f"cos_after_restore_with_self="
                f"{metrics['cos_after_restore_with_self']:.6f}"
            )

        all_g_valid = bool(torch.all((full_g >= 0.0) & (full_g <= 1.0)))
        all_diagnostic_finite = all(bool(torch.isfinite(tensor).all()) for tensor in (
            full_sigma,
            full_energy,
            full_g,
            full_a,
            full_b,
            full_gb,
            cancellation_ratio,
        ))
        near_zero_v = int(torch.sum(torch.abs(full_h) <= log_zero).item())
        print(
            f"  checks: mask_symmetric={mask_symmetric} self_mask={self_mask_valid} "
            f"selected_g_in_range={strength_in_range} all_g_in_range={all_g_valid} "
            f"finite={finite_ok and all_diagnostic_finite} "
            f"near_zero_v={near_zero_v} "
            f"g_formula_max_error={strength_formula_max_error.item():.3e} "
            f"svd_reconstruction_error={reconstruction_error.item():.3e}"
        )

    def _safe_cosine(self, first, second):
        denominator = torch.norm(first) * torch.norm(second)
        if denominator <= 1e-12:
            return 0.0
        return float((torch.dot(first, second) / denominator).item())

    def _weighted_average_vectors(self, vectors, weights):
        avg = torch.zeros_like(vectors[0])
        for weight, vec in zip(weights, vectors):
            avg += vec * float(weight)
        return avg

    def _update_personal_residuals(self, name, residuals, global_weight):
        mu = float(getattr(self.args, "personal_residual_mu", 0.9))
        gamma = float(getattr(self.args, "personal_residual_gamma", 0.5))
        clip_ratio = float(getattr(self.args, "personal_residual_clip", 0.0))
        eps = 1e-12

        for cid, residual in zip(self.uploaded_ids, residuals):
            client_residuals = self.personal_residuals.setdefault(cid, {})
            if name in client_residuals:
                previous = client_residuals[name].to(residual.device)
            else:
                previous = torch.zeros_like(residual)
            updated = mu * previous + gamma * residual

            if clip_ratio > 0:
                residual_norm = torch.norm(updated)
                limit = clip_ratio * torch.norm(global_weight.detach())
                if residual_norm > limit:
                    updated = updated * (limit / (residual_norm + eps))

            client_residuals[name] = updated.detach().cpu()

    def _save_sign_personalized_models(self, global_model, personalized_updates):
        for cid in range(self.num_clients):
            personalized_model = copy.deepcopy(global_model).to(self.device)
            if cid in personalized_updates:
                param_dict = dict(personalized_model.named_parameters())
                start_params = self.client_start_full_weights[cid]
                for name, personalized_delta in personalized_updates[cid].items():
                    if name not in param_dict or name not in start_params:
                        continue
                    param_dict[name].data.copy_(
                        start_params[name].to(param_dict[name].device)
                        + personalized_delta.to(param_dict[name].device)
                    )
            save_item(personalized_model, self.role, f"model_{cid}", self.save_folder_name)

    def _save_sign_personalized_weight_models(
        self,
        global_model,
        personalized_weights,
    ):
        """Save absolute personalized weights without adding a client start."""
        for cid in range(self.num_clients):
            personalized_model = copy.deepcopy(global_model).to(self.device)
            if cid in personalized_weights:
                param_dict = dict(personalized_model.named_parameters())
                for name, personalized_weight in personalized_weights[cid].items():
                    if name not in param_dict:
                        continue
                    param_dict[name].data.copy_(
                        personalized_weight.to(param_dict[name].device)
                    )
            save_item(
                personalized_model,
                self.role,
                f"model_{cid}",
                self.save_folder_name,
            )

    def _save_personalized_models_from_global(self, global_model, use_residual, instant_residuals=None):
        for cid in range(self.num_clients):
            personalized_model = copy.deepcopy(global_model).to(self.device)
            if use_residual:
                param_dict = dict(personalized_model.named_parameters())
                for name, residual in self.personal_residuals.get(cid, {}).items():
                    if name in param_dict:
                        param_dict[name].data += residual.to(param_dict[name].device)
            if instant_residuals is not None:
                param_dict = dict(personalized_model.named_parameters())
                for name, residual in instant_residuals.get(cid, {}).items():
                    if name in param_dict:
                        param_dict[name].data += residual.to(param_dict[name].device)
            save_item(personalized_model, self.role, f"model_{cid}", self.save_folder_name)

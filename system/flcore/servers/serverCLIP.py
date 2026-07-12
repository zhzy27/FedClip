import csv
import copy
import math
import os
import random
import time

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

    def _is_sign_projection_diagnostic_round(self):
        first_projection_round = self._projection_warmup_rounds() + 1
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

    def _aggregate_sign_projection_variant(
        self,
        mode_name,
        group_renorm,
        norm_restore,
    ):
        assert len(self.uploaded_ids) > 0

        mode_labels = {
            "sign_personalized_projection": "符号一致性个性化 Projection",
            "sign_projection_norm_restore": "符号 Projection + 整体范数恢复",
            "sign_projection_no_group_renorm": "符号 Projection（无组内归一化）+ 整体范数恢复",
        }
        print(f"🚀 执行 CNN {mode_labels[mode_name]} 聚合")
        uploaded_full_param_dicts = []
        delta_param_dicts = []
        projectable_weight_names = set()

        for cid in self.uploaded_ids:
            if cid not in self.client_start_full_weights:
                raise RuntimeError(
                    f"Sign Personalized Projection 缺少 Client_{cid} 本轮训练起点 S_i^t。"
                )

            client = self.clients[cid]
            uploaded_low_rank_model = load_item(client.role, "model", client.save_folder_name)
            if uploaded_low_rank_model is None:
                raise RuntimeError(f"无法加载 Client_{cid} 的上传模型。")

            projectable_weight_names.update(
                self._projectable_weight_names_from_low_rank_model(uploaded_low_rank_model)
            )
            uploaded_full_model = copy.deepcopy(uploaded_low_rank_model).to(self.device)
            self._recover_if_needed(uploaded_full_model)
            uploaded_full_model = uploaded_full_model.to(self.device)
            uploaded_params = dict(uploaded_full_model.named_parameters())

            start_params = self.client_start_full_weights[cid]
            delta_params = {}
            for name, uploaded_param in uploaded_params.items():
                if name in start_params:
                    delta_params[name] = (
                        uploaded_param.data.detach().clone()
                        - start_params[name].to(uploaded_param.device)
                    )

            uploaded_full_param_dicts.append(uploaded_params)
            delta_param_dicts.append(delta_params)

        global_model = load_item(self.role, "model", self.save_folder_name).to(self.device)
        self._recover_if_needed(global_model)
        global_model = global_model.to(self.device)
        global_param_dict = dict(global_model.named_parameters())
        alpha = [float(weight) for weight in self.uploaded_weights]
        personalized_updates = {cid: {} for cid in self.uploaded_ids}

        ordered_projectable_names = [
            name for name in global_param_dict if name in projectable_weight_names
        ]
        diagnostic_round = (
            bool(ordered_projectable_names)
            and self._is_sign_projection_diagnostic_round()
        )

        for name, global_param in global_param_dict.items():
            can_project = (
                name in projectable_weight_names
                and all(name in delta_params for delta_params in delta_param_dicts)
                and all(
                    name in self.client_start_full_weights[cid]
                    for cid in self.uploaded_ids
                )
            )
            if can_project:
                personalized_deltas, average_delta = (
                    self._sign_personalized_update_for_layer(
                        name,
                        delta_param_dicts,
                        alpha,
                        global_param.data.shape,
                        log_diagnostics=diagnostic_round,
                        console_diagnostics=(
                            diagnostic_round
                            and self._is_sign_projection_console_layer(name)
                        ),
                        group_renorm=group_renorm,
                        norm_restore=norm_restore,
                        mode_name=mode_name,
                    )
                )
                global_param.data += average_delta.to(global_param.device)
                for cid, personalized_delta in zip(self.uploaded_ids, personalized_deltas):
                    personalized_updates[cid][name] = personalized_delta.detach().cpu()
            else:
                global_param.data.zero_()
                for weight, uploaded_params in zip(alpha, uploaded_full_param_dicts):
                    if name in uploaded_params:
                        global_param.data += weight * uploaded_params[name].data

        save_item(global_model, self.role, "model", self.save_folder_name)
        self.personal_residuals = {cid: {} for cid in range(self.num_clients)}
        self._save_sign_personalized_models(global_model, personalized_updates)
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
        cumulative = torch.cumsum(eigvals, dim=0) / (eigvals.sum() + eps)
        k_energy = int((cumulative >= energy_threshold).nonzero(as_tuple=False)[0].item()) + 1
        k = max(1, min(k_energy, k_max, int(positive.sum().item())))

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
        cumulative = torch.cumsum(eigvals, dim=0) / (eigvals.sum() + eps)
        k_energy = int((cumulative >= energy_threshold).nonzero(as_tuple=False)[0].item()) + 1
        k = max(1, min(k_energy, k_max, int(positive.sum().item())))

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

    def _sign_personalized_update_for_layer(
        self,
        name,
        delta_param_dicts,
        alpha,
        target_shape,
        log_diagnostics=False,
        console_diagnostics=False,
        group_renorm=True,
        norm_restore=False,
        mode_name="sign_personalized_projection",
    ):
        eps = 1e-12
        log_zero = 1e-8
        device = self.device
        raw_vecs = [
            delta_dict[name].detach().to(device).reshape(-1).float()
            for delta_dict in delta_param_dicts
        ]
        if not all(bool(torch.isfinite(vec).all()) for vec in raw_vecs):
            raise FloatingPointError(f"层 {name} 的客户端 delta 出现 NaN 或 Inf。")
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
            print(f"⚠️ 层 {name} Gram 分解失败，退回该层个性化 DeltaAvg。")
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
        cumulative = torch.cumsum(eigvals, dim=0) / (eigvals.sum() + eps)
        k_energy = int((cumulative >= energy_threshold).nonzero(as_tuple=False)[0].item()) + 1
        k = max(1, min(k_energy, k_max, int(positive.sum().item())))

        h = eigvecs[:, :k]
        sigma = torch.sqrt(eigvals[:k].clamp_min(eps))
        alpha_tensor = torch.tensor(alpha, device=device, dtype=average_delta.dtype)

        # Recover the selected left singular directions so target participation
        # strengths can be checked directly in the original update space.
        left_directions = []
        for direction_idx in range(k):
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

        direct_strengths_unclamped = torch.stack([
            torch.stack([
                torch.abs(torch.dot(unit_vec, direction))
                for direction in left_directions
            ])
            for unit_vec in unit_vecs
        ])
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
            (num_clients, k),
            device=device,
            dtype=average_delta.dtype,
        )
        group_coefficients_with_renorm = torch.zeros(
            (num_clients, k),
            device=device,
            dtype=average_delta.dtype,
        )
        group_coefficients_without_renorm = torch.zeros(
            (num_clients, k),
            device=device,
            dtype=average_delta.dtype,
        )
        personalized_coefficients = torch.zeros(
            (num_clients, k),
            device=device,
            dtype=average_delta.dtype,
        )
        mask_symmetric = True
        self_mask_valid = True
        for direction_idx in range(k):
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

        scaled_personalized_coefficients = (
            target_strengths * personalized_coefficients
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
            unscaled_source_coefficients = h @ (
                personalized_coefficients[target_idx] / sigma
            )
            scaled_source_coefficients = h @ (
                scaled_personalized_coefficients[target_idx] / sigma
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
            direct_strengths_unclamped,
            target_strengths,
            svd_strengths,
            strength_formula_max_error,
            same_sign_weight_masses,
            group_coefficients_with_renorm,
            group_coefficients_without_renorm,
            personalized_coefficients,
            scaled_personalized_coefficients,
            gamma_raw_values,
            gamma_used_values,
            *unscaled_personalized_vecs,
            *personalized_vecs_before_restore,
            *personalized_vecs,
        ]
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
                same_sign_weight_masses=same_sign_weight_masses,
                group_coefficients_with_renorm=group_coefficients_with_renorm,
                group_coefficients_without_renorm=group_coefficients_without_renorm,
                personalized_coefficients=personalized_coefficients,
                target_strengths=target_strengths,
                scaled_personalized_coefficients=scaled_personalized_coefficients,
                gamma_raw_values=gamma_raw_values,
                gamma_used_values=gamma_used_values,
                group_renorm=group_renorm,
                norm_restore=norm_restore,
                mode_name=mode_name,
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
        same_sign_weight_masses,
        group_coefficients_with_renorm,
        group_coefficients_without_renorm,
        personalized_coefficients,
        target_strengths,
        scaled_personalized_coefficients,
        gamma_raw_values,
        gamma_used_values,
        group_renorm,
        norm_restore,
        mode_name,
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

        eps = 1e-12
        k = h.shape[1]
        rank_r = int(positive.sum().item())
        num_clients = len(raw_vecs)
        full_eigvals = eigvals[:rank_r]
        full_sigma = torch.sqrt(full_eigvals.clamp_min(eps))
        full_h = eigvecs[:, :rank_r]
        full_energy = full_eigvals / (full_eigvals.sum() + eps)
        full_cumulative = torch.cumsum(full_energy, dim=0)

        sqrt_alpha = torch.sqrt(alpha_tensor.clamp_min(eps)).unsqueeze(1)
        signed_unit_coefficients = (
            full_sigma.unsqueeze(0) * full_h / (sqrt_alpha + eps)
        )
        full_g = torch.clamp(torch.abs(signed_unit_coefficients), 0.0, 1.0)
        full_g[:, :k] = target_strengths

        raw_norms = torch.stack([torch.norm(vec) for vec in raw_vecs])
        full_a = signed_unit_coefficients * (raw_norms + eps).unsqueeze(1)
        full_a[:, :k] = direction_projections
        average_a = alpha_tensor @ full_a

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
        full_b_with_renorm[:, :k] = group_coefficients_with_renorm
        full_b_without_renorm[:, :k] = group_coefficients_without_renorm
        full_b[:, :k] = personalized_coefficients
        full_gb = full_g * full_b
        full_gb[:, :k] = scaled_personalized_coefficients
        full_restored_gb = full_gb * gamma_used_values.unsqueeze(1)

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
            norm_before_g = torch.norm(
                unscaled_personalized_vecs[target_idx]
            )
            norm_after_g_before_restore = torch.norm(
                personalized_vecs_before_restore[target_idx]
            )
            norm_after_restore = torch.norm(personalized_vecs[target_idx])
            cos_after_restore_with_avg = self._safe_cosine(
                personalized_vecs[target_idx],
                average_delta,
            )
            cos_after_restore_with_self = self._safe_cosine(
                personalized_vecs[target_idx],
                raw_vecs[target_idx],
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
                "norm_after_restore": float(norm_after_restore.item()),
                "cos_after_restore_with_avg": cos_after_restore_with_avg,
                "cos_after_restore_with_self": cos_after_restore_with_self,
            }
            client_metrics.append(metrics)
            client_rows.append({
                "round": self.cur_ground,
                "layer": name,
                "client_id": client_id,
                "aggregation_mode": mode_name,
                "group_renorm": int(group_renorm),
                "norm_restore": int(norm_restore),
                "rank_R": rank_r,
                "selected_K": k,
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
                "max_g_after_K": max_g_after_k,
                "mean_g": float(torch.mean(target_g).item()),
                "min_g": float(torch.min(target_g).item()),
                "norm_self": float(self_norm.item()),
                "norm_avg": float(avg_norm.item()),
                "norm_delta_avg": float(avg_norm.item()),
                "norm_before_g": float(norm_before_g.item()),
                "norm_after_g_before_restore": float(
                    norm_after_g_before_restore.item()
                ),
                "gamma_raw": float(gamma_raw_values[target_idx].item()),
                "gamma_used": float(gamma_used_values[target_idx].item()),
                "norm_after_restore": float(norm_after_restore.item()),
                "cos_after_restore_with_avg": cos_after_restore_with_avg,
                "cos_after_restore_with_self": cos_after_restore_with_self,
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
                restored_coefficient = full_restored_gb[
                    target_idx, direction_idx
                ]
                ratio_denominator = torch.abs(avg_coefficient) + eps
                all_clients_same_sign = bool(
                    full_same_sign_count[target_idx, direction_idx].item()
                    == num_clients
                )
                direction_rows.append({
                    "round": self.cur_ground,
                    "layer": name,
                    "client_id": client_id,
                    "aggregation_mode": mode_name,
                    "group_renorm": int(group_renorm),
                    "norm_restore": int(norm_restore),
                    "k": direction_idx + 1,
                    "rank_R": rank_r,
                    "selected_K": k,
                    "selected_by_current_K": int(direction_idx < k),
                    "sigma": float(full_sigma[direction_idx].item()),
                    "energy": float(full_energy[direction_idx].item()),
                    "cumulative_energy": float(full_cumulative[direction_idx].item()),
                    "g": float(full_g[target_idx, direction_idx].item()),
                    "a_self": float(full_a[target_idx, direction_idx].item()),
                    "a_avg": float(avg_coefficient.item()),
                    "b_sign": float(sign_coefficient.item()),
                    "group_coeff_with_renorm": float(
                        full_b_with_renorm[target_idx, direction_idx].item()
                    ),
                    "group_coeff_without_renorm": float(
                        full_b_without_renorm[target_idx, direction_idx].item()
                    ),
                    "g_times_b": float(final_coefficient.item()),
                    "gamma_used": float(gamma_used_values[target_idx].item()),
                    "coefficient_after_restore": float(
                        restored_coefficient.item()
                    ),
                    "same_sign_count": int(
                        full_same_sign_count[target_idx, direction_idx].item()
                    ),
                    "same_sign_mass": float(same_sign_mass.item()),
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

        client_csv = os.path.join(
            self.save_folder_name,
            "projection_client_diagnostics.csv",
        )
        direction_csv = os.path.join(
            self.save_folder_name,
            "projection_direction_diagnostics.csv",
        )
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
                group_renorm=group_renorm,
                norm_restore=norm_restore,
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
        group_renorm,
        norm_restore,
        strength_formula_max_error,
        strength_in_range,
        reconstruction_error,
        mask_symmetric,
        self_mask_valid,
        finite_ok,
        log_zero,
    ):
        print(
            f"[SignProjection诊断] mode={mode_name} "
            f"round={self.cur_ground} layer={name} "
            f"clients={len(self.uploaded_ids)} rank_R={rank_r} "
            f"selected_K={selected_k} group_renorm={group_renorm} "
            f"norm_restore={norm_restore}"
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
            )

        sampled_targets = min(3, len(self.uploaded_ids))
        for target_idx in range(sampled_targets):
            metrics = client_metrics[target_idx]
            target_cid = metrics["client_id"]
            target_g = metrics["target_g"]
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
            for direction_idx in range(selected_k):
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
                ratio_denominator = torch.abs(avg_value) + 1e-12
                all_same = int(
                    full_same_sign_count[target_idx, direction_idx].item()
                    == len(self.uploaded_ids)
                )
                print(
                    f"      k={direction_idx + 1} g={target_g[direction_idx].item():.6f} "
                    f"a_self={full_a[target_idx, direction_idx].item():.6f} "
                    f"a_avg={avg_value.item():.6f} b={b_value.item():.6f} "
                    f"g*b={gb_value.item():.6f} "
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
                    f"      {prefix_name}: norm_self_original="
                    f"{raw_norms[target_idx].item():.6f} "
                    f"norm_delta_avg={avg_norm.item():.6f} "
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
                f"      norm_restore: norm_delta_avg={avg_norm.item():.6f} "
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

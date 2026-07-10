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
            self._aggregation_mode() in {"delta_avg", "sign_personalized_projection"}
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
        assert len(self.uploaded_ids) > 0

        print("🚀 执行 CNN 符号一致性个性化 Projection 聚合")
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
        diagnostic_names = set()
        if self.cur_ground % 10 == 0 and ordered_projectable_names:
            diagnostic_names.add(ordered_projectable_names[0])
            diagnostic_names.add(ordered_projectable_names[-1])

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
                        log_diagnostics=name in diagnostic_names,
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
        print(f"符号一致性个性化 Projection 聚合完成，样本量权重为 {self.uploaded_weights}")

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

        direction_projections = []
        for vec in raw_vecs:
            d_transpose_vec = torch.stack([
                torch.dot(weighted_unit_vec, vec)
                for weighted_unit_vec in weighted_unit_vecs
            ])
            direction_projections.append((h.t() @ d_transpose_vec) / sigma)
        direction_projections = torch.stack(direction_projections)

        masks = []
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
                personalized_coefficients[target_idx, direction_idx] = (
                    numerator / (denominator + eps)
                )

        personalized_vecs = []
        for target_idx in range(num_clients):
            source_coefficients = h @ (
                personalized_coefficients[target_idx] / sigma
            )
            personalized_vec = torch.zeros_like(average_delta)
            for coefficient, weighted_unit_vec in zip(
                source_coefficients,
                weighted_unit_vecs,
            ):
                personalized_vec += coefficient * weighted_unit_vec
            personalized_vecs.append(personalized_vec)

        finite_ok = (
            torch.isfinite(eigvals).all()
            and torch.isfinite(direction_projections).all()
            and torch.isfinite(personalized_coefficients).all()
            and all(torch.isfinite(vec).all() for vec in personalized_vecs)
        )
        if not bool(finite_ok):
            raise FloatingPointError(f"层 {name} 的符号一致性聚合出现 NaN 或 Inf。")
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
                average_delta=average_delta,
                personalized_vecs=personalized_vecs,
                alpha_tensor=alpha_tensor,
                eigvals=eigvals,
                h=h,
                sigma=sigma,
                masks=masks,
                direction_projections=direction_projections,
                personalized_coefficients=personalized_coefficients,
                reconstruction_error=reconstruction_error,
                mask_symmetric=mask_symmetric,
                self_mask_valid=self_mask_valid,
                finite_ok=bool(finite_ok),
                log_zero=log_zero,
            )

        personalized = [vec.reshape(target_shape) for vec in personalized_vecs]
        return personalized, average_delta.reshape(target_shape)

    def _print_sign_projection_diagnostics(
        self,
        name,
        raw_vecs,
        average_delta,
        personalized_vecs,
        alpha_tensor,
        eigvals,
        h,
        sigma,
        masks,
        direction_projections,
        personalized_coefficients,
        reconstruction_error,
        mask_symmetric,
        self_mask_valid,
        finite_ok,
        log_zero,
    ):
        k = h.shape[1]
        energy_ratio = eigvals[:k] / (eigvals.sum() + 1e-12)
        cumulative_ratio = torch.cumsum(energy_ratio, dim=0)
        print(
            f"[SignProjection诊断] round={self.cur_ground} layer={name} "
            f"clients={len(self.uploaded_ids)} K={k}"
        )
        print(
            "  singular="
            + str([round(value, 6) for value in sigma.detach().cpu().tolist()])
            + " energy="
            + str([round(value, 6) for value in energy_ratio.detach().cpu().tolist()])
            + " cumulative="
            + str([round(value, 6) for value in cumulative_ratio.detach().cpu().tolist()])
        )

        for direction_idx in range(k):
            v = h[:, direction_idx]
            positive_count = int(torch.sum(v > log_zero).item())
            negative_count = int(torch.sum(v < -log_zero).item())
            near_zero_count = len(self.uploaded_ids) - positive_count - negative_count
            top_count = min(3, len(self.uploaded_ids))
            top_indices = torch.topk(v.square(), k=top_count).indices.tolist()
            top_clients = [self.uploaded_ids[index] for index in top_indices]
            top_v2 = [round(float(v[index].square().item()), 6) for index in top_indices]
            print(
                f"  dir={direction_idx + 1} pos={positive_count} neg={negative_count} "
                f"near0={near_zero_count} top_clients={top_clients} v2={top_v2}"
            )

        sampled_targets = min(3, len(self.uploaded_ids))
        for target_idx in range(sampled_targets):
            target_cid = self.uploaded_ids[target_idx]
            for direction_idx in range(k):
                active_indices = torch.nonzero(
                    masks[direction_idx][target_idx],
                    as_tuple=False,
                ).flatten().tolist()
                active_ids = [self.uploaded_ids[index] for index in active_indices]
                active = masks[direction_idx][target_idx].to(alpha_tensor.dtype)
                denominator = torch.sum(alpha_tensor * active).item()
                coefficient = personalized_coefficients[
                    target_idx,
                    direction_idx,
                ].item()
                print(
                    f"    target={target_cid} dir={direction_idx + 1} "
                    f"same_sign={len(active_ids)} ids={active_ids} "
                    f"denom={denominator:.6f} b={coefficient:.6f}"
                )

            self_delta = raw_vecs[target_idx]
            personalized_delta = personalized_vecs[target_idx]
            print(
                f"    target={target_cid} norms(self/pers/delta_avg)="
                f"{torch.norm(self_delta).item():.6f}/"
                f"{torch.norm(personalized_delta).item():.6f}/"
                f"{torch.norm(average_delta).item():.6f} "
                f"cos(pers,self)={self._safe_cosine(personalized_delta, self_delta):.6f} "
                f"cos(pers,delta_avg)={self._safe_cosine(personalized_delta, average_delta):.6f}"
            )

        print(
            f"  checks: mask_symmetric={mask_symmetric} self_mask={self_mask_valid} "
            f"finite={finite_ok} svd_reconstruction_error={reconstruction_error.item():.3e}"
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

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
            if self._should_use_projection():
                self.aggregate_common_residual_projection_cnn()
            else:
                if self._is_resnet_model():
                    print("当前公共-残差投影只实现 CNN 路径，ResNet 暂时使用 FedAvg 聚合。")
                elif self._is_projection_warmup_round():
                    print(
                        f"公共-残差投影 warm-up: round={self.cur_ground}, "
                        f"warmup_rounds={self._projection_warmup_rounds()}，使用 FedAvg。"
                    )
                self.aggregate_avg(save_personalized=True)
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
        for client in self.selected_clients:
            start_time = time.time()
            client.set_parameters()
            client.send_time_cost["num_rounds"] += 1
            client.send_time_cost["total_cost"] += 2 * (time.time() - start_time)

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

    def _projection_warmup_rounds(self):
        ratio = float(getattr(self.args, "projection_warmup_ratio", 0.2))
        ratio = min(1.0, max(0.0, ratio))
        return int(round(self.global_rounds * ratio))

    def _is_projection_warmup_round(self):
        return self.cur_ground <= self._projection_warmup_rounds()

    def _should_use_projection(self):
        return (
            bool(getattr(self.args, "use_common_residual_projection", 1))
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

    def _load_old_start_model(self, cid, rank_rate):
        old_model = load_item(self.role, f"model_{cid}", self._low_rank_start_folder())
        if old_model is None:
            old_model = load_item(self.role, f"model_{cid}", self.save_folder_name)
        if old_model is None:
            old_model = load_item(self.role, "model", self.save_folder_name)
        old_model = copy.deepcopy(old_model).to(self.device)
        if not self._has_low_rank_params(old_model):
            self._decompose_if_needed(old_model, rank_rate)
        self._recover_if_needed(old_model)
        return old_model.to(self.device)

    def _client_full_models_and_deltas(self):
        uploaded_full_models = []
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

            uploaded_full_models.append(current_full)
            uploaded_full_param_dicts.append(current_dict)
            full_delta_param_dicts.append(delta_dict)

        return (
            uploaded_full_models,
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

    def aggregate_common_residual_projection_cnn(self):
        assert len(self.uploaded_ids) > 0

        print("🚀 执行 CNN 公共-残差投影聚合")
        (
            _uploaded_full_models,
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
                    self._update_personal_residuals(name, residuals, global_param.data)
            else:
                global_param.data.zero_()
                for weight, client_param_dict in zip(alpha, uploaded_full_param_dicts):
                    if name in client_param_dict:
                        global_param.data += client_param_dict[name].data * weight

        save_item(global_model, self.role, "model", self.save_folder_name)
        self._save_personalized_models_from_global(global_model, projection_use_residual)

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

    def _save_personalized_models_from_global(self, global_model, use_residual):
        for cid in range(self.num_clients):
            personalized_model = copy.deepcopy(global_model).to(self.device)
            if use_residual:
                param_dict = dict(personalized_model.named_parameters())
                for name, residual in self.personal_residuals.get(cid, {}).items():
                    if name in param_dict:
                        param_dict[name].data += residual.to(param_dict[name].device)
            save_item(personalized_model, self.role, f"model_{cid}", self.save_folder_name)

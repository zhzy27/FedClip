import csv
import copy
import os
import random
import time

import numpy as np
import torch
import torch.nn.functional as F

from flcore.clients.clientCLIP import clientCLIP
from flcore.clients.clientbase import load_item, save_item
from flcore.servers.serverbase import Server
from flcore.trainmodel.models import Model_Distribe


class FedCLIP(Server):
    _CLASSIFIER_MODE_ALIASES = {
        "classifier": "flat",
        "classifier_delta": "flat_delta",
    }

    def __init__(self, args, times):
        super().__init__(args, times)

        self.set_slow_clients()
        self.set_clients(clientCLIP)

        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print("Finished creating server and clients.")

        self.Budget = []
        global_model = Model_Distribe(args, -1, is_global=True).to(self.device)
        global_model.recover_larger_model()
        save_item(global_model, self.role, 'model', self.save_folder_name)

    def train(self):
        for round_idx in range(self.global_rounds + 1):
            self.cur_ground = round_idx
            round_start = time.time()
            self.selected_clients = self.select_clients()

            if round_idx > 0 and round_idx % self.eval_gap == 0:
                print(f"\n-------------Round number: {round_idx} 聚合前-------------")
                print("\nEvaluate heterogeneous models")
                self.evaluate(epoch=round_idx)

            self.send_parameters()

            if torch.cuda.is_available() and str(self.device).startswith("cuda"):
                torch.cuda.synchronize(self.device)
            local_train_wall_start = time.time()
            client_train_times = []
            for client in self.selected_clients:
                client_train_time = client.train(current_round=round_idx)
                if client_train_time is None:
                    client_train_time = getattr(client, "last_train_time_cost", 0.0)
                client_train_times.append((client.id, float(client_train_time)))
            if torch.cuda.is_available() and str(self.device).startswith("cuda"):
                torch.cuda.synchronize(self.device)

            local_train_wall_time = time.time() - local_train_wall_start
            local_train_sum_time = sum(train_time for _, train_time in client_train_times)
            print(
                f"⏱️ [Round {round_idx:03d}] 本地训练总耗时: "
                f"sum_client={local_train_sum_time:.3f}s | "
                f"wall={local_train_wall_time:.3f}s | "
                f"clients={len(client_train_times)}"
            )
            print(
                "⏱️ [Round {:03d}] 客户端训练耗时明细: {}".format(
                    round_idx,
                    ", ".join(
                        f"Client_{client_id}:{train_time:.3f}s"
                        for client_id, train_time in client_train_times
                    ),
                )
            )

            self.receive_ids()
            if torch.cuda.is_available() and str(self.device).startswith("cuda"):
                torch.cuda.synchronize(self.device)
            aggregation_wall_start = time.time()
            similarity_mode = getattr(
                self.args, "classifier_similarity_mode", "none"
            )
            if similarity_mode == "none":
                self.aggregate_avg()
            else:
                self.aggregate_classifier_similarity(similarity_mode)
            if torch.cuda.is_available() and str(self.device).startswith("cuda"):
                torch.cuda.synchronize(self.device)
            aggregation_wall_time = time.time() - aggregation_wall_start
            print(
                f"⏱️ [Round {round_idx:03d}] 聚合总墙钟耗时: "
                f"{aggregation_wall_time:.3f}s"
            )

            self.Budget.append(time.time() - round_start)
            print('-' * 25, 'time cost', '-' * 25, self.Budget[-1])

            if self.auto_break and self.check_done(
                acc_lss=[self.rs_test_acc], top_cnt=self.top_cnt
            ):
                break

        print("\nBest accuracy.")
        print(max(self.rs_test_acc))

        print("\nAverage time cost per round.")
        if len(self.Budget) > 1:
            print(sum(self.Budget[1:]) / len(self.Budget[1:]))
        else:
            print(self.Budget[0])

        self.save_results()
        self.save_json_file()

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
        for idx, weight in enumerate(self.uploaded_weights):
            self.uploaded_weights[idx] = weight / total_samples

    def send_parameters(self):
        assert len(self.selected_clients) > 0

        for client in self.selected_clients:
            start_time = time.time()
            client.set_parameters()
            client.send_time_cost['num_rounds'] += 1
            client.send_time_cost['total_cost'] += 2 * (time.time() - start_time)

    @staticmethod
    def _has_low_rank_params(model):
        return any(
            name.endswith('conv_v') or name.endswith('weight_v')
            for name, _ in model.named_parameters()
        )

    def _recover_if_needed(self, model):
        if self._has_low_rank_params(model):
            model.recover_larger_model()
        return model

    def aggregate_avg(self):
        """Sample-weighted averaging of complete recovered models."""
        assert len(self.uploaded_ids) > 0
        print("🚀 开始 Avg 聚合：恢复满秩后按客户端样本量聚合完整模型")

        uploaded_full_param_dicts = []
        for client_id in self.uploaded_ids:
            client = self.clients[client_id]
            client_model = load_item(
                client.role, 'model', client.save_folder_name
            )
            if client_model is None:
                raise RuntimeError(
                    f"Client_{client_id} uploaded model is missing."
                )
            full_model = copy.deepcopy(client_model).to(self.device)
            self._recover_if_needed(full_model)
            full_model = full_model.to(self.device)
            uploaded_full_param_dicts.append(
                dict(full_model.named_parameters())
            )

        self._save_sample_weighted_global(uploaded_full_param_dicts)
        print(f"✅ Avg 聚合完成，样本量权重: {self.uploaded_weights}")

    def _save_sample_weighted_global(self, uploaded_full_param_dicts):
        global_model = load_item(
            self.role, 'model', self.save_folder_name
        )
        if global_model is None:
            raise RuntimeError(
                "Server global model is missing before Avg aggregation."
            )
        global_model = global_model.to(self.device)
        self._recover_if_needed(global_model)
        global_model = global_model.to(self.device)
        global_params = dict(global_model.named_parameters())

        reference_names = global_params.keys()
        for source_idx, source_params in enumerate(
            uploaded_full_param_dicts
        ):
            if source_params.keys() != reference_names:
                raise RuntimeError(
                    f"Client_{self.uploaded_ids[source_idx]} full model is "
                    "incompatible with the Avg global model."
                )

        for global_param in global_params.values():
            global_param.data.zero_()
        for source_idx, weight in enumerate(self.uploaded_weights):
            for name, global_param in global_params.items():
                source_param = uploaded_full_param_dicts[source_idx][name]
                if source_param.shape != global_param.shape:
                    raise RuntimeError(
                        f"Avg shape mismatch for {name}: "
                        f"global={tuple(global_param.shape)}, "
                        f"client={tuple(source_param.shape)}"
                    )
                global_param.data += source_param.data * weight

        save_item(global_model, self.role, 'model', self.save_folder_name)

    def _load_recovered_uploaded_models(self):
        recovered_models = []
        for client_id in self.uploaded_ids:
            client = self.clients[client_id]
            client_model = load_item(
                client.role, 'model', client.save_folder_name
            )
            if client_model is None:
                raise RuntimeError(
                    f"Client_{client_id} uploaded model is missing."
                )
            recovered_model = copy.deepcopy(client_model).to(self.device)
            self._recover_if_needed(recovered_model)
            recovered_model = recovered_model.to(self.device)
            recovered_models.append(recovered_model)
        return recovered_models

    @staticmethod
    def _flatten_classifier(model):
        if not hasattr(model, "head"):
            raise RuntimeError("FedCLIP model has no head for classifier similarity.")
        classifier_params = [
            param.detach().reshape(-1)
            for param in model.head.parameters()
        ]
        if not classifier_params:
            raise RuntimeError("FedCLIP classifier head has no parameters.")
        return torch.cat(classifier_params)

    @classmethod
    def _canonical_classifier_mode(cls, mode):
        return cls._CLASSIFIER_MODE_ALIASES.get(mode, mode)

    @staticmethod
    def _classifier_parameter_names(model):
        if not hasattr(model, "head"):
            raise RuntimeError("FedCLIP model has no classifier head.")
        head_parameter_ids = {id(param) for param in model.head.parameters()}
        names = {
            name
            for name, param in model.named_parameters()
            if id(param) in head_parameter_ids
        }
        if not names:
            raise RuntimeError(
                "FedCLIP classifier head has no parameters in the full model."
            )
        return names

    def _classifier_weight(self, model):
        if not hasattr(model, "head"):
            raise RuntimeError("FedCLIP model has no head for class-wise similarity.")
        candidates = [
            (name, param)
            for name, param in model.head.named_parameters()
            if name.endswith("weight")
            and param.ndim == 2
            and int(param.shape[0]) == int(self.num_classes)
        ]
        if len(candidates) != 1:
            candidate_shapes = [
                (name, tuple(param.shape))
                for name, param in model.head.named_parameters()
                if name.endswith("weight") and param.ndim == 2
            ]
            raise RuntimeError(
                "Class-wise classifier similarity requires exactly one "
                f"[num_classes, feature_dim] weight, found {candidate_shapes}."
            )
        return candidates[0]

    def _classifier_delta_weight(self, model, client_id):
        weight_name, end_weight = self._classifier_weight(model)
        start_state = getattr(
            self.clients[client_id], "classifier_start_state", None
        )
        if start_state is None or weight_name not in start_state:
            raise RuntimeError(
                f"Client_{client_id} has no current-round start snapshot "
                f"for classifier weight {weight_name}."
            )
        start_weight = start_state[weight_name].to(
            device=end_weight.device,
            dtype=end_weight.dtype,
        )
        if start_weight.shape != end_weight.shape:
            raise RuntimeError(
                f"Client_{client_id} class-wise classifier shape mismatch: "
                f"start={tuple(start_weight.shape)}, "
                f"end={tuple(end_weight.shape)}."
            )
        return end_weight.detach() - start_weight

    def _flatten_classifier_delta(self, model, client_id):
        client = self.clients[client_id]
        start_state = getattr(client, "classifier_start_state", None)
        if start_state is None:
            raise RuntimeError(
                f"Client_{client_id} has no current-round classifier start snapshot."
            )

        delta_parts = []
        end_names = []
        for name, end_param in model.head.named_parameters():
            end_names.append(name)
            if name not in start_state:
                raise RuntimeError(
                    f"Client_{client_id} classifier start snapshot is missing {name}."
                )
            start_param = start_state[name].to(
                device=end_param.device, dtype=end_param.dtype
            )
            if start_param.shape != end_param.shape:
                raise RuntimeError(
                    f"Client_{client_id} classifier shape mismatch for {name}: "
                    f"start={tuple(start_param.shape)}, end={tuple(end_param.shape)}."
                )
            delta_parts.append((end_param.detach() - start_param).reshape(-1))

        if set(end_names) != set(start_state):
            extra_names = sorted(set(start_state) - set(end_names))
            raise RuntimeError(
                f"Client_{client_id} classifier start snapshot has extra "
                f"parameters: {extra_names}."
            )
        if not delta_parts:
            raise RuntimeError("FedCLIP classifier head has no parameters.")
        return torch.cat(delta_parts)

    @staticmethod
    def _pairwise_cosine(vectors):
        num_clients = len(vectors)
        similarity = torch.empty(
            (num_clients, num_clients),
            device=vectors[0].device,
            dtype=vectors[0].dtype,
        )
        for row in range(num_clients):
            for col in range(row, num_clients):
                cosine = F.cosine_similarity(
                    vectors[row], vectors[col], dim=0, eps=1e-8
                )
                similarity[row, col] = cosine
                similarity[col, row] = cosine
        return torch.nan_to_num(similarity, nan=0.0, posinf=1.0, neginf=-1.0)

    @staticmethod
    def _pairwise_classwise_cosine(weight_matrices):
        reference_shape = weight_matrices[0].shape
        if any(weight.shape != reference_shape for weight in weight_matrices):
            raise RuntimeError(
                "Class-wise classifier weights must have the same shape, got "
                f"{[tuple(weight.shape) for weight in weight_matrices]}."
            )
        stacked = torch.stack(weight_matrices, dim=0)
        normalized = F.normalize(stacked, p=2, dim=2, eps=1e-8)
        per_class_similarity = torch.einsum(
            "icd,jcd->ijc", normalized, normalized
        )
        per_class_similarity = torch.nan_to_num(
            per_class_similarity,
            nan=0.0,
            posinf=1.0,
            neginf=-1.0,
        )
        return per_class_similarity.mean(dim=2), per_class_similarity

    @staticmethod
    def _off_diagonal_values(matrix):
        if matrix.shape[0] <= 1:
            return matrix.reshape(-1)
        mask = ~torch.eye(
            matrix.shape[0], device=matrix.device, dtype=torch.bool
        )
        return matrix[mask]

    def _save_classifier_diagnostics(
        self,
        mode,
        similarity,
        lambda_weights,
        per_class_similarity=None,
    ):
        entropy = -(
            lambda_weights
            * torch.log(lambda_weights.clamp_min(1e-12))
        ).sum(dim=1)
        lambda_self_weight = torch.diagonal(lambda_weights)
        off_diagonal = self._off_diagonal_values(similarity)

        summary = {
            "round": int(self.cur_ground),
            "classifier_similarity_mode": mode,
            "local_classifier": int(
                bool(getattr(self.args, "local_classifier", 0))
            ),
            "mean_offdiag_similarity": float(off_diagonal.mean().item()),
            "std_offdiag_similarity": float(
                off_diagonal.std(unbiased=False).item()
            ),
            "min_similarity": float(off_diagonal.min().item()),
            "max_similarity": float(off_diagonal.max().item()),
            "mean_self_weight": float(lambda_self_weight.mean().item()),
            "mean_entropy": float(entropy.mean().item()),
        }
        if per_class_similarity is not None:
            if per_class_similarity.shape[0] > 1:
                classwise_mask = ~torch.eye(
                    per_class_similarity.shape[0],
                    device=per_class_similarity.device,
                    dtype=torch.bool,
                )
                per_class_pairs = per_class_similarity[classwise_mask]
            else:
                per_class_pairs = per_class_similarity.reshape(
                    -1, per_class_similarity.shape[2]
                )
            per_class_means = per_class_pairs.mean(dim=0)
            per_class_stds = per_class_pairs.std(dim=0, unbiased=False)
            summary["mean_per_class_similarity"] = float(
                per_class_means.mean().item()
            )
            summary["mean_per_class_similarity_std"] = float(
                per_class_stds.mean().item()
            )
        print(
            "[Classifier Similarity] "
            f"round={summary['round']} | mode={mode} | "
            f"local_classifier={summary['local_classifier']} | "
            f"offdiag_mean={summary['mean_offdiag_similarity']:.6f} | "
            f"offdiag_std={summary['std_offdiag_similarity']:.6f} | "
            f"sim_min={summary['min_similarity']:.6f} | "
            f"sim_max={summary['max_similarity']:.6f} | "
            f"self_weight_mean={summary['mean_self_weight']:.6f} | "
            f"entropy_mean={summary['mean_entropy']:.6f}"
        )

        diagnostic_dir = os.path.join(
            self.save_folder_name, "classifier_similarity_diagnostics"
        )
        os.makedirs(diagnostic_dir, exist_ok=True)
        summary_path = os.path.join(diagnostic_dir, "summary.csv")
        write_header = not os.path.exists(summary_path)
        with open(summary_path, "a", newline="", encoding="utf-8") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=list(summary))
            if write_header:
                writer.writeheader()
            writer.writerow(summary)

        similarity_np = similarity.detach().cpu().numpy()
        weights_np = lambda_weights.detach().cpu().numpy()
        round_tag = f"round_{self.cur_ground:03d}"
        client_ids_np = np.asarray(self.uploaded_ids, dtype=np.int64)
        np.save(
            os.path.join(diagnostic_dir, f"similarity_{round_tag}.npy"),
            similarity_np,
        )
        np.save(
            os.path.join(diagnostic_dir, f"lambda_{round_tag}.npy"),
            weights_np,
        )
        np.save(
            os.path.join(diagnostic_dir, f"client_ids_{round_tag}.npy"),
            client_ids_np,
        )
        for matrix_name, matrix in (
            ("similarity", similarity_np),
            ("lambda", weights_np),
        ):
            matrix_path = os.path.join(
                diagnostic_dir, f"{matrix_name}_{round_tag}.csv"
            )
            with open(
                matrix_path, "w", newline="", encoding="utf-8"
            ) as matrix_file:
                writer = csv.writer(matrix_file)
                writer.writerow(["client_id", *self.uploaded_ids])
                for client_id, row in zip(self.uploaded_ids, matrix):
                    writer.writerow([client_id, *row.tolist()])

        should_save_classwise = (
            per_class_similarity is not None
            and (
                self.cur_ground % 10 == 0
                or self.cur_ground == self.global_rounds
            )
        )
        if should_save_classwise:
            per_class_np = per_class_similarity.detach().cpu().numpy()
            if len(self.uploaded_ids) > 1:
                mask = ~np.eye(len(self.uploaded_ids), dtype=bool)
                per_class_pairs = per_class_np[mask]
            else:
                per_class_pairs = per_class_np.reshape(
                    -1, per_class_np.shape[-1]
                )
            per_class_mean = per_class_pairs.mean(axis=0)
            per_class_std = per_class_pairs.std(axis=0)
            per_class_path = os.path.join(
                diagnostic_dir,
                f"per_class_summary_{round_tag}.csv",
            )
            with open(
                per_class_path, "w", newline="", encoding="utf-8"
            ) as per_class_file:
                writer = csv.writer(per_class_file)
                writer.writerow(
                    ["class_id", "mean_similarity", "std_similarity"]
                )
                for class_id, (class_mean, class_std) in enumerate(
                    zip(per_class_mean, per_class_std)
                ):
                    writer.writerow([class_id, class_mean, class_std])
            print(
                "[Classifier Classwise] "
                f"round={self.cur_ground} | "
                f"class_mean={per_class_mean.mean():.6f} | "
                f"class_std_mean={per_class_std.mean():.6f}"
            )

    def aggregate_classifier_similarity(self, mode):
        assert len(self.uploaded_ids) > 0
        mode = self._canonical_classifier_mode(mode)
        supported_modes = {
            "flat",
            "flat_delta",
            "classwise",
            "classwise_delta",
        }
        if mode not in supported_modes:
            raise ValueError(f"Unsupported classifier similarity mode: {mode}")

        tau_value = getattr(self.args, "classifier_similarity_tau", None)
        if tau_value is None:
            tau_value = getattr(self.args, "aggregate_tau", 1.0)
        tau = float(tau_value)
        if tau <= 0.0:
            raise ValueError(
                f"classifier_similarity_tau must be positive, got {tau}."
            )

        recovered_models = self._load_recovered_uploaded_models()
        per_class_similarity = None
        if mode == "flat":
            classifier_vectors = [
                self._flatten_classifier(model)
                for model in recovered_models
            ]
            similarity = self._pairwise_cosine(classifier_vectors)
        elif mode == "flat_delta":
            classifier_vectors = [
                self._flatten_classifier_delta(model, client_id)
                for model, client_id in zip(
                    recovered_models, self.uploaded_ids
                )
            ]
            similarity = self._pairwise_cosine(classifier_vectors)
        elif mode == "classwise":
            classifier_weights = [
                self._classifier_weight(model)[1].detach()
                for model in recovered_models
            ]
            similarity, per_class_similarity = (
                self._pairwise_classwise_cosine(classifier_weights)
            )
        else:
            classifier_delta_weights = [
                self._classifier_delta_weight(model, client_id)
                for model, client_id in zip(
                    recovered_models, self.uploaded_ids
                )
            ]
            similarity, per_class_similarity = (
                self._pairwise_classwise_cosine(
                    classifier_delta_weights
                )
            )

        lambda_weights = torch.softmax(similarity / tau, dim=1)
        sample_weights = torch.tensor(
            self.uploaded_weights,
            device=self.device,
            dtype=lambda_weights.dtype,
        )
        row_sums = lambda_weights.sum(dim=1)
        if not torch.allclose(
            row_sums, torch.ones_like(row_sums), atol=1e-6, rtol=1e-6
        ):
            raise RuntimeError(
                "Classifier personalized aggregation weights do not sum to one."
            )

        # Keep the ordinary sample-weighted model for first-round and
        # non-participating-client fallback.
        global_model = load_item(
            self.role, 'model', self.save_folder_name
        )
        if global_model is None:
            raise RuntimeError(
                "Server global model is missing before classifier aggregation."
            )
        global_model = global_model.to(self.device)
        local_classifier = bool(
            int(getattr(self.args, "local_classifier", 0))
        )
        source_param_dicts = None
        if not local_classifier:
            for param in global_model.parameters():
                param.data.zero_()
            for sample_weight, source_model in zip(
                sample_weights, recovered_models
            ):
                for server_param, client_param in zip(
                    global_model.parameters(), source_model.parameters()
                ):
                    server_param.data += (
                        client_param.data.clone() * sample_weight
                    )
        else:
            global_params = dict(global_model.named_parameters())
            source_param_dicts = [
                dict(source_model.named_parameters())
                for source_model in recovered_models
            ]
            classifier_names = self._classifier_parameter_names(global_model)
            for source_idx, source_params in enumerate(source_param_dicts):
                if source_params.keys() != global_params.keys():
                    raise RuntimeError(
                        f"Client_{self.uploaded_ids[source_idx]} recovered "
                        "model is incompatible with the server model."
                    )
            for name, global_param in global_params.items():
                if name in classifier_names:
                    continue
                global_param.data.zero_()
                for sample_weight, source_params in zip(
                    sample_weights, source_param_dicts
                ):
                    global_param.data += (
                        source_params[name].data.clone() * sample_weight
                    )
        save_item(global_model, self.role, 'model', self.save_folder_name)

        for target_idx, target_client_id in enumerate(self.uploaded_ids):
            personalized_model = copy.deepcopy(recovered_models[target_idx])
            if not local_classifier:
                for param in personalized_model.parameters():
                    param.data.zero_()
                for source_weight, source_model in zip(
                    lambda_weights[target_idx], recovered_models
                ):
                    for target_param, source_param in zip(
                        personalized_model.parameters(), source_model.parameters()
                    ):
                        target_param.data += (
                            source_param.data.clone() * source_weight
                        )
            else:
                target_params = dict(personalized_model.named_parameters())
                classifier_names = self._classifier_parameter_names(
                    personalized_model
                )
                for name, target_param in target_params.items():
                    if name in classifier_names:
                        continue
                    target_param.data.zero_()
                    for source_weight, source_params in zip(
                        lambda_weights[target_idx], source_param_dicts
                    ):
                        target_param.data += (
                            source_params[name].data.clone() * source_weight
                        )
            save_item(
                personalized_model,
                self.role,
                f'model_{target_client_id}',
                self.save_folder_name,
            )

        self._save_classifier_diagnostics(
            mode,
            similarity,
            lambda_weights,
            per_class_similarity=per_class_similarity,
        )

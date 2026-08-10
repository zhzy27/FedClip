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

    def aggregate_avg(self):
        assert len(self.uploaded_ids) > 0

        global_model = load_item(
            self.role, 'model', self.save_folder_name
        ).to(self.device)
        for param in global_model.parameters():
            param.data.zero_()

        uploaded_base_models = []
        for client_id in self.uploaded_ids:
            client = self.clients[client_id]
            client_model = load_item(
                client.role, 'model', client.save_folder_name
            )
            recovered_model = copy.deepcopy(client_model)
            recovered_model.recover_larger_model()
            recovered_model.to(self.device)
            uploaded_base_models.append(recovered_model.base)

        print(f"执行权重聚合，聚合权重为{self.uploaded_weights}")
        for weight, base_model in zip(
            self.uploaded_weights, uploaded_base_models
        ):
            weight_tensor = torch.tensor(weight, device=self.device)
            for server_param, client_param in zip(
                global_model.base.parameters(), base_model.parameters()
            ):
                server_param.data += client_param.data.clone() * weight_tensor

        save_item(global_model, self.role, 'model', self.save_folder_name)

    def _load_recovered_uploaded_models(self):
        recovered_models = []
        for client_id in self.uploaded_ids:
            client = self.clients[client_id]
            client_model = load_item(
                client.role, 'model', client.save_folder_name
            )
            recovered_model = copy.deepcopy(client_model)
            recovered_model.recover_larger_model()
            recovered_model.to(self.device)
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

    def _save_classifier_diagnostics(
        self, mode, similarity, lambda_weights, final_weights
    ):
        entropy = -(
            lambda_weights
            * torch.log(lambda_weights.clamp_min(1e-12))
        ).sum(dim=1)
        lambda_self_weight = torch.diagonal(lambda_weights)
        final_self_weight = torch.diagonal(final_weights)

        summary = {
            "round": int(self.cur_ground),
            "classifier_similarity_mode": mode,
            "mean_cosine": float(similarity.mean().item()),
            "min_cosine": float(similarity.min().item()),
            "max_cosine": float(similarity.max().item()),
            "mean_self_weight": float(lambda_self_weight.mean().item()),
            "mean_final_self_weight": float(final_self_weight.mean().item()),
            "mean_entropy": float(entropy.mean().item()),
        }
        print(
            "[Classifier Similarity] "
            f"round={summary['round']} | mode={mode} | "
            f"cos_mean={summary['mean_cosine']:.6f} | "
            f"cos_min={summary['min_cosine']:.6f} | "
            f"cos_max={summary['max_cosine']:.6f} | "
            f"self_weight_mean={summary['mean_self_weight']:.6f} | "
            f"final_self_weight_mean={summary['mean_final_self_weight']:.6f} | "
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

        should_save_matrix = (
            self.cur_ground % 10 == 0
            or self.cur_ground == self.global_rounds
        )
        if not should_save_matrix:
            return

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

    def aggregate_classifier_similarity(self, mode):
        assert len(self.uploaded_ids) > 0
        if mode not in {"classifier", "classifier_delta"}:
            raise ValueError(f"Unsupported classifier similarity mode: {mode}")

        tau = float(getattr(self.args, "aggregate_tau", 1.0))
        d_max = float(getattr(self.args, "d_max", 0.7))
        if tau <= 0.0:
            raise ValueError(f"aggregate_tau must be positive, got {tau}.")
        if not 0.0 <= d_max <= 1.0:
            raise ValueError(f"d_max must be in [0, 1], got {d_max}.")

        recovered_models = self._load_recovered_uploaded_models()
        if mode == "classifier":
            classifier_vectors = [
                self._flatten_classifier(model)
                for model in recovered_models
            ]
        else:
            classifier_vectors = [
                self._flatten_classifier_delta(model, client_id)
                for model, client_id in zip(
                    recovered_models, self.uploaded_ids
                )
            ]

        similarity = self._pairwise_cosine(classifier_vectors)
        lambda_weights = torch.softmax(similarity / tau, dim=1)
        sample_weights = torch.tensor(
            self.uploaded_weights,
            device=self.device,
            dtype=lambda_weights.dtype,
        )
        final_weights = (
            (1.0 - d_max) * sample_weights.unsqueeze(0)
            + d_max * lambda_weights
        )
        row_sums = final_weights.sum(dim=1)
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
        ).to(self.device)
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
        save_item(global_model, self.role, 'model', self.save_folder_name)

        for target_idx, target_client_id in enumerate(self.uploaded_ids):
            personalized_model = copy.deepcopy(recovered_models[target_idx])
            for param in personalized_model.parameters():
                param.data.zero_()
            for source_weight, source_model in zip(
                final_weights[target_idx], recovered_models
            ):
                for target_param, source_param in zip(
                    personalized_model.parameters(), source_model.parameters()
                ):
                    target_param.data += (
                        source_param.data.clone() * source_weight
                    )
            save_item(
                personalized_model,
                self.role,
                f'model_{target_client_id}',
                self.save_folder_name,
            )

        self._save_classifier_diagnostics(
            mode, similarity, lambda_weights, final_weights
        )

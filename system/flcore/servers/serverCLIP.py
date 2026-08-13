import random
import time

import torch

from flcore.clients.clientCLIP import clientCLIP
from flcore.clients.clientbase import load_item, save_item
from flcore.servers.serverbase import Server
from flcore.trainmodel.models import Model_Distribe


class FedCLIP(Server):
    """FedCLIP server using sample-weighted averaging in full-weight space."""

    def __init__(self, args, times):
        super().__init__(args, times)
        self.set_slow_clients()
        self.set_clients(clientCLIP)
        print(
            f"\nJoin ratio / total clients: "
            f"{self.join_ratio} / {self.num_clients}"
        )
        print("Finished creating server and clients.")

        self.Budget = []
        global_model = Model_Distribe(args, -1, is_global=True).to(self.device)
        global_model.recover_larger_model()
        save_item(global_model, self.role, "model", self.save_folder_name)

    def train(self):
        for current_round in range(self.global_rounds + 1):
            self.cur_ground = current_round
            round_start = time.time()
            self.selected_clients = self.select_clients()

            if current_round > 0 and current_round % self.eval_gap == 0:
                print(f"\n-------------Round number: {current_round}-------------")
                print("\nEvaluate heterogeneous models")
                self.evaluate(epoch=current_round)

            self.send_parameters()
            self._train_selected_clients(current_round)
            self.receive_ids()

            self._synchronize_cuda()
            aggregation_start = time.time()
            self.aggregate_parameters_avg()
            self._synchronize_cuda()
            print(
                f"[Round {current_round:03d}] aggregation time: "
                f"{time.time() - aggregation_start:.3f}s"
            )

            self.Budget.append(time.time() - round_start)
            print(
                "-" * 25,
                "time cost",
                "-" * 25,
                self.Budget[-1],
            )
            if self.auto_break and self.check_done(
                acc_lss=[self.rs_test_acc], top_cnt=self.top_cnt
            ):
                break

        if self.rs_test_acc:
            print("\nBest accuracy.")
            print(max(self.rs_test_acc))
        measured_rounds = self.Budget[1:] or self.Budget
        if measured_rounds:
            print("\nAverage time cost per round.")
            print(sum(measured_rounds) / len(measured_rounds))

        self.save_results()
        self.save_json_file()

    def _train_selected_clients(self, current_round):
        self._synchronize_cuda()
        wall_start = time.time()
        client_times = []
        subspace_stats = []
        for client in self.selected_clients:
            train_time = client.train(current_round=current_round)
            client_times.append(float(train_time or 0.0))
            if client.last_u_subspace_stats is not None:
                subspace_stats.append(client.last_u_subspace_stats)
        self._synchronize_cuda()
        print(
            f"[Round {current_round:03d}] local training: "
            f"sum={sum(client_times):.3f}s | "
            f"wall={time.time() - wall_start:.3f}s | "
            f"clients={len(client_times)}"
        )
        if subspace_stats:
            mean_loss = sum(
                item["mean_loss"] for item in subspace_stats
            ) / len(subspace_stats)
            mean_drift = sum(
                item["drift_norm"] for item in subspace_stats
            ) / len(subspace_stats)
            print(
                f"[USubspaceRegSummary] round={current_round} "
                f"mean_loss={mean_loss:.6e} "
                f"mean_drift_norm={mean_drift:.6e} "
                f"clients={len(subspace_stats)}"
            )

    def _synchronize_cuda(self):
        if torch.cuda.is_available() and str(self.device).startswith("cuda"):
            torch.cuda.synchronize(self.device)

    def receive_ids(self):
        if not self.selected_clients:
            raise RuntimeError("No clients were selected for aggregation.")
        num_active = int(
            (1 - self.client_drop_rate) * self.current_num_join_clients
        )
        if num_active <= 0:
            raise RuntimeError("No active clients remain after client dropout.")
        active_clients = random.sample(self.selected_clients, num_active)

        total_samples = sum(client.train_samples for client in active_clients)
        if total_samples <= 0:
            raise RuntimeError("Active clients contain no training samples.")
        self.uploaded_ids = [client.id for client in active_clients]
        self.uploaded_weights = [
            client.train_samples / total_samples for client in active_clients
        ]

    def send_parameters(self):
        if not self.selected_clients:
            raise RuntimeError("No clients were selected for model dispatch.")
        for client in self.selected_clients:
            start_time = time.time()
            client.set_parameters()
            client.send_time_cost["num_rounds"] += 1
            client.send_time_cost["total_cost"] += 2 * (
                time.time() - start_time
            )

    @staticmethod
    def _has_low_rank_params(model):
        return any(
            name.endswith("conv_v") or name.endswith("weight_v")
            for name, _ in model.named_parameters()
        )

    def _recover_if_needed(self, model):
        if self._has_low_rank_params(model):
            model.recover_larger_model()
        return model

    def aggregate_parameters_avg(self):
        if not self.uploaded_ids:
            raise RuntimeError("No client model was uploaded for Avg.")

        global_model = load_item(self.role, "model", self.save_folder_name)
        if global_model is None:
            raise RuntimeError("Server global model is missing before Avg.")
        global_model = global_model.to(self.device)
        self._recover_if_needed(global_model)
        global_params = dict(global_model.named_parameters())
        expected_names = list(global_params)

        with torch.no_grad():
            for global_param in global_params.values():
                global_param.zero_()

        for client_id, weight in zip(
            self.uploaded_ids, self.uploaded_weights
        ):
            client = self.clients[client_id]
            client_model = load_item(
                client.role, "model", client.save_folder_name
            )
            if client_model is None:
                raise RuntimeError(
                    f"Client_{client_id} uploaded model is missing."
                )
            full_model = client_model.to(self.device)
            self._recover_if_needed(full_model)
            source_params = dict(full_model.named_parameters())
            if list(source_params) != expected_names:
                raise RuntimeError(
                    f"Client_{client_id} full model is "
                    "incompatible with the server model."
                )
            with torch.no_grad():
                for name, global_param in global_params.items():
                    source_param = source_params[name]
                    if source_param.shape != global_param.shape:
                        raise RuntimeError(
                            f"Avg shape mismatch for {name}: "
                            f"server={tuple(global_param.shape)}, "
                            f"client={tuple(source_param.shape)}"
                        )
                    global_param.add_(source_param, alpha=weight)

        save_item(global_model, self.role, "model", self.save_folder_name)
        print(f"Avg weights: {self.uploaded_weights}")

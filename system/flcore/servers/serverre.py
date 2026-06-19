import random
import time

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from flcore.clients.clientbase import load_item, save_item
from flcore.clients.clientre import clientRE
from flcore.servers.serverbase import Server


def _sanitize_tensor(tensor, tag, clamp_value=1e4):
    if torch.isfinite(tensor).all():
        return tensor
    print(f"⚠️ FedRE {tag} 出现 NaN/Inf，已执行 nan_to_num。")
    return torch.nan_to_num(tensor, nan=0.0, posinf=clamp_value, neginf=-clamp_value)


def _sanitize_module_parameters_(module, tag, clamp_value=1e4):
    with torch.no_grad():
        for name, param in module.named_parameters():
            if param is not None and not torch.isfinite(param).all():
                print(f"⚠️ FedRE {tag}.{name} 参数出现 NaN/Inf，已执行 nan_to_num。")
                param.data = torch.nan_to_num(param.data, nan=0.0, posinf=clamp_value, neginf=-clamp_value)


def _sanitize_module_gradients_(module, tag, clamp_value=1e4):
    for name, param in module.named_parameters():
        if param.grad is not None and not torch.isfinite(param.grad).all():
            print(f"⚠️ FedRE {tag}.{name} 梯度出现 NaN/Inf，已执行 nan_to_num。")
            param.grad.data = torch.nan_to_num(param.grad.data, nan=0.0, posinf=clamp_value, neginf=-clamp_value)


def _soft_cross_entropy(logits, soft_targets):
    log_probs = F.log_softmax(logits, dim=1)
    return -(soft_targets * log_probs).sum(dim=1).mean()


class FedRE(Server):
    def __init__(self, args, times):
        super().__init__(args, times)

        self.set_slow_clients()
        self.set_clients(clientRE)

        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print("Finished creating server and clients.")

        self.Budget = []
        self.server_learning_rate = args.server_learning_rate
        self.server_epochs = args.server_epochs

        head = load_item(self.clients[0].role, 'model', self.clients[0].save_folder_name).head.to(self.device)
        _sanitize_module_parameters_(head, "Server.head")
        save_item(head, 'Server', 'head', self.save_folder_name)

    def train(self):
        for i in range(self.global_rounds + 1):
            s_t = time.time()
            self.selected_clients = self.select_clients()

            if i % self.eval_gap == 0:
                print(f"\n-------------Round number: {i}-------------")
                print("\nEvaluate heterogeneous models")
                self.evaluate(epoch=i)

            self.send_select_client_parameters()
            for client in self.selected_clients:
                client.train()

            self.receive_entangled_reps()
            self.train_head()

            self.Budget.append(time.time() - s_t)
            print('-' * 50, self.Budget[-1])

            if self.auto_break and self.check_done(acc_lss=[self.rs_test_acc], top_cnt=self.top_cnt):
                break

        print("\nBest accuracy.")
        print(max(self.rs_test_acc))
        print(sum(self.Budget[1:]) / len(self.Budget[1:]))

        self.save_results()
        self.save_json_file()

    def send_select_client_parameters(self):
        assert (len(self.clients) > 0)

        for client in self.selected_clients:
            start_time = time.time()
            client.set_parameters()

            client.send_time_cost['num_rounds'] += 1
            client.send_time_cost['total_cost'] += 2 * (time.time() - start_time)

    def receive_entangled_reps(self):
        assert (len(self.selected_clients) > 0)
        active_clients = random.sample(
            self.selected_clients, int((1 - self.client_drop_rate) * self.current_num_join_clients))

        self.uploaded_ids = []
        self.uploaded_weights = []
        tot_samples = 0
        uploaded_reps = []

        for client in active_clients:
            entangled_items = load_item(client.role, 'entangled_reps', client.save_folder_name)
            if entangled_items is None or len(entangled_items) == 0:
                print(f"⚠️ FedRE 未读取到 {client.role} 的 entangled representation，跳过该客户端。")
                continue

            tot_samples += client.train_samples
            self.uploaded_ids.append(client.id)
            self.uploaded_weights.append(client.train_samples)

            for rep, soft_label in entangled_items:
                rep = _sanitize_tensor(rep.detach().to(self.device), f"{client.role} entangled rep")
                soft_label = _sanitize_tensor(soft_label.detach().to(self.device), f"{client.role} entangled label")
                label_sum = soft_label.sum().clamp_min(1e-12)
                soft_label = soft_label / label_sum
                uploaded_reps.append((rep, soft_label))

        if tot_samples == 0 or len(uploaded_reps) == 0:
            print("⚠️ FedRE 本轮没有可用 entangled representation，服务器 head 将保持不变。")
            save_item([], self.role, 'uploaded_entangled_reps', self.save_folder_name)
            return

        for i, w in enumerate(self.uploaded_weights):
            self.uploaded_weights[i] = w / tot_samples
        save_item(uploaded_reps, self.role, 'uploaded_entangled_reps', self.save_folder_name)

    def train_head(self):
        uploaded_reps = load_item(self.role, 'uploaded_entangled_reps', self.save_folder_name)
        if uploaded_reps is None or len(uploaded_reps) == 0:
            print("⚠️ FedRE 没有 uploaded_entangled_reps，跳过服务器 head 训练。")
            return

        rep_loader = DataLoader(uploaded_reps, self.batch_size, drop_last=False, shuffle=True)
        head = load_item('Server', 'head', self.save_folder_name).to(self.device)
        _sanitize_module_parameters_(head, "Server.head")

        opt_h = torch.optim.SGD(head.parameters(), lr=self.server_learning_rate)

        for _ in range(self.server_epochs):
            for rep, soft_label in rep_loader:
                rep = _sanitize_tensor(rep.to(self.device), "server entangled rep batch")
                soft_label = _sanitize_tensor(soft_label.to(self.device), "server entangled label batch")
                soft_label = soft_label / soft_label.sum(dim=1, keepdim=True).clamp_min(1e-12)

                out = head(rep)
                loss = _soft_cross_entropy(out, soft_label)
                if not torch.isfinite(loss):
                    print("⚠️ FedRE 服务器 head loss 非有限，跳过当前 entangled batch 并清理 head 参数。")
                    _sanitize_module_parameters_(head, "Server.head")
                    continue

                opt_h.zero_grad()
                loss.backward()
                _sanitize_module_gradients_(head, "Server.head")
                opt_h.step()
                _sanitize_module_parameters_(head, "Server.head")

        save_item(head, 'Server', 'head', self.save_folder_name)

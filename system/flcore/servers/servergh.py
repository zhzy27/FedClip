import time
import random
import torch
import torch.nn as nn
from flcore.clients.clientgh import clientGH
from flcore.servers.serverbase import Server
from flcore.clients.clientbase import load_item, save_item
from threading import Thread
from torch.utils.data import DataLoader


def _sanitize_tensor(tensor, tag, clamp_value=1e4):
    if torch.isfinite(tensor).all():
        return tensor
    print(f"⚠️ FedGH {tag} 出现 NaN/Inf，已执行 nan_to_num。")
    return torch.nan_to_num(tensor, nan=0.0, posinf=clamp_value, neginf=-clamp_value)


def _sanitize_module_parameters_(module, tag, clamp_value=1e4):
    with torch.no_grad():
        for name, param in module.named_parameters():
            if param is not None and not torch.isfinite(param).all():
                print(f"⚠️ FedGH {tag}.{name} 参数出现 NaN/Inf，已执行 nan_to_num。")
                param.data = torch.nan_to_num(param.data, nan=0.0, posinf=clamp_value, neginf=-clamp_value)


def _sanitize_module_gradients_(module, tag, clamp_value=1e4):
    for name, param in module.named_parameters():
        if param.grad is not None and not torch.isfinite(param.grad).all():
            print(f"⚠️ FedGH {tag}.{name} 梯度出现 NaN/Inf，已执行 nan_to_num。")
            param.grad.data = torch.nan_to_num(param.grad.data, nan=0.0, posinf=clamp_value, neginf=-clamp_value)


class FedGH(Server):
    def __init__(self, args, times):
        super().__init__(args, times)

        # select slow clients
        self.set_slow_clients()
        #设置客户端对象
        self.set_clients(clientGH)

        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print("Finished creating server and clients.")

        # self.load_model()
        self.Budget = []
        self.CEloss = nn.CrossEntropyLoss()
        #服务器学习率设置
        self.server_learning_rate = args.server_learning_rate
        #服务器训练的批次
        self.server_epochs = args.server_epochs
        #服务器聚合的head参数
        head = load_item(self.clients[0].role, 'model', self.clients[0].save_folder_name).head.to(self.device)
        _sanitize_module_parameters_(head, "Server.head")
        save_item(head, 'Server', 'head', self.save_folder_name)


    def train(self):
        for i in range(self.global_rounds+1):
            s_t = time.time()
            self.selected_clients = self.select_clients()
            # self.send_parameters()
            # self.send_select_client_parameters()
            if i%self.eval_gap == 0:
                print(f"\n-------------Round number: {i}-------------")
                print("\nEvaluate heterogeneous models")
                self.evaluate(epoch=i)
            self.send_select_client_parameters() #先测再分发相当于测试本地上传前的
            for client in self.selected_clients:
                client.train()
                client.collect_protos()

            # threads = [Thread(target=client.train)
            #            for client in self.selected_clients]
            # [t.start() for t in threads]
            # [t.join() for t in threads]
            #客户端接受原型
            self.receive_protos()
            #服务器使用接受的原型训练head
            self.train_head()

            self.Budget.append(time.time() - s_t)
            print('-'*50, self.Budget[-1])

            if self.auto_break and self.check_done(acc_lss=[self.rs_test_acc], top_cnt=self.top_cnt):
                break

        print("\nBest accuracy.")
        # self.print_(max(self.rs_test_acc), max(
        #     self.rs_train_acc), min(self.rs_train_loss))
        print(max(self.rs_test_acc))
        print(sum(self.Budget[1:])/len(self.Budget[1:]))

        self.save_results()
        # self.writer.close()
        self.save_json_file()
    #发送模型参数（之后可能会修改，因为测试方法要保持一致，训练完后测试个性化性能）
    def send_select_client_parameters(self):
        assert (len(self.clients) > 0)
        #选中的客户端接受base参数
        for client in self.selected_clients:
            start_time = time.time()
            client.set_parameters()

            client.send_time_cost['num_rounds'] += 1
            client.send_time_cost['total_cost'] += 2 * (time.time() - start_time)
    def receive_protos(self):
        assert (len(self.selected_clients) > 0)
        active_clients = random.sample(
            self.selected_clients, int((1-self.client_drop_rate) * self.current_num_join_clients))

        self.uploaded_ids = []
        self.uploaded_weights = []
        tot_samples = 0
        uploaded_protos = []
        for client in active_clients:
            tot_samples += client.train_samples
            self.uploaded_ids.append(client.id)
            self.uploaded_weights.append(client.train_samples)
            protos = load_item(client.role, 'protos', client.save_folder_name)
            if protos is None:
                print(f"⚠️ FedGH 未读取到 {client.role} 的原型，跳过该客户端。")
                continue
            for cc in protos.keys():
                y = torch.tensor(cc, dtype=torch.int64, device=self.device)
                proto = protos[cc].detach().to(self.device)
                proto = _sanitize_tensor(proto, f"{client.role} class {cc} proto")
                uploaded_protos.append((proto, y))
        if tot_samples == 0 or len(uploaded_protos) == 0:
            print("⚠️ FedGH 本轮没有可用原型，服务器 head 将保持不变。")
            save_item([], self.role, 'uploaded_protos', self.save_folder_name)
            return
        for i, w in enumerate(self.uploaded_weights):
            self.uploaded_weights[i] = w / tot_samples
        save_item(uploaded_protos, self.role, 'uploaded_protos', self.save_folder_name)
    #训练head
    def train_head(self):
        uploaded_protos = load_item(self.role, 'uploaded_protos', self.save_folder_name)
        if uploaded_protos is None or len(uploaded_protos) == 0:
            print("⚠️ FedGH 没有 uploaded_protos，跳过服务器 head 训练。")
            return
        proto_loader = DataLoader(uploaded_protos, self.batch_size, drop_last=False, shuffle=True)
        head = load_item('Server', 'head', self.save_folder_name).to(self.device)
        _sanitize_module_parameters_(head, "Server.head")
        
        opt_h = torch.optim.SGD(head.parameters(), lr=self.server_learning_rate)

        for _ in range(self.server_epochs):
            for p, y in proto_loader:
                p = _sanitize_tensor(p.to(self.device), "server proto batch")
                y = y.to(self.device)
                out = head(p)
                loss = self.CEloss(out, y)
                if not torch.isfinite(loss):
                    print("⚠️ FedGH 服务器 head loss 非有限，跳过当前 proto batch 并清理 head 参数。")
                    _sanitize_module_parameters_(head, "Server.head")
                    continue
                opt_h.zero_grad()
                loss.backward()
                _sanitize_module_gradients_(head, "Server.head")
                opt_h.step()
                _sanitize_module_parameters_(head, "Server.head")

        save_item(head, 'Server', 'head', self.save_folder_name)

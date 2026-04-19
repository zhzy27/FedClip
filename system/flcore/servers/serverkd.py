import copy
import os
import random
import time

import numpy as np
from flcore.clients.clientkd import clientKD, recover, decomposition
from flcore.servers.serverbase import Server
from flcore.clients.clientbase import load_item, save_item
from threading import Thread
from flcore.trainmodel.models import BaseHeadSplit
from flcore.trainmodel.models import Model_Distribe


class FedKD(Server):
    def __init__(self, args, times):
        super().__init__(args, times)
        if args.save_folder_name == 'temp' or 'temp' not in args.save_folder_name:
            if hasattr(args, 'global_model'):
                # global_model = BaseHeadSplit(args, is_global=True).to(args.device)
                global_model = Model_Distribe(args, -1, is_global=True).to(self.device)
            else:
                # global_model = BaseHeadSplit(args, 0).to(args.device)
                raise "没有指定全局模型"
            save_item(global_model, self.role, 'global_model', self.save_folder_name)
        #记录每个通信轮次传递的参数数量
        self.comm_params = []
        # select slow clients
        self.set_slow_clients()
        self.set_clients(clientKD)

        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print("Finished creating server and clients.")

        # self.load_model()
        self.Budget = []
        self.T_start = args.T_start
        self.T_end = args.T_end
        self.energy = self.T_start

    def train(self):
        for i in range(self.global_rounds + 1):
            s_t = time.time()
            self.selected_clients = self.select_clients()

            if i % self.eval_gap == 0:
                print(f"\n-------------Round number: {i}-------------")
                print("\nEvaluate heterogeneous models")
                self.evaluate(epoch=i)

            for client in self.selected_clients:
                client.train()

            # threads = [Thread(target=client.train)
            #            for client in self.selected_clients]
            # [t.start() for t in threads]
            # [t.join() for t in threads]

            self.receive_ids()
            self.aggregate_parameters()
            print(f"----------------已经花费的总的通信开销为{sum(self.comm_params)}")
            self.send_parameters()

            self.Budget.append(time.time() - s_t)
            print('-' * 25, 'time cost', '-' * 25, self.Budget[-1])

            if self.auto_break and self.check_done(acc_lss=[self.rs_test_acc], top_cnt=self.top_cnt):
                break

            self.energy = self.T_start + ((1 + i) / self.global_rounds) * (self.T_end - self.T_start)
            for client in self.clients:
                client.energy = self.energy

        print("\nBest accuracy.")
        # self.print_(max(self.rs_test_acc), max(
        #     self.rs_train_acc), min(self.rs_train_loss))
        print(max(self.rs_test_acc))
        print("\nAverage time cost per round.")
        print(sum(self.Budget[1:]) / len(self.Budget[1:]))

        self.save_results()
        # self.writer.close()
        self.save_json_file()

    def aggregate_parameters(self):
        assert (len(self.uploaded_ids) > 0)

        global_model = load_item(self.role, 'global_model', self.save_folder_name)
        global_param = {name: param.detach().cpu().numpy()
                        for name, param in global_model.named_parameters()}
        for k in global_param.keys():
            global_param[k] = np.zeros_like(global_param[k])
        clients_params = []
        for cid in self.uploaded_ids:
            client = self.clients[cid]
            compressed_param = load_item(client.role, 'compressed_param', client.save_folder_name)
            #计算客户端上传的参数开销
            param_nums = self.count_compressed_params(compressed_param)
            print(f"客户端{cid}上传的参数数量为：{param_nums}")
            clients_params.append(param_nums)
            client_param = recover(compressed_param)
            for server_k, client_k in zip(global_param.keys(), client_param.keys()):
                global_param[server_k] += client_param[client_k] * 1 / len(self.uploaded_ids)
        receive_total_params = sum(clients_params)
        print(f"所有客户端上传参数的数量为:{receive_total_params}")
        compressed_param = decomposition(global_param.items(), self.energy)
        send_params_nums= self.count_compressed_params(compressed_param)*len(self.uploaded_ids)
        #计算下发后的模型参数
        print(f"服务器下发给所有客户端模型的参数数量为:",send_params_nums)
        self.comm_params.append(receive_total_params+send_params_nums)
        print(f"该通信轮次总的通信参数量为：{receive_total_params+send_params_nums}")
        save_item(compressed_param, self.role, 'compressed_param', self.save_folder_name)

    def count_compressed_params(self,compressed_param):

        total_params = 0

        for name, param in compressed_param.items():
            if isinstance(param, list) and len(param) == 3:
                # SVD压缩的参数 [u, sigma, v]
                u, sigma, v = param
                total_params += np.prod(u.shape) + np.prod(sigma.shape) + np.prod(v.shape)
                print(f"{u.shape},{sigma.shape},{v.shape},{total_params}")
            else:
                # 未压缩的参数
                total_params += np.prod(param.shape)

        return total_params
    def save_json_file(self):
        dict = {
            "train_loss": self.rs_train_loss,
            "test_acc": self.rs_test_acc,
            "comm_cost":self.comm_params
        }
        filename = self.args.exp_name + ".json"
        filepath = os.path.join("./json", filename)
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        self.save_json(file_path=filepath, dict=dict)
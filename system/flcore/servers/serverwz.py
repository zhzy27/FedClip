import copy
import random
import time
from flcore.clients.clientwz import clientWZ
from flcore.servers.serverbase import Server
from flcore.clients.clientbase import load_item, save_item
from threading import Thread
from flcore.trainmodel.models import  Model_Distribe
import torch
class FedWZ(Server):
    def __init__(self, args, times):
        super().__init__(args, times)

        # select slow clients
        self.set_slow_clients()
        self.set_clients(clientWZ)

        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print("Finished creating server and clients.")

        # self.load_model()
        self.Budget = []
        #创建全局base用于之后聚合
        global_mdoel = Model_Distribe(args, -1,is_global=True).to(self.device)
        global_mdoel.recover_larger_model()
        save_item(global_mdoel, self.role, 'model', self.save_folder_name)


    def train(self):
        for i in range(self.global_rounds+1):
            s_t = time.time()
            self.selected_clients = self.select_clients()
            # #分发全局base  这个相当于测全局base+个性化head
            # self.send_parameters()
            #这个测试的是全局base+个性化head之后可能需要修改
            if i%self.eval_gap == 0:
                print(f"\n-------------Round number: {i}-------------")
                print("\nEvaluate heterogeneous models")
                self.evaluate(epoch=i)
            #这种逻辑是测试客户端微调的base+个性化head
            self.send_select_client_parameters()
            #被选中的客户端先进行参数对齐
            for client in self.selected_clients:
                if self.args.align_method ==0:
                    #mse对齐
                    client.align_base()
                else:
                    #mse+kl对齐
                    client.align_base_kl()
            for client in self.selected_clients:
                client.train()

            # threads = [Thread(target=client.train)
            #            for client in self.selected_clients]
            # [t.start() for t in threads]
            # [t.join() for t in threads]

            self.receive_ids()
            self.aggregate_parameters()

            self.Budget.append(time.time() - s_t)
            print('-'*25, 'time cost', '-'*25, self.Budget[-1])

            if self.auto_break and self.check_done(acc_lss=[self.rs_test_acc], top_cnt=self.top_cnt):
                break

        print("\nBest accuracy.")
        # self.print_(max(self.rs_test_acc), max(
        #     self.rs_train_acc), min(self.rs_train_loss))
        print(max(self.rs_test_acc))
        print("\nAverage time cost per round.")
        print(sum(self.Budget[1:])/len(self.Budget[1:]))

        self.save_results()
        # self.writer.close()
        self.save_json_file()

    #发送模型参数（之后可能会修改，因为测试方法要保持一致，训练完后测试个性化性能）
    def send_parameters(self):
        assert (len(self.clients) > 0)

        for client in self.clients:
            start_time = time.time()
            #有的客户端会实现
            client.set_parameters()

            client.send_time_cost['num_rounds'] += 1
            client.send_time_cost['total_cost'] += 2 * (time.time() - start_time)
    #发送模型参数（之后可能会修改，因为测试方法要保持一致，训练完后测试个性化性能）
    def send_select_client_parameters(self):
        assert (len(self.clients) > 0)
        #选中的客户端接受base参数
        for client in self.selected_clients:
            start_time = time.time()
            #有的客户端会实现
            client.set_parameters()

            client.send_time_cost['num_rounds'] += 1
            client.send_time_cost['total_cost'] += 2 * (time.time() - start_time)
    #从客户顿接受id信息和样本数信息
    def receive_ids(self):
        assert (len(self.selected_clients) > 0)

        active_clients = random.sample(
            self.selected_clients, int((1-self.client_drop_rate) * self.current_num_join_clients))

        self.uploaded_ids = []
        self.uploaded_weights = []
        tot_samples = 0
        for client in active_clients:
            tot_samples += client.train_samples
            self.uploaded_ids.append(client.id)
            self.uploaded_weights.append(client.train_samples)
        for i, w in enumerate(self.uploaded_weights):
            self.uploaded_weights[i] = w / tot_samples
    #客户端base进行参数对齐并且进行对齐后聚合
    def aggregate_parameters(self):
        assert (len(self.uploaded_ids) > 0)
        #载入全局模型,全局模型是完整模型状态
        global_model = load_item(self.role, 'model', self.save_folder_name).to(self.device)
        for param in global_model.parameters():
            param.data.zero_()
        #记录客户端恢复形状后的base模型
        self.uploaded_base_model = []

        for cid in  self.uploaded_ids:
            client = self.clients[cid]
            client_model = load_item(client.role, 'model', client.save_folder_name)
            #创建临时模型用于模型参数恢复
            model = copy.deepcopy(client_model)
            model.recover_larger_model()
            model.to(self.device)
            self.uploaded_base_model.append(model.base)
        for w,base_model in zip(self.uploaded_weights,self.uploaded_base_model):
            #将模型参数聚合
            for server_param, client_param in zip(global_model.base.parameters(), base_model.parameters()):
                w = torch.tensor(w).to(self.device)
                server_param.data += client_param.data.clone() * w

        save_item(global_model, self.role, 'model', self.save_folder_name)
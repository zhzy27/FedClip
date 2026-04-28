import copy
import random
import time
from flcore.clients.clientCLIP import clientCLIP
from flcore.servers.serverbase import Server
from flcore.clients.clientbase import load_item, save_item
from threading import Thread
from flcore.trainmodel.models import  Model_Distribe
import torch
from torch.utils.data import DataLoader
from sklearn.preprocessing import label_binarize
from sklearn import metrics
from utils.data_utils import read_client_data
from utils.get_clip_text_encoder import get_clip_class_embeddings
import numpy as np
class FedCLIP(Server):
    def __init__(self, args, times):
        super().__init__(args, times)

        # select slow clients
        self.set_slow_clients()
        self.set_clients(clientCLIP)

        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print("Finished creating server and clients.")

        # self.load_model()
        self.Budget = []
        #创建全局base用于之后聚合
        global_model = Model_Distribe(args, -1,is_global=True).to(self.device)
        global_model.recover_larger_model()
        self.global_acc=[]
        save_item(global_model, self.role, 'model', self.save_folder_name)
        clip_text_features,clip_text_features_norm = get_clip_class_embeddings(self.dataset,model_name= "ViT-B/32",prompt_template= "a photo of {}",device = self.device)
        self.clip_text_features,self.clip_text_features_norm = clip_text_features.float(),clip_text_features_norm.float()


    def train(self):
        for i in range(self.global_rounds+1):
            s_t = time.time()
            self.selected_clients = self.select_clients()
            # 下发就测试
            # self.send_parameters()
            if i%self.eval_gap == 0: # 测试间隔
                print(f"\n-------------Round number: {i}-------------")
                print("\nEvaluate heterogeneous models")
                self.evaluate(epoch=i)
            self.send_parameters()
            for client in self.selected_clients:
                client.train(current_round=i)

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
        
        print("\nBest Global accuracy.")
        # self.print_(max(self.rs_test_acc), max(
        #     self.rs_train_acc), min(self.rs_train_loss))
        if len(self.global_acc) > 0:
            print(max(self.global_acc))
        else:
            print("未记录 Global accuracy")
        print("\nAverage time cost per round.")
        print(sum(self.Budget[1:])/len(self.Budget[1:]))

        self.save_results()
        self.save_json_file()


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
    #发送模型参数（之后可能会修改，因为测试方法要保持一致，训练完后测试个性化性能）
    def send_parameters(self):
        assert (len(self.selected_clients) > 0)

        for client in self.selected_clients:
            start_time = time.time()
            #有的客户端会实现
            client.set_parameters()

            client.send_time_cost['num_rounds'] += 1
            client.send_time_cost['total_cost'] += 2 * (time.time() - start_time)


    # 客户端base进行参数对齐并且进行对齐后聚合
    # 客户端base进行个性化余弦相似度聚合
    def aggregate_parameters(self):
        assert (len(self.uploaded_ids) > 0)
        
        # 1. 记录客户端恢复形状后的 base 模型
        self.uploaded_base_model = []
        for cid in self.uploaded_ids:
            client = self.clients[cid]
            client_model = load_item(client.role, 'model', client.save_folder_name)
            # 创建临时模型用于模型参数恢复
            model = copy.deepcopy(client_model)
            model.recover_larger_model()
            model.to(self.device)
            self.uploaded_base_model.append(model.base)
        
        # 目的：为那些上一轮没参与、没有专属模型的客户端提供最新的“通用全局模型”做兜底
        general_global_model = load_item(self.role, 'model', self.save_folder_name).to(self.device)
        for param in general_global_model.parameters():
            param.data.zero_()
        for w, base_model in zip(self.uploaded_weights, self.uploaded_base_model):
            for server_param, client_param in zip(general_global_model.base.parameters(), base_model.parameters()):
                w_tensor = torch.tensor(w).to(self.device)
                server_param.data += client_param.data.clone() * w_tensor
        save_item(general_global_model, self.role, 'model', self.save_folder_name)

        # 2. 将每个模型的参数展平为 1D 向量，方便计算余弦相似度
        flat_params = []
        for base_model in self.uploaded_base_model:
            # 拼接所有参数为一个长向量
            vec = torch.cat([p.data.view(-1) for p in base_model.parameters()])
            flat_params.append(vec)

        print(f"执行基于余弦相似度的个性化聚合，参与客户端: {self.uploaded_ids}")
        
        # 温度系数：因为神经网络高维参数向量的余弦相似度通常都非常接近 1 (例如 0.999 和 0.998)
        # 如果直接 Softmax 会退化为平均权重。调小 tau (如 0.05 - 0.1) 可以放大相似度差异，使得更相似的模型获得显著更大的权重。
        tau = 0.00005

        # 3. 为每个上传的客户端计算专属的聚合权重，并生成个性化全局模型
        for i, target_cid in enumerate(self.uploaded_ids):
            sims = []
            # 3.1 计算第 i 个模型与其他所有参与聚合模型的余弦相似度
            for j in range(len(self.uploaded_ids)):
                sim = torch.nn.functional.cosine_similarity(flat_params[i], flat_params[j], dim=0)
                sims.append(sim)
            
            sims = torch.stack(sims) # [num_uploaded_clients]
            # 1. 减去最大值，防止除以极小的 tau 后指数爆炸 (Softmax平移不变性)
            # 2. 除以极小的 tau 放大细微差异
            sims_scaled = (sims - torch.max(sims)) / tau
            
            # 3.2 使用带温度系数的 Softmax 将相似度转化为权重分布 (和为1)
            weights = torch.nn.functional.softmax(sims_scaled, dim=0)
            
            # (可选) 打印出每个客户端的个性化权重分布，方便你观察
            print(f"  -> 客户端 {target_cid} 的聚合权重: {weights.cpu().numpy().round(3)}")

            # 3.3 载入一个干净的全局模型作为聚合容器
            personalized_global_model = load_item(self.role, 'model', self.save_folder_name).to(self.device)
            for param in personalized_global_model.parameters():
                param.data.zero_()

            # 3.4 根据刚刚算出的个性化权重，对所有模型进行加权求和
            for w, base_model in zip(weights, self.uploaded_base_model):
                for server_param, client_param in zip(personalized_global_model.base.parameters(), base_model.parameters()):
                    server_param.data += client_param.data.clone() * w.item()

            # 3.5 保存为该客户端的【专属全局模型】(例如命名为 model_1, model_2)
            save_item(personalized_global_model, self.role, f'model_{target_cid}', self.save_folder_name)



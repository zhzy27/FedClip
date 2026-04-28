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
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from datetime import datetime

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
            self.cur_ground = i
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


    # 客户端base进行个性化余弦相似度聚合
    def aggregate_parameters(self):
        assert (len(self.uploaded_ids) > 0)
        
        self.uploaded_base_model = []
        delta_params = [] # 用于存放精准的参数增量 Delta W

        # 1. 遍历所有上传的客户端，计算它们各自的 Delta W
        for cid in self.uploaded_ids:
            client = self.clients[cid]
            
            # --- 1.1 加载客户端本地训练后的最新模型 ---
            client_model = load_item(client.role, 'model', client.save_folder_name)
            model = copy.deepcopy(client_model)
            model.recover_larger_model()
            model.to(self.device)
            self.uploaded_base_model.append(model.base)
            
            # --- 1.2 加载该客户端在训练前【原本的起点模型】 ---
            # 优先加载它的旧版专属个性化模型
            old_start_model = load_item(self.role, f'model_{cid}', self.save_folder_name)
            
            if old_start_model is not None:
                old_start_model = old_start_model.to(self.device)
            else:
                # 如果没有专属模型（比如第一轮），说明它本地训练的起点是通用的全局模型
                old_start_model = load_item(self.role, 'model', self.save_folder_name).to(self.device)
            
            # --- 1.3 展平参数，计算真正的本地更新增量 Delta W ---
            flat_updated = torch.cat([p.data.view(-1) for p in model.base.parameters()])
            flat_old_start = torch.cat([p.data.view(-1) for p in old_start_model.base.parameters()])
            
            delta_w = flat_updated - flat_old_start
            delta_params.append(delta_w)


        # ================== 保留基础的 FedAvg 通用聚合 (兜底机制) ==================
        general_global_model = load_item(self.role, 'model', self.save_folder_name).to(self.device)
        for param in general_global_model.parameters():
            param.data.zero_()
        for w, base_model in zip(self.uploaded_weights, self.uploaded_base_model):
            for server_param, client_param in zip(general_global_model.base.parameters(), base_model.parameters()):
                w_tensor = torch.tensor(w).to(self.device)
                server_param.data += client_param.data.clone() * w_tensor
        save_item(general_global_model, self.role, 'model', self.save_folder_name)
        # =====================================================================


        print(f"执行基于参数增量(Delta W)的个性化聚合，参与客户端: {self.uploaded_ids}")
        
        # 此时的相似度是基于纯粹的 Delta W 计算的，差异已经非常明显。
        # tau 可以先设为 0.05（如果你想两极分化更严重，可以下调到 0.01）
        tau = 1 

        # 2. 为每个上传的客户端计算专属的聚合权重，并生成新的专属模型
        for i, target_cid in enumerate(self.uploaded_ids):
            sims = []
            for j in range(len(self.uploaded_ids)):
                # 使用 delta_params 计算相似度
                sim = torch.nn.functional.cosine_similarity(delta_params[i], delta_params[j], dim=0)
                sims.append(sim)
            
            sims = torch.stack(sims) 
            
            # 稳定版 Softmax (减去最大值防止数值溢出，并除以 tau 放大差异)
            sims_scaled = (sims - torch.max(sims)) / tau
            weights = torch.nn.functional.softmax(sims_scaled, dim=0)
            
            # (可选) 观察打印出的权重，看看是不是完美拉开了差距
            print(f"  -> 客户端 {target_cid} 的聚合权重: {weights.cpu().numpy().round(3)}")

            # 载入一个干净的全局模型作为聚合容器
            personalized_global_model = load_item(self.role, 'model', self.save_folder_name).to(self.device)
            for param in personalized_global_model.parameters():
                param.data.zero_()

            # 使用刚算出的个性化权重，把各个客户端【完整的最新模型】加权组合，形成目标客户端的新一代专属模型
            for w, base_model in zip(weights, self.uploaded_base_model):
                for server_param, client_param in zip(personalized_global_model.base.parameters(), base_model.parameters()):
                    server_param.data += client_param.data.clone() * w.item()

            save_item(personalized_global_model, self.role, f'model_{target_cid}', self.save_folder_name)


import copy
import random
import time
from collections import defaultdict

import numpy as np
from flcore.clients.clientARA2 import clientARA2
from flcore.servers.serverbase import Server
from flcore.clients.clientbase import load_item, save_item
from threading import Thread
from flcore.trainmodel.models import  Model_Distribe
from sklearn.metrics.pairwise import cosine_similarity
import torch

class FedARA2(Server):
    def __init__(self, args, times):
        super().__init__(args, times)

        # select slow clients
        self.set_slow_clients()
        self.set_clients(clientARA2)

        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print("Finished creating server and clients.")

        # self.load_model()
        self.Budget = []
        #创建全局base用于之后聚合
        global_model = Model_Distribe(args, -1,is_global=True).to(self.device)
        global_model.recover_larger_model()
        save_item(global_model, self.role, 'model', self.save_folder_name)


    def train(self):
        for i in range(self.global_rounds+1):
            s_t = time.time()
            # self.send_parameters()
            self.selected_clients = self.select_clients()
            if i%self.eval_gap == 0:
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
            self.aggregate_parameters_proto()

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
        assert (len(self.selected_clients) > 0)

        for client in self.selected_clients:
            start_time = time.time()
            #有的客户端会实现
            client.set_parameters()

            client.send_time_cost['num_rounds'] += 1
            client.send_time_cost['total_cost'] += 2 * (time.time() - start_time)
    #从客户顿接受id信息和样本数信息以及原型
    def receive_ids(self):
        assert (len(self.selected_clients) > 0)

        active_clients = random.sample(
            self.selected_clients, int((1-self.client_drop_rate) * self.current_num_join_clients))

        self.uploaded_ids = []
        self.uploaded_weights = []
        self.uploaded_protos = []
        tot_samples = 0
        for client in active_clients:
            tot_samples += client.train_samples
            self.uploaded_ids.append(client.id)
            self.uploaded_weights.append(client.train_samples)
            protos = load_item(client.role, 'protos', client.save_folder_name)
            self.uploaded_protos.append(protos)

        for i, w in enumerate(self.uploaded_weights):
            self.uploaded_weights[i] = w / tot_samples


    #客户端base进行参数对齐并且进行对齐后聚合
    def aggregate_parameters_proto(self):
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
        #聚合原型
        # 载入历史原型
        history_global_proto = load_item('Server', 'global_protos', self.save_folder_name)
        global_protos = proto_aggregation(history_global_proto, self.uploaded_protos)
        self.current_global_protos = global_protos
        #计算并保存全局一致的距离矩阵
        self.generate_prototype_matrices()
        save_item(global_protos, self.role, 'global_protos', self.save_folder_name)
        save_item(global_model, self.role, 'model', self.save_folder_name)

    #之后要处理参与率不为1的时候可能会出问题之后要增加稳健性
    def generate_prototype_matrices(self):
        """
        基于全局原型生成三个矩阵：
        1. MSE距离矩阵
        2. 余弦相似度矩阵
        3. Gamma距离矩阵（基于高斯核的相似度）
        """
        if not hasattr(self, 'current_global_protos') or not self.current_global_protos:
            print("Warning: No global prototypes available for matrix generation")
            return

        # 获取所有类别并排序
        labels = sorted(self.current_global_protos.keys())
        n_classes = len(labels)

        if n_classes == 0:
            print("Warning: No classes found in global prototypes")
            return

        # 将原型转换为矩阵形式
        proto_matrix = []
        for label in labels:
            proto = self.current_global_protos[label]
            if isinstance(proto, torch.Tensor):
                proto_matrix.append(proto.cpu().numpy())
            else:
                proto_matrix.append(proto)

        proto_matrix = np.array(proto_matrix)  # shape: (n_classes, feature_dim)

        # 1. 计算MSE距离矩阵
        self.mse_distance_matrix = self._compute_mse_distance_matrix(proto_matrix, labels)

        # 2. 计算余弦相似度矩阵
        self.similarity_matrix = self._compute_cosine_similarity_matrix(proto_matrix, labels)

        # # 3. 计算Gamma距离矩阵（基于高斯核的相似度）
        # self.gamma_distance_matrix = self._compute_gamma_similarity_matrix(proto_matrix, labels)

        #保存全局原型的距离
        save_item(self.mse_distance_matrix, self.role, 'global_mse_matrix', self.save_folder_name)
        save_item(self.similarity_matrix, self.role, 'global_sim_matrix', self.save_folder_name)
        # save_item(self.gamma_distance_matrix, self.role, 'global_gama_matrix', self.save_folder_name)
        # 打印矩阵信息
        print(f"\nGenerated matrices for {n_classes} classes:")
        print(f"MSE Distance Matrix shape: {self.mse_distance_matrix.shape}")
        print(f"Similarity Matrix shape: {self.similarity_matrix.shape}")
        # print(f"Gamma Distance Matrix shape: {self.gamma_distance_matrix.shape}")

    def _compute_mse_distance_matrix(self, proto_matrix, labels):
        """计算MSE距离矩阵"""
        n = len(labels)
        mse_matrix = np.zeros((n, n))

        for i in range(n):
            for j in range(n):
                # 计算两个原型之间的MSE距离
                mse_matrix[i, j] = np.mean((proto_matrix[i] - proto_matrix[j]) ** 2)

        return mse_matrix

    def _compute_cosine_similarity_matrix(self, proto_matrix, labels):
        """计算余弦相似度矩阵"""
        # 使用sklearn的cosine_similarity
        similarity_matrix = cosine_similarity(proto_matrix)
        return similarity_matrix

    def _compute_gamma_similarity_matrix(self, proto_matrix, labels, gamma=None):
        """计算Gamma相似度矩阵（基于高斯核）"""
        n = len(labels)

        # 如果没有提供gamma，自动计算
        if gamma is None:
            # 使用启发式方法计算gamma：1 / (2 * sigma^2)
            # 其中sigma是原型间距离的中位数
            distances = []
            for i in range(n):
                for j in range(i + 1, n):
                    dist = np.linalg.norm(proto_matrix[i] - proto_matrix[j])
                    distances.append(dist)
            if distances:
                median_dist = np.median(distances)
                gamma = 1.0 / (2.0 * (median_dist ** 2)) if median_dist > 0 else 1.0
            else:
                gamma = 1.0

        print(f"Using gamma value: {gamma}")

        gamma_matrix = np.zeros((n, n))

        for i in range(n):
            for j in range(n):
                # 计算欧氏距离
                dist = np.linalg.norm(proto_matrix[i] - proto_matrix[j])
                # 应用高斯核：exp(-gamma * dist^2)
                gamma_matrix[i, j] = np.exp(-gamma * (dist ** 2))

        return gamma_matrix



#如果本回合聚合出的没有全部类别的全局原型使用历史原型代替,平均加权（不按照样本数加权）
def proto_aggregation(history_global_proto,local_protos_list):
    # 保留历史全局原型
    if history_global_proto is None:
        agg_protos_label = {}
    else:
        agg_protos_label = history_global_proto.copy()
    # 存储新收集的原型和样本数
    new_protos = {}
    # 收集本轮上传的本地原型
    for idx, local_protos in enumerate(local_protos_list):
        for label, proto in local_protos.items():
            if label not in new_protos:
                new_protos[label] = []
            new_protos[label].append(proto)

    # 平均聚合生成新原型
    for label in new_protos.keys():
        proto_list = new_protos[label]
        global_proto = torch.zeros_like(proto_list[0])
        for i in range(len(proto_list)):
            #平均聚合
            global_proto +=  proto_list[i]
        #对旧的全局原型覆盖，对新的全局原型追加
        agg_protos_label[label] = global_proto / len(new_protos[label])
    return agg_protos_label



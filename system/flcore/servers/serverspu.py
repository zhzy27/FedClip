import copy
import random
import time
from collections import OrderedDict
import numpy as np
from flcore.clients.clientspu import clientSPU
from flcore.servers.serverbase import Server
from threading import Thread
from flcore.trainmodel.models import Model_Distribe
import torch
from utils.data_utils import read_client_data
# from torch.utils.tensorboard import SummaryWriter
import json
from flcore.clients.clientbase import load_item, save_item

class FedSPU(Server):
    def __init__(self, args, times):
        super().__init__(args, times)

        # select slow clients
        self.set_slow_clients()
        # 压缩的几个比例为 
        if 'Cifar10' in args.dataset:
            self.drop_rates = [1.0, 0.85, 0.7, 0.6, 0.45] 
        else:
            self.drop_rates = [1.0,0.75,0.7,0.65,0.45]  
        # 存储客户端的压缩比例
        self.clients_drop_rates = []
        # 设置客户端训练集相关信息
        self.set_clients(clientSPU)
        print(f"客户端们设置的压缩比例为{self.clients_drop_rates}")
        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print("Finished creating server and clients.")

        # self.load_model()
        self.Budget = []
        # 创建全局模型
        global_model = Model_Distribe(args, -1, is_global=True).to(self.device)
        print(f"服务器的全局模型为{global_model}")
        #保存全局模型
        save_item(global_model, self.role, 'model', self.save_folder_name)

    def set_clients(self, clientObj):
        for i, train_slow, send_slow in zip(range(self.num_clients), self.train_slow_clients, self.send_slow_clients):
            train_data = read_client_data(self.dataset, i, is_train=True, few_shot=self.few_shot)
            test_data = read_client_data(self.dataset, i, is_train=False, few_shot=self.few_shot)
            client = clientObj(self.args,
                               id=i,
                               train_samples=len(train_data),
                               test_samples=len(test_data),
                               train_slow=train_slow,
                               send_slow=send_slow)
            drop_rate = self.drop_rates[client.id % len(self.drop_rates)]
            client.drop_rate = drop_rate
            self.clients_drop_rates.append(drop_rate)
            self.clients.append(client)

    def train(self):
        for i in range(self.global_rounds + 1):
            s_t = time.time()
            # 选择客户端参与训练
            self.selected_clients = self.select_clients()
            # 评估客户端个性化模型性能
            if i % self.eval_gap == 0:
                print(f"\n-------------Round number: {i}-------------")
                print("\nEvaluate heterogeneous models")
                self.evaluate(epoch=i)
            # 给客户端分发子模型参数以及掩码,并将选中客户端的掩码保留下来用于之后聚合使用
            self.send_parameters()

            for client in self.selected_clients:
                print(f"客户端{client.id}本地训练")
                client.train(current_round=i)
            global_model = load_item(self.role, 'model', self.save_folder_name).to(self.device)
            current_parameter = self.get_filters(global_model)
            parameters_aggregated = self.aggregate_parameters(current_parameter)
            # 更新全局参数
            self.set_filters(global_model, parameters_aggregated)
            self.Budget.append(time.time() - s_t)
            print('-' * 25, 'time cost', '-' * 25, self.Budget[-1])

            if self.auto_break and self.check_done(acc_lss=[self.rs_test_acc], top_cnt=self.top_cnt):
                break

        print("\nBest accuracy.")
        # self.print_(max(self.rs_test_acc), max(
        #     self.rs_train_acc), min(self.rs_train_loss))
        print(max(self.rs_test_acc))
        print("\nAverage time cost per round.")
        print(sum(self.Budget[1:]) / len(self.Budget[1:]))

        self.save_results()
        # self.writer.close()
        self.save_json_file()

    # 给客户端分发子模型参数以及掩码,并将选中客户端的掩码保留下来用于之后聚合使用
    def send_parameters(self):
        assert (len(self.clients) > 0)
        for client in self.selected_clients:
            start_time = time.time()
            # 根据客户端的压缩比例随机选择全局模型的子模型，并设置相应的掩码
            print(f"为客户端{client.id}设置对应的掩码")
            global_model = load_item(self.role, 'model', self.save_folder_name).to(self.device)
            drop_info, sub_parameters, base_7_weight_in_dince = self.generate_filters_random(global_model,
                                                                                             client.drop_rate)
            # 服务器传递保留索引
            client.drop_info = drop_info
            # 向客户端传递保留索引对应的全局参数
            client.base_7_weight_in_dince = base_7_weight_in_dince
            client.subparamters = sub_parameters
            # 客户端介绍子模型参数并初始化本地模型参数
            client.set_parameters()
            # 单独存储base7的输入保留索引
            client.send_time_cost['num_rounds'] += 1
            client.send_time_cost['total_cost'] += 2 * (time.time() - start_time)

    def set_filters(self, net, parameters):  # modify the parameters of a neural network
        param_set_index = 0
        all_names = []
        all_params = []
        old_param_dict = net.state_dict()
        for k, _ in old_param_dict.items():
            all_params.append(parameters[param_set_index])
            all_names.append(k)
            param_set_index += 1
        params_dict = zip(all_names, all_params)
        state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
        net.load_state_dict(state_dict, strict=False)
        save_item(net, self.role, 'model', self.save_folder_name)

    # 获取模型参数列表[] 只含参数不含键
    def get_filters(self, net):
        params_list = []
        for k, v in net.state_dict().items():
            params_list.append(v.cpu().numpy())
        return params_list

    def generate_filters_random(self, global_model, rate):
        drop_information = {}
        # 直接不剪枝
        if rate >= 0.99:
            return drop_information, self.get_filters(global_model), torch.tensor(list(range(800)), device=self.device)
        # 获取全局模型参数字典
        param_dict = global_model.state_dict()
        old_indices = None  # 初始化old_indices为None，用于记录上一层的滤波器索引
        base_7_weight_in_dince = None
        # 子参数集合
        subparams = []
        # 对每一层按照比例剪枝（剪枝输入通道数）
        for name in param_dict.keys():
            # 逐层剪枝
            w = param_dict[name]
            device = w.device
            # 输出滤波器
            num_filters = w.shape[0]
            num_selected_filters = max(1, int(num_filters * rate))
            # 最后一层全连接层只剪枝输入通道，输出通道不剪枝,这个逻辑有点问题不是严格意义上的剪枝
            if name == 'head.weight':
                non_masked_filter_ids = list(range(self.args.num_classes)) # 输出不剪
                # 确保索引张量在正确的设备上
                non_masked_filter_ids = torch.tensor(non_masked_filter_ids, device=device)
                sub_param_1 = torch.index_select(w, 0, torch.tensor(non_masked_filter_ids))
                sub_param = torch.index_select(sub_param_1, 1, torch.tensor(old_indices))  # 找出输入通道的保存索引
                old_indices = non_masked_filter_ids  # 给出保留的输出通道索引，作为下一层保留的输入通道索引
            elif name == 'base.7.weight':  # 要单独处理，更具上一个轮次的输出通道保留输入太少了（上一个卷积层的输出总共才32）
                non_masked_filter_ids = sorted(
                    random.sample(list(range(num_filters)), num_selected_filters))  # 先找输出的保存索引
                # 确保索引张量在正确的设备上
                non_masked_filter_ids = torch.tensor(non_masked_filter_ids, device=device)
                sub_param_1 = torch.index_select(w, 0, torch.tensor(non_masked_filter_ids))
                # 它的保留输入通道索引要单独保留一下
                indins = torch.tensor(sorted(random.sample(list(range(800)), int(800 * rate))), device=device)
                base_7_weight_in_dince = indins
                sub_param = torch.index_select(sub_param_1, 1, indins)  # 找出输入通道的保存索引
                old_indices = non_masked_filter_ids  # 给出保留的输出通道索引，作为下一层保留的输入通道索引
            elif name == "head.bias":
                non_masked_filter_ids = list(range(self.args.num_classes))
                # 确保索引张量在正确的设备上
                non_masked_filter_ids = torch.tensor(non_masked_filter_ids, device=device)
                sub_param = torch.index_select(w, 0, torch.tensor(list(range(self.args.num_classes)), device=device))
            # 第一个权重层只剪枝输出维度输入不剪
            elif name == "base.0.weight":
                non_masked_filter_ids = sorted(random.sample(list(range(num_filters)), num_selected_filters))
                # 确保索引张量在正确的设备上
                non_masked_filter_ids = torch.tensor(non_masked_filter_ids, device=device)
                sub_param = torch.index_select(w, 0, torch.tensor(non_masked_filter_ids))
                old_indices = non_masked_filter_ids  # 更新剪枝掉的输出通道索引给下一层剪枝使用
            elif 'bias' in name:  # 偏置单独处理
                non_masked_filter_ids = old_indices
                # 确保索引张量在正确的设备上
                non_masked_filter_ids = torch.tensor(non_masked_filter_ids, device=device)
                sub_param = torch.index_select(w, 0, torch.tensor(non_masked_filter_ids))
            else:  # 其他的层输入输出都要剪枝
                non_masked_filter_ids = sorted(random.sample(list(range(num_filters)), num_selected_filters))  # 先找输出的保存索引
                # 确保索引张量在正确的设备上
                non_masked_filter_ids = torch.tensor(non_masked_filter_ids, device=device)
                sub_param_1 = torch.index_select(w, 0, torch.tensor(non_masked_filter_ids))
                sub_param = torch.index_select(sub_param_1, 1, torch.tensor(old_indices))  # 找出输入通道的保存索引
                old_indices = non_masked_filter_ids  # 给出保留的输出通道索引，作为下一层保留的输入通道索引
            drop_information[name] = non_masked_filter_ids  # 存储剪枝索引
            subparams.append(sub_param.cpu().numpy())
        return drop_information, subparams, base_7_weight_in_dince  # 返回保留的参数索引信息和子参数 #（当前保存索引，前一层的索引）就是当前层保留的参数索引 第一层和最后一层不是这样要单独处理

    # 对接收到的子参数进行聚合
    def aggregate_parameters(self, global_param):
        Aggregation_Dict = {}  # 存储待聚合的参数片段：键为参数位置索引
        Aggregated_params = {}  # 存储聚合后的参数片段：键为参数位置索引，值为聚合结果
        full_results = []  # 存储未剪枝客户端的完整参数（用于后续直接聚合）
        for client in self.selected_clients:
            # 解析客户端返回的参数、样本数、剪枝信息
            param, num, merge_info = client.get_updated_parameters(), client.train_samples, client.drop_info
            print(f"服务器收集客户端{client.id}更新的参数")
            # 3. 处理“未剪枝”的客户端（merge_info为空，即客户端使用完整模型）
            if len(merge_info) == 0:
                full_results.append((param, num))
                # 遍历完整参数的每个片段，记录到Aggregation_Dict
                for l1 in range(len(param)):  # l1：层所有输出通道索引索引
                    layer = param[l1]  # 当前层的参数
                    for l2 in range(len(layer)):  # 输出通道所有索引
                        filter = layer[l2]  # 当前一个滤波器的权重
                        if len(layer.shape) >1: #全连接层和卷积层一起处理
                            for l3 in range(len(filter)):  # 对每个输出通道权重遍历
                                # print("----------全连接层或卷积层追加-------------------")
                                # print((l1, l2, l3))
                                if (l1, l2, l3) in Aggregation_Dict.keys():
                                    # 键为（层索引，输出索引，输入通道索引），值为（参数值，样本数）
                                    Aggregation_Dict[(l1, l2, l3)].append((filter[l3], num))  # 保留一个卷积二维权重
                                else:
                                    Aggregation_Dict[(l1, l2, l3)] = [(filter[l3], num)]
                        # bias
                        else:
                            # 键为（层索引，输出通道）
                            if (l1, l2) in Aggregation_Dict.keys():
                                Aggregation_Dict[(l1, l2)].append((filter, num))
                            else:
                                Aggregation_Dict[(l1, l2)] = [(filter, num)]
            # 4. 处理“有剪枝”的客户端（merge_info非空，即客户端使用子网络）
            else:
                last_layer_indices = list(range(3))  # 上一层保留的输入通道索引（初始为输入图像通道）
                layer_count = 0  # 当前处理的层索引（与剪枝信息中的层对应）
                # 遍历剪枝信息中的每一层
                for k in merge_info.keys():
                    print(f"客户端获取第{k}层更新参数")
                    # print(f"layer name == {k}")
                    selected_filters = merge_info[k]  # 当前层保留的输出通道，最后一层全连接是输入通道
                    layer = param[layer_count]  # 客户端返回的上传的子网络层
                    i1 = 0  # 标明输出通道索引的
                    if 'bias' in k:
                        for f in selected_filters:
                            if (layer_count, f) in Aggregation_Dict.keys():
                                Aggregation_Dict[(layer_count, f)].append((layer[i1], num))
                            else:
                                Aggregation_Dict[(layer_count, f)] = [(layer[i1], num)]
                            i1+=1
                    elif k == "base.7.weight":
                        # 遍历当前层保留的滤波器（输出通道）
                        for f in selected_filters:
                            j1 = 0  #
                            # 遍历上一层保留的输出通道权重（确保与上一层剪枝对齐）
                            for j in client.base_7_weight_in_dince:
                                if (layer_count, f, j) in Aggregation_Dict.keys():
                                    # 键：（层索引，保留输出，保留的输入）
                                    Aggregation_Dict[(layer_count, f, j)].append((layer[i1][j1], num))
                                else:
                                    Aggregation_Dict[(layer_count, f, j)] = [(layer[i1][j1], num)]
                                j1 += 1
                            i1 += 1
                    else:
                        # 遍历当前层保留的滤波器（输出通道）
                        for f in selected_filters:
                            j1 = 0  #
                            # 遍历上一层保留的输出通道权重（确保与上一层剪枝对齐）
                            for j in last_layer_indices:
                                if (layer_count, f, j) in Aggregation_Dict.keys():
                                    # 键：（层索引，保留输出，保留的输入）
                                    Aggregation_Dict[(layer_count, f, j)].append((layer[i1][j1], num))
                                else:
                                    Aggregation_Dict[(layer_count, f, j)] = [(layer[i1][j1], num)]
                                j1 += 1
                            i1 += 1
                    layer_count += 1
                    last_layer_indices = selected_filters
        print("服务器实现客户端参数聚合")
        # 5. 聚合所有剪枝子网络的参数片段（按样本数加权平均）
        for z, p in Aggregation_Dict.items():  # p是一个列表,列表内容是（param,weight）
            # print(f"要聚合的参数键为{z}")
            Aggregated_params[z] = self.aggregate(p)
        # 完整的聚合参数作为要替换的目标
        full_param = self.aggregate_full(full_results) if len(full_results) > 0 else copy.deepcopy(global_param)
        # 7. 将聚合后的参数片段更新到完整全局参数中
        for Key in Aggregated_params.keys():
            if len(Key) == 2:  # 2维bias
                layer_idx, filter = Key
                full_param[layer_idx][filter] = Aggregated_params[Key]
            else:
                layer_idx, filter, last_filter = Key  # 3维键：（层索引，输出索引，输入索引）→ 如卷积层、全连接层权重
                full_param[layer_idx][filter][last_filter] = Aggregated_params[Key]
        return full_param
    def aggregate(self, param_nums_list):
        """
        聚合参数，按客户端数据量加权平均
        param_nums_list: [(param, num), ...] 每个param是一个参数张量
        返回: 聚合后的参数张量
        """
        if not param_nums_list:
            return None

        # 计算总样本数
        total_samples = sum(n for (_, n) in param_nums_list)

        if total_samples == 0:
            return None

        # 获取第一个参数作为模板
        first_param, _ = param_nums_list[0]

        # 根据参数类型创建零张量
        if hasattr(first_param, 'cpu'):  # torch tensor
            result = torch.zeros_like(first_param)
        else:  # numpy array
            result = np.zeros_like(first_param)

        # 加权平均聚合
        for param, n in param_nums_list:
            weight = n / total_samples
            result += param * weight

        return result
    def aggregate_full(self, param_nums_list):
        """
        聚合参数，按客户端数据量加权平均
        param_nums_list: [([param1, param2], num), ...] 形式的三层结构
        返回: [param1_agg, param2_agg, ...] 形式的聚合结果
        """
        if not param_nums_list:
            return []

        # 计算总样本数
        total_samples = sum(n for (_, n) in param_nums_list)

        if total_samples == 0:
            return []

        # 获取参数列表的长度
        param_count = len(param_nums_list[0][0])

        # 初始化结果列表
        first_param_list = param_nums_list[0][0]
        result = [torch.zeros_like(p) if hasattr(p, 'cpu') else np.zeros_like(p)
                  for p in first_param_list]

        # 加权平均聚合
        for (param_list, n) in param_nums_list:
            weight = n / total_samples
            for i in range(param_count):
                result[i] += param_list[i] * weight

        return result #返回完整模型参数列表


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
            train_data = read_client_data(self.dataset, i, is_train=True, few_shot=self.few_shot, args=self.args)
            test_data = read_client_data(self.dataset, i, is_train=False, few_shot=self.few_shot, args=self.args)
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
            all_params.append(torch.as_tensor(
                parameters[param_set_index],
                dtype=old_param_dict[k].dtype,
                device=old_param_dict[k].device,
            ))
            all_names.append(k)
            param_set_index += 1
        params_dict = zip(all_names, all_params)
        state_dict = OrderedDict({k: v for k, v in params_dict})
        net.load_state_dict(state_dict, strict=False)
        save_item(net, self.role, 'model', self.save_folder_name)

    # 获取模型参数列表[] 只含参数不含键
    def get_filters(self, net):
        params_list = []
        for k, v in net.state_dict().items():
            params_list.append(v.cpu().numpy())
        return params_list

    def generate_filters_random(self, global_model, rate):
        if self._is_resnet_spu_model():
            return self._generate_filters_random_resnet(global_model, rate)

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

    def _is_resnet_spu_model(self):
        return "resnet" in getattr(self.args, "model_family", "").lower()

    def _sample_indices(self, num_filters, rate, device):
        num_selected_filters = max(1, int(num_filters * rate))
        selected = sorted(random.sample(list(range(num_filters)), num_selected_filters))
        return torch.tensor(selected, dtype=torch.long, device=device)

    def _index_sub_parameter(self, weight, info):
        if info["kind"] == "full":
            return weight

        out_idx = torch.as_tensor(info["out"], dtype=torch.long, device=weight.device)
        if weight.dim() == 1:
            return torch.index_select(weight, 0, out_idx)

        in_idx = torch.as_tensor(info["in"], dtype=torch.long, device=weight.device)
        sub_weight = torch.index_select(weight, 0, out_idx)
        return torch.index_select(sub_weight, 1, in_idx)

    def _generate_filters_random_resnet(self, global_model, rate):
        drop_information = OrderedDict()
        if rate >= 0.99:
            return drop_information, self.get_filters(global_model), None

        param_dict = global_model.state_dict()
        subparams = []
        input_indices = None
        current_feature_indices = None
        last_output_indices = None
        block_inputs = {}
        block_outputs = {}

        for name, weight in param_dict.items():
            device = weight.device
            info = {"kind": "full"}

            if weight.dim() == 0:
                info = {"kind": "full"}
            elif name == "conv1.weight":
                input_indices = torch.arange(weight.shape[1], dtype=torch.long, device=device)
                output_indices = self._sample_indices(weight.shape[0], rate, device)
                info = {
                    "kind": "matrix",
                    "out": output_indices.cpu().tolist(),
                    "in": input_indices.cpu().tolist(),
                }
                current_feature_indices = output_indices
                last_output_indices = output_indices
            elif name.startswith("bn1."):
                info = {"kind": "vector", "out": last_output_indices.cpu().tolist()}
            elif ".conv1.weight" in name:
                block_name = name.rsplit(".conv1.weight", 1)[0]
                block_inputs[block_name] = current_feature_indices
                input_indices = current_feature_indices.to(device)
                output_indices = self._sample_indices(weight.shape[0], rate, device)
                info = {
                    "kind": "matrix",
                    "out": output_indices.cpu().tolist(),
                    "in": input_indices.cpu().tolist(),
                }
                last_output_indices = output_indices
            elif ".bn1." in name:
                info = {"kind": "vector", "out": last_output_indices.cpu().tolist()}
            elif ".conv2.weight" in name:
                block_name = name.rsplit(".conv2.weight", 1)[0]
                input_indices = last_output_indices.to(device)
                output_indices = self._sample_indices(weight.shape[0], rate, device)
                block_outputs[block_name] = output_indices
                info = {
                    "kind": "matrix",
                    "out": output_indices.cpu().tolist(),
                    "in": input_indices.cpu().tolist(),
                }
                current_feature_indices = output_indices
                last_output_indices = output_indices
            elif ".bn2." in name:
                info = {"kind": "vector", "out": last_output_indices.cpu().tolist()}
            elif ".shortcut.0.weight" in name:
                block_name = name.rsplit(".shortcut.0.weight", 1)[0]
                input_indices = block_inputs[block_name].to(device)
                output_indices = block_outputs[block_name].to(device)
                info = {
                    "kind": "matrix",
                    "out": output_indices.cpu().tolist(),
                    "in": input_indices.cpu().tolist(),
                }
                last_output_indices = output_indices
            elif ".shortcut.1." in name:
                block_name = name.rsplit(".shortcut.1.", 1)[0]
                output_indices = block_outputs[block_name].to(device)
                info = {"kind": "vector", "out": output_indices.cpu().tolist()}
                last_output_indices = output_indices
            elif name.endswith(".weight") and weight.dim() == 2:
                if name == "head.weight":
                    output_indices = torch.arange(weight.shape[0], dtype=torch.long, device=device)
                else:
                    output_indices = self._sample_indices(weight.shape[0], rate, device)

                input_indices = current_feature_indices.to(device)
                info = {
                    "kind": "matrix",
                    "out": output_indices.cpu().tolist(),
                    "in": input_indices.cpu().tolist(),
                }
                current_feature_indices = output_indices
                last_output_indices = output_indices
            elif name.endswith(".bias") and weight.dim() == 1:
                if name == "head.bias":
                    output_indices = torch.arange(weight.shape[0], dtype=torch.long, device=device)
                else:
                    output_indices = last_output_indices.to(device)
                info = {"kind": "vector", "out": output_indices.cpu().tolist()}

            drop_information[name] = info
            subparams.append(self._index_sub_parameter(weight, info).cpu().numpy())

        return drop_information, subparams, None

    # 对接收到的子参数进行聚合
    # 对接收到的子参数进行聚合 (彻底消灭循环，使用全局张量累加)
    # 聚合参数
    def aggregate_parameters(self, global_param):
        if self._is_resnet_spu_model():
            return self._aggregate_parameters_resnet(global_param)

        sum_params = [torch.zeros_like(torch.tensor(p, device=self.device)) for p in global_param]
        count_params = [torch.zeros_like(torch.tensor(p, device=self.device)) for p in global_param]
        
        print("服务器开始收集客户端参数并累加...")
        for client in self.selected_clients:
            param = client.get_updated_parameters()
            num = client.train_samples
            merge_info = client.drop_info
            
            if len(merge_info) == 0:
                for l_idx, layer in enumerate(param):
                    t_layer = torch.tensor(layer, device=self.device)
                    sum_params[l_idx] += t_layer * num
                    count_params[l_idx] += num
            else:
                last_layer_indices = list(range(3))
                layer_count = 0
                for k in merge_info.keys():
                    selected_filters = merge_info[k]
                    t_layer = torch.tensor(param[layer_count], device=self.device)
                    out_idx = torch.tensor(selected_filters, dtype=torch.long, device=self.device)
                    
                    if 'bias' in k:
                        sum_params[layer_count][out_idx] += t_layer * num
                        count_params[layer_count][out_idx] += num
                    elif k == "base.7.weight":
                        in_idx = torch.tensor(client.base_7_weight_in_dince, dtype=torch.long, device=self.device)
                        # 统一张量切片
                        sum_params[layer_count][out_idx[:, None], in_idx[None, :]] += t_layer * num
                        count_params[layer_count][out_idx[:, None], in_idx[None, :]] += num
                    else:
                        in_idx = torch.tensor(last_layer_indices, dtype=torch.long, device=self.device)
                        sum_params[layer_count][out_idx[:, None], in_idx[None, :]] += t_layer * num
                        count_params[layer_count][out_idx[:, None], in_idx[None, :]] += num
                    
                    layer_count += 1
                    last_layer_indices = selected_filters

        print("服务器计算加权平均并合并参数...")
        full_param = copy.deepcopy(global_param)
        for i in range(len(full_param)):
            valid_mask = count_params[i] > 0
            if valid_mask.any():
                avg_layer = sum_params[i] / count_params[i].clamp(min=1e-9)
                t_full = torch.tensor(full_param[i], device=self.device)
                t_full[valid_mask] = avg_layer[valid_mask]
                full_param[i] = t_full.cpu().numpy()
                
        return full_param

    def _add_resnet_param_update(self, sum_param, count_param, sub_param, info, num):
        if info["kind"] == "full":
            sum_param += sub_param.float() * num
            count_param += num
            return

        out_idx = torch.as_tensor(info["out"], dtype=torch.long, device=self.device)
        if sub_param.dim() == 1:
            sum_param[out_idx] += sub_param.float() * num
            count_param[out_idx] += num
            return

        in_idx = torch.as_tensor(info["in"], dtype=torch.long, device=self.device)
        if sub_param.dim() == 2:
            sum_param[out_idx[:, None], in_idx[None, :]] += sub_param.float() * num
            count_param[out_idx[:, None], in_idx[None, :]] += num
        elif sub_param.dim() == 4:
            sum_param[out_idx[:, None], in_idx[None, :], :, :] += sub_param.float() * num
            count_param[out_idx[:, None], in_idx[None, :], :, :] += num

    def _aggregate_parameters_resnet(self, global_param):
        sum_params = [
            torch.zeros_like(torch.as_tensor(p, device=self.device), dtype=torch.float32)
            for p in global_param
        ]
        count_params = [
            torch.zeros_like(torch.as_tensor(p, device=self.device), dtype=torch.float32)
            for p in global_param
        ]

        print("服务器开始按 ResNet-SPU 掩码收集客户端参数并累加...")
        for client in self.selected_clients:
            param = client.get_updated_parameters()
            num = client.train_samples
            merge_info = client.drop_info

            if len(merge_info) == 0:
                for layer_count, layer in enumerate(param):
                    t_layer = torch.as_tensor(layer, device=self.device)
                    sum_params[layer_count] += t_layer.float() * num
                    count_params[layer_count] += num
            else:
                for layer_count, (_, info) in enumerate(merge_info.items()):
                    t_layer = torch.as_tensor(param[layer_count], device=self.device)
                    self._add_resnet_param_update(sum_params[layer_count], count_params[layer_count], t_layer, info, num)

        print("服务器计算 ResNet-SPU 加权平均并合并参数...")
        full_param = copy.deepcopy(global_param)
        for i in range(len(full_param)):
            valid_mask = count_params[i] > 0
            if valid_mask.any():
                avg_layer = sum_params[i] / count_params[i].clamp(min=1e-9)
                old_layer = torch.as_tensor(full_param[i], device=self.device)
                new_layer = old_layer.clone()
                new_layer[valid_mask] = avg_layer.to(dtype=old_layer.dtype)[valid_mask]
                full_param[i] = new_layer.cpu().numpy()

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


import copy
import re
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
        old_param_dict = net.state_dict()
        if isinstance(parameters, dict):
            state_dict = OrderedDict()
            for k, old_v in old_param_dict.items():
                if k in parameters:
                    new_v = torch.as_tensor(parameters[k], device=old_v.device, dtype=old_v.dtype)
                    state_dict[k] = self._paste_common_shape(old_v.clone(), new_v)
                else:
                    state_dict[k] = old_v
        else:
            all_names = []
            all_params = []
            for param_set_index, (k, _) in enumerate(old_param_dict.items()):
                all_params.append(parameters[param_set_index])
                all_names.append(k)
            params_dict = zip(all_names, all_params)
            state_dict = OrderedDict({k: torch.as_tensor(v, device=old_param_dict[k].device, dtype=old_param_dict[k].dtype) for k, v in params_dict})
        net.load_state_dict(state_dict, strict=False)
        save_item(net, self.role, 'model', self.save_folder_name)

    # 获取模型参数列表[] 只含参数不含键
    def get_filters(self, net):
        params = OrderedDict()
        for k, v in net.state_dict().items():
            params[k] = v.detach().cpu().numpy()
        return params

    def generate_filters_random(self, global_model, rate):
        drop_information = OrderedDict()
        # 直接不剪枝
        if rate >= 0.99:
            return drop_information, self.get_filters(global_model), None
        # 获取全局模型参数字典
        param_dict = global_model.state_dict()
        if self._is_resnet_model(global_model):
            subparams = self._generate_resnet_subnet(param_dict, rate, drop_information)
            return drop_information, subparams, None
        subparams = self._generate_sequential_subnet(param_dict, rate, drop_information)
        return drop_information, subparams, None  # 返回保留的参数索引信息和子参数

    def _generate_sequential_subnet(self, param_dict, rate, drop_information):
        old_indices = None  # 初始化old_indices为None，用于记录上一层的滤波器索引
        subparams = OrderedDict()
        for name, w in param_dict.items():
            if w.dim() == 0:
                info = self._full_info()
                drop_information[name] = info
                subparams[name] = self._slice_with_spu_info(w, info).detach().cpu().numpy()
                continue
            if name == 'head.weight':
                info = self._spu_info(list(range(self.args.num_classes)), old_indices)
            elif name == "head.bias":
                info = self._spu_info(list(range(self.args.num_classes)), None)
            elif name == 'base.7.weight':
                out_indices = self._sample_indices(w.shape[0], rate)
                in_indices = self._sample_indices(w.shape[1], rate)
                info = self._spu_info(out_indices, in_indices)
                old_indices = out_indices
            # 第一个权重层只剪枝输出维度输入不剪
            elif name == "base.0.weight":
                out_indices = self._sample_indices(w.shape[0], rate)
                info = self._spu_info(out_indices, None)
                old_indices = out_indices
            elif w.dim() == 1:
                info = self._spu_info(old_indices, None)
            else:  # 其他的层输入输出都要剪枝
                out_indices = self._sample_indices(w.shape[0], rate)
                info = self._spu_info(out_indices, old_indices)
                old_indices = out_indices
            drop_information[name] = info
            subparams[name] = self._slice_with_spu_info(w, info).detach().cpu().numpy()
        return subparams

    def _generate_resnet_subnet(self, param_dict, rate, drop_information):
        subparams = OrderedDict()
        current_indices = None
        block_context = {}
        for name, w in param_dict.items():
            info = self._resnet_spu_info(name, w, rate, current_indices, block_context)
            if name == "conv1.weight":
                current_indices = info["out"]
            block_match = re.match(r"layer\d+\.\d+\.conv2\.weight", name)
            if block_match:
                current_indices = info["out"]
            neck_match = re.match(r"neck\.0\.weight", name)
            if neck_match:
                current_indices = info["out"]
            drop_information[name] = info
            subparams[name] = self._slice_with_spu_info(w, info).detach().cpu().numpy()
        return subparams

    def _resnet_spu_info(self, name, tensor, rate, current_indices, block_context):
        if tensor.dim() == 0:
            return self._full_info()
        if name == "conv1.weight":
            return self._spu_info(self._sample_indices(tensor.shape[0], rate), None)
        if name.startswith("bn1."):
            return self._spu_info(current_indices, None)
        block_match = re.match(r"(layer\d+\.\d+)\.(conv1|bn1|conv2|bn2|shortcut\.0|shortcut\.1)\.(.+)", name)
        if block_match:
            block_name, module_name, _ = block_match.groups()
            context = block_context.setdefault(block_name, {})
            if module_name == "conv1":
                context["input"] = current_indices
                context["hidden"] = self._sample_indices(tensor.shape[0], rate)
                return self._spu_info(context["hidden"], context["input"])
            if module_name == "bn1":
                return self._spu_info(context["hidden"], None)
            if module_name == "conv2":
                context["output"] = self._sample_indices(tensor.shape[0], rate)
                return self._spu_info(context["output"], context["hidden"])
            if module_name == "bn2":
                return self._spu_info(context["output"], None)
            if module_name == "shortcut.0":
                return self._spu_info(context["output"], context["input"])
            if module_name == "shortcut.1":
                return self._spu_info(context["output"], None)
        if name == "neck.0.weight":
            return self._spu_info(self._sample_indices(tensor.shape[0], rate), current_indices)
        if name == "neck.0.bias":
            return self._spu_info(current_indices, None)
        if name == "head.weight":
            return self._spu_info(list(range(self.args.num_classes)), current_indices)
        if name == "head.bias":
            return self._spu_info(list(range(self.args.num_classes)), None)
        if tensor.dim() == 1:
            return self._spu_info(current_indices, None)
        return self._full_info()

    # 对接收到的子参数进行聚合
    # 对接收到的子参数进行聚合 (彻底消灭循环，使用全局张量累加)
    # 聚合参数
    def aggregate_parameters(self, global_param):
        sum_params = {
            name: torch.zeros_like(torch.as_tensor(param, device=self.device), dtype=torch.float32)
            for name, param in global_param.items()
        }
        count_params = {
            name: torch.zeros_like(torch.as_tensor(param, device=self.device), dtype=torch.float32)
            for name, param in global_param.items()
        }

        print("服务器开始收集客户端参数并累加...")
        for client in self.selected_clients:
            param = client.get_updated_parameters()
            num = client.train_samples
            merge_info = client.drop_info

            if len(merge_info) == 0:
                for name, layer in param.items():
                    if name not in sum_params:
                        continue
                    t_layer = torch.as_tensor(layer, device=self.device)
                    self._add_full_tensor(sum_params[name], count_params[name], t_layer, num)
            else:
                for name, info in merge_info.items():
                    if name not in param or name not in sum_params:
                        continue
                    t_layer = torch.as_tensor(param[name], device=self.device)
                    self._add_spu_tensor(sum_params[name], count_params[name], t_layer, num, info)

        print("服务器计算加权平均并合并参数...")
        full_param = copy.deepcopy(global_param)
        for name in full_param.keys():
            if count_params[name].dim() == 0:
                if count_params[name].item() > 0:
                    avg_layer = sum_params[name] / count_params[name].clamp(min=1e-9)
                    t_full = torch.as_tensor(full_param[name], device=self.device)
                    if not t_full.is_floating_point():
                        avg_layer = avg_layer.round().to(dtype=t_full.dtype)
                    else:
                        avg_layer = avg_layer.to(dtype=t_full.dtype)
                    full_param[name] = avg_layer.cpu().numpy()
                continue
            valid_mask = count_params[name] > 0
            if valid_mask.any():
                avg_layer = sum_params[name] / count_params[name].clamp(min=1e-9)
                t_full = torch.as_tensor(full_param[name], device=self.device)
                if not t_full.is_floating_point():
                    avg_layer = avg_layer.round().to(dtype=t_full.dtype)
                else:
                    avg_layer = avg_layer.to(dtype=t_full.dtype)
                t_full[valid_mask] = avg_layer[valid_mask]
                full_param[name] = t_full.cpu().numpy()

        return full_param

    def _is_resnet_model(self, model):
        return all(hasattr(model, name) for name in ["conv1", "layer1", "layer2", "layer3", "layer4"])

    def _sample_indices(self, num_filters, rate):
        num_selected_filters = max(1, int(num_filters * rate))
        return sorted(random.sample(list(range(num_filters)), num_selected_filters))

    def _full_info(self):
        return {"mode": "full", "out": None, "in": None}

    def _spu_info(self, out_indices, in_indices):
        mode = "out_in" if in_indices is not None else "out"
        return {"mode": mode, "out": out_indices, "in": in_indices}

    def _index_tensor(self, index):
        if index is None:
            return None
        if isinstance(index, torch.Tensor):
            return index.to(self.device, dtype=torch.long)
        return torch.tensor(index, dtype=torch.long, device=self.device)

    def _common_slices(self, shape_a, shape_b):
        return tuple(slice(0, min(a, b)) for a, b in zip(shape_a, shape_b))

    def _paste_common_shape(self, target, source):
        if target.shape == source.shape:
            return source
        if target.dim() == 0 or source.dim() == 0:
            return source.to(dtype=target.dtype) if target.shape == source.shape else target
        common = self._common_slices(target.shape, source.shape)
        target[common] = source[common].to(dtype=target.dtype)
        return target

    def _slice_with_spu_info(self, tensor, info):
        if info.get("mode", "full") == "full" or tensor.dim() == 0:
            return tensor
        out_idx = self._index_tensor(info.get("out"))
        in_idx = self._index_tensor(info.get("in"))
        if tensor.dim() == 1:
            return tensor[out_idx]
        if in_idx is None:
            return torch.index_select(tensor, 0, out_idx)
        return tensor[out_idx[:, None], in_idx[None, :]]

    def _add_full_tensor(self, sum_tensor, count_tensor, value, num):
        if value.dim() == 0 and sum_tensor.dim() == 0:
            sum_tensor += value.to(dtype=sum_tensor.dtype) * num
            count_tensor += num
            return
        common = self._common_slices(sum_tensor.shape, value.shape)
        sum_tensor[common] += value[common].to(dtype=sum_tensor.dtype) * num
        count_tensor[common] += num

    def _add_spu_tensor(self, sum_tensor, count_tensor, value, num, info):
        if info.get("mode", "full") == "full" or sum_tensor.dim() == 0:
            self._add_full_tensor(sum_tensor, count_tensor, value, num)
            return
        out_idx = self._index_tensor(info.get("out"))
        in_idx = self._index_tensor(info.get("in"))
        value = value.to(dtype=sum_tensor.dtype)
        if sum_tensor.dim() == 1:
            sum_tensor[out_idx] += value * num
            count_tensor[out_idx] += num
        elif in_idx is None:
            sum_tensor[out_idx] += value * num
            count_tensor[out_idx] += num
        else:
            sum_tensor[out_idx[:, None], in_idx[None, :]] += value * num
            count_tensor[out_idx[:, None], in_idx[None, :]] += num

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


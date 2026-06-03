import copy
from collections import OrderedDict

import torch
import torch.nn as nn
import numpy as np
import time
from flcore.clients.clientbase import Client
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import label_binarize
from sklearn import metrics
from flcore.trainmodel.models import BaseHeadSplit, Model_Distribe
from flcore.clients.clientbase import load_item, save_item

class clientSPU(Client):
    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        super().__init__(args, id, train_samples, test_samples, **kwargs)
        torch.manual_seed(0)
        self.drop_rate = 1.0  # 先默认之后会设置不同的比例
        self.drop_info = None  # 保留剪枝后的索引
        self.mask = None
        self.subparamters = None
        self.base_7_weight_in_dince = None  # 兼容旧接口；当前剪枝输入索引保存在 drop_info 中
        self.hook_handles = []

    # 本地训练
    def train(self, current_round=0):
        # 生成冻结掩码以冻结参数梯度
        print(f"客户端{self.id}创建掩码")
        model = load_item(self.role, 'model', self.save_folder_name).to(self.device)
        self.mask = self.mask_gradients(model)
        # 根据掩码生成钩子函数冻结梯度
        print(f"客户端{self.id}冻结梯度")
        self.freeze_filters(model,self.mask)
        trainloader = self.load_train_data()
        optimizer = torch.optim.SGD(model.parameters(), lr=self.learning_rate)
        # model.to(self.device)
        start_time = time.time()
        max_local_epochs = self.local_epochs
        print(f"客户端{self.id}开始本地训练")
        if self.train_slow:
            max_local_epochs = np.random.randint(1, max_local_epochs // 2)
        for step in range(max_local_epochs):
            for i, (x, y) in enumerate(trainloader):
                optimizer.zero_grad()
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                if self.train_slow:
                    time.sleep(0.1 * np.abs(np.random.rand()))
                output = model(x)
                loss = self.loss(output, y)
                loss.backward()
                optimizer.step()
        self.remove_hooks()
        save_item(model, self.role, 'model', self.save_folder_name)
        self.train_time_cost['num_rounds'] += 1
        self.train_time_cost['total_cost'] += time.time() - start_time

    # 从服务器接受全局模型子参数更新本地参数
    def set_parameters(self):
        merged_parameters = self.merge_subnet()  # 接受了全局参数后的本地参数
        # 形成初始参数
        model = load_item(self.role, 'model', self.save_folder_name).to(self.device)
        self.set_filters(model, merged_parameters)

    # 获得更新后的参数
# 获得更新后的参数
    def get_updated_parameters(self, C=3):
        model = load_item(self.role, 'model', self.save_folder_name).to(self.device)
        if len(self.drop_info) == 0:
            return self.get_filters(model)

        sub_params = OrderedDict()
        full_params = self.get_filters(model)

        for name, info in self.drop_info.items():
            if name not in full_params:
                continue
            full_layer = torch.as_tensor(full_params[name], device=self.device)
            sub_layer = self._slice_with_spu_info(full_layer, info)
            sub_params[name] = sub_layer.detach().cpu().numpy()

        return sub_params


    # 合并子网络参数
    def merge_subnet(self, C=3):
        if len(self.drop_info) == 0:
            return self.subparamters

        model = load_item(self.role, 'model', self.save_folder_name).to(self.device)
        full_params = self.get_filters(model)
        result = OrderedDict()

        for name, info in self.drop_info.items():
            if name not in full_params or name not in self.subparamters:
                continue
            full_layer = torch.as_tensor(full_params[name], device=self.device)
            sub_layer = torch.as_tensor(self.subparamters[name], device=self.device, dtype=full_layer.dtype)
            merged_layer = self._merge_with_spu_info(full_layer, sub_layer, info)
            result[name] = merged_layer.detach().cpu().numpy()

        return result


    # 生成掩码
    def mask_gradients(self, model, C=3):
        if len(self.drop_info) == 0:
            return {name: torch.ones_like(param, device=self.device) for name, param in model.named_parameters()}

        masks = {}
        for name, param in model.named_parameters():
            gradient_mask = torch.zeros_like(param, device=self.device)
            if name in self.drop_info:
                gradient_mask = self._mask_with_spu_info(gradient_mask, self.drop_info[name])
            masks[name] = gradient_mask

        return masks

    def get_filters(self, net):
        params = OrderedDict()
        for k, v in net.state_dict().items():
            params[k] = v.detach().cpu().numpy()
        return params

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

    # 从服务器全局接受的子参数更新本地个性化参数


    def create_fc_mask(self,weight_shape, non_mask_filters, last_layer_indices):
        """
        根据保留的输入输出索引生成全连接层掩码

        Args:
            weight_shape: 权重形状 [out_features, in_features]
            non_mask_filters: 当前层要保留的输出通道索引
            last_layer_indices: 上一层保留的输出通道索引（当前层的输入通道保留索引）

        Returns:
            mask: 与weight_shape相同的0/1掩码
        """
        out_features, in_features = weight_shape

        # 创建输出通道掩码 - 使用布尔类型进行计算
        output_mask = torch.zeros(out_features, dtype=torch.bool)
        if len(non_mask_filters) != 0:  # 确保列表不为空
            output_mask[non_mask_filters] = True

        # 创建输入通道掩码 - 使用布尔类型进行计算
        input_mask = torch.zeros(in_features, dtype=torch.bool)
        if len(last_layer_indices) != 0:  # 确保列表不为空
            input_mask[last_layer_indices] = True

        # 广播生成最终布尔掩码
        bool_mask = output_mask.unsqueeze(1) & input_mask.unsqueeze(0)

        # 将布尔掩码转换为0/1掩码
        mask = bool_mask.float()

        return mask

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

    def _merge_with_spu_info(self, full_layer, sub_layer, info):
        if info.get("mode", "full") == "full" or full_layer.dim() == 0:
            return self._paste_common_shape(full_layer.clone(), sub_layer)
        full_layer = full_layer.clone()
        out_idx = self._index_tensor(info.get("out"))
        in_idx = self._index_tensor(info.get("in"))
        if full_layer.dim() == 1:
            full_layer[out_idx] = sub_layer
        elif in_idx is None:
            full_layer[out_idx] = sub_layer
        else:
            full_layer[out_idx[:, None], in_idx[None, :]] = sub_layer
        return full_layer

    def _mask_with_spu_info(self, mask, info):
        if info.get("mode", "full") == "full" or mask.dim() == 0:
            mask.fill_(1.0)
            return mask
        out_idx = self._index_tensor(info.get("out"))
        in_idx = self._index_tensor(info.get("in"))
        if mask.dim() == 1:
            mask[out_idx] = 1.0
        elif in_idx is None:
            mask[out_idx] = 1.0
        else:
            mask[out_idx[:, None], in_idx[None, :]] = 1.0
        return mask

    def test_metrics(self):
        testloader = self.load_test_data()
        model = load_item(self.role, 'model', self.save_folder_name).to(self.device)
        model.eval()
        test_acc = 0
        test_num = 0
        y_prob = []
        y_true = []

        with torch.no_grad():
            for x, y in testloader:
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                output = model(x)

                test_acc += (torch.sum(torch.argmax(output, dim=1) == y)).item()
                test_num += y.shape[0]

                y_prob.append(output.detach().cpu().numpy())
                nc = self.num_classes
                if self.num_classes == 2:
                    nc += 1
                lb = label_binarize(y.detach().cpu().numpy(), classes=np.arange(nc))
                if self.num_classes == 2:
                    lb = lb[:, :2]
                y_true.append(lb)

        y_prob = np.concatenate(y_prob, axis=0)
        y_true = np.concatenate(y_true, axis=0)

        auc = metrics.roc_auc_score(y_true, y_prob, average='micro')

        return test_acc, test_num, auc

    def train_metrics(self):
        trainloader = self.load_train_data()
        model = load_item(self.role, 'model', self.save_folder_name).to(self.device)
        model.eval()

        train_num = 0
        losses = 0
        with torch.no_grad():
            for x, y in trainloader:
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                output = model(x)
                loss = self.loss(output, y)
                train_num += y.shape[0]
                losses += loss.item() * y.shape[0]

        return losses, train_num

    def freeze_filters(self,model, masks):
        print("创建钩子函数")
        for name, param in model.named_parameters():
            if name not in masks:
                continue
            mask = masks[name].to(param.device)
            self.hook_handles.append(param.register_hook(lambda grad, mask=mask: grad * mask))

    def _get_head_linear(self, model):
        if isinstance(model.head, nn.Linear):
            return model.head
        if isinstance(model.head, nn.Sequential):
            for layer in reversed(model.head):
                if isinstance(layer, nn.Linear):
                    return layer
        raise AttributeError(f"Unsupported FedSPU head type: {type(model.head).__name__}")

    def remove_hooks(self):
        """移除所有钩子"""
        if hasattr(self, 'hook_handles') and self.hook_handles:
            print(f"客户端{self.id}移除钩子函数")
            for handle in self.hook_handles:
                handle.remove()
            self.hook_handles.clear()

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
        self.base_7_weight_in_dince = None  # 单独保留一下 base.7.weight的保留的输出通道（不能根据前一个卷积层的输出通道决定）
        self.hook_handles = []

    # 本地训练
    def train(self, current_round=0):
        # 生成冻结掩码以冻结参数梯度
        print(f"客户端{self.id}创建掩码")
        model = load_item(self.role, 'model', self.save_folder_name).to(self.device)
        total_params = sum(p.numel() for p in model.parameters())
        # 为了方便阅读，将其转换为 百万 (Million, M) 级别
        print(f"[{self.role}] 当前模型参数量为: {total_params} ({total_params / 1e6:.3f} M)")
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
            
        sub_params = []
        full_params = self.get_filters(model)
        layer_count = 0
        last_layer_indices = list(range(C))
        
        for k in self.drop_info.keys():
            full_layer = torch.tensor(full_params[layer_count], device=self.device)
            out_idx = torch.tensor(self.drop_info[k], dtype=torch.long, device=self.device)
            
            if 'bias' in k:
                sub_layer = full_layer[out_idx]
            elif k == "base.7.weight":
                in_idx = torch.tensor(self.base_7_weight_in_dince, dtype=torch.long, device=self.device)
                # 统一的切片方式，自动兼容 2D 和 4D 张量
                sub_layer = full_layer[out_idx[:, None], in_idx[None, :]]
                last_layer_indices = self.drop_info[k]
            else:
                in_idx = torch.tensor(last_layer_indices, dtype=torch.long, device=self.device)
                sub_layer = full_layer[out_idx[:, None], in_idx[None, :]]
                last_layer_indices = self.drop_info[k]
                
            sub_params.append(sub_layer.cpu().numpy())
            layer_count += 1
            
        return sub_params


    # 合并子网络参数
    def merge_subnet(self, C=3):
        if len(self.drop_info) == 0:
            return self.subparamters
            
        model = load_item(self.role, 'model', self.save_folder_name).to(self.device)
        full_params = self.get_filters(model)
        layer_count = 0
        result = []
        last_layer_indices = list(range(C))
        
        for k in self.drop_info.keys():
            selected_filters = self.drop_info[k]
            full_layer = torch.tensor(full_params[layer_count], device=self.device)
            sub_layer = torch.tensor(self.subparamters[layer_count], device=self.device)
            
            out_idx = torch.tensor(selected_filters, dtype=torch.long, device=self.device)
            
            if k == "head.bias":  
                full_layer[:] = sub_layer[:]
            elif "bias" in k:
                full_layer[out_idx] = sub_layer
            elif k == "base.7.weight":
                in_idx = torch.tensor(self.base_7_weight_in_dince, dtype=torch.long, device=self.device)
                full_layer[out_idx[:, None], in_idx[None, :]] = sub_layer
            else:
                in_idx = torch.tensor(last_layer_indices, dtype=torch.long, device=self.device)
                full_layer[out_idx[:, None], in_idx[None, :]] = sub_layer
                    
            result.append(full_layer.cpu().numpy())
            layer_count += 1
            last_layer_indices = selected_filters
            
        return result


    # 生成掩码
    def mask_gradients(self, model, C=3):
        weights = []
        params = model.state_dict()
        for k, v in params.items():
            weights.append(v)
            
        if len(self.drop_info) == 0:
            return [torch.ones(w.shape, device=self.device) for w in weights]
            
        last_layer_indices = list(range(C))
        Masks = []
        l = 0
        
        for k in self.drop_info.keys():
            gradient_mask = torch.zeros(weights[l].shape, device=self.device)
            non_mask_filters = self.drop_info[k]
            out_idx = torch.tensor(non_mask_filters, dtype=torch.long, device=self.device)
            
            if 'bias' in k:
                gradient_mask[out_idx] = 1.0
            elif k == "base.7.weight":
                in_idx = torch.tensor(self.base_7_weight_in_dince, dtype=torch.long, device=self.device)
                gradient_mask[out_idx[:, None], in_idx[None, :]] = 1.0
                last_layer_indices = non_mask_filters
            else:
                in_idx = torch.tensor(last_layer_indices, dtype=torch.long, device=self.device)
                gradient_mask[out_idx[:, None], in_idx[None, :]] = 1.0
                last_layer_indices = non_mask_filters
                
            Masks.append(gradient_mask)
            l += 1
            
        return Masks

    def get_filters(self, net):
        params_list = []
        for k, v in net.state_dict().items():
            params_list.append(v.cpu().numpy())
        return params_list

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
        head_layer = self._get_head_linear(model)
        # 注册新钩子并保存句柄
        self.hook_handles.extend([
            model.base[0].weight.register_hook(lambda grad: grad * masks[0]),
            model.base[0].bias.register_hook(lambda grad: grad * masks[1]),
            model.base[3].weight.register_hook(lambda grad: grad * masks[2]),
            model.base[3].bias.register_hook(lambda grad: grad * masks[3]),
            model.base[7].weight.register_hook(lambda grad: grad * masks[4]),
            model.base[7].bias.register_hook(lambda grad: grad * masks[5]),
            model.base[9].weight.register_hook(lambda grad: grad * masks[6]),
            model.base[9].bias.register_hook(lambda grad: grad * masks[7]),
            head_layer.weight.register_hook(lambda grad: grad * masks[8]),
            head_layer.bias.register_hook(lambda grad: grad * masks[9])
        ])

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

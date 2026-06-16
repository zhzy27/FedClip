import torch
import numpy as np
import time
from flcore.clients.clientbase import Client, load_item, save_item
from collections import defaultdict


def _sanitize_tensor(tensor, tag, clamp_value=1e4):
    if torch.isfinite(tensor).all():
        return tensor
    print(f"⚠️ FedGH {tag} 出现 NaN/Inf，已执行 nan_to_num。")
    return torch.nan_to_num(tensor, nan=0.0, posinf=clamp_value, neginf=-clamp_value)


def _sanitize_module_parameters_(module, tag, clamp_value=1e4):
    with torch.no_grad():
        for name, param in module.named_parameters():
            if param is not None and not torch.isfinite(param).all():
                print(f"⚠️ FedGH {tag}.{name} 参数出现 NaN/Inf，已执行 nan_to_num。")
                param.data = torch.nan_to_num(param.data, nan=0.0, posinf=clamp_value, neginf=-clamp_value)


def _sanitize_module_gradients_(module, tag, clamp_value=1e4):
    for name, param in module.named_parameters():
        if param.grad is not None and not torch.isfinite(param.grad).all():
            print(f"⚠️ FedGH {tag}.{name} 梯度出现 NaN/Inf，已执行 nan_to_num。")
            param.grad.data = torch.nan_to_num(param.grad.data, nan=0.0, posinf=clamp_value, neginf=-clamp_value)


class clientGH(Client):
    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        super().__init__(args, id, train_samples, test_samples, **kwargs)
        torch.manual_seed(0)
    #客户端本地训练
    def train(self):
        trainloader = self.load_train_data()
        model = load_item(self.role, 'model', self.save_folder_name).to(self.device)
        _sanitize_module_parameters_(model, f"{self.role}.model")
        total_params = sum(p.numel() for p in model.parameters())
        print(f"[{self.role}] 当前模型参数量为: {total_params} ({total_params / 1e6:.3f} M)")
        
        optimizer = torch.optim.SGD(model.parameters(), lr=self.learning_rate)
        # model.to(self.device)
        model.train()
        
        start_time = time.time()

        max_local_epochs = self.local_epochs
        if self.train_slow:
            max_local_epochs = np.random.randint(1, max_local_epochs // 2)

        for step in range(max_local_epochs):
            for i, (x, y) in enumerate(trainloader):
                
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                if self.train_slow:
                    time.sleep(0.1 * np.abs(np.random.rand()))
                output = model(x)
                loss = self.loss(output, y)
                if not torch.isfinite(loss):
                    print(f"⚠️ FedGH {self.role} 本地 loss 非有限，跳过当前 batch 并清理模型参数。")
                    _sanitize_module_parameters_(model, f"{self.role}.model")
                    continue
                optimizer.zero_grad()
                loss.backward()
                _sanitize_module_gradients_(model, f"{self.role}.model")
                optimizer.step()
                _sanitize_module_parameters_(model, f"{self.role}.model")
        #模型参数一直在保存和读写
        save_item(model, self.role, 'model', self.save_folder_name)
        
        self.train_time_cost['num_rounds'] += 1
        self.train_time_cost['total_cost'] += time.time() - start_time

    #从服务器下载head参数
    def set_parameters(self):
        model = load_item(self.role, 'model', self.save_folder_name).to(self.device)
        head = load_item('Server', 'head', self.save_folder_name).to(self.device)
        _sanitize_module_parameters_(head, "Server.head")
        for new_param, old_param in zip(head.parameters(), model.head.parameters()):
            old_param.data = new_param.data.clone()
        save_item(model, self.role, 'model', self.save_folder_name)
    #计算本地原型用于之后上传
    def collect_protos(self):
        trainloader = self.load_train_data()
        model = load_item(self.role, 'model', self.save_folder_name).to(self.device)
        _sanitize_module_parameters_(model, f"{self.role}.model")
        model.eval()

        protos = defaultdict(list)
        with torch.no_grad():
            for i, (x, y) in enumerate(trainloader):
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                if self.train_slow:
                    time.sleep(0.1 * np.abs(np.random.rand()))
                rep = model.base(x)
                rep = _sanitize_tensor(rep, f"{self.role} prototype feature")

                for i, yy in enumerate(y):
                    y_c = yy.item()
                    protos[y_c].append(rep[i, :].detach().data)
        #原型保存
        save_item(agg_func(protos), self.role, 'protos', self.save_folder_name)

#聚合原型
# https://github.com/yuetan031/fedproto/blob/main/lib/utils.py#L205
def agg_func(protos):
    """
    Returns the average of the weights.
    """

    for [label, proto_list] in protos.items():
        if len(proto_list) > 1:
            proto = 0 * proto_list[0].data
            for i in proto_list:
                proto += _sanitize_tensor(i.data, f"class {label} proto")
            protos[label] = proto / len(proto_list)
        else:
            protos[label] = _sanitize_tensor(proto_list[0], f"class {label} proto")

    return protos

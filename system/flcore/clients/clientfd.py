import torch
import torch.nn as nn
import numpy as np
import time
from flcore.clients.clientbase import Client, load_item, save_item
from collections import defaultdict


def _sanitize_tensor(tensor, tag, clamp_value=1e4):
    if torch.isfinite(tensor).all():
        return tensor
    print(f"⚠️ FD {tag} 出现 NaN/Inf，已执行 nan_to_num。")
    return torch.nan_to_num(tensor, nan=0.0, posinf=clamp_value, neginf=-clamp_value)


def _sanitize_module_parameters_(module, tag, clamp_value=1e4):
    with torch.no_grad():
        for name, param in module.named_parameters():
            if param is not None and not torch.isfinite(param).all():
                print(f"⚠️ FD {tag}.{name} 参数出现 NaN/Inf，已执行 nan_to_num。")
                param.data = torch.nan_to_num(param.data, nan=0.0, posinf=clamp_value, neginf=-clamp_value)


def _sanitize_module_gradients_(module, tag, clamp_value=1e4):
    for name, param in module.named_parameters():
        if param.grad is not None and not torch.isfinite(param.grad).all():
            print(f"⚠️ FD {tag}.{name} 梯度出现 NaN/Inf，已执行 nan_to_num。")
            param.grad.data = torch.nan_to_num(param.grad.data, nan=0.0, posinf=clamp_value, neginf=-clamp_value)


def _get_global_logit(global_logits, label, device, expected_shape):
    if global_logits is None or label not in global_logits:
        return None
    logit = global_logits[label]
    if isinstance(logit, list):
        return None
    logit = _sanitize_tensor(logit.detach().to(device), f"class {label} global logit")
    if logit.shape != expected_shape:
        print(f"⚠️ FD class {label} global logit 形状不匹配: {tuple(logit.shape)} != {tuple(expected_shape)}，跳过该类蒸馏。")
        return None
    return logit


class clientFD(Client):
    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        super().__init__(args, id, train_samples, test_samples, **kwargs)
        torch.manual_seed(0)

        self.lamda = args.lamda


    def train(self):
        trainloader = self.load_train_data()
        model = load_item(self.role, 'model', self.save_folder_name).to(self.device)
        _sanitize_module_parameters_(model, f"{self.role}.model")
        optimizer = torch.optim.SGD(model.parameters(), lr=self.learning_rate)
        global_logits = load_item('Server', 'global_logits', self.save_folder_name)
        
        start_time = time.time()

        model.train()

        max_local_epochs = self.local_epochs
        if self.train_slow:
            max_local_epochs = np.random.randint(1, max_local_epochs // 2)

        logits = defaultdict(list)
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
                if not torch.isfinite(output).all():
                    print(f"⚠️ FD {self.role} 输出出现 NaN/Inf，跳过当前 batch 并清理模型参数。")
                    _sanitize_module_parameters_(model, f"{self.role}.model")
                    continue
                loss = self.loss(output, y)

                if global_logits is not None:
                    logit_new = output.detach().clone()
                    for j, y_c in enumerate(y.detach().cpu().tolist()):
                        global_logit = _get_global_logit(global_logits, y_c, self.device, logit_new[j, :].shape)
                        if global_logit is not None:
                            logit_new[j, :] = global_logit.data
                    loss += self.loss(output, logit_new.softmax(dim=1)) * self.lamda

                if not torch.isfinite(loss):
                    print(f"⚠️ FD {self.role} 本地 loss 非有限，跳过当前 batch 并清理模型参数。")
                    _sanitize_module_parameters_(model, f"{self.role}.model")
                    continue

                for j, y_c in enumerate(y.detach().cpu().tolist()):
                    logits[y_c].append(_sanitize_tensor(output[j, :].detach().data, f"{self.role} class {y_c} logit"))

                optimizer.zero_grad()
                loss.backward()
                _sanitize_module_gradients_(model, f"{self.role}.model")
                optimizer.step()
                _sanitize_module_parameters_(model, f"{self.role}.model")

        save_item(model, self.role, 'model', self.save_folder_name)
        save_item(agg_func(logits), self.role, 'logits', self.save_folder_name)

        self.train_time_cost['num_rounds'] += 1
        self.train_time_cost['total_cost'] += time.time() - start_time


    def train_metrics(self):
        trainloader = self.load_train_data()
        model = load_item(self.role, 'model', self.save_folder_name).to(self.device)
        _sanitize_module_parameters_(model, f"{self.role}.model")
        global_logits = load_item('Server', 'global_logits', self.save_folder_name)
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
                if not torch.isfinite(output).all():
                    print(f"⚠️ FD {self.role} train_metrics 输出非有限，跳过当前 batch。")
                    continue
                loss = self.loss(output, y)

                if global_logits is not None:
                    logit_new = output.detach().clone()
                    for j, y_c in enumerate(y.detach().cpu().tolist()):
                        global_logit = _get_global_logit(global_logits, y_c, self.device, logit_new[j, :].shape)
                        if global_logit is not None:
                            logit_new[j, :] = global_logit.data
                    loss += self.loss(output, logit_new.softmax(dim=1)) * self.lamda

                if not torch.isfinite(loss):
                    print(f"⚠️ FD {self.role} train_metrics loss 非有限，跳过当前 batch。")
                    continue

                train_num += y.shape[0]
                losses += loss.item() * y.shape[0]

        return losses, train_num


# https://github.com/yuetan031/fedlogit/blob/main/lib/utils.py#L205
def agg_func(logits):
    """
    Returns the average of the weights.
    """

    for [label, logit_list] in logits.items():
        if len(logit_list) > 1:
            logit = 0 * logit_list[0].data
            for i in logit_list:
                logit += _sanitize_tensor(i.data, f"class {label} logit")
            logits[label] = logit / len(logit_list)
        else:
            logits[label] = _sanitize_tensor(logit_list[0], f"class {label} logit")

    return logits

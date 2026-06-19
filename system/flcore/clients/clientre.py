import time
from collections import defaultdict

import numpy as np
import torch

from flcore.clients.clientbase import Client, load_item, save_item


def _sanitize_tensor(tensor, tag, clamp_value=1e4):
    if torch.isfinite(tensor).all():
        return tensor
    print(f"⚠️ FedRE {tag} 出现 NaN/Inf，已执行 nan_to_num。")
    return torch.nan_to_num(tensor, nan=0.0, posinf=clamp_value, neginf=-clamp_value)


def _sanitize_module_parameters_(module, tag, clamp_value=1e4):
    with torch.no_grad():
        for name, param in module.named_parameters():
            if param is not None and not torch.isfinite(param).all():
                print(f"⚠️ FedRE {tag}.{name} 参数出现 NaN/Inf，已执行 nan_to_num。")
                param.data = torch.nan_to_num(param.data, nan=0.0, posinf=clamp_value, neginf=-clamp_value)


def _sanitize_module_gradients_(module, tag, clamp_value=1e4):
    for name, param in module.named_parameters():
        if param.grad is not None and not torch.isfinite(param.grad).all():
            print(f"⚠️ FedRE {tag}.{name} 梯度出现 NaN/Inf，已执行 nan_to_num。")
            param.grad.data = torch.nan_to_num(param.grad.data, nan=0.0, posinf=clamp_value, neginf=-clamp_value)


class clientRE(Client):
    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        super().__init__(args, id, train_samples, test_samples, **kwargs)
        torch.manual_seed(0)
        self.re_samples = args.re_samples

    def train(self):
        trainloader = self.load_train_data()
        model = load_item(self.role, 'model', self.save_folder_name).to(self.device)
        _sanitize_module_parameters_(model, f"{self.role}.model")
        optimizer = torch.optim.SGD(model.parameters(), lr=self.learning_rate)
        model.train()

        start_time = time.time()

        max_local_epochs = self.local_epochs
        if self.train_slow:
            max_local_epochs = np.random.randint(1, max_local_epochs // 2)

        class_reps = defaultdict(list)
        for step in range(max_local_epochs):
            for i, (x, y) in enumerate(trainloader):
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                if self.train_slow:
                    time.sleep(0.1 * np.abs(np.random.rand()))

                rep = model.base(x)
                rep = _sanitize_tensor(rep, f"{self.role} feature")
                output = model.head(rep)
                if not torch.isfinite(output).all():
                    print(f"⚠️ FedRE {self.role} 输出出现 NaN/Inf，跳过当前 batch 并清理模型参数。")
                    _sanitize_module_parameters_(model, f"{self.role}.model")
                    continue

                loss = self.loss(output, y)
                if not torch.isfinite(loss):
                    print(f"⚠️ FedRE {self.role} 本地 loss 非有限，跳过当前 batch 并清理模型参数。")
                    _sanitize_module_parameters_(model, f"{self.role}.model")
                    continue

                optimizer.zero_grad()
                loss.backward()
                _sanitize_module_gradients_(model, f"{self.role}.model")
                optimizer.step()
                _sanitize_module_parameters_(model, f"{self.role}.model")

                for j, y_c in enumerate(y.detach().cpu().tolist()):
                    class_reps[y_c].append(rep[j, :].detach().data)

        save_item(model, self.role, 'model', self.save_folder_name)
        save_item(self._build_entangled_representations(class_reps), self.role, 'entangled_reps', self.save_folder_name)

        self.train_time_cost['num_rounds'] += 1
        self.train_time_cost['total_cost'] += time.time() - start_time

    def set_parameters(self):
        model = load_item(self.role, 'model', self.save_folder_name).to(self.device)
        head = load_item('Server', 'head', self.save_folder_name)
        if head is None:
            save_item(model, self.role, 'model', self.save_folder_name)
            return
        head = head.to(self.device)
        _sanitize_module_parameters_(head, "Server.head")
        for new_param, old_param in zip(head.parameters(), model.head.parameters()):
            old_param.data = new_param.data.clone()
        save_item(model, self.role, 'model', self.save_folder_name)

    def _build_entangled_representations(self, class_reps):
        class_protos = self._average_class_reps(class_reps)
        if len(class_protos) == 0:
            print(f"⚠️ FedRE {self.role} 没有可用类表示，本轮不上传 entangled representation。")
            return []

        labels = sorted(class_protos.keys())
        proto_stack = torch.stack([class_protos[label] for label in labels], dim=0).to(self.device)
        entangled_items = []
        sample_num = max(1, self.re_samples)

        for _ in range(sample_num):
            weights = torch.rand(len(labels), device=self.device)
            weights = weights / weights.sum().clamp_min(1e-12)

            entangled_rep = torch.sum(proto_stack * weights.view(-1, 1), dim=0)
            entangled_label = torch.zeros(self.num_classes, device=self.device)
            for idx, label in enumerate(labels):
                entangled_label[label] = weights[idx]

            entangled_rep = _sanitize_tensor(entangled_rep.detach(), f"{self.role} entangled rep")
            entangled_label = _sanitize_tensor(entangled_label.detach(), f"{self.role} entangled label")
            entangled_items.append((entangled_rep, entangled_label))

        return entangled_items

    def _average_class_reps(self, class_reps):
        class_protos = {}
        for label, rep_list in class_reps.items():
            if len(rep_list) == 0:
                continue
            proto = 0 * rep_list[0].data
            for rep in rep_list:
                proto += _sanitize_tensor(rep.data, f"{self.role} class {label} feature")
            class_protos[label] = proto / len(rep_list)
        return class_protos

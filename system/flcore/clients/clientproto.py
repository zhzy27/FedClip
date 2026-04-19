import copy
import torch
import torch.nn as nn
import numpy as np
import time
from flcore.clients.clientbase import Client, load_item, save_item
from collections import defaultdict


class clientProto(Client):
    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        super().__init__(args, id, train_samples, test_samples, **kwargs)
        torch.manual_seed(0)

        self.loss_mse = nn.MSELoss()
        self.lamda = args.lamda


    def train(self):
        trainloader = self.load_train_data()
        model = load_item(self.role, 'model', self.save_folder_name)
        global_protos = load_item('Server', 'global_protos', self.save_folder_name)
        optimizer = torch.optim.SGD(model.parameters(), lr=self.learning_rate)
        # 跑VIT使用
        # optimizer = torch.optim.SGD(model.parameters(), lr=self.learning_rate,momentum=0.9,weight_decay=1e-4,nesterov=True)
        # model.to(self.device)
        model.train()

        start_time = time.time()

        max_local_epochs = self.local_epochs
        if self.train_slow:
            max_local_epochs = np.random.randint(1, max_local_epochs // 2)

        protos = defaultdict(list)
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
                output = model.head(rep)
                loss = self.loss(output, y)

                if global_protos is not None:
                    proto_new = copy.deepcopy(rep.detach())
                    for i, yy in enumerate(y):
                        y_c = yy.item()
                        #训练稳健性操作
                        if y_c in global_protos.keys():
                            proto_new[i, :] = global_protos[y_c].data
                    loss += self.loss_mse(proto_new, rep) * self.lamda

                for i, yy in enumerate(y):
                    y_c = yy.item()
                    protos[y_c].append(rep[i, :].detach().data)

                optimizer.zero_grad()
                loss.backward()
                # total_norm =torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()

        save_item(agg_func(protos), self.role, 'protos', self.save_folder_name)
        save_item(model, self.role, 'model', self.save_folder_name)

        self.train_time_cost['num_rounds'] += 1
        self.train_time_cost['total_cost'] += time.time() - start_time


    def test_metrics(self):
        testloader = self.load_test_data()
        model = load_item(self.role, 'model', self.save_folder_name)
        global_protos = load_item('Server', 'global_protos', self.save_folder_name)
        # print("原型测试")
        model.eval()

        test_acc = 0
        test_num = 0
        
        if global_protos is not None:
            with torch.no_grad():
                for x, y in testloader:
                    if type(x) == type([]):
                        x[0] = x[0].to(self.device)
                    else:
                        x = x.to(self.device)
                    y = y.to(self.device)
                    rep = model.base(x)

                    output = float('inf') * torch.ones(y.shape[0], self.num_classes).to(self.device)
                    for i, r in enumerate(rep):
                        for j, pro in global_protos.items():
                            if type(pro) != type([]):
                                output[i, j] = self.loss_mse(r, pro)

                    test_acc += (torch.sum(torch.argmin(output, dim=1) == y)).item()
                    test_num += y.shape[0]

            return test_acc, test_num, 0
        else:
            return 0, 1e-5, 0

    # def test_metrics(self, use_proto=True):
    #     testloader = self.load_test_data()
    #     model = load_item(self.role, 'model', self.save_folder_name)
    #     global_protos = load_item('Server', 'global_protos', self.save_folder_name)
    #     model.eval()
    #     correct = 0
    #     total = 0
        
    #     # 定义总类别数（例如CIFAR100为100）
    #     total_num_classes = self.num_classes  
        
    #     # 1. 检查是否应该使用原型测试
    #     should_use_proto = use_proto and global_protos
        
    #     if should_use_proto:
    #         # 检查全局原型是否覆盖了所有类别
    #         if not isinstance(global_protos, dict):
    #             print("警告: 全局原型类型错误，将使用传统分类")
    #             should_use_proto = False
    #         elif len(global_protos) < total_num_classes:
    #             # 原型不全，使用传统方法
    #             missing_classes = total_num_classes - len(global_protos)
    #             print(f"警告: 全局原型不全，缺少{missing_classes}个类别，将使用传统分类")
    #             should_use_proto = False
    #         else:
    #             # 检查原型字典中是否包含所有类别（0到total_num_classes-1）
    #             missing_classes = []
    #             for class_idx in range(total_num_classes):
    #                 if class_idx not in global_protos:
    #                     missing_classes.append(class_idx)
                
    #             if missing_classes:
    #                 print(f"警告: 全局原型缺少类别{missing_classes}，将使用传统分类")
    #                 should_use_proto = False
    #             else:
    #                 # 原型齐全，预计算原型张量
    #                 try:
    #                     proto_labels = list(range(total_num_classes))  # 按顺序0,1,2,...,total_num_classes-1
    #                     proto_tensor = torch.stack([global_protos[k] for k in proto_labels])
    #                     proto_tensor = proto_tensor.to(self.device)
    #                     print(f"使用原型测试方法，共有{len(proto_labels)}个类别的原型")
    #                 except Exception as e:
    #                     print(f"原型预处理错误: {e}，将使用传统分类")
    #                     should_use_proto = False
        
    #     with torch.no_grad():
    #         for inputs, labels in testloader:
    #             inputs, labels = inputs.to(self.device), labels.to(self.device)
    #             features = model.base(inputs)
    #             batch_size = labels.size(0)
                
    #             if should_use_proto:
    #                 # 原型齐全，使用原型测试方法
    #                 # 计算距离矩阵
    #                 dists = torch.cdist(features, proto_tensor, p=2)
    #                 min_indices = torch.argmin(dists, dim=1)
                    
    #                 # 将原型索引转换为实际标签（索引就是标签值）
    #                 predictions = min_indices.long()
    #             else:
    #                 # 使用传统分类方法
    #                 outputs = model.head(features)
    #                 _, predictions = torch.max(outputs, 1)
                
    #             # 计算正确率
    #             correct += (predictions == labels).sum().item()
    #             total += batch_size
        
 
    #     return correct, total, 0

    def train_metrics(self):
        trainloader = self.load_train_data()
        model = load_item(self.role, 'model', self.save_folder_name)
        global_protos = load_item('Server', 'global_protos', self.save_folder_name)
        # model.to(self.device)
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
                rep = model.base(x)
                output = model.head(rep)
                loss = self.loss(output, y)

                if global_protos is not None:
                    proto_new = copy.deepcopy(rep.detach())
                    for i, yy in enumerate(y):
                        y_c = yy.item()
                        if type(global_protos[y_c]) != type([]):
                            proto_new[i, :] = global_protos[y_c].data
                    loss += self.loss_mse(proto_new, rep) * self.lamda
                train_num += y.shape[0]
                losses += loss.item() * y.shape[0]

        return losses, train_num


# https://github.com/yuetan031/fedproto/blob/main/lib/utils.py#L205
def agg_func(protos):
    """
    Returns the average of the weights.
    """

    for [label, proto_list] in protos.items():
        if len(proto_list) > 1:
            proto = 0 * proto_list[0].data
            for i in proto_list:
                proto += i.data
            protos[label] = proto / len(proto_list)
        else:
            protos[label] = proto_list[0]

    return protos
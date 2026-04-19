import copy

import torch
import torch.nn as nn
import numpy as np
import time
from flcore.clients.clientbase import Client, load_item, save_item
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import label_binarize
from sklearn import metrics
class clientadra(Client):
    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        super().__init__(args, id, train_samples, test_samples, **kwargs)
        torch.manual_seed(0)
        self.mse_fn = torch.nn.MSELoss()
    # def train(self,current_round=0):
    #     trainloader = self.load_train_data()
    #     if current_round ==0: 
    #         model = load_item(self.role, 'model', self.save_folder_name) 
    #         optimizer = torch.optim.SGD(model.parameters(), lr=self.learning_rate)
    #         model.to(self.device)
    #         model.train()
    #         start_time = time.time()
    
    #         max_local_epochs = self.local_epochs
    #         if self.train_slow:
    #             max_local_epochs = np.random.randint(1, max_local_epochs // 2)
    
    #         for step in range(max_local_epochs):
    #             for i, (x, y) in enumerate(trainloader):
    #                 optimizer.zero_grad()
    #                 if type(x) == type([]):
    #                     x[0] = x[0].to(self.device)
    #                 else:
    #                     x = x.to(self.device)
    #                 y = y.to(self.device)
    #                 if self.train_slow:
    #                     time.sleep(0.1 * np.abs(np.random.rand()))
    #                 output = model(x)
    #                 loss = self.loss(output, y)
    #                 loss.backward()
    #                 optimizer.step()
    #         save_item(model, self.role, 'model', self.save_folder_name)
    
    #         self.train_time_cost['num_rounds'] += 1
    #         self.train_time_cost['total_cost'] += time.time() - start_time
    #     else:
    #         print("使用全局base防止泛化知识遗忘")
    #         model = load_item(self.role, 'model', self.save_folder_name) 
    #         optimizer = torch.optim.SGD(model.parameters(), lr=self.learning_rate)
    #         model.to(self.device)
    #         #接受的全局模型base参数副本
    #         global_model = copy.deepcopy(model)
    #         local_prototypes = [[] for _ in range(self.num_classes)]
    #         trainloader = self.load_train_data()
    #         # 全局模型提取本地数据的类原型
    #         for x_batch, y_batch in trainloader:
    #             x_batch = x_batch.to(self.device)
    #             y_batch = y_batch.to(self.device)

    #             with torch.no_grad():
    #                 proto_batch = global_model.base(x_batch)

    #             # Scatter the prototypes based on their labels
    #             for proto, y in zip(proto_batch, y_batch):
    #                 local_prototypes[y.item()].append(proto)

    #         mean_prototypes = []
    #         # 计算历史模型的全局原型之后进行参数对齐
    #         for class_prototypes in local_prototypes:

    #             if not class_prototypes == []:
    #                 # Stack the tensors for the current class
    #                 stacked_protos = torch.stack(class_prototypes)

    #                 # Compute the mean tensor for the current class
    #                 mean_proto = torch.mean(stacked_protos, dim=0)
    #                 mean_prototypes.append(mean_proto)
    #             else:
    #                 mean_prototypes.append(None)
    #         start_time = time.time()
    
    #         max_local_epochs = self.local_epochs
    #         if self.train_slow:
    #             max_local_epochs = np.random.randint(1, max_local_epochs // 2)
    
    #         for step in range(max_local_epochs):
    #             for i, (x, y) in enumerate(trainloader):
    #                 optimizer.zero_grad()
    #                 if type(x) == type([]):
    #                     x[0] = x[0].to(self.device)
    #                 else:
    #                     x = x.to(self.device)
    #                 y = y.to(self.device)
    #                 if self.train_slow:
    #                     time.sleep(0.1 * np.abs(np.random.rand()))
    #                 feature_S = model.base(x)
    #                 output = model.head(feature_S)
    #                 ce_loss = self.loss(output, y)
    #                 # 计算每个样本的对应的教师特征原型
    #                 labels_list = y.tolist()
    #                 target_prototypes_list = [mean_prototypes[label] for label in labels_list]
    #                 feature_T = torch.stack(target_prototypes_list).to(self.device)
    #                 mse_loss = self.mse_fn(feature_S, feature_T)
    #                 loss = ce_loss + self.args.mse_lamda * mse_loss
    #                 loss.backward()
    #                 optimizer.step()
    #         save_item(model, self.role, 'model', self.save_folder_name)
    
    #         self.train_time_cost['num_rounds'] += 1
    #         self.train_time_cost['total_cost'] += time.time() - start_time

    def train(self,current_round=0):
        trainloader = self.load_train_data()
        if current_round ==0: 
            model = load_item(self.role, 'model', self.save_folder_name) 
            optimizer = torch.optim.SGD(model.parameters(), lr=self.learning_rate)
            model.to(self.device)
            model.train()
            start_time = time.time()
    
            max_local_epochs = self.local_epochs
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
            save_item(model, self.role, 'model', self.save_folder_name)
    
            self.train_time_cost['num_rounds'] += 1
            self.train_time_cost['total_cost'] += time.time() - start_time
        else:
            print("使用全局base防止泛化知识遗忘")
            model = load_item(self.role, 'model', self.save_folder_name) 
            optimizer = torch.optim.SGD(model.parameters(), lr=self.learning_rate)
            model.to(self.device)
            
            # 动态λ参数计算：前10轮线性增长，之后保持不变
            max_rounds_for_growth = 10  # λ增长的最大轮次

            if current_round <= max_rounds_for_growth:
                # 线性增长：从0到设定的mse_lamda
                mse_lamda= self.args.mse_lamda * (current_round / max_rounds_for_growth)
            else:
                # 10轮后保持最大值
                mse_lamda = self.args.mse_lamda
            #接受的全局模型base参数副本
            global_model = copy.deepcopy(model)
            local_prototypes = [[] for _ in range(self.num_classes)]
            trainloader = self.load_train_data()
            # 全局模型提取本地数据的类原型
            for x_batch, y_batch in trainloader:
                x_batch = x_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                with torch.no_grad():
                    proto_batch = global_model.base(x_batch)

                # Scatter the prototypes based on their labels
                for proto, y in zip(proto_batch, y_batch):
                    local_prototypes[y.item()].append(proto)

            mean_prototypes = []
            # 计算历史模型的全局原型之后进行参数对齐
            for class_prototypes in local_prototypes:

                if not class_prototypes == []:
                    # Stack the tensors for the current class
                    stacked_protos = torch.stack(class_prototypes)

                    # Compute the mean tensor for the current class
                    mean_proto = torch.mean(stacked_protos, dim=0)
                    mean_prototypes.append(mean_proto)
                else:
                    mean_prototypes.append(None)
            start_time = time.time()
    
            max_local_epochs = self.local_epochs
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
                    feature_S = model.base(x)
                    output = model.head(feature_S)
                    ce_loss = self.loss(output, y)
                    # 计算每个样本的对应的教师特征原型
                    labels_list = y.tolist()
                    target_prototypes_list = [mean_prototypes[label] for label in labels_list]
                    feature_T = torch.stack(target_prototypes_list).to(self.device)
                    logits_T = model.head(feature_T)
                    #计算KL损失
                    # 使用温度参数软化概率分布
                    temperature = 1.0
                    soft_targets = F.softmax(logits_T / temperature, dim=1)
                    log_probs = F.log_softmax(output / temperature, dim=1)
                    KL_loss = F.kl_div(log_probs, soft_targets, reduction='batchmean') * (temperature ** 2)
                    mse_loss = self.mse_fn(feature_S, feature_T)
                    loss = ce_loss + mse_lamda * mse_loss +self.args.kl_lamda*KL_loss
                    if self.args.is_regular==1:
                        # print("使用正则化")
                        loss += self.args.regular_lamda*model.frobenius_decay()
                    loss.backward()
                    
                    # print(f"总损失为：{loss},ce_loss:{ce_loss}_占比为{ce_loss/loss},mse_loss:{mse_lamda * mse_loss}_占比为{self.args.mse_lamda * mse_loss/loss},kl_loss:{self.args.kl_lamda*KL_loss}_占比为{self.args.kl_lamda*KL_loss/loss}")
                    optimizer.step()
            save_item(model, self.role, 'model', self.save_folder_name)
    
            self.train_time_cost['num_rounds'] += 1
            self.train_time_cost['total_cost'] += time.time() - start_time
    # 从服务器接受全局模型参数
    def set_parameters(self):
        model = load_item(self.role, 'model', self.save_folder_name).to(self.device)
        global_name = "Server"+ str(model.ratio_LR)
        global_model = load_item(global_name, 'model', self.save_folder_name).to(self.device)
        print(f"客户端{self.role}接收服务器模型参数")
        for new_param, old_param in zip(global_model.base.parameters(), model.base.parameters()):
            old_param.data = new_param.data.clone()
        save_item(model, self.role, 'model', self.save_folder_name)

    # def test_metrics(self):
    #     testloader = self.load_test_data()
    #     client_model = load_item(self.role, 'model', self.save_folder_name).to(self.device)
    #     personalized_head_state_dict = client_model.head.state_dict()
    #     model = load_item('Server', 'model', self.save_folder_name)
    #     #测试模型由全局未分解base+个性化head组成
    #     model.head.load_state_dict(personalized_head_state_dict)
    #     model.to(self.device)
    #     model.eval()
    #     test_acc = 0
    #     test_num = 0
    #     y_prob = []
    #     y_true = []

    #     with torch.no_grad():
    #         for x, y in testloader:
    #             if type(x) == type([]):
    #                 x[0] = x[0].to(self.device)
    #             else:
    #                 x = x.to(self.device)
    #             y = y.to(self.device)
    #             output = model(x)

    #             test_acc += (torch.sum(torch.argmax(output, dim=1) == y)).item()
    #             test_num += y.shape[0]

    #             y_prob.append(output.detach().cpu().numpy())
    #             nc = self.num_classes
    #             if self.num_classes == 2:
    #                 nc += 1
    #             lb = label_binarize(y.detach().cpu().numpy(), classes=np.arange(nc))
    #             if self.num_classes == 2:
    #                 lb = lb[:, :2]
    #             y_true.append(lb)

    #     y_prob = np.concatenate(y_prob, axis=0)
    #     y_true = np.concatenate(y_true, axis=0)

    #     auc = metrics.roc_auc_score(y_true, y_prob, average='micro')

    #     return test_acc, test_num, auc

    # def train_metrics(self):
    #     trainloader = self.load_train_data()
    #     client_model = load_item(self.role, 'model', self.save_folder_name).to(self.device)
    #     personalized_head_state_dict = client_model.head.state_dict()
    #     model = load_item('Server', 'model', self.save_folder_name)
    #     #测试模型由全局未分解base+个性化head组成
    #     model.head.load_state_dict(personalized_head_state_dict)
    #     model.to(self.device)
    #     model.eval()

    #     train_num = 0
    #     losses = 0
    #     with torch.no_grad():
    #         for x, y in trainloader:
    #             if type(x) == type([]):
    #                 x[0] = x[0].to(self.device)
    #             else:
    #                 x = x.to(self.device)
    #             y = y.to(self.device)
    #             output = model(x)
    #             loss = self.loss(output, y)
    #             train_num += y.shape[0]
    #             losses += loss.item() * y.shape[0]

    #     return losses, train_num
    def test_metrics(self):
        testloader = self.load_test_data()
        model = load_item(self.role, 'model', self.save_folder_name).to(self.device)
        model.to(self.device)
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
        model.to(self.device)
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
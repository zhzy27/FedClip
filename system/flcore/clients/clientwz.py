import copy

import torch
import torch.nn as nn
import numpy as np
import time
from flcore.clients.clientbase import Client, load_item, save_item
import torch.nn as nn
import torch.nn.functional as F


class clientWZ(Client):
    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        super().__init__(args, id, train_samples, test_samples, **kwargs)
        torch.manual_seed(0)
        self.history_model = None
        self.have_history_model = False
        self.mse_fn = F.mse_loss
    def train(self):
        trainloader = self.load_train_data()
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
        # 存储历史模型用于之后进行对齐
        self.history_model = copy.deepcopy(model)
        self.have_history_model = True
        save_item(model, self.role, 'model', self.save_folder_name)

        self.train_time_cost['num_rounds'] += 1
        self.train_time_cost['total_cost'] += time.time() - start_time

    # 从服务器接受全局模型参数
    def set_parameters(self):
        model = load_item(self.role, 'model', self.save_folder_name).to(self.device)
        global_model = load_item('Server', 'model', self.save_folder_name).to(self.device)
        # 从全局模型中分解出低秩模型base给客户端
        global_model.decom_larger_model(model.ratio_LR)
        print(f"客户端{self.role}接收服务器模型参数")
        for new_param, old_param in zip(global_model.base.parameters(), model.base.parameters()):
            old_param.data = new_param.data.clone()
        save_item(model, self.role, 'model', self.save_folder_name)

    def align_base(self):
        if self.args.wo_local == 1 or self.have_history_model == False:
            if self.have_history_model == False:
                print("客户端没有历史模型，跳过对齐过程")
        else:
            print("实现全局参数对齐")
            # 全局参数本地化
            # 客户端模型（全局base参数和个性化head参数）
            model = load_item(self.role, 'model', self.save_folder_name).to(self.device)
            local_prototypes = [[] for _ in range(self.num_classes)]
            trainloader = self.load_train_data()
            # 历史模型提取对应的logits和原型
            for x_batch, y_batch in trainloader:
                x_batch = x_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                with torch.no_grad():
                    proto_batch = self.history_model.base(x_batch)

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
            # Align global model's prototype with the local prototype
            alignment_optimizer = torch.optim.SGD(model.base.parameters(),
                                                  lr=self.args.align_lr)
            alignment_loss_fn = torch.nn.MSELoss()
            for _ in range(self.args.align_epoch):
                for x_batch, y_batch in trainloader:
                    
                    x_batch = x_batch.to(self.device)
                    y_batch = y_batch.to(self.device)
                    global_proto_batch = model.base(x_batch)
                    loss = 0
                    for label in y_batch.unique():
                        if mean_prototypes[label.item()] is not None:
                            loss += alignment_loss_fn(global_proto_batch[y_batch == label],
                                                      mean_prototypes[label.item()])
                    # 使用正则化
                    if self.args.is_regular == 1:
                        loss1 = self.args.regular_lamda * model.frobenius_decay()
                        loss += loss1
                        print(f"低秩比为{self.p_rate}对齐过程中总损失为：", loss, "正则项损失为：", loss1)
                    alignment_optimizer.zero_grad()
                    loss.backward()
                    # self.check_gradients(model)
                    alignment_optimizer.step()

            save_item(model, self.role, 'model', self.save_folder_name)

    def kl_alignment_loss(self, feature_T, feature_S, logits_T, logits_S, KL_T=1.0, lamda=0.1):
        # MSE loss between features
        mse_loss = F.mse_loss(feature_S, feature_T)
    
        # KL divergence between softened logits
        p_T = F.softmax(logits_T / KL_T, dim=-1)
        p_S = F.log_softmax(logits_S / KL_T, dim=-1)
        kl_loss = F.kl_div(p_S, p_T, reduction='batchmean') * (KL_T ** 2)
    
        # Combined loss
        loss = mse_loss + lamda * kl_loss
        return loss
    
    # 原型级别的对齐并且使用KL散度
    def align_base_kl(self):
    
        if self.args.wo_local == 1 or self.have_history_model == False:
            if self.have_history_model == False:
                print("客户端没有历史模型，跳过对齐过程")
        else:
            print("实现全局参数KL对齐")
            # 这个model是全局base+个性化head
            model = load_item(self.role, 'model', self.save_folder_name).to(self.device)
            # 全局参数本地化
            trainloader = self.load_train_data()
            local_prototypes = [[] for _ in range(self.num_classes)]
            local_logits = [[] for _ in range(self.num_classes)]
            
            self.history_model.eval()
            # print(f'client{id}')
            for x_batch, y_batch in trainloader:
                x_batch = x_batch.to(self.device)
                y_batch = y_batch.to(self.device)
    
                with torch.no_grad():
                    proto_batch = self.history_model.base(x_batch)
                    logit_batch = self.history_model.head(proto_batch)
    
                # Scatter the prototypes based on their labels
                for proto, y in zip(proto_batch, y_batch):
                    local_prototypes[y.item()].append(proto)
                for logit, y in zip(logit_batch, y_batch):
                    local_logits[y.item()].append(logit)
    
            mean_prototypes = []
            mean_logits = []
            # 计算历史模型的全局原型之后进行参数对齐
            # print(f'client{self.id}')
            for class_prototypes in local_prototypes:
    
                if not class_prototypes == []:
                    # Stack the tensors for the current class
                    stacked_protos = torch.stack(class_prototypes)
    
                    # Compute the mean tensor for the current class
                    mean_proto = torch.mean(stacked_protos, dim=0)
                    mean_prototypes.append(mean_proto)
                else:
                    mean_prototypes.append(None)
            for class_logits in local_logits:
    
                if not class_logits == []:
                    # Stack the tensors for the current class
                    stacked_logits = torch.stack(class_logits)
    
                    # Compute the mean tensor for the current class
                    mean_logit = torch.mean(stacked_logits, dim=0)
                    mean_logits.append(mean_logit)
                else:
                    mean_logits.append(None)
            # self.model是历史共享模型参数，model是从服务器接受的聚合后的全局共享参数
            # Align global model's prototype with the local prototype
            alignment_optimizer = torch.optim.SGD(model.base.parameters(),
                                                  lr=self.args.align_lr)  # Adjust learning rate and optimizer as needed
            for _ in range(self.args.align_epoch):  # Iterate for 1 epochs; adjust as needed
                for x_batch, y_batch in trainloader:
                    alignment_optimizer.zero_grad()
                    x_batch = x_batch.to(self.device)
                    y_batch = y_batch.to(self.device)
                    global_proto_batch = model.base(x_batch)
                    global_logit_batch = model.head(global_proto_batch)
                    loss = 0
                    for label in y_batch.unique():
                        if mean_prototypes[label.item()] is not None:
                            proto_T = mean_prototypes[label.item()].expand_as(global_proto_batch[y_batch == label])
                            logit_T = mean_logits[label.item()].expand_as(global_logit_batch[y_batch == label])
                            loss += self.kl_alignment_loss(proto_T, global_proto_batch[y_batch == label], logit_T,
                                                           global_logit_batch[y_batch == label], KL_T=self.args.kl_Tim,
                                                           lamda=self.args.kl_lamda)
                                        #使用正则化
                    if self.args.is_regular==1:
                        # print("使用正则化")
                        loss += self.args.regular_lamda*model.frobenius_decay()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)  # 添加梯度裁剪
                    alignment_optimizer.step()
            # 保存base对齐过后模型参数
            save_item(model, self.role, 'model', self.save_folder_name)


    # def kl_loss(self, logits_T, logits_S, KL_T=1.0):
    #     # KL divergence between softened logits
    #     p_T = F.softmax(logits_T / KL_T, dim=-1)
    #     p_S = F.log_softmax(logits_S / KL_T, dim=-1)
    #     kl_loss = F.kl_div(p_S, p_T, reduction='batchmean') * (KL_T ** 2)

    #     return kl_loss
    
    # #和特征MSE使用KL散度
    # def align_base_kl(self):
    
    #     if self.args.wo_local == 1 or self.have_history_model == False:
    #         if self.have_history_model == False:
    #             print("客户端没有历史模型，跳过对齐过程")
    #     else:
    #         print("实现全局参数特征MSE+KL对齐")
    #         # 这个model是全局base+个性化head
    #         model = load_item(self.role, 'model', self.save_folder_name).to(self.device)
    #         # 全局参数本地化
    #         trainloader = self.load_train_data()
    #         self.history_model.eval()
    #         # self.model是历史共享模型参数，model是从服务器接受的聚合后的全局共享参数
    #         # Align global model's prototype with the local prototype
    #         alignment_optimizer = torch.optim.SGD(model.base.parameters(),
    #                                               lr=self.args.align_lr)  # Adjust learning rate and optimizer as needed
        
    #         for _ in range(self.args.align_epoch):  # Iterate for 1 epochs; adjust as needed
    #             for x_batch, y_batch in trainloader:
    #                 alignment_optimizer.zero_grad()
    #                 x_batch = x_batch.to(self.device)
    #                 y_batch = y_batch.to(self.device)
    #                 feature_S = model.base(x_batch)
    #                 logits_S = model.head(feature_S)
    #                 with torch.no_grad():
    #                     feature_T = self.history_model.base(x_batch)
    #                     logits_T = self.history_model.head(feature_T)
    #                 kl_loss = self.kl_loss(logits_T=logits_T,logits_S=logits_S,KL_T=self.args.kl_Tim)
    #                 mse_loss = self.mse_fn(feature_S,feature_T)
    #                 loss = kl_loss+mse_loss
    #                                     #使用正则化
    #                 if self.args.is_regular==1:
    #                     # print("使用正则化")
    #                     loss += self.args.regular_lamda*model.frobenius_decay()
    #                 loss.backward()
    #                 alignment_optimizer.step()
    #         # 保存base对齐过后模型参数
    #         save_item(model, self.role, 'model', self.save_folder_name)
    
    # #原型特征对齐+细粒度logits对齐
    # def kl_loss(self, logits_T, logits_S, KL_T=1.0, lamda=0.1):
    #     p_T = F.softmax(logits_T / KL_T, dim=-1)
    #     p_S = F.log_softmax(logits_S / KL_T, dim=-1)
    #     kl_loss = F.kl_div(p_S, p_T, reduction='batchmean') * (KL_T ** 2)
    #     return kl_loss

    # def align_base_kl(self):
    #     if self.args.wo_local == 1 or not self.have_history_model:
    #         if not self.have_history_model:
    #             print("客户端没有历史模型，跳过对齐过程")
    #     else:
    #         print("实现全局参数细粒度KL对齐")
    #         model = load_item(self.role, 'model', self.save_folder_name).to(self.device)
    #         trainloader = self.load_train_data()
    #         local_prototypes = [[] for _ in range(self.num_classes)]
    #         self.history_model.eval()
    #         for x_batch, y_batch in trainloader:
    #             x_batch = x_batch.to(self.device)
    #             y_batch = y_batch.to(self.device)
    #             with torch.no_grad():
    #                 proto_batch = self.history_model.base(x_batch)
    #             for proto, y in zip(proto_batch, y_batch):
    #                 local_prototypes[y.item()].append(proto)

    #         mean_prototypes = []
    #         for class_prototypes in local_prototypes:
    #             mean_prototypes.append(torch.mean(torch.stack(class_prototypes), dim=0) if class_prototypes else None)
    #         alignment_loss_fn = torch.nn.MSELoss()
    #         alignment_optimizer = torch.optim.SGD(model.base.parameters(), lr=self.args.align_lr)

    #         for epoch in range(self.args.align_epoch):
    #             epoch_loss = 0.0
    #             num_batches = 0
    #             for x_batch, y_batch in trainloader:
    #                 alignment_optimizer.zero_grad()
    #                 x_batch = x_batch.to(self.device)
    #                 y_batch = y_batch.to(self.device)
    #                 global_proto_batch = model.base(x_batch)
    #                 logit_S = model.head(global_proto_batch)
    #                 with torch.no_grad():
    #                     logit_T = self.history_model(x_batch)
    #                 loss = 0
    #                 for label in y_batch.unique():
    #                     if mean_prototypes[label.item()] is not None:
    #                         loss += alignment_loss_fn(global_proto_batch[y_batch == label],
    #                                                   mean_prototypes[label.item()])
    #                 kl_loss = self.kl_loss(logit_T,logit_S,KL_T=self.args.kl_Tim)
    #                 loss += self.args.kl_lamda*kl_loss
    #                 if self.args.is_regular == 1:
    #                     loss += self.args.regular_lamda * model.frobenius_decay()
    #                 loss.backward()

    #                 alignment_optimizer.step()
    #                 epoch_loss += loss.item()
    #                 num_batches += 1
    #             if num_batches > 0:
    #                 print(f"[对齐] Epoch {epoch+1}/{self.args.align_epoch}: 平均Loss={epoch_loss/num_batches:.6f}")
    #         save_item(model, self.role, 'model', self.save_folder_name)
    # def kl_alignment_loss(self, feature_T, feature_S, logits_T, logits_S, KL_T=1.0, lamda=0.1):
    #     # MSE loss between features
    #     mse_loss = F.mse_loss(feature_S, feature_T)

    #     # KL divergence between softened logits
    #     p_T = F.softmax(logits_T / KL_T, dim=-1)
    #     p_S = F.log_softmax(logits_S / KL_T, dim=-1)
    #     kl_loss = F.kl_div(p_S, p_T, reduction='batchmean') * (KL_T ** 2)

    #     # === 监控数值 ===
    #     if torch.isnan(mse_loss) or torch.isinf(mse_loss):
    #         print(f"[警告] MSE 损失出现异常: {mse_loss.item()}")
    #     if torch.isnan(kl_loss) or torch.isinf(kl_loss):
    #         print(f"[警告] KL 损失出现异常: {kl_loss.item()}")
    #     if torch.isnan(p_T).any() or torch.isinf(p_T).any():
    #         print(f"[警告] p_T 包含异常值 (min={p_T.min().item():.4f}, max={p_T.max().item():.4f})")
    #     if torch.isnan(p_S).any() or torch.isinf(p_S).any():
    #         print(f"[警告] p_S 包含异常值 (min={p_S.min().item():.4f}, max={p_S.max().item():.4f})")

    #     # Combined loss
    #     loss = mse_loss + lamda * kl_loss
    #     return loss

    # def align_base_kl(self):
    #     if self.args.wo_local == 1 or not self.have_history_model:
    #         if not self.have_history_model:
    #             print("客户端没有历史模型，跳过对齐过程")
    #     else:
    #         print("实现全局参数KL对齐")
    #         model = load_item(self.role, 'model', self.save_folder_name).to(self.device)
    #         trainloader = self.load_train_data()
    #         local_prototypes = [[] for _ in range(self.num_classes)]
    #         local_logits = [[] for _ in range(self.num_classes)]

    #         self.history_model.eval()
    #         for x_batch, y_batch in trainloader:
    #             x_batch = x_batch.to(self.device)
    #             y_batch = y_batch.to(self.device)
    #             with torch.no_grad():
    #                 proto_batch = self.history_model.base(x_batch)
    #                 logit_batch = self.history_model.head(proto_batch)
    #             for proto, y in zip(proto_batch, y_batch):
    #                 local_prototypes[y.item()].append(proto)
    #             for logit, y in zip(logit_batch, y_batch):
    #                 local_logits[y.item()].append(logit)

    #         mean_prototypes = []
    #         mean_logits = []
    #         for class_prototypes in local_prototypes:
    #             mean_prototypes.append(torch.mean(torch.stack(class_prototypes), dim=0) if class_prototypes else None)
    #         for class_logits in local_logits:
    #             mean_logits.append(torch.mean(torch.stack(class_logits), dim=0) if class_logits else None)

    #         alignment_optimizer = torch.optim.SGD(model.base.parameters(), lr=self.args.align_lr)

    #         for epoch in range(self.args.align_epoch):
    #             epoch_loss = 0.0
    #             num_batches = 0
    #             for x_batch, y_batch in trainloader:
    #                 alignment_optimizer.zero_grad()
    #                 x_batch = x_batch.to(self.device)
    #                 y_batch = y_batch.to(self.device)
    #                 global_proto_batch = model.base(x_batch)
    #                 global_logit_batch = model.head(global_proto_batch)

    #                 # === 监控 forward 结果 ===
    #                 if torch.isnan(global_logit_batch).any() or torch.isinf(global_logit_batch).any():
    #                     print(f"[警告] Epoch {epoch} logits 出现异常 (min={global_logit_batch.min().item():.4f}, max={global_logit_batch.max().item():.4f})")

    #                 loss = 0
    #                 for label in y_batch.unique():
    #                     if mean_prototypes[label.item()] is not None:
    #                         proto_T = mean_prototypes[label.item()].expand_as(global_proto_batch[y_batch == label])
    #                         logit_T = mean_logits[label.item()].expand_as(global_logit_batch[y_batch == label])
    #                         loss += self.kl_alignment_loss(proto_T, global_proto_batch[y_batch == label], logit_T,
    #                                                        global_logit_batch[y_batch == label], KL_T=self.args.kl_Tim,
    #                                                        lamda=self.args.kl_lamda)
    #                 if self.args.is_regular == 1:
    #                     loss += self.args.regular_lamda * model.frobenius_decay()

    #                 # === 监控 loss ===
    #                 if torch.isnan(loss) or torch.isinf(loss):
    #                     print(f"[警告] Epoch {epoch} Loss 异常: {loss.item()}")
    #                     continue

    #                 loss.backward()

    #                 # === 监控梯度 ===
    #                 for name, param in model.named_parameters():
    #                     if param.grad is not None and (torch.isnan(param.grad).any() or torch.isinf(param.grad).any()):
    #                         print(f"[警告] 梯度异常: {name}")
    #                         alignment_optimizer.zero_grad()
    #                         break

    #                 alignment_optimizer.step()
    #                 epoch_loss += loss.item()
    #                 num_batches += 1
    #             if num_batches > 0:
    #                 print(f"[对齐] Epoch {epoch+1}/{self.args.align_epoch}: 平均Loss={epoch_loss/num_batches:.6f}")
    #         save_item(model, self.role, 'model', self.save_folder_name)
import copy
from collections import defaultdict

import torch
import torch.nn as nn
import numpy as np
import time
from flcore.clients.clientbase import Client, load_item, save_item
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import label_binarize
from sklearn import metrics


class clientARA2(Client):
    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        super().__init__(args, id, train_samples, test_samples, **kwargs)
        torch.manual_seed(0)
        self.mse_fn = torch.nn.MSELoss()

    def train(self, current_round=0):
        trainloader = self.load_train_data()
        if current_round == 0:
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
            # 收集并计算原型
            self.collect_protos()
            self.train_time_cost['num_rounds'] += 1
            self.train_time_cost['total_cost'] += time.time() - start_time
        else:
            # 接受全局原型进行下一步操作
            global_protos = load_item('Server', 'global_protos', self.save_folder_name)
            # 接受全局位置矩阵
            global_sim_matrix = load_item('Server', 'global_sim_matrix', self.save_folder_name)

            # 将全局矩阵转换为tensor并移动到设备
            global_sim_tensor = torch.tensor(global_sim_matrix, dtype=torch.float32).to(self.device)

            # 获取全局原型tensor和类别顺序
            sorted_labels = sorted(global_protos.keys())
            global_protos_tensor = torch.stack([global_protos[label] for label in sorted_labels]).to(self.device)

            # 使用全局原型做约束不使用模型生成的一致性锚点做约束
            print("使用对比学习+余弦全局约束+MSE对齐")
            model = load_item(self.role, 'model', self.save_folder_name)
            optimizer = torch.optim.SGD(model.parameters(), lr=self.learning_rate)
            model.to(self.device)

            # 动态λ参数计算：前10轮线性增长，之后保持不变
            max_rounds_for_growth = 10  # λ增长的最大轮次

            # if current_round <= max_rounds_for_growth:
            #     # 线性增长：从0到设定的mse_lamda
            #     mse_lamda = self.args.mse_lamda * (current_round / max_rounds_for_growth)
            #     contrast_lamda = self.args.Con_lamda * (current_round / max_rounds_for_growth)  # 对比学习权重也动态增长
            # else:
            #     # 10轮后保持最大值
            #     mse_lamda = self.args.mse_lamda
            #     contrast_lamda = self.args.Con_lamda  # 对比学习权重
            
            if current_round <= max_rounds_for_growth:
                # MSE权重：线性增长从0到设定的mse_lamda
                mse_lamda = self.args.mse_lamda * (current_round / max_rounds_for_growth)
                
                # 对比学习权重：线性递减从Con_lamda到Con_lamda * 0.1
                contrast_lamda = self.args.Con_lamda * (1.0 - 0.8 * (current_round / max_rounds_for_growth))
            else:
                # 10轮后MSE保持最大值，对比学习保持较小值
                mse_lamda = self.args.mse_lamda
                contrast_lamda = self.args.Con_lamda * 0.2  # 对比学习权重递减到40%
            
            start_time = time.time()
            max_local_epochs = self.local_epochs
            if self.train_slow:
                max_local_epochs = np.random.randint(1, max_local_epochs // 2)

            # Local Training
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

                    # 计算余弦相似度矩阵
                    norm_features = F.normalize(feature_S, p=2, dim=1)
                    norm_protos = F.normalize(global_protos_tensor, p=2, dim=1)
                    sample_cos_similarities = torch.mm(norm_features, norm_protos.t())

                    # ========== 对比学习损失 ==========
                    if self.args.which_con =="sim":
                        contrastive_loss = self.global_consistent_contrastive_learning(
                        feature_S, global_protos_tensor, y, sorted_labels, temperature=0.1
                    )
                    else:
                        contrastive_loss=self.mse_distance_contrastive_learning(
                        feature_S, global_protos_tensor, y, sorted_labels, temperature=0.1
                    )

                    # ========== 余弦距离全局约束 ==========
                    # cosine_constraint_loss = self.cosine_global_constraint(
                    #     sample_cos_similarities, global_sim_tensor, y, sorted_labels
                    # )
                    cosine_constraint_loss = self.cosine_global_constraint(
                        sample_cos_similarities, global_sim_tensor, y, sorted_labels,0.01
                    )

                    # Calculate the corresponding teacher feature prototype for each sample
                    labels_list = y.tolist()

                    # 处理全局原型中不存在的类别
                    valid_indices = []
                    target_prototypes_list = []

                    for idx, label in enumerate(labels_list):
                        if label in global_protos:
                            valid_indices.append(idx)
                            target_prototypes_list.append(global_protos[label])

                    # 计算MSE损失（只针对在全局原型中存在的类别）- 保留样本与其对应原型的MSE对齐损失
                    if target_prototypes_list:
                        valid_feature_S = feature_S[valid_indices]
                        feature_T = torch.stack(target_prototypes_list).to(self.device)
                        mse_loss = self.mse_fn(valid_feature_S, feature_T)
                    else:
                        mse_loss = torch.tensor(0.0).to(self.device)

                    # 总损失 = 分类损失 + 传统MSE对齐损失 + 对比学习损失 + 余弦全局约束
                    loss = (ce_loss +
                            mse_lamda * mse_loss +
                            contrast_lamda * contrastive_loss +
                            self.args.Rel_lamda * cosine_constraint_loss)  # 余弦约束使用较小权重

                    if self.args.is_regular == 1:
                        loss += self.args.regular_lamda * model.frobenius_decay()

                    loss.backward()

                    # # 打印详细的损失信息
                    # if i % 50 == 0:
                    #     total = loss.item()
                    #     ce_ratio = ce_loss.item() / total
                    #     mse_ratio = (mse_lamda * mse_loss.item()) / total
                    #     contrast_ratio = (contrast_lamda * contrastive_loss.item()) / total
                    #     cosine_ratio = (self.args.Rel_lamda * cosine_constraint_loss.item()) / total

                    #     print(f"Round {current_round}, Step {step}, Batch {i}:")
                    #     print(f"  Total: {total:.4f}")
                    #     print(f"  CE: {ce_loss.item():.4f} ({ce_ratio:.1%})")
                    #     print(f"  MSE Align: {mse_lamda * mse_loss.item():.4f} ({mse_ratio:.1%})")
                    #     print(f"  Contrast: {contrast_lamda * contrastive_loss.item():.4f} ({contrast_ratio:.1%})")
                    #     print(f"  Cosine Constraint: {self.args.Rel_lamda * cosine_constraint_loss.item():.4f} ({cosine_ratio:.1%})")

                    optimizer.step()

            save_item(model, self.role, 'model', self.save_folder_name)
            # 收集并计算原型
            self.collect_protos()

            self.train_time_cost['num_rounds'] += 1
            self.train_time_cost['total_cost'] += time.time() - start_time

    def mse_distance_contrastive_learning(self, feature_S, global_protos_tensor, y,
                                          sorted_labels, temperature=1.0, hard_negative_mining=False):
        """
        基于MSE距离的对比损失实现
        使用MSE距离替代余弦相似度进行对比学习

        Args:
            feature_S: 本地样本特征 [batch_size, feat_dim]
            global_protos_tensor: 全局原型tensor [num_classes, feat_dim]
            y: 样本标签 [batch_size]
            sorted_labels: 排序后的标签列表
            temperature: 温度参数，控制对比的尖锐程度
            hard_negative_mining: 是否使用硬负样本挖掘
        """
        batch_size, feat_dim = feature_S.shape
        num_classes = global_protos_tensor.shape[0]

        # 计算MSE距离矩阵 (batch_size, num_classes)
        expanded_features = feature_S.unsqueeze(1)  # (batch_size, 1, feat_dim)
        expanded_protos = global_protos_tensor.unsqueeze(0)  # (1, num_classes, feat_dim)
        mse_distances = torch.mean((expanded_features - expanded_protos) ** 2, dim=2)  # (batch_size, num_classes)

        # 将距离转换为相似度（距离越小，相似度越高）
        # 使用负距离并缩放，确保数值稳定性
        similarities = -mse_distances / temperature
        # 裁剪相似度值，防止指数运算溢出
        similarities = torch.clamp(similarities, min=-50, max=50)

        contrastive_loss = 0.0
        valid_count = 0

        for batch_idx in range(batch_size):
            true_label = y[batch_idx].item()

            if true_label in sorted_labels:
                true_label_idx = sorted_labels.index(true_label)
                valid_count += 1

                # 获取当前样本与所有原型的相似度
                sample_similarities = similarities[batch_idx]

                # 正样本相似度（对应真实类别）
                positive_sim = sample_similarities[true_label_idx]

                if hard_negative_mining :  # 只在类别数较多时使用硬负样本挖掘(选择最近似的40%负样本)
                    # 硬负样本挖掘：选择与正样本最相似的top_k个负样本
                    k = int(min(self.args.num_classes*0.4, num_classes - 1))  # 选择3个最难负样本或所有负样本

                    # 创建掩码排除正样本
                    mask = torch.ones(num_classes, dtype=torch.bool, device=feature_S.device)
                    mask[true_label_idx] = False

                    # 获取负样本相似度
                    negative_sims = sample_similarities[mask]

                    # 选择最相似的top_k个负样本（最难负样本）
                    topk_negatives, _ = torch.topk(negative_sims, k)

                    denominator = torch.exp(positive_sim) + torch.exp(topk_negatives).sum()
                else:
                    # 标准InfoNCE：使用所有负样本
                    denominator = torch.exp(sample_similarities).sum()

                # 对比损失
                instance_loss = -torch.log(torch.exp(positive_sim) / (denominator + 1e-8))
                contrastive_loss += instance_loss

        return contrastive_loss / max(valid_count, 1) if valid_count > 0 else torch.tensor(0.0)


    def global_consistent_contrastive_learning(self, feature_S, global_protos_tensor, y,
                                               sorted_labels, temperature=0.1):
        """改进的对比损失实现"""
    
        batch_size = feature_S.shape[0]
        contrastive_loss = 0.0
        valid_count = 0
    
        # 添加归一化 - 确保计算的是余弦相似度
        norm_features = F.normalize(feature_S, p=2, dim=1)
        norm_protos = F.normalize(global_protos_tensor, p=2, dim=1)
    
        # 计算特征-原型相似度（余弦相似度）
        feature_proto_sim = torch.mm(norm_features, norm_protos.t()) / temperature
        # 裁剪相似度值，防止指数运算溢出
        feature_proto_sim = torch.clamp(feature_proto_sim, min=-50, max=50)
        for batch_idx in range(batch_size):
            true_label = y[batch_idx].item()
    
            if true_label in sorted_labels:
                true_label_idx = sorted_labels.index(true_label)
                valid_count += 1
    
                # 获取当前样本与所有原型的相似度
                sample_similarities = feature_proto_sim[batch_idx]
    
                # 正样本相似度
                positive_sim = sample_similarities[true_label_idx]
    
                # 标准InfoNCE损失
                numerator = torch.exp(positive_sim)
                denominator = torch.exp(sample_similarities).sum()
    
                instance_loss = -torch.log(numerator / denominator)
                contrastive_loss += instance_loss
    
        return contrastive_loss / max(valid_count, 1) if valid_count > 0 else torch.tensor(0.0)

    # # 余弦距离全局约束方法
    # def cosine_global_constraint(self, sample_cos_similarities, global_sim_tensor, y, sorted_labels):
    #     """余弦距离全局约束"""

    #     batch_size = sample_cos_similarities.shape[0]
    #     constraint_loss = 0.0
    #     valid_count = 0

    #     for batch_idx in range(batch_size):
    #         true_label = y[batch_idx].item()

    #         if true_label in sorted_labels:
    #             true_label_idx = sorted_labels.index(true_label)
    #             valid_count += 1

    #             # 获取当前样本的余弦相似度关系
    #             sample_cos_relation = sample_cos_similarities[batch_idx]

    #             # 获取全局余弦相似度关系
    #             global_cos_relation = global_sim_tensor[true_label_idx]
    #             loss = F.mse_loss(sample_cos_relation, global_cos_relation)
    #             constraint_loss += loss

    #     return constraint_loss / max(valid_count, 1)
    
    def cosine_global_constraint(self, sample_cos_similarities, global_sim_tensor, y, sorted_labels, temperature=1.0):
        """基于KL散度的余弦全局约束"""

        batch_size = sample_cos_similarities.shape[0]
        constraint_loss = 0.0
        valid_count = 0

        for batch_idx in range(batch_size):
            true_label = y[batch_idx].item()

            if true_label in sorted_labels:
                true_label_idx = sorted_labels.index(true_label)
                valid_count += 1

                # 获取当前样本的余弦相似度关系
                sample_cos_relation = sample_cos_similarities[batch_idx]
                # 获取全局余弦相似度关系
                global_cos_relation = global_sim_tensor[true_label_idx]
  
                # 使用KL散度计算分布差异
                # KL(P||Q) = sum(P * log(P/Q))
                # F.kl_div要求输入是log概率，所以我们需要对sample_probs取log
                kl_loss = F.kl_div(
                    input=F.log_softmax(sample_cos_relation / temperature, dim=0),
                    target=F.softmax(global_cos_relation / temperature, dim=0),
                    reduction='batchmean'
                )

                constraint_loss += kl_loss
        # print(f"全局约束损失：{constraint_loss}")
        return constraint_loss / max(valid_count, 1) if valid_count > 0 else torch.tensor(0.0)

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

    def static_rank(self):
        # Prepare the model
        model = copy.deepcopy(load_item(self.role, 'model', self.save_folder_name).to(self.device))
        if model.ratio_LR < 1.0 and self.id == 3:
            # Reconstruct full weights
            W1 = model.fc1.reconstruct_full_weight()
            W2 = model.fc2.reconstruct_full_weight()

            # Compute all singular values of W1 and W2
            with torch.no_grad():
                # Compute full SVD
                U1, S1, V1 = torch.svd(W1)
                U2, S2, V2 = torch.svd(W2)
                # Store complete SVD results
                RANK_1 = S1.cpu().numpy()
                RANK_2 = S2.cpu().numpy()

                self.W1_RANK.append(RANK_1)
                self.W2_RANK.append(RANK_2)
            print(f"客户端{self.id}的FC1的奇异值分布为{np.cumsum(RANK_1) / sum(RANK_1)}")
            print(f"客户端{self.id}的FC2的奇异值分布为{np.cumsum(RANK_2) / sum(RANK_2)}")

    # 收集本地原型用于进行相似度矩阵计算
    def collect_protos(self):
        trainloader = self.load_train_data()
        model = load_item(self.role, 'model', self.save_folder_name)
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

                for i, yy in enumerate(y):
                    y_c = yy.item()
                    protos[y_c].append(rep[i, :].detach().data)
        save_item(agg_func(protos), self.role, 'protos', self.save_folder_name)


# 聚和出对应的原型
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
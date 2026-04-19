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


class clientDAR(Client):
    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        super().__init__(args, id, train_samples, test_samples, **kwargs)
        torch.manual_seed(0)
        self.mse_fn = torch.nn.MSELoss()


    def train(self, current_round=0):
        trainloader = self.load_train_data()
        if current_round == 0:
            model = load_item(self.role, 'model', self.save_folder_name)
            # 跑VIT使用
            # optimizer = torch.optim.SGD(model.parameters(), lr=self.learning_rate,momentum=0.9,weight_decay=1e-4,nesterov=True)
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
            model = load_item(self.role, 'model', self.save_folder_name)
            optimizer = torch.optim.SGD(model.parameters(), lr=self.learning_rate)
            # 跑VIT使用
            # optimizer = torch.optim.SGD(model.parameters(), lr=self.learning_rate,momentum=0.9,weight_decay=1e-4,nesterov=True)
            model.to(self.device)
            mse_lamda = self.args.mse_lamda
            contrast_lamda = self.args.Con_lamda
            # 接受的全局模型base参数副本
            global_model = copy.deepcopy(model)
            local_prototypes = {}
            trainloader = self.load_train_data()
            global_model.eval()
            # 全局模型提取本地数据的类原型
            for x_batch, y_batch in trainloader:
                x_batch = x_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                with torch.no_grad():
                    proto_batch = global_model.base(x_batch)
                # Scatter the prototypes based on their labels
                for proto, y in zip(proto_batch, y_batch):
                    if y not in local_prototypes.keys():
                        local_prototypes[y.item()] = []
                        local_prototypes[y.item()].append(proto)
                    else:
                        local_prototypes[y.item()].append(proto)
            mean_prototypes = {}
            # 计算一致性锚点
            for label,class_prototypes in local_prototypes.items():
                stacked_protos = torch.stack(class_prototypes)
                # Compute the mean tensor for the current class
                mean_proto = torch.mean(stacked_protos, dim=0)
                mean_prototypes[label]=mean_proto

            sorted_labels = sorted(mean_prototypes.keys())
            protos_tensor = torch.stack([mean_prototypes[label] for label in sorted_labels]).to(self.device)

            start_time = time.time()
            max_local_epochs = self.local_epochs
            if self.train_slow:
                max_local_epochs = np.random.randint(1, max_local_epochs // 2)
            # 正式本地训练
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
                    #mse损失
                    mse_loss = self.mse_fn(feature_S, feature_T)
                    #对比损失
                    # ========== 对比学习损失 ==========  and contrast_lamda > 0.0
                    if self.args.which_con == "sim":
                        # contrastive_loss = self.consistent_contrastive_learning(
                        #     feature_S, protos_tensor, y, sorted_labels, temperature=self.args.Con_T,
                        #     hard_negative_mining=self.args.hard_negative_mining, top_k=self.args.topk
                        # )
                        contrastive_loss = self.consistent_contrastive_learning(
                            feature_S, protos_tensor, y, sorted_labels, temperature=self.args.Con_T
                        )
                    else:
                        contrastive_loss = self.mse_distance_contrastive_learning(
                            feature_S, protos_tensor, y, sorted_labels, temperature=self.args.Con_T
                        )
                    # 总损失 = 分类损失 + MSE对齐损失 + 对比学习损失 
                    loss = ce_loss + mse_lamda * mse_loss + contrast_lamda * contrastive_loss
                    if self.args.is_regular == 1:
                        # print("使用正则化")
                        loss += self.args.regular_lamda * model.frobenius_decay()
                    loss.backward()
                    # total_norm =torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                    #TT分解要加梯度裁剪防止出现NAN
                    # if self.args.is_TT_Decom:   []
                    #     total_norm =torch.nn.utils.clip_grad_norm_(model.parameters(), 50.0)
                    #     print(f"Gradient norm: {total_norm}")
                    # if i%50==0:
                    #     print(f"Batch_id:{i}")
                    #     print(f"总损失为：{loss},ce_loss:{ce_loss}_占比为{ce_loss/loss},mse_loss:{mse_lamda * mse_loss}_占比为{mse_lamda * mse_loss/loss},con_loss:{contrast_lamda * contrastive_loss}_占比为{contrast_lamda * contrastive_loss/loss}")
                    optimizer.step()
            save_item(model, self.role, 'model', self.save_folder_name)
            self.train_time_cost['num_rounds'] += 1
            self.train_time_cost['total_cost'] += time.time() - start_time
            
    def classifier_consist(self, protos, head_param):
        """
        分类器一致性损失函数
        Args:
            protos: 字典，键为标签，值为原型向量 {label: prototype}
            head_param: 分类器权重参数，形状为 (num_classes, feature_dim)
        Returns:
            loss: 一致性损失
        """
        loss = torch.tensor(0.0).to(head_param.device)
        count = torch.tensor(0.0).to(head_param.device)

        for label, proto in protos.items():
            # 确保原型向量的维度正确 (feature_dim,) -> (1, feature_dim)
            if proto.dim() == 1:
                proto = proto.unsqueeze(0)  # (1, feature_dim)

            # 计算相似度向量: proto * head_param
            # proto: (1, feature_dim), head_param: (num_classes, feature_dim)
            # 结果: (1, num_classes)
            sim_vector = torch.matmul(proto, head_param.t())  # 矩阵乘法

            # 确保标签在有效范围内
            if label >= sim_vector.size(1):
                continue

            # 计算softmax概率
            probabilities = F.softmax(sim_vector, dim=1)

            # 计算交叉熵损失: -log(p_true)
            class_loss = -torch.log(probabilities[0, label] + 1e-8)  # 添加小值防止log(0)

            loss = loss + class_loss
            count = count + 1

        # 如果count为0，返回0损失
        if count == 0:
            return torch.tensor(0.0).to(head_param.device)

        return loss / count



    def mse_distance_contrastive_learning(self, feature_S, protos_tensor, y,
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
        #本地类别少就不对比
        if len(sorted_labels)<=1:
            return torch.tensor(0.0)
        batch_size, feat_dim = feature_S.shape
        num_classes = protos_tensor.shape[0]

        # 计算MSE距离矩阵 (batch_size, num_classes)
        expanded_features = feature_S.unsqueeze(1)  # (batch_size, 1, feat_dim)
        expanded_protos = protos_tensor.unsqueeze(0)  # (1, num_classes, feat_dim)
        mse_distances = torch.mean((expanded_features - expanded_protos) ** 2, dim=2)  # (batch_size, num_classes)

        # 将距离转换为相似度（距离越小，相似度越高）
        # 使用负距离并缩放，确保数值稳定性
        similarities = -mse_distances / temperature

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
                # print(f"当前样本与正样本相似度为{positive_sim},整体相似度为{sample_similarities}")

                if hard_negative_mining and num_classes > 5:  # 只在类别数较多时使用硬负样本挖掘
                    # 硬负样本挖掘：选择与正样本最相似的top_k个负样本
                    k = min(3, num_classes - 1)  # 选择3个最难负样本或所有负样本

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

    def consistent_contrastive_learning(self, feature_S, protos_tensor, y,
                                               sorted_labels, temperature=0.1):
        """改进的对比损失实现"""
        if len(sorted_labels)<=1:
            return torch.tensor(0.0)
        batch_size = feature_S.shape[0]
        contrastive_loss = 0.0
        valid_count = 0

        # 添加归一化 - 确保计算的是余弦相似度
        norm_features = F.normalize(feature_S, p=2, dim=1)
        norm_protos = F.normalize(protos_tensor, p=2, dim=1)

        # 计算特征-原型相似度（余弦相似度）
        feature_proto_sim = torch.mm(norm_features, norm_protos.t()) / temperature

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


    # def consistent_contrastive_learning(self, feature_S, protos_tensor, y,
    #                                     sorted_labels, temperature=0.1,
    #                                     hard_negative_mining=False, top_k=10):
    #     """改进的对比损失实现 - 添加负样本top-k选择"""
    #     if len(sorted_labels) <= 1:
    #         return torch.tensor(0.0, device=feature_S.device)

    #     batch_size = feature_S.shape[0]
    #     contrastive_loss = 0.0
    #     valid_count = 0

    #     # 添加归一化 - 确保计算的是余弦相似度
    #     norm_features = F.normalize(feature_S, p=2, dim=1)
    #     norm_protos = F.normalize(protos_tensor, p=2, dim=1)

    #     # 计算特征-原型相似度（余弦相似度）
    #     feature_proto_sim = torch.mm(norm_features, norm_protos.t()) / temperature

    #     num_prototypes = len(sorted_labels)

    #     for batch_idx in range(batch_size):
    #         true_label = y[batch_idx].item()

    #         if true_label in sorted_labels:
    #             true_label_idx = sorted_labels.index(true_label)
    #             valid_count += 1

    #             # 获取当前样本与所有原型的相似度
    #             sample_similarities = feature_proto_sim[batch_idx]
                

    #             # 正样本相似度
    #             positive_sim = sample_similarities[true_label_idx]
    #             print(f"当前样本与正样本相似度为{positive_sim},整体相似度为{sample_similarities}")
    #             # 负样本处理
    #             if hard_negative_mining and top_k > 0:
    #                 # print("使用负样本挖掘")
    #                 # 创建负样本掩码（排除正样本）
    #                 negative_mask = torch.ones(num_prototypes, dtype=torch.bool, device=feature_S.device)
    #                 negative_mask[true_label_idx] = False

    #                 # 获取所有负样本相似度
    #                 negative_sims = sample_similarities[negative_mask]

    #                 # 确定要选择的负样本数量
    #                 num_available_negatives = len(negative_sims)
    #                 if num_available_negatives > 0:
    #                     num_negatives = min(top_k, num_available_negatives)

    #                     # 选择最难负样本（相似度最高的负样本）
    #                     hardest_negatives, _ = torch.topk(negative_sims, k=num_negatives, largest=True)

    #                     # 重新构建logits用于InfoNCE损失
    #                     selected_logits = torch.cat([
    #                         positive_sim.unsqueeze(0),
    #                         hardest_negatives
    #                     ])

    #                     # 分母只包括正样本和选中的负样本
    #                     denominator = torch.exp(selected_logits).sum()
    #                 else:
    #                     # 如果没有负样本，只有正样本
    #                     denominator = torch.exp(positive_sim)
    #             else:
    #                 # 使用所有负样本（原始InfoNCE）
    #                 denominator = torch.exp(sample_similarities).sum()

    #             # 计算分子
    #             numerator = torch.exp(positive_sim)

    #             # 添加数值稳定性处理
    #             eps = 1e-8
    #             denominator = torch.clamp(denominator, min=eps)
    #             numerator = torch.clamp(numerator, min=eps)

    #             # 计算损失
    #             instance_loss = -torch.log(numerator / denominator)
    #             contrastive_loss += instance_loss

    #     return contrastive_loss / max(valid_count, 1) if valid_count > 0 else torch.tensor(0.0, device=feature_S.device)

    #从服务器接受全局模型参数
    def set_parameters(self):
        model = load_item(self.role, 'model', self.save_folder_name).to(self.device)
        global_model = load_item('Server', 'model', self.save_folder_name).to(self.device)
        # 从全局模型中分解出低秩模型base给客户端
        global_model.decom_larger_model(model.ratio_LR)
        print(f"客户端{self.role}接收服务器模型参数")
        for new_param, old_param in zip(global_model.base.parameters(), model.base.parameters()):
            old_param.data = new_param.data.clone()
        save_item(model, self.role, 'model', self.save_folder_name)
    
    # TT分解实现
    def set_parameters_TT(self):
        model = load_item(self.role, 'model', self.save_folder_name).to(self.device)
        global_model = load_item('Server', 'model', self.save_folder_name).to(self.device)
        # 从全局模型中分解出低秩模型base给客户端
        global_model.TT_decom_larger_model(model.ratio_LR)
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

        # auc = metrics.roc_auc_score(y_true, y_prob, average='micro')
        auc = 0

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


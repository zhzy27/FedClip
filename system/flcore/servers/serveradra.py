import copy
import random
import time
from flcore.clients.clientadra import clientadra
from flcore.servers.serverbase import Server
from flcore.clients.clientbase import load_item, save_item
from threading import Thread
from flcore.trainmodel.models import  Model_Distribe
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import numpy as np
import torch.optim as optim
from tqdm import tqdm
import torch.nn.functional as F
# from flcore.trainmodel.models import FactorizedConv,FactorizedLinear
class ADRALPFL(Server):
    def __init__(self, args, times):
        super().__init__(args, times)

        # select slow clients
        self.set_slow_clients()
        self.set_clients(clientadra)

        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print("Finished creating server and clients.")

        # self.load_model()
        self.Budget = []
        #创建全局base用于之后聚合
        global_model = Model_Distribe(args, -1,is_global=True).to(self.device)
        global_model.recover_larger_model()
        self.lowrank_rates = [1.0,0.5,0.35,0.25,0.15]
        #创建要分解的低秩模型结构
        for rate in self.lowrank_rates:
            model = copy.deepcopy(global_model)
            model.decom_larger_model(rate)
            role_name = self.role+ str(rate)
            save_item(model, role_name, 'model', self.save_folder_name)


    def train(self):
        for i in range(self.global_rounds+1):
            s_t = time.time()
            # self.send_parameters()
            self.selected_clients = self.select_clients()
            #这个测试的是全局base+个性化head之后可能需要修改
            if i%self.eval_gap == 0:
                print(f"\n-------------Round number: {i}-------------")
                print("\nEvaluate heterogeneous models")
                self.evaluate(epoch=i)
            self.send_parameters()
            for client in self.selected_clients:
                client.train(current_round=i)

            # threads = [Thread(target=client.train)
            #            for client in self.selected_clients]
            # [t.start() for t in threads]
            # [t.join() for t in threads]

            self.receive_ids()
            self.aggregate_parameters()

            self.Budget.append(time.time() - s_t)
            print('-'*25, 'time cost', '-'*25, self.Budget[-1])

            if self.auto_break and self.check_done(acc_lss=[self.rs_test_acc], top_cnt=self.top_cnt):
                break

        print("\nBest accuracy.")
        # self.print_(max(self.rs_test_acc), max(
        #     self.rs_train_acc), min(self.rs_train_loss))
        print(max(self.rs_test_acc))
        print("\nAverage time cost per round.")
        print(sum(self.Budget[1:])/len(self.Budget[1:]))

        self.save_results()
        # self.writer.close()
        self.save_json_file()

    #发送模型参数，被选中的客户端会接受模型参数（参与率变化需要重跑实验）
    def send_parameters(self):
        assert (len(self.clients) > 0)

        for client in  self.selected_clients:
            start_time = time.time()
            #有的客户端会实现
            client.set_parameters()
            client.send_time_cost['num_rounds'] += 1
            client.send_time_cost['total_cost'] += 2 * (time.time() - start_time)
    #从客户顿接受id信息和样本数信息
    def receive_ids(self):
        assert (len(self.selected_clients) > 0)

        active_clients = random.sample(
            self.selected_clients, int((1-self.client_drop_rate) * self.current_num_join_clients))

        self.uploaded_ids = []
        self.uploaded_weights = []
        tot_samples = 0
        for client in active_clients:
            tot_samples += client.train_samples
            self.uploaded_ids.append(client.id)
            self.uploaded_weights.append(client.train_samples)
        for i, w in enumerate(self.uploaded_weights):
            self.uploaded_weights[i] = w / tot_samples
    #客户端base进行参数对齐并且进行对齐后聚合
    def aggregate_parameters(self):
        assert (len(self.uploaded_ids) > 0)
        #载入全局模型,全局模型是完整模型状态
        global_model = load_item(self.role+"1.0", 'model', self.save_folder_name).to(self.device)
        for param in global_model.parameters():
            param.data.zero_()
        #记录客户端恢复形状后的base模型
        self.uploaded_base_model = []

        for cid in  self.uploaded_ids:
            client = self.clients[cid]
            client_model = load_item(client.role, 'model', client.save_folder_name)
            #创建临时模型用于模型参数恢复
            model = copy.deepcopy(client_model)
            model.recover_larger_model()
            model.to(self.device)
            self.uploaded_base_model.append(model.base)
        for w,base_model in zip(self.uploaded_weights,self.uploaded_base_model):
            #将模型参数聚合
            for server_param, client_param in zip(global_model.base.parameters(), base_model.parameters()):
                w = torch.tensor(w).to(self.device)
                server_param.data += client_param.data.clone() * w
        is_distill = 1
        if is_distill:
            self.distill_Knowledge_compensation(global_model,"Cifar100")
        else:
            for rate in self.lowrank_rates:
                model = copy.deepcopy(global_model)
                model.decom_larger_model(rate)
                role_name = self.role + str(rate)   # 修改这里，将rate转换为字符串
                save_item(model, role_name, 'model', self.save_folder_name)


    def get_distill_dataloader(self,dataset_name="Cifar100", batch_size=256, sample_size=10000):
        # 定义数据预处理
        transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

        # 根据数据集名称加载相应的数据集
        if dataset_name == "Cifar100":
            train_dataset = datasets.CIFAR100(
                root='../dataset/Cifar100/rawdata', train=True, download=True, transform=transform
            )

        elif dataset_name == "Cifar10":
            train_dataset = datasets.CIFAR10(
                root='../dataset/Cifar10/rawdata', train=True, download=True, transform=transform
            )

        elif dataset_name == "MNIST":
            transform = transforms.Compose([
                transforms.Resize((32, 32)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.1307], std=[0.3081])
            ])
            train_dataset = datasets.MNIST(
                root='../dataset/Mnist/rawdata', train=True, download=True, transform=transform
            )

        else:
            raise ValueError(f"不支持的数据集: {dataset_name}")

        # 对训练集进行采样，减小数据集规模用于蒸馏
        if sample_size > 0 and sample_size < len(train_dataset):
            # 确保采样数量不超过数据集总数
            sample_size = min(sample_size, len(train_dataset))
            # 随机采样
            indices = np.random.choice(len(train_dataset), sample_size, replace=False)
            train_dataset = Subset(train_dataset, indices)

        # 创建数据加载器
        train_loader = DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            shuffle=True, 
            num_workers=2,
            pin_memory=True if torch.cuda.is_available() else False
        )
        


        return train_loader
        
    # 使用蒸馏补充低秩分解损失的知识
    def distill_Knowledge_compensation(self, global_model, distill_dataset_name="Cifar100"):
        distill_data = self.get_distill_dataloader(distill_dataset_name)

        for rate in self.lowrank_rates:
            if rate < 0.35:
                # 创建学生模型并分解
                model = copy.deepcopy(global_model)
                model.decom_larger_model(rate)

                # 执行特征蒸馏
                self.feature_distill(model, global_model, distill_data, rate=rate)

                print(f"低秩分解比{rate}蒸馏弥补知识完成！")
                role_name = self.role + str(rate)
                save_item(model, role_name, 'model', self.save_folder_name)
            else:
                # 对于完整模型，直接保存
                model = copy.deepcopy(global_model)
                model.decom_larger_model(rate)
                role_name = self.role + str(rate)
                save_item(model, role_name, 'model', self.save_folder_name)

    def feature_distill(self, S_model, T_model, dataloader, epochs=5, lr=0.01, rate=0.5):
        # 设置设备
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        S_model.to(device)
        T_model.to(device)
        T_model.eval()
        S_model.train()

        optimizer = optim.SGD(S_model.parameters(), lr=lr, momentum=0.9)

        # 学习率调度器
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs * len(dataloader))

        # 蒸馏训练循环
        for epoch in range(epochs):
            running_loss = 0.0
            running_feature_loss = 0.0
            running_regular_loss = 0.0

            # 使用tqdm显示进度
            pbar = tqdm(enumerate(dataloader), total=len(dataloader))
            for i, (X, _) in pbar:
                X = X.to(device)
                optimizer.zero_grad()

                with torch.no_grad():
                    T_features = T_model.base(X)

                S_features = S_model.base(X)

                # 计算特征蒸馏损失
                feature_loss = self.feature_distill_loss(T_features, S_features)
                regular_loss = 1e-3 * self.weight_regular(S_model, T_model)
                # 组合损失
                loss = feature_loss + regular_loss
                # 反向传播和参数更新
                loss.backward()
                optimizer.step()
                scheduler.step()

                # 累计损失
                running_loss += loss.item()
                running_feature_loss += feature_loss.item()
                running_regular_loss += regular_loss.item()

                # 更新进度条
                pbar.set_description(f"Epoch [{epoch+1}/{epochs}], Batch [{i+1}/{len(dataloader)}]")
                pbar.set_postfix({
                    'loss': f'{loss.item():.6f}',
                    'feat_loss': f'{feature_loss.item():.6f}',
                    'reg_loss': f'{regular_loss.item():.6f}' 
                })

            # 计算本轮平均损失
            epoch_loss = running_loss / len(dataloader)
            epoch_feature_loss = running_feature_loss / len(dataloader)
            epoch_regular_loss = running_regular_loss / len(dataloader) 

            print(f"Epoch {epoch+1}/{epochs}, Average Loss: {epoch_loss:.6f}, "
                  f"Feature Loss: {epoch_feature_loss:.6f}, Regular Loss: {epoch_regular_loss:.6f}")

        print("特征蒸馏完成!")

    def weight_regular(self, model, global_model):
        """
        计算分解模型与完整模型之间权重的L2范数差异作为正则化项
        使用named_modules确保正确匹配对应层
        """
        device = next(model.parameters()).device
        loss = torch.tensor(0.0, device=device, requires_grad=True)

        # 使用named_modules确保遍历所有层
        model_modules = dict(model.base.named_modules())
        global_modules = dict(global_model.base.named_modules())

        # 遍历学生模型的所有模块
        for name, module in model.base.named_modules():
            # 跳过没有权重的模块
            if not hasattr(module, 'weight') and not hasattr(module, 'reconstruct_full_weight'):
                continue

            # 获取对应的全局模型模块
            if name not in global_modules:
                continue

            global_module = global_modules[name]

            # 获取学生模型的权重（分解层需要重建）
            if hasattr(module, 'reconstruct_full_weight'):
                model_weight = module.reconstruct_full_weight()
            else:
                model_weight = module.weight

            # 获取全局模型的权重
            if hasattr(global_module, 'weight'):
                global_weight = global_module.weight.detach()  # 不计算梯度

                # 确保形状匹配
                if model_weight.shape != global_weight.shape:
                    try:
                        model_weight = model_weight.view(global_weight.shape)
                    except:
                        print(f"警告: 参数形状不匹配且无法调整: {model_weight.shape} vs {global_weight.shape}")
                        continue

                # 计算L2范数差异
                loss = loss + torch.norm(model_weight - global_weight, p=2) ** 2

        return loss

    # def feature_distill_loss(self, feature_T, feature_S, temperature=3, kl_weight=0.5):
    #     """
    #     改进的特征蒸馏损失函数，结合KL散度和MSE损失
    #     """
    #     # 计算KL散度损失
    #     feature_T_soft = F.softmax(feature_T / temperature, dim=1)
    #     feature_S_soft = F.log_softmax(feature_S / temperature, dim=1)

    #     kl_loss = F.kl_div(
    #         feature_S_soft, 
    #         feature_T_soft, 
    #         reduction='batchmean'
    #     ) * (temperature ** 2)

    #     # 计算MSE损失
    #     mse_loss = F.mse_loss(feature_S, feature_T)

    #     # 计算余弦相似度损失（增加方向一致性约束）
    #     cosine_loss = 1 - F.cosine_similarity(
    #         feature_S.view(feature_S.size(0), -1),
    #         feature_T.view(feature_T.size(0), -1),
    #         dim=1
    #     ).mean()

    #     # 组合损失
    #     total_loss = (
    #         kl_weight * kl_loss + 
    #         (0.8 - kl_weight) * mse_loss + 
    #         0.2 * cosine_loss
    #     )

    #     return total_loss
    def feature_distill_loss(self, feature_T, feature_S,weight=0.8):

        # 计算MSE损失
        mse_loss = F.mse_loss(feature_S, feature_T)

        # 计算余弦相似度损失（增加方向一致性约束）
        cosine_loss = 1 - F.cosine_similarity(
            feature_S.view(feature_S.size(0), -1),
            feature_T.view(feature_T.size(0), -1),
            dim=1
        ).mean()

        # 组合损失
        total_loss = (
            weight  * mse_loss + 
            (1-weight)* cosine_loss
        )

        return total_loss
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
from utils.get_clip_text_encoder import get_clip_class_embeddings
from utils.get_clip_text_encoder import get_clip_v_encoder
from utils.data_utils import read_client_data
from torch.utils.data import DataLoader, Dataset
import os


# 新增一个辅助类：将预计算的视觉特征绑定到原生数据集上
class DatasetWithVFeat(Dataset):
    """用于将预计算的视觉特征与原始数据集绑定"""
    def __init__(self, dataset, v_features):
        self.dataset = dataset
        self.v_features = v_features
        
    def __getitem__(self, index):
        x, y = self.dataset[index]
        v_feat = self.v_features[index]
        return x, y, v_feat
        
    def __len__(self):
        return len(self.dataset)

class clientCLIP(Client):
    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        super().__init__(args, id, train_samples, test_samples, **kwargs)
        torch.manual_seed(0)
        self.mse_fn = torch.nn.MSELoss()
        clip_text_features,clip_text_features_norm = get_clip_class_embeddings(self.dataset,model_name= "ViT-B/32",prompt_template= "a photo of {}",device = self.device)
        self.clip_text_features,self.clip_text_features_norm = clip_text_features.float(),clip_text_features_norm.float()

    def load_train_data(self, batch_size=None):
        if batch_size is None:
            batch_size = self.batch_size
            
        # 如果当前实例的内存中还未缓存数据集
        if not hasattr(self, 'cached_train_data'):
            # 1. 调用底层的读取逻辑获取原始数据集 (这部分只加载图片/索引，速度很快)
            raw_train_data = read_client_data(self.dataset, self.id, is_train=True, few_shot=self.few_shot, args=self.args)
            
            # ================= 核心增强：构建本地持久化缓存路径 =================
            dataset_name = getattr(self.args, 'dataset', 'UnknownData')
            partition = getattr(self.args, 'partition', 'dir')
            alpha = getattr(self.args, 'dir_alpha', getattr(self.args, 'alpha', 0.1))
            
            # 在当前目录下创建一个专门存视觉特征的仓库
            cache_base_dir = os.path.join(".", "Visual_Features_Cache", f"{dataset_name}_{partition}_{alpha}")
            os.makedirs(cache_base_dir, exist_ok=True)
            
            # 文件名加入 few_shot 标志防重合
            fs_tag = f"_fs{self.few_shot}" if hasattr(self, 'few_shot') and self.few_shot else ""
            cache_filepath = os.path.join(cache_base_dir, f"client_{self.id}{fs_tag}_vfeat.pt")
            # ====================================================================

            # 2. 判断本地磁盘是否已经有该客户端的特征
            if os.path.exists(cache_filepath):
                print(f"[{self.role} {self.id}] 🎯 命中磁盘缓存！直接读取预计算特征...")
                # 直接从磁盘加载特征
                clip_visual_features = torch.load(cache_filepath)
            else:
                print(f"[{self.role} {self.id}] ⚠️ 磁盘无缓存，正在初始化 ViT 模型预计算视觉特征...")
                
                # 3. 加载视觉 Teacher 模型
                v_encoder = get_clip_v_encoder(model_name="ViT-B/32", device=self.device)
                v_encoder.eval()
                
                # 建立临时非打乱的 Loader 提取特征
                temp_loader = DataLoader(raw_train_data, batch_size=batch_size, shuffle=False)
                all_v_features = []
                
                with torch.no_grad():
                    # 直接精准获取视觉模型第一层卷积的网络精度 (通常是 torch.float16)
                    clip_dtype = v_encoder.conv1.weight.dtype 
                    
                    import torch.nn.functional as F
                    for x, _ in temp_loader:
                        if type(x) == type([]):
                            x_input = x[0].to(self.device)
                        else:
                            x_input = x.to(self.device)

                        # 将输入图片强制插值到 224x224
                        if x_input.shape[-1] != 224 or x_input.shape[-2] != 224:
                            x_input = F.interpolate(x_input, size=(224, 224), mode='bicubic', align_corners=False)

                        # 提取特征
                        feat = v_encoder(x_input.to(clip_dtype))
                        all_v_features.append(feat.cpu())
                        
                clip_visual_features = torch.cat(all_v_features, dim=0).float()
                
                # 4. 释放显存
                del v_encoder
                torch.cuda.empty_cache()
                
                # ================= 将提取的特征持久化到磁盘 =================
                torch.save(clip_visual_features, cache_filepath)
                print(f"[{self.role} {self.id}] 💾 视觉特征预计算完成并保存至: {cache_filepath}")
                # ==========================================================

            # 5. 包装成新的数据集，并挂载到当前客户端实例中
            self.cached_train_data = DatasetWithVFeat(raw_train_data, clip_visual_features)

        # 之后每轮联邦训练都直接用带特征的缓存数据集生成 DataLoader
        return DataLoader(self.cached_train_data, batch_size, drop_last=False, shuffle=True)
    
    def train_metrics(self):
        trainloader = self.load_train_data()
        model = load_item(self.role, 'model', self.save_folder_name).to(self.device)
        # model.to(self.device)
        model.eval()

        train_num = 0
        losses = 0
        with torch.no_grad():
            for x, y, _ in trainloader:
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

    def train(self, current_round=0):
        trainloader = self.load_train_data()
        model = load_item(self.role, 'model', self.save_folder_name)
        model.to(self.device)
        optimizer = torch.optim.SGD(model.parameters(), lr=self.learning_rate)
        model.train()
        start_time = time.time()
        max_local_epochs = self.local_epochs
        if self.train_slow:
            max_local_epochs = np.random.randint(1, max_local_epochs // 2)
        for step in range(max_local_epochs):
            for i, (x, y, target_v_features) in enumerate(trainloader):
                optimizer.zero_grad()
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                target_v_features = target_v_features.to(self.device)
                if self.train_slow:
                    time.sleep(0.1 * np.abs(np.random.rand()))

                features = model.base(x)  # 图像特征 [B, 512]
                # features_norm = F.normalize(features, dim=-1)
                logits = model.head(features)

                # 2. 视觉对比损失
                v_mse_loss = self.mse_fn(features, target_v_features)

                #图像特征和文本特征距离度量损失
                mse_loss = self.mse_fn(features,self.clip_text_features[y])

                #角度度量损失
                # cos_loss = (1 - F.cosine_similarity(features_norm, self.clip_text_features_norm[y], dim=-1)).mean()
                #图像特征和文本特征
                loss = self.loss(logits, y) + self.args.mse_lamda * mse_loss + self.args.v_mse_lamda * v_mse_loss
                if self.args.is_regular==1:
                    loss += self.args.regular_lamda*model.frobenius_decay()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
                optimizer.step()
        save_item(model, self.role, 'model', self.save_folder_name)
        self.train_time_cost['num_rounds'] += 1
        self.train_time_cost['total_cost'] += time.time() - start_time


# 从服务器接受专属全局模型参数
    def set_parameters(self):
        model = load_item(self.role, 'model', self.save_folder_name).to(self.device)
        
        # 尝试加载专属模型 (注意：文件不存在时 load_item 会返回 None)
        global_model = load_item('Server', f'model_{self.id}', self.save_folder_name)
        
        if global_model is not None:
            global_model = global_model.to(self.device)
            print(f"客户端{self.role}成功接收基于余弦相似度的专属聚合参数")
        else:
            # 如果没有专属模型（如第一轮，或该客户端上一轮未参与），拉取最新的通用全局模型
            global_model = load_item('Server', 'model', self.save_folder_name).to(self.device)
            print(f"客户端{self.role}接收最新的通用服务器模型参数")

        # 从全局模型中分解出低秩模型base给客户端
        global_model.decom_larger_model(model.ratio_LR)
        
        for new_param, old_param in zip(global_model.parameters(), model.parameters()):
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

                features = model.base(x)  # 图像特征 [B, 512]
                output = model.head(features)
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

        return test_acc, test_num, 0

    

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
from utils.data_utils import read_client_data

class clientCLIP(Client):
    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        super().__init__(args, id, train_samples, test_samples, **kwargs)
        torch.manual_seed(0)
        self.mse_fn = torch.nn.MSELoss()
        clip_text_features,clip_text_features_norm = get_clip_class_embeddings(self.dataset,model_name= "ViT-B/32",prompt_template= "a photo of {}",device = self.device)
        self.clip_text_features,self.clip_text_features_norm = clip_text_features.float(),clip_text_features_norm.float()

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
            for i, (x, y) in enumerate(trainloader):
                optimizer.zero_grad()
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                if self.train_slow:
                    time.sleep(0.1 * np.abs(np.random.rand()))

                features = model.base(x)  # 图像特征 [B, 512]
                features_norm = F.normalize(features, dim=-1)
                logits = model.head(features)

                #图像特征和文本特征距离度量损失
                mse_loss = self.mse_fn(features,self.clip_text_features[y])

                #角度度量损失
                cos_loss = (1 - F.cosine_similarity(features_norm, self.clip_text_features_norm[y], dim=-1)).mean()
                #图像特征和文本特征
                loss = self.loss(logits, y) + self.args.mse_lamda * mse_loss + self.args.Cos_lamda*cos_loss
                if self.args.is_regular==1:
                    loss += self.args.regular_lamda*model.frobenius_decay()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
                optimizer.step()
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
                features = model.base(x)  # 图像特征 [B, 512]
                output = model.head(features)
                loss = self.loss(output, y)
                train_num += y.shape[0]
                losses += loss.item() * y.shape[0]

        return losses, train_num
    

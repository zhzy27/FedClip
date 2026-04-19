import copy
import torch
import torch.nn as nn
import numpy as np
import time
import torch.nn.functional as F
from flcore.clients.clientbase import Client, load_item, save_item
from sklearn.preprocessing import label_binarize
from sklearn import metrics
from flcore.trainmodel.models import vector_alpha

class clientAFM(Client):
    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        super().__init__(args, id, train_samples, test_samples, **kwargs)
        torch.manual_seed(0)
        #输入同构表征获得对应比例的输出
        ALPHA = vector_alpha()
        save_item(ALPHA, self.role, 'alpha', self.save_folder_name)

    def train(self):
        trainloader = self.load_train_data()
        model = load_item(self.role, 'model', self.save_folder_name)
        alpha_model = load_item(self.role, 'alpha', self.save_folder_name)
        global_model = load_item('Server', 'global_model', self.save_folder_name)
        #异构模型参数优化器
        optimizer = torch.optim.SGD(model.parameters(), lr=self.learning_rate)
        #参数权重优化器
        optimizer_alpha = torch.optim.SGD(alpha_model.parameters(), lr=self.args.alpha_lr)
        #全局同构优化器
        optimizer_g = torch.optim.SGD(global_model.parameters(), lr=self.learning_rate)
        # model.to(self.device)
        model.train()
        global_model.train()
        
        start_time = time.time()

        max_local_epochs = self.local_epochs
        if self.train_slow:
            max_local_epochs = np.random.randint(1, max_local_epochs // 2)
        #优化异构大模型和alpha
        for step in range(max_local_epochs):
            for i, (x, y) in enumerate(trainloader):
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                if self.train_slow:
                    time.sleep(0.1 * np.abs(np.random.rand()))
                _,global_rep = global_model(x)
                alpha = alpha_model.alpha
                w_global_rep = alpha_model(global_rep)
                output,mix_feature = model(x,w_global_rep,alpha)
                loss = self.loss(output, y) 

                optimizer.zero_grad()
                optimizer_alpha.zero_grad()
                optimizer_g.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 50)
                # torch.nn.utils.clip_grad_norm_(alpha_model.parameters(), 1)
                alpha_model.alpha.data = torch.clamp(alpha_model.alpha.data, 0, 1)
                optimizer.step()
                optimizer_alpha.step()
                #让alpha的值落在正常范围内
                alpha_model.alpha.data = torch.clamp(alpha_model.alpha.data, 0, 1)
        for step in range(max_local_epochs):
            #只更新全局小模型的特征提取器
            for i, (x, y) in enumerate(trainloader):
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                if self.train_slow:
                    time.sleep(0.1 * np.abs(np.random.rand()))
                _,global_rep = global_model(x)
                # output,mix_feature = model(x,global_rep, torch.zeros(500))
                output,mix_feature = model(x,global_rep, torch.zeros(512))
                loss = self.loss(output, y) 
                optimizer_g.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(global_model.parameters(), 50)
                optimizer_g.step()


        # model.cpu()
        save_item(model, self.role, 'model', self.save_folder_name)
        #保留本地训练过的小模型参数用于之后聚合
        save_item(global_model, self.role, 'global_model', self.save_folder_name)
        self.train_time_cost['num_rounds'] += 1
        self.train_time_cost['total_cost'] += time.time() - start_time


    def test_metrics(self):
        testloader = self.load_test_data()
        model = load_item(self.role, 'model', self.save_folder_name)
        alpha_model = load_item(self.role, 'alpha', self.save_folder_name)
        global_model = load_item('Server', 'global_model', self.save_folder_name)
        model.to(self.device)
        model.eval()
        global_model.to(self.device)
        global_model.eval()
        alpha_model.to(self.device)
        alpha_model.eval()
        global_model.eval()

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
                _,rep_g = global_model(x)
                alpha = alpha_model.alpha
                w_rep_g = alpha_model(rep_g)
                output,mix_rep = model(x,w_rep_g,alpha)
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
        model = load_item(self.role, 'model', self.save_folder_name)
        alpha_model = load_item(self.role, 'alpha', self.save_folder_name)
        global_model = load_item('Server', 'global_model', self.save_folder_name)
        model.to(self.device)
        model.eval()
        global_model.to(self.device)
        global_model.eval()
        alpha_model.to(self.device)
        alpha_model.eval()
        global_model.eval()

        train_num = 0
        losses = 0
        with torch.no_grad():
            for x, y in trainloader:
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                _,rep_g = global_model(x)
                alpha = alpha_model.alpha
                w_rep_g = alpha_model(rep_g)
                output,mix_rep = model(x,w_rep_g,alpha)
                loss = self.loss(output, y)
                train_num += y.shape[0]
                losses += loss.item() * y.shape[0]

        return losses, train_num
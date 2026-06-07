import torch
import numpy as np
import time
from flcore.clients.clientbase import Client, load_item, save_item
from sklearn.preprocessing import label_binarize
from utils.get_clip_text_encoder import get_clip_class_embeddings, get_clip_class_depth_embeddings


class clientCLIP(Client):
    _clip_text_cache = {}
    _clip_depth_text_cache = {}

    @staticmethod
    def _limit_torch_cpu_threads(max_threads):
        if max_threads is None or max_threads <= 0:
            return
        current_threads = torch.get_num_threads()
        if current_threads > max_threads:
            torch.set_num_threads(max_threads)

    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        super().__init__(args, id, train_samples, test_samples, **kwargs)
        torch.manual_seed(0)
        self._limit_torch_cpu_threads(getattr(args, "clip_cpu_threads", 4))
        self.mse_fn = torch.nn.MSELoss()
        self.use_resnet_multilevel_clip = "resnet" in getattr(args, "model_family", "").lower()
        if self.use_resnet_multilevel_clip:
            cache_key = (self.dataset, "ViT-B/32", "a photo of {}", str(self.device), 4)
            if cache_key not in clientCLIP._clip_depth_text_cache:
                clientCLIP._clip_depth_text_cache[cache_key] = get_clip_class_depth_embeddings(
                    self.dataset,
                    model_name="ViT-B/32",
                    prompt_template="a photo of {}",
                    device=self.device,
                    num_depths=4
                )
            clip_depth_features, clip_depth_features_norm = clientCLIP._clip_depth_text_cache[cache_key]
            self.clip_text_depth_features = clip_depth_features.float()
            self.clip_text_depth_features_norm = clip_depth_features_norm.float()
            self.clip_text_features = self.clip_text_depth_features[-1]
            self.clip_text_features_norm = self.clip_text_depth_features_norm[-1]
            self.resnet_clip_aligners = None
            self._resnet_stage_end_cache = {}
        else:
            cache_key = (self.dataset, "ViT-B/32", "a photo of {}", str(self.device))
            if cache_key not in clientCLIP._clip_text_cache:
                clientCLIP._clip_text_cache[cache_key] = get_clip_class_embeddings(self.dataset,model_name= "ViT-B/32",prompt_template= "a photo of {}",device = self.device)
            clip_text_features,clip_text_features_norm = clientCLIP._clip_text_cache[cache_key]
            self.clip_text_features,self.clip_text_features_norm = clip_text_features.float(),clip_text_features_norm.float()

    def _ensure_resnet_clip_aligners(self, stage_features):
        target_dim = self.clip_text_depth_features.shape[-1]
        stage_dims = [stage_feature.shape[-1] for stage_feature in stage_features]
        need_rebuild = self.resnet_clip_aligners is None
        if not need_rebuild:
            need_rebuild = len(self.resnet_clip_aligners) != len(stage_dims)
        if not need_rebuild:
            need_rebuild = any(
                aligner.in_features != stage_dim or aligner.out_features != target_dim
                for aligner, stage_dim in zip(self.resnet_clip_aligners, stage_dims)
            )

        if need_rebuild:
            self.resnet_clip_aligners = torch.nn.ModuleList([
                torch.nn.Linear(stage_dim, target_dim)
                for stage_dim in stage_dims
            ]).to(self.device)
        else:
            self.resnet_clip_aligners = self.resnet_clip_aligners.to(self.device)
        return self.resnet_clip_aligners

    def _forward_resnet_multilevel_features(self, model, x):
        base = model.base
        input_x = x
        x = base.conv1(x)
        x = base.bn1(x)
        x = base.relu(x)
        if hasattr(base, "maxpool"):
            x = base.maxpool(x)

        if hasattr(base, "stages"):
            stage_features = []
            for stage in base.stages:
                x = stage(x)
                stage_features.append(base.avgpool(x))
            while len(stage_features) < 4:
                stage_features.append(stage_features[-1])
            final_features = stage_features[-1]
            if hasattr(base, "projection"):
                final_features = base.projection(final_features)
                stage_features[-1] = final_features
            return final_features, stage_features[:4]

        if not hasattr(base, "layers"):
            final_features = model.base(input_x)
            return final_features, [final_features] * 4

        num_layers = len(base.layers)
        if num_layers not in self._resnet_stage_end_cache:
            self._resnet_stage_end_cache[num_layers] = {
                max(0, ((num_layers * (stage_idx + 1) + 3) // 4) - 1)
                for stage_idx in range(4)
            }
        stage_end_indices = self._resnet_stage_end_cache[num_layers]

        stage_features = []
        for layer_idx in range(num_layers):
            layer = getattr(base, f'layer_{layer_idx}')
            x = layer(x)
            if layer_idx in stage_end_indices:
                stage_features.append(base.avgpool(x))

        while len(stage_features) < 4:
            stage_features.append(stage_features[-1])

        final_features = stage_features[-1]
        if hasattr(base, "projection"):
            final_features = base.projection(final_features)
            stage_features[-1] = final_features

        return final_features, stage_features[:4]

    def _resnet_multilevel_clip_loss(self, stage_features, y):
        aligners = self._ensure_resnet_clip_aligners(stage_features)
        losses = []
        for stage_idx, (stage_feature, aligner) in enumerate(zip(stage_features, aligners)):
            anchor = self.clip_text_depth_features[stage_idx][y].to(stage_feature.device)
            aligned_feature = aligner(stage_feature)
            losses.append(self.mse_fn(aligned_feature, anchor))
        return sum(losses) / len(losses)
    
    def train_metrics(self):
        trainloader = self.load_train_data()
        model = load_item(self.role, 'model', self.save_folder_name).to(self.device)
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
                output = model(x)
                loss = self.loss(output, y)
                train_num += y.shape[0]
                losses += loss.item() * y.shape[0]

        return losses, train_num

    def train(self, current_round=0):
        trainloader = self.load_train_data()
        model = load_item(self.role, 'model', self.save_folder_name)
        model.to(self.device)
        # ================= 增加模型大小打印 =================
        total_params = sum(p.numel() for p in model.parameters())
        # 为了方便阅读，将其转换为 百万 (Million, M) 级别
        print(f"[{self.role}] 当前模型参数量为: {total_params} ({total_params / 1e6:.3f} M)")
        
        # ================= 新增：非对称学习率 (Asymmetric LR) 分组 =================
        u_params = []
        v_params = []
        other_params = []

        # 遍历模型的所有参数，根据命名后缀进行分发
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            
            # 筛选 U 矩阵参数 ( FactorizedLinear 和 FactorizedConv )
            if name.endswith('weight_u') or name.endswith('conv_u'):
                u_params.append(param)
            # 筛选 V 矩阵参数
            elif name.endswith('weight_v') or name.endswith('conv_v'):
                v_params.append(param)
            # 基础网络、分类头、偏置(bias)等其他参数
            else:
                other_params.append(param)

        # 设置 U 的学习率衰减系数，默认 U 的学习率是 V 的 0.1 倍 (即 0.0005)
        # 建议在 main.py 的 argparse 中加入这个参数以便调参，比如 args.u_lr_ratio
        u_lr_ratio = getattr(self.args, 'u_lr_ratio', 0.1) 

        # 构建优化器参数组
        optimizer = torch.optim.SGD([
            {'params': v_params, 'lr': self.learning_rate},               # V 使用正常学习率 (0.005)
            {'params': u_params, 'lr': self.learning_rate * u_lr_ratio},  # U 使用极低学习率 (0.005 * 0.1)
            {'params': other_params, 'lr': self.learning_rate}            # 其他参数使用正常学习率
        ])
        aligner_params_added = False
        clip_params = list(model.parameters())
        if self.use_resnet_multilevel_clip and self.resnet_clip_aligners is not None:
            optimizer.add_param_group({'params': self.resnet_clip_aligners.parameters(), 'lr': self.learning_rate})
            clip_params.extend(list(self.resnet_clip_aligners.parameters()))
            aligner_params_added = True
        # =========================================================================
        
        model.train()
        if self.use_resnet_multilevel_clip and self.resnet_clip_aligners is not None:
            self.resnet_clip_aligners.train()
        if torch.cuda.is_available() and str(self.device).startswith("cuda"):
            torch.cuda.synchronize(self.device)
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

                if self.use_resnet_multilevel_clip:
                    features, stage_features = self._forward_resnet_multilevel_features(model, x)
                    logits = model.head(features)
                    mse_loss = self._resnet_multilevel_clip_loss(stage_features, y)
                    if self.resnet_clip_aligners is not None and not aligner_params_added:
                        optimizer.add_param_group({'params': self.resnet_clip_aligners.parameters(), 'lr': self.learning_rate})
                        clip_params.extend(list(self.resnet_clip_aligners.parameters()))
                        aligner_params_added = True
                else:
                    features = model.base(x)  # 图像特征 [B, 512]
                    # features_norm = F.normalize(features, dim=-1)
                    logits = model.head(features)

                    #图像特征和文本特征距离度量损失
                    mse_loss = self.mse_fn(features,self.clip_text_features[y])

                #角度度量损失
                # cos_loss = (1 - F.cosine_similarity(features_norm, self.clip_text_features_norm[y], dim=-1)).mean()
                #图像特征和文本特征
                loss = self.loss(logits, y) + self.args.mse_lamda * mse_loss
                if self.args.is_regular==1:
                    loss += self.args.regular_lamda*model.frobenius_decay()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(clip_params, 10.0)
                optimizer.step()
        if torch.cuda.is_available() and str(self.device).startswith("cuda"):
            torch.cuda.synchronize(self.device)
        local_train_time = time.time() - start_time
        save_item(model, self.role, 'model', self.save_folder_name)
        self.train_time_cost['num_rounds'] += 1
        self.train_time_cost['total_cost'] += local_train_time
        self.last_train_time_cost = local_train_time
        print(
            f"⏱️ [Round {current_round:03d}] {self.role} 本地训练耗时: "
            f"{local_train_time:.3f}s | local_epochs={max_local_epochs} | train_samples={self.train_samples}"
        )
        return local_train_time


# 从服务器接受专属全局模型参数
    def set_parameters(self):
        model = load_item(self.role, 'model', self.save_folder_name)   # 本地的低秩模型，参数还是未聚合的
        model = model.to(self.device)
        
        # 尝试加载聚合后的模型
        global_model = load_item('Server', f'model_{self.id}', self.save_folder_name)
        
        if global_model is not None:
            global_model = global_model.to(self.device)
            print(f"客户端{self.role}成功接收基于余弦相似度的专属聚合参数")
        else:
            # 如果没有专属模型（如第一轮，或该客户端上一轮未参与），拉取最新的通用全局模型
            global_model = load_item('Server', 'model', self.save_folder_name).to(self.device)
            print(f"客户端{self.role}接收最新的通用服务器模型参数")

        # 从全局模型中分解出低秩模型base给客户端，并将其参数存起来在训练中使用
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

    

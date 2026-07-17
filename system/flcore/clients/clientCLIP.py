import copy
import random
from contextlib import contextmanager

import torch
import numpy as np
import time
import os
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

    def _disable_rank_dropout(self, model):
        for module in model.modules():
            if hasattr(module, "rank_dropout_enabled"):
                module.rank_dropout_enabled = False
            if hasattr(module, "rank_dropout_schedule"):
                module.rank_dropout_schedule = None

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
            shared_init_seed = getattr(
                self,
                "_local_view_shared_aligner_init_seed",
                None,
            )
            if shared_init_seed is None:
                self.resnet_clip_aligners = torch.nn.ModuleList([
                    torch.nn.Linear(stage_dim, target_dim)
                    for stage_dim in stage_dims
                ]).to(self.device)
            else:
                cuda_devices = []
                if torch.cuda.is_available() and str(self.device).startswith("cuda"):
                    device_index = torch.device(self.device).index
                    cuda_devices = [
                        torch.cuda.current_device()
                        if device_index is None
                        else device_index
                    ]
                with torch.random.fork_rng(devices=cuda_devices):
                    torch.manual_seed(shared_init_seed)
                    if cuda_devices:
                        torch.cuda.manual_seed(shared_init_seed)
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

    @staticmethod
    def _local_update_view_seed(current_round, client_id, view_index):
        modulus = (1 << 31) - 1
        seed = (
            104729
            + (int(current_round) + 1) * 1000003
            + (int(client_id) + 1) * 10007
            + int(view_index) * 1009
        ) % modulus
        return seed if seed > 0 else 1

    @contextmanager
    def _isolated_local_update_seed(self, seed):
        python_state = random.getstate()
        numpy_state = np.random.get_state()
        torch_state = torch.get_rng_state()
        use_cuda = torch.cuda.is_available()
        cuda_states = torch.cuda.get_rng_state_all() if use_cuda else None
        random.seed(seed)
        np.random.seed(seed % (1 << 32))
        torch.manual_seed(seed)
        if use_cuda:
            torch.cuda.manual_seed_all(seed)
        try:
            yield
        finally:
            random.setstate(python_state)
            np.random.set_state(numpy_state)
            torch.set_rng_state(torch_state)
            if cuda_states is not None:
                torch.cuda.set_rng_state_all(cuda_states)

    def _train_model_view(
        self,
        model,
        trainloader,
        current_round,
        max_local_epochs=None,
    ):
        model.to(self.device)
        if self.use_resnet_multilevel_clip:
            if hasattr(model, "set_rank_dropout_context"):
                model.set_rank_dropout_context(current_round, self.args.global_rounds)
        else:
            self._disable_rank_dropout(model)
        # ================= 增加模型大小打印 =================
        total_params = sum(p.numel() for p in model.parameters())
        # 为了方便阅读，将其转换为 百万 (Million, M) 级别
        print(f"[{self.role}] 当前模型参数量为: {total_params} ({total_params / 1e6:.3f} M)")
        
        if self.use_resnet_multilevel_clip:
            u_params = []
            v_params = []
            other_params = []
            for name, param in model.named_parameters():
                if not param.requires_grad:
                    continue
                if name.endswith('weight_u') or name.endswith('conv_u'):
                    u_params.append(param)
                elif name.endswith('weight_v') or name.endswith('conv_v'):
                    v_params.append(param)
                else:
                    other_params.append(param)
            u_lr_ratio = getattr(self.args, 'u_lr_ratio', 0.1)
            optimizer = torch.optim.SGD([
                {'params': v_params, 'lr': self.learning_rate},
                {'params': u_params, 'lr': self.learning_rate * u_lr_ratio},
                {'params': other_params, 'lr': self.learning_rate},
            ])
        else:
            optimizer = torch.optim.SGD(model.parameters(), lr=self.learning_rate)
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
        if max_local_epochs is None:
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
        return model, local_train_time, max_local_epochs

    def _record_local_train_time(
        self,
        current_round,
        local_train_time,
        max_local_epochs,
        view_description=None,
    ):
        self.train_time_cost['num_rounds'] += 1
        self.train_time_cost['total_cost'] += local_train_time
        self.last_train_time_cost = local_train_time
        view_suffix = "" if view_description is None else f" | {view_description}"
        print(
            f"⏱️ [Round {current_round:03d}] {self.role} 本地训练耗时: "
            f"{local_train_time:.3f}s | local_epochs={max_local_epochs} | "
            f"train_samples={self.train_samples}{view_suffix}"
        )

    def train(self, current_round=0):
        local_update_views = int(getattr(self.args, "local_update_views", 1))
        if local_update_views not in (1, 2):
            raise ValueError("local_update_views must be 1 or 2.")

        if local_update_views == 1:
            trainloader = self.load_train_data()
            model = load_item(self.role, 'model', self.save_folder_name)
            model, local_train_time, max_local_epochs = self._train_model_view(
                model,
                trainloader,
                current_round,
            )
            save_item(model, self.role, 'model', self.save_folder_name)
            self.local_update_view_b_round = None
            self.last_local_update_view_seeds = None
            self._record_local_train_time(
                current_round,
                local_train_time,
                max_local_epochs,
            )
            return local_train_time

        start_model = load_item(self.role, 'model', self.save_folder_name)
        if start_model is None:
            raise RuntimeError(
                f"{self.role} cannot create two local views without a start model."
            )
        start_model = start_model.to("cpu")
        seed_a = self._local_update_view_seed(current_round, self.id, 0)
        seed_b = self._local_update_view_seed(current_round, self.id, 1)
        if seed_a == seed_b:
            raise AssertionError("Local-update view seeds must be different.")
        self.last_local_update_view_seeds = (seed_a, seed_b)

        if self.train_slow:
            with self._isolated_local_update_seed(seed_a):
                max_local_epochs = np.random.randint(1, self.local_epochs // 2)
        else:
            max_local_epochs = self.local_epochs

        aligner_start = copy.deepcopy(
            getattr(self, "resnet_clip_aligners", None)
        )
        self._local_view_shared_aligner_init_seed = (
            self._local_update_view_seed(current_round, self.id, 2)
        )
        aligner_after_a = None
        try:
            self.resnet_clip_aligners = copy.deepcopy(aligner_start)
            with self._isolated_local_update_seed(seed_a):
                generator_a = torch.Generator()
                generator_a.manual_seed(seed_a)
                trainloader_a = self.load_train_data(generator=generator_a)
                model_a, time_a, epochs_a = self._train_model_view(
                    copy.deepcopy(start_model),
                    trainloader_a,
                    current_round,
                    max_local_epochs=max_local_epochs,
                )
                model_a = model_a.to("cpu")
            aligner_after_a = self.resnet_clip_aligners

            self.resnet_clip_aligners = copy.deepcopy(aligner_start)
            with self._isolated_local_update_seed(seed_b):
                generator_b = torch.Generator()
                generator_b.manual_seed(seed_b)
                trainloader_b = self.load_train_data(generator=generator_b)
                model_b, time_b, epochs_b = self._train_model_view(
                    copy.deepcopy(start_model),
                    trainloader_b,
                    current_round,
                    max_local_epochs=max_local_epochs,
                )
                model_b = model_b.to("cpu")
        finally:
            if aligner_after_a is not None:
                self.resnet_clip_aligners = aligner_after_a
            if hasattr(self, "_local_view_shared_aligner_init_seed"):
                del self._local_view_shared_aligner_init_seed

        if epochs_a != epochs_b:
            raise AssertionError("Local-update views must use the same local epochs.")
        save_item(model_a, self.role, 'model', self.save_folder_name)
        save_item(model_b, self.role, 'model_view_b', self.save_folder_name)
        self.local_update_view_b_round = int(current_round)

        total_train_time = time_a + time_b
        self._record_local_train_time(
            current_round,
            total_train_time,
            max_local_epochs,
            view_description=(
                f"views=2 A={time_a:.3f}s B={time_b:.3f}s "
                f"seeds={seed_a}/{seed_b}"
            ),
        )
        return total_train_time


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

        # 额外缓存“本轮下发后的低秩起点模型”，下一轮服务器聚合可直接用它算低秩 delta。
        low_rank_start_folder = os.path.join(self.save_folder_name, 'low_rank_start')
        save_item(model, 'Server', f'model_{self.id}', low_rank_start_folder)

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

    

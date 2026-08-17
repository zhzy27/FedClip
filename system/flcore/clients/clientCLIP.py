import time

import numpy as np
import torch

from flcore.clients.clientbase import Client, load_item, save_item
from utils.get_clip_text_encoder import (
    get_clip_class_depth_embeddings,
    get_clip_class_embeddings,
)
from utils.ce_anchor_diagnostics import (
    collect_ce_anchor_gradient_diagnostics,
)


class clientCLIP(Client):
    """FedCLIP client with matched learning rates and CLIP alignment."""

    _clip_text_cache = {}
    _clip_depth_text_cache = {}

    @staticmethod
    def _limit_torch_cpu_threads(max_threads):
        if max_threads is not None and max_threads > 0:
            torch.set_num_threads(min(torch.get_num_threads(), max_threads))

    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        super().__init__(args, id, train_samples, test_samples, **kwargs)
        torch.manual_seed(0)
        self._limit_torch_cpu_threads(getattr(args, "clip_cpu_threads", 4))
        self.mse_fn = torch.nn.MSELoss()
        self.use_resnet_multilevel_clip = (
            "resnet" in getattr(args, "model_family", "").lower()
        )
        self.last_ce_anchor_diag = None

        if self.use_resnet_multilevel_clip:
            cache_key = (
                self.dataset,
                "ViT-B/32",
                "a photo of {}",
                str(self.device),
                4,
            )
            if cache_key not in self._clip_depth_text_cache:
                self._clip_depth_text_cache[cache_key] = (
                    get_clip_class_depth_embeddings(
                        self.dataset,
                        model_name="ViT-B/32",
                        prompt_template="a photo of {}",
                        device=self.device,
                        num_depths=4,
                    )
                )
            depth_features, _ = self._clip_depth_text_cache[cache_key]
            self.clip_text_depth_features = depth_features.float()
            self.resnet_clip_aligners = None
            self._resnet_stage_end_cache = {}
        else:
            cache_key = (
                self.dataset,
                "ViT-B/32",
                "a photo of {}",
                str(self.device),
            )
            if cache_key not in self._clip_text_cache:
                self._clip_text_cache[cache_key] = get_clip_class_embeddings(
                    self.dataset,
                    model_name="ViT-B/32",
                    prompt_template="a photo of {}",
                    device=self.device,
                )
            features, _ = self._clip_text_cache[cache_key]
            self.clip_text_features = features.float()

    def _ensure_resnet_clip_aligners(self, stage_features):
        target_dim = self.clip_text_depth_features.shape[-1]
        stage_dims = [feature.shape[-1] for feature in stage_features]
        needs_rebuild = self.resnet_clip_aligners is None
        if not needs_rebuild:
            needs_rebuild = len(self.resnet_clip_aligners) != len(stage_dims)
        if not needs_rebuild:
            needs_rebuild = any(
                aligner.in_features != stage_dim
                or aligner.out_features != target_dim
                for aligner, stage_dim in zip(
                    self.resnet_clip_aligners, stage_dims
                )
            )

        if needs_rebuild:
            self.resnet_clip_aligners = torch.nn.ModuleList(
                [
                    torch.nn.Linear(stage_dim, target_dim)
                    for stage_dim in stage_dims
                ]
            ).to(self.device)
        else:
            self.resnet_clip_aligners = self.resnet_clip_aligners.to(
                self.device
            )
        return self.resnet_clip_aligners

    def _forward_resnet_multilevel_features(self, model, inputs):
        base = model.base
        original_inputs = inputs
        features = base.relu(base.bn1(base.conv1(inputs)))
        if hasattr(base, "maxpool"):
            features = base.maxpool(features)

        if hasattr(base, "stages"):
            stage_features = []
            for stage in base.stages:
                features = stage(features)
                stage_features.append(base.avgpool(features))
        elif hasattr(base, "layers"):
            num_layers = len(base.layers)
            if num_layers not in self._resnet_stage_end_cache:
                self._resnet_stage_end_cache[num_layers] = {
                    max(0, ((num_layers * (idx + 1) + 3) // 4) - 1)
                    for idx in range(4)
                }
            stage_end_indices = self._resnet_stage_end_cache[num_layers]
            stage_features = []
            for layer_idx in range(num_layers):
                features = getattr(base, f"layer_{layer_idx}")(features)
                if layer_idx in stage_end_indices:
                    stage_features.append(base.avgpool(features))
        else:
            final_features = model.base(original_inputs)
            return final_features, [final_features] * 4

        while len(stage_features) < 4:
            stage_features.append(stage_features[-1])
        final_features = stage_features[-1]
        if hasattr(base, "projection"):
            final_features = base.projection(final_features)
            stage_features[-1] = final_features
        return final_features, stage_features[:4]

    def _resnet_multilevel_clip_loss(self, stage_features, labels):
        aligners = self._ensure_resnet_clip_aligners(stage_features)
        losses = []
        for stage_idx, (stage_feature, aligner) in enumerate(
            zip(stage_features, aligners)
        ):
            anchor = self.clip_text_depth_features[stage_idx][labels].to(
                stage_feature.device
            )
            losses.append(self.mse_fn(aligner(stage_feature), anchor))
        return sum(losses) / len(losses)

    def train(self, current_round=0):
        trainloader = self.load_train_data()
        model = load_item(self.role, "model", self.save_folder_name)
        if model is None:
            raise RuntimeError(f"{self.role} model is missing before training.")
        model = model.to(self.device)

        optimizer = torch.optim.SGD(
            (param for param in model.parameters() if param.requires_grad),
            lr=self.learning_rate,
        )
        clip_params = list(model.parameters())
        aligners_added = False
        if self.use_resnet_multilevel_clip and self.resnet_clip_aligners is not None:
            optimizer.add_param_group(
                {
                    "params": self.resnet_clip_aligners.parameters(),
                    "lr": self.learning_rate,
                }
            )
            clip_params.extend(self.resnet_clip_aligners.parameters())
            aligners_added = True

        model.train()
        if (
            self.use_resnet_multilevel_clip
            and self.resnet_clip_aligners is not None
        ):
            self.resnet_clip_aligners.train()
        if torch.cuda.is_available() and str(self.device).startswith("cuda"):
            torch.cuda.synchronize(self.device)
        start_time = time.time()

        max_local_epochs = self.local_epochs
        if self.train_slow:
            max_local_epochs = np.random.randint(
                1, max(2, max_local_epochs // 2 + 1)
            )

        diagnostic_enabled = bool(getattr(self.args, "ce_anchor_diag", 0))
        self.last_ce_anchor_diag = None
        diagnostic_gradients = None
        diagnostic_batch_seen = False
        ce_loss_sum = None
        anchor_loss_sum = None
        loss_batch_count = 0
        shared_named_parameters = None
        if diagnostic_enabled:
            shared_named_parameters = [
                (name, parameter)
                for name, parameter in model.base.named_parameters()
                if parameter.requires_grad
            ]

        for _ in range(max_local_epochs):
            for inputs, labels in trainloader:
                optimizer.zero_grad()
                if isinstance(inputs, list):
                    inputs[0] = inputs[0].to(self.device)
                else:
                    inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                if self.train_slow:
                    time.sleep(0.1 * np.abs(np.random.rand()))

                if self.use_resnet_multilevel_clip:
                    features, stage_features = (
                        self._forward_resnet_multilevel_features(model, inputs)
                    )
                    logits = model.head(features)
                    alignment_loss = self._resnet_multilevel_clip_loss(
                        stage_features, labels
                    )
                    if not aligners_added:
                        optimizer.add_param_group(
                            {
                                "params": self.resnet_clip_aligners.parameters(),
                                "lr": self.learning_rate,
                            }
                        )
                        clip_params.extend(self.resnet_clip_aligners.parameters())
                        aligners_added = True
                else:
                    features = model.base(inputs)
                    logits = model.head(features)
                    alignment_loss = self.mse_fn(
                        features, self.clip_text_features[labels]
                    )

                ce_loss = self.loss(logits, labels)
                if diagnostic_enabled:
                    detached_ce = ce_loss.detach()
                    detached_anchor = alignment_loss.detach()
                    ce_loss_sum = (
                        detached_ce
                        if ce_loss_sum is None
                        else ce_loss_sum + detached_ce
                    )
                    anchor_loss_sum = (
                        detached_anchor
                        if anchor_loss_sum is None
                        else anchor_loss_sum + detached_anchor
                    )
                    loss_batch_count += 1
                    if not diagnostic_batch_seen:
                        diagnostic_gradients = (
                            collect_ce_anchor_gradient_diagnostics(
                                True,
                                ce_loss=ce_loss,
                                anchor_loss=alignment_loss,
                                named_shared_parameters=shared_named_parameters,
                                mse_lambda=self.args.mse_lamda,
                            )
                        )
                        diagnostic_batch_seen = True

                loss = ce_loss
                loss += self.args.mse_lamda * alignment_loss
                if self.args.is_regular == 1:
                    loss += (
                        self.args.regular_lamda * model.frobenius_decay()
                    )
                loss.backward()
                torch.nn.utils.clip_grad_norm_(clip_params, 10.0)
                optimizer.step()

        if torch.cuda.is_available() and str(self.device).startswith("cuda"):
            torch.cuda.synchronize(self.device)
        local_train_time = time.time() - start_time
        if diagnostic_gradients is not None and loss_batch_count > 0:
            mean_ce_loss = float((ce_loss_sum / loss_batch_count).item())
            mean_anchor_loss = float(
                (anchor_loss_sum / loss_batch_count).item()
            )
            mse_lambda = float(self.args.mse_lamda)
            weighted_anchor_loss = mse_lambda * mean_anchor_loss
            try:
                capacity_ratio = float(model.ratio_LR)
            except (AttributeError, TypeError, ValueError):
                capacity_ratio = float("nan")
            self.last_ce_anchor_diag = {
                "round": int(current_round),
                "client_id": int(self.id),
                "capacity_ratio": capacity_ratio,
                "mse_lambda": mse_lambda,
                "mean_ce_loss": mean_ce_loss,
                "mean_anchor_loss": mean_anchor_loss,
                "weighted_anchor_loss": weighted_anchor_loss,
                "anchor_to_ce_loss_ratio": weighted_anchor_loss
                / (mean_ce_loss + 1e-12),
                **diagnostic_gradients,
            }
        save_item(model, self.role, "model", self.save_folder_name)
        self.train_time_cost["num_rounds"] += 1
        self.train_time_cost["total_cost"] += local_train_time
        self.last_train_time_cost = local_train_time
        print(
            f"[Round {current_round:03d}] {self.role} local training: "
            f"{local_train_time:.3f}s"
        )
        return local_train_time

    def set_parameters(self):
        model = load_item(self.role, "model", self.save_folder_name)
        if model is None:
            raise RuntimeError(f"{self.role} low-rank model is missing.")
        model = model.to(self.device)

        global_model = load_item("Server", "model", self.save_folder_name)
        if global_model is None:
            raise RuntimeError("Server Avg model is missing.")
        global_model = global_model.to(self.device)
        global_model.decom_larger_model(model.ratio_LR)

        for source_param, target_param in zip(
            global_model.parameters(), model.parameters()
        ):
            target_param.data.copy_(source_param.data)
        save_item(model, self.role, "model", self.save_folder_name)

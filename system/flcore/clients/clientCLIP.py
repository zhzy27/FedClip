import time

import numpy as np
import torch

from flcore.clients.clientbase import Client, load_item, save_item
from utils.get_clip_text_encoder import (
    get_clip_class_depth_embeddings,
    get_clip_class_embeddings,
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
        self.last_u_subspace_stats = None
        self.last_u_subspace_layer_stats = []
        self.use_resnet_multilevel_clip = (
            "resnet" in getattr(args, "model_family", "").lower()
        )

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

    @staticmethod
    def _is_u_parameter(name):
        return name.endswith("weight_u") or name.endswith("conv_u")

    @staticmethod
    def _is_v_parameter(name):
        return name.endswith("weight_v") or name.endswith("conv_v")

    @staticmethod
    def _paired_v_name(u_name):
        if u_name.endswith("weight_u"):
            return f"{u_name[:-len('weight_u')]}weight_v"
        if u_name.endswith("conv_u"):
            return f"{u_name[:-len('conv_u')]}conv_v"
        raise ValueError(f"Cannot find the V factor paired with {u_name}.")

    @staticmethod
    def _validate_u_parameter(name, parameter):
        if parameter.ndim != 2:
            raise RuntimeError(
                f"U parameter {name} must be two-dimensional, got "
                f"shape={tuple(parameter.shape)}."
            )
        if parameter.shape[0] < parameter.shape[1]:
            raise RuntimeError(
                f"U parameter {name} has invalid low-rank shape "
                f"{tuple(parameter.shape)}."
            )
        if not torch.isfinite(parameter).all():
            norm = torch.linalg.vector_norm(parameter.detach().float()).item()
            raise RuntimeError(
                f"U parameter {name} contains NaN/Inf: "
                f"shape={tuple(parameter.shape)}, rank={parameter.shape[1]}, "
                f"norm={norm}."
            )

    def _build_optimizer(self, model):
        if not bool(self.args.use_asymmetric_lr):
            return torch.optim.SGD(
                (param for param in model.parameters() if param.requires_grad),
                lr=self.learning_rate,
            )

        u_ratio = float(self.args.u_lr_ratio)
        v_ratio = float(self.args.v_lr_ratio)
        if u_ratio < 0.0 or v_ratio < 0.0:
            raise ValueError(
                "u_lr_ratio and v_lr_ratio must be non-negative, got "
                f"{u_ratio} and {v_ratio}."
            )

        u_params = []
        v_params = []
        other_params = []
        for name, parameter in model.named_parameters():
            if not parameter.requires_grad:
                continue
            if self._is_u_parameter(name):
                u_params.append(parameter)
            elif self._is_v_parameter(name):
                v_params.append(parameter)
            else:
                other_params.append(parameter)

        param_groups = []
        if other_params:
            param_groups.append(
                {"params": other_params, "lr": self.learning_rate}
            )
        if u_params:
            param_groups.append(
                {
                    "params": u_params,
                    "lr": self.learning_rate * u_ratio,
                }
            )
        if v_params:
            param_groups.append(
                {
                    "params": v_params,
                    "lr": self.learning_rate * v_ratio,
                }
            )
        return torch.optim.SGD(param_groups, lr=self.learning_rate)

    def _capture_u_start_subspaces(self, model):
        subspaces = {}
        with torch.no_grad():
            for name, parameter in model.named_parameters():
                if not self._is_u_parameter(name):
                    continue
                self._validate_u_parameter(name, parameter)
                q_start, _ = torch.linalg.qr(
                    parameter.detach().float(), mode="reduced"
                )
                subspaces[name] = q_start.detach()
        if not subspaces:
            raise RuntimeError(
                "U-subspace tracking was requested, but the model contains "
                "no weight_u or conv_u parameters."
            )
        return subspaces

    def _capture_factor_starts(self, model, start_subspaces):
        named_parameters = dict(model.named_parameters())
        factor_starts = {}
        with torch.no_grad():
            for u_name in start_subspaces:
                v_name = self._paired_v_name(u_name)
                if v_name not in named_parameters:
                    raise RuntimeError(
                        f"V parameter {v_name}, paired with {u_name}, is missing."
                    )
                u_parameter = named_parameters[u_name]
                v_parameter = named_parameters[v_name]
                self._validate_u_parameter(u_name, u_parameter)
                if v_parameter.ndim != 2 or not torch.isfinite(v_parameter).all():
                    raise RuntimeError(
                        f"V parameter {v_name} is invalid: "
                        f"shape={tuple(v_parameter.shape)}, "
                        f"norm={torch.linalg.vector_norm(v_parameter.detach().float()).item()}."
                    )
                factor_starts[u_name] = {
                    "u": u_parameter.detach().float().clone(),
                    "v_name": v_name,
                    "v": v_parameter.detach().float().clone(),
                }
        return factor_starts

    def _u_subspace_loss(self, model, start_subspaces):
        named_parameters = dict(model.named_parameters())
        layer_losses = []
        for name, q_start in start_subspaces.items():
            if name not in named_parameters:
                raise RuntimeError(f"U parameter {name} disappeared during training.")
            parameter = named_parameters[name]
            self._validate_u_parameter(name, parameter)
            q_current, _ = torch.linalg.qr(parameter.float(), mode="reduced")
            rank = q_current.shape[1]
            overlap_sq = torch.sum((q_start.T @ q_current) ** 2)
            layer_loss = 1.0 - overlap_sq / rank
            layer_losses.append(torch.clamp(layer_loss, min=0.0, max=1.0))
        if not layer_losses:
            raise RuntimeError("No U subspace loss could be computed.")
        return torch.stack(layer_losses).mean()

    def _u_subspace_drift_norm(self, model, start_subspaces):
        named_parameters = dict(model.named_parameters())
        drift_sq = 0.0
        rank_scale = 0
        with torch.no_grad():
            for name, q_start in start_subspaces.items():
                parameter = named_parameters[name]
                self._validate_u_parameter(name, parameter)
                q_end, _ = torch.linalg.qr(
                    parameter.detach().float(), mode="reduced"
                )
                rank = q_end.shape[1]
                overlap_sq = torch.sum((q_start.T @ q_end) ** 2).item()
                drift_sq += max(2.0 * rank - 2.0 * overlap_sq, 0.0)
                rank_scale += 2 * rank
        if rank_scale == 0:
            raise RuntimeError("No U subspace drift could be computed.")
        return (drift_sq / rank_scale) ** 0.5

    @staticmethod
    def _gradient_diagnostics(base_loss, subspace_loss, u_parameters, weight):
        base_gradients = torch.autograd.grad(
            base_loss,
            u_parameters,
            retain_graph=True,
            allow_unused=True,
        )
        subspace_gradients = torch.autograd.grad(
            subspace_loss,
            u_parameters,
            retain_graph=True,
            allow_unused=True,
        )

        zero = base_loss.detach().new_zeros(())
        base_sq = zero.clone()
        subspace_sq = zero.clone()
        dot = zero.clone()
        for base_gradient, subspace_gradient in zip(
            base_gradients, subspace_gradients
        ):
            if base_gradient is not None:
                base_sq += torch.sum(base_gradient.detach().float() ** 2)
            if subspace_gradient is not None:
                subspace_sq += torch.sum(
                    subspace_gradient.detach().float() ** 2
                )
            if base_gradient is not None and subspace_gradient is not None:
                dot += torch.sum(
                    base_gradient.detach().float()
                    * subspace_gradient.detach().float()
                )

        base_norm = torch.sqrt(base_sq).item()
        subspace_norm = torch.sqrt(subspace_sq).item()
        weighted_norm = weight * subspace_norm
        ratio = weighted_norm / (base_norm + 1e-12)
        cosine = dot.item() / (base_norm * subspace_norm + 1e-12)
        return {
            "u_base_grad_norm": base_norm,
            "u_sub_grad_norm": subspace_norm,
            "u_sub_weighted_grad_norm": weighted_norm,
            "u_sub_to_base_grad_ratio": ratio,
            "u_base_sub_grad_cos": cosine,
        }

    def _factor_learning_rates(self):
        if bool(self.args.use_asymmetric_lr):
            return (
                self.learning_rate * float(self.args.u_lr_ratio),
                self.learning_rate * float(self.args.v_lr_ratio),
            )
        return self.learning_rate, self.learning_rate

    def _u_subspace_layer_diagnostics(
        self,
        model,
        start_subspaces,
        factor_starts,
        current_round,
    ):
        named_parameters = dict(model.named_parameters())
        u_lr, _ = self._factor_learning_rates()
        effective_lambda = (
            float(self.args.u_subspace_lambda)
            if bool(self.args.u_subspace_reg)
            else 0.0
        )
        rows = []
        with torch.no_grad():
            for u_name, q_start in start_subspaces.items():
                u_end = named_parameters[u_name]
                self._validate_u_parameter(u_name, u_end)
                start = factor_starts[u_name]
                v_end = named_parameters[start["v_name"]]
                if not torch.isfinite(v_end).all():
                    raise RuntimeError(
                        f"V parameter {start['v_name']} contains NaN/Inf."
                    )

                u_end_float = u_end.detach().float()
                v_end_float = v_end.detach().float()
                q_end, _ = torch.linalg.qr(u_end_float, mode="reduced")
                rank = q_end.shape[1]
                overlap = q_start.T @ q_end
                overlap_sq = torch.sum(overlap ** 2).item()
                drift_sq = min(
                    max(1.0 - overlap_sq / rank, 0.0), 1.0
                )

                principal_cosines = torch.linalg.svdvals(overlap).clamp(
                    0.0, 1.0
                )
                angles_deg = torch.rad2deg(torch.acos(principal_cosines))
                u_singular_values = torch.linalg.svdvals(u_end_float)
                sigma_min = u_singular_values.min().item()
                sigma_max = u_singular_values.max().item()

                u_relative_change = (
                    torch.linalg.vector_norm(u_end_float - start["u"]).item()
                    / (
                        torch.linalg.vector_norm(start["u"]).item()
                        + 1e-12
                    )
                )
                v_relative_change = (
                    torch.linalg.vector_norm(v_end_float - start["v"]).item()
                    / (
                        torch.linalg.vector_norm(start["v"]).item()
                        + 1e-12
                    )
                )
                rows.append(
                    {
                        "round": current_round,
                        "client_id": self.id,
                        "layer_name": u_name,
                        "rank": rank,
                        "u_lr": u_lr,
                        "lambda_sub": effective_lambda,
                        "subspace_drift_sq": drift_sq,
                        "subspace_drift_norm": drift_sq ** 0.5,
                        "principal_angle_mean_deg": angles_deg.mean().item(),
                        "principal_angle_max_deg": angles_deg.max().item(),
                        "principal_angle_median_deg": angles_deg.median().item(),
                        "R_U": u_relative_change,
                        "R_V": v_relative_change,
                        "u_sigma_min": sigma_min,
                        "u_sigma_max": sigma_max,
                        "u_condition_number": sigma_max / (sigma_min + 1e-12),
                    }
                )
        return rows

    def train(self, current_round=0):
        trainloader = self.load_train_data()
        model = load_item(self.role, "model", self.save_folder_name)
        if model is None:
            raise RuntimeError(f"{self.role} model is missing before training.")
        model = model.to(self.device)

        optimizer = self._build_optimizer(model)
        use_u_subspace_reg = bool(self.args.u_subspace_reg)
        use_u_subspace_diag = bool(
            getattr(self.args, "u_subspace_diag", 0)
        )
        track_u_subspace = use_u_subspace_reg or use_u_subspace_diag
        u_start_subspaces = None
        factor_starts = None
        if use_u_subspace_reg:
            if self.args.u_subspace_lambda < 0.0:
                raise ValueError(
                    "u_subspace_lambda must be non-negative, got "
                    f"{self.args.u_subspace_lambda}."
                )
        if track_u_subspace:
            u_start_subspaces = self._capture_u_start_subspaces(model)
        if use_u_subspace_diag:
            factor_starts = self._capture_factor_starts(
                model, u_start_subspaces
            )
        self.last_u_subspace_stats = None
        self.last_u_subspace_layer_stats = []
        subspace_loss_sum = 0.0
        subspace_loss_steps = 0
        gradient_diagnostic_done = False
        optimizer_steps_completed = 0
        gradient_stats = {
            "u_base_grad_norm": float("nan"),
            "u_sub_grad_norm": float("nan"),
            "u_sub_weighted_grad_norm": float("nan"),
            "u_sub_to_base_grad_ratio": float("nan"),
            "u_base_sub_grad_cos": float("nan"),
        }
        u_parameters = []
        if use_u_subspace_diag:
            u_parameters = [
                parameter
                for name, parameter in model.named_parameters()
                if self._is_u_parameter(name) and parameter.requires_grad
            ]
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

                base_loss = self.loss(logits, labels)
                base_loss += self.args.mse_lamda * alignment_loss
                if self.args.is_regular == 1:
                    base_loss += (
                        self.args.regular_lamda * model.frobenius_decay()
                    )
                subspace_loss = None
                if use_u_subspace_reg:
                    subspace_loss = self._u_subspace_loss(
                        model, u_start_subspaces
                    )
                    subspace_loss_sum += subspace_loss.detach().item()
                    subspace_loss_steps += 1
                elif (
                    use_u_subspace_diag
                    and not gradient_diagnostic_done
                    and optimizer_steps_completed >= 1
                ):
                    subspace_loss = self._u_subspace_loss(
                        model, u_start_subspaces
                    )

                if (
                    use_u_subspace_diag
                    and not gradient_diagnostic_done
                    and optimizer_steps_completed >= 1
                ):
                    if subspace_loss is None:
                        subspace_loss = self._u_subspace_loss(
                            model, u_start_subspaces
                        )
                    diagnostic_weight = (
                        float(self.args.u_subspace_lambda)
                        if use_u_subspace_reg
                        else 0.0
                    )
                    gradient_stats = self._gradient_diagnostics(
                        base_loss,
                        subspace_loss,
                        u_parameters,
                        diagnostic_weight,
                    )
                    gradient_diagnostic_done = True

                loss = base_loss
                if use_u_subspace_reg:
                    loss = (
                        loss
                        + self.args.u_subspace_lambda * subspace_loss
                    )
                loss.backward()
                torch.nn.utils.clip_grad_norm_(clip_params, 10.0)
                optimizer.step()
                optimizer_steps_completed += 1

        if torch.cuda.is_available() and str(self.device).startswith("cuda"):
            torch.cuda.synchronize(self.device)
        local_train_time = time.time() - start_time
        if use_u_subspace_diag:
            layer_stats = self._u_subspace_layer_diagnostics(
                model,
                u_start_subspaces,
                factor_starts,
                current_round,
            )
            self.last_u_subspace_layer_stats = layer_stats
            u_lr, v_lr = self._factor_learning_rates()
            mean_subspace_loss = (
                subspace_loss_sum / max(subspace_loss_steps, 1)
                if use_u_subspace_reg
                else float("nan")
            )
            self.last_u_subspace_stats = {
                "round": current_round,
                "client_id": self.id,
                "u_lr": u_lr,
                "v_lr": v_lr,
                "lambda_sub": (
                    float(self.args.u_subspace_lambda)
                    if use_u_subspace_reg
                    else 0.0
                ),
                "mean_loss": mean_subspace_loss,
                "mean_subspace_drift_norm": sum(
                    row["subspace_drift_norm"] for row in layer_stats
                ) / len(layer_stats),
                "max_subspace_drift_norm": max(
                    row["subspace_drift_norm"] for row in layer_stats
                ),
                "mean_principal_angle_deg": sum(
                    row["principal_angle_mean_deg"] for row in layer_stats
                ) / len(layer_stats),
                "max_principal_angle_deg": max(
                    row["principal_angle_max_deg"] for row in layer_stats
                ),
                "mean_R_U": sum(row["R_U"] for row in layer_stats)
                / len(layer_stats),
                "mean_R_V": sum(row["R_V"] for row in layer_stats)
                / len(layer_stats),
                "mean_u_sigma_min": sum(
                    row["u_sigma_min"] for row in layer_stats
                ) / len(layer_stats),
                "mean_u_condition_number": sum(
                    row["u_condition_number"] for row in layer_stats
                ) / len(layer_stats),
                "diag_enabled": True,
                "reg_enabled": use_u_subspace_reg,
                **gradient_stats,
            }
        elif use_u_subspace_reg:
            mean_subspace_loss = subspace_loss_sum / max(
                subspace_loss_steps, 1
            )
            drift_norm = self._u_subspace_drift_norm(
                model, u_start_subspaces
            )
            self.last_u_subspace_stats = {
                "client_id": self.id,
                "mean_loss": mean_subspace_loss,
                "drift_norm": drift_norm,
                "diag_enabled": False,
                "reg_enabled": True,
            }
            if self.id == 0:
                print(
                    f"[USubspaceReg] round={current_round} client={self.id} "
                    f"lambda={self.args.u_subspace_lambda:g} "
                    f"train_loss={mean_subspace_loss:.6e} "
                    f"u_subspace_drift_norm={drift_norm:.6e}"
                )
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

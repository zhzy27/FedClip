import torch
import numpy as np
import time
import itertools
from flcore.clients.clientbase import Client, load_item, save_item
from sklearn.preprocessing import label_binarize
from utils.get_clip_text_encoder import get_clip_class_embeddings, get_clip_class_depth_embeddings
from utils.factor_loss_diagnostics import (
    VIRTUAL_FIELDS,
    collect_factor_loss_diagnostics,
    factor_kind,
    gradient_clip_diagnostics,
    named_factor_parameters,
    scaled_u_gradients,
)
from utils.agg_path_diagnostics import (
    aggregation_human_round,
    collect_agg_path_updates,
    diagnostic_round_selected,
)


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
        self.last_factor_update_stats = None
        self.last_ce_anchor_diagnostics = []
        self.last_agg_path_updates = {}
        # Optional ResNet state is defined for every client so CNN diagnostics
        # can safely share the same lifecycle code.
        self.resnet_clip_aligners = None
        self._resnet_stage_end_cache = {}
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

    @staticmethod
    def _factor_kind(param_name):
        if param_name.endswith('weight_u') or param_name.endswith('conv_u'):
            return 'u'
        if param_name.endswith('weight_v') or param_name.endswith('conv_v'):
            return 'v'
        return None

    @staticmethod
    def _paired_v_name(u_name):
        if u_name.endswith('weight_u'):
            return f"{u_name[:-len('weight_u')]}weight_v"
        if u_name.endswith('conv_u'):
            return f"{u_name[:-len('conv_u')]}conv_v"
        raise ValueError(f"Not a recognized U-factor parameter: {u_name}")

    @staticmethod
    def _product_frobenius_sq(left, right, max_chunk_elements=4_000_000):
        """Return ||left @ right||_F^2 without materializing a huge product."""
        if left.ndim != 2 or right.ndim != 2:
            raise ValueError(
                "Factor contribution diagnostics require two-dimensional "
                f"matrices, got {tuple(left.shape)} and {tuple(right.shape)}."
            )
        if left.shape[1] != right.shape[0]:
            raise ValueError(
                "Factor contribution shapes are incompatible: "
                f"{tuple(left.shape)} @ {tuple(right.shape)}."
            )

        output_columns = int(right.shape[1])
        rows_per_chunk = max(
            1,
            min(
                int(left.shape[0]),
                max_chunk_elements // max(output_columns, 1),
            ),
        )
        total_sq = left.new_zeros(())
        for row_start in range(0, int(left.shape[0]), rows_per_chunk):
            product = left[row_start:row_start + rows_per_chunk] @ right
            total_sq += torch.sum(product * product)
        return total_sq

    @staticmethod
    def _weight_update_frobenius_sq(
        u_start,
        v_start,
        u_end,
        v_end,
        max_chunk_elements=4_000_000,
    ):
        """Return ||U_end V_end - U_start V_start||_F^2 exactly."""
        factors = (u_start, v_start, u_end, v_end)
        if any(factor.ndim != 2 for factor in factors):
            raise ValueError(
                "Effective-weight diagnostics require two-dimensional "
                f"factors, got {[tuple(factor.shape) for factor in factors]}."
            )
        if u_start.shape != u_end.shape or v_start.shape != v_end.shape:
            raise ValueError(
                "Start/end factor shapes must match, got "
                f"U {tuple(u_start.shape)}/{tuple(u_end.shape)} and "
                f"V {tuple(v_start.shape)}/{tuple(v_end.shape)}."
            )
        if u_start.shape[1] != v_start.shape[0]:
            raise ValueError(
                "Effective-weight factor shapes are incompatible: "
                f"{tuple(u_start.shape)} @ {tuple(v_start.shape)}."
            )

        output_columns = int(v_start.shape[1])
        rows_per_chunk = max(
            1,
            min(
                int(u_start.shape[0]),
                max_chunk_elements // max(output_columns, 1),
            ),
        )
        total_sq = u_start.new_zeros(())
        for row_start in range(0, int(u_start.shape[0]), rows_per_chunk):
            row_end = row_start + rows_per_chunk
            w_start = u_start[row_start:row_end] @ v_start
            w_end = u_end[row_start:row_end] @ v_end
            delta_w = w_end - w_start
            total_sq += torch.sum(delta_w * delta_w)
        return total_sq

    @staticmethod
    def _u_subspace_drift_sq(u_start, u_end):
        """Return ||Q_end Q_end^T - Q_start Q_start^T||_F^2."""
        if u_start.ndim != 2 or u_end.ndim != 2:
            raise ValueError(
                "U subspace diagnostics require two-dimensional matrices, "
                f"got {tuple(u_start.shape)} and {tuple(u_end.shape)}."
            )
        q_start, _ = torch.linalg.qr(u_start, mode='reduced')
        q_end, _ = torch.linalg.qr(u_end, mode='reduced')
        overlap_sq = torch.sum((q_start.transpose(0, 1) @ q_end) ** 2)
        drift_sq = q_start.shape[1] + q_end.shape[1] - 2.0 * overlap_sq
        return torch.clamp(drift_sq, min=0.0)

    def _snapshot_factor_parameters(self, model):
        return {
            name: param.detach().clone()
            for name, param in model.named_parameters()
            if self._factor_kind(name) is not None
        }

    @staticmethod
    def _parse_int_list(value):
        if value is None:
            return None
        if isinstance(value, (list, tuple, set)):
            return {int(item) for item in value}
        value = str(value).strip()
        if not value:
            return None
        return {int(item.strip()) for item in value.split(',') if item.strip()}

    def _diagnostic_target(self, current_round):
        enabled = bool(getattr(self.args, 'enable_ce_anchor_diagnostics', 0))
        virtual_enabled = bool(
            getattr(self.args, 'enable_virtual_step_diagnostics', 0)
        )
        if not (enabled or virtual_enabled):
            return False
        rounds = self._parse_int_list(
            getattr(self.args, 'diagnostic_rounds', '1,20,50')
        )
        client_ids = self._parse_int_list(
            getattr(self.args, 'diagnostic_client_ids', '0,10,19')
        )
        human_round = int(current_round) + 1
        return (
            (rounds is None or human_round in rounds)
            and (client_ids is None or int(self.id) in client_ids)
        )

    def _agg_path_diagnostic_target(self, current_round):
        if not bool(getattr(self.args, "enable_agg_path_diagnostics", 0)):
            return False
        return diagnostic_round_selected(
            aggregation_human_round(current_round),
            getattr(
                self.args,
                "agg_diagnostic_rounds",
                "1,5,10,20,30,40,50,60,70,80,90,100",
            ),
        )

    def _move_batch_to_device(self, batch):
        x, y = batch
        if isinstance(x, list):
            x[0] = x[0].to(self.device)
        else:
            x = x.to(self.device)
        return x, y.to(self.device)

    def _forward_clip_losses(self, model, x, y):
        if self.use_resnet_multilevel_clip:
            features, stage_features = self._forward_resnet_multilevel_features(
                model, x
            )
            logits = model.head(features)
            anchor_loss = self._resnet_multilevel_clip_loss(stage_features, y)
        else:
            features = model.base(x)
            logits = model.head(features)
            anchor_loss = self.mse_fn(features, self.clip_text_features[y])
        return self.loss(logits, y), anchor_loss

    def _virtual_step_diagnostics(
        self,
        model,
        probe_batch,
        gradients,
        actual_u_lr,
        actual_v_lr,
    ):
        results = {field: float('nan') for field in VIRTUAL_FIELDS}
        if probe_batch is None:
            return results

        x_probe, y_probe = self._move_batch_to_device(probe_batch)
        with torch.no_grad():
            baseline_ce, baseline_anchor = self._forward_clip_losses(
                model, x_probe, y_probe
            )
            baseline_ce = float(baseline_ce.item())
            baseline_anchor = float(baseline_anchor.item())

        parameter_map = dict(named_factor_parameters(model))
        virtual_scale = float(getattr(self.args, 'virtual_step_scale', 1.0))
        if virtual_scale < 0.0:
            raise ValueError(
                f"virtual_step_scale must be non-negative, got {virtual_scale}."
            )

        common_probe_lr = float(self.learning_rate)
        step_modes = (
            ('', {'u': float(actual_u_lr), 'v': float(actual_v_lr)}),
            (
                'common_',
                {'u': common_probe_lr, 'v': common_probe_lr},
            ),
        )
        for field_prefix, group_lrs in step_modes:
            for source_name in ('ce', 'anchor'):
                for group_name in ('u', 'v'):
                    selected = {
                        name: parameter
                        for name, parameter in parameter_map.items()
                        if factor_kind(name) == group_name
                    }
                    originals = {
                        name: parameter.detach().clone()
                        for name, parameter in selected.items()
                    }
                    try:
                        with torch.no_grad():
                            for name, parameter in selected.items():
                                gradient = gradients[source_name].get(name)
                                if gradient is not None:
                                    parameter.add_(
                                        gradient,
                                        alpha=(
                                            -group_lrs[group_name]
                                            * virtual_scale
                                        ),
                                    )
                            changed_ce, changed_anchor = (
                                self._forward_clip_losses(
                                    model, x_probe, y_probe
                                )
                            )
                            changed_ce = float(changed_ce.item())
                            changed_anchor = float(changed_anchor.item())
                    finally:
                        with torch.no_grad():
                            for name, parameter in selected.items():
                                parameter.copy_(originals[name])

                    prefix = (
                        f"virtual_{field_prefix}{source_name}_to_"
                        f"{group_name}_delta"
                    )
                    results[f"{prefix}_ce"] = changed_ce - baseline_ce
                    results[f"{prefix}_anchor"] = (
                        changed_anchor - baseline_anchor
                    )
        return results

    def _run_loss_diagnostics(
        self,
        model,
        diagnostic_batch,
        probe_batch,
        current_round,
        actual_u_lr,
        actual_v_lr,
    ):
        model_was_training = model.training
        resnet_clip_aligners = getattr(
            self, 'resnet_clip_aligners', None
        )
        aligners_were_training = (
            None
            if resnet_clip_aligners is None
            else resnet_clip_aligners.training
        )
        model.eval()
        if resnet_clip_aligners is not None:
            resnet_clip_aligners.eval()
        try:
            x, y = self._move_batch_to_device(diagnostic_batch)
            ce_loss, anchor_loss = self._forward_clip_losses(model, x, y)
            regularization_coefficient = (
                float(self.args.regular_lamda)
                if int(self.args.is_regular) == 1
                else 0.0
            )
            regularization_loss = (
                model.frobenius_decay()
                if int(self.args.is_regular) == 1
                else None
            )
            try:
                capacity = float(model.ratio_LR)
            except (AttributeError, TypeError, ValueError):
                capacity = float('nan')
            rows, gradients = collect_factor_loss_diagnostics(
                model=model,
                ce_loss=ce_loss,
                anchor_loss=anchor_loss,
                regularization_loss=regularization_loss,
                anchor_coefficient=float(self.args.mse_lamda),
                regularization_coefficient=regularization_coefficient,
                round_number=int(current_round) + 1,
                client_id=self.id,
                capacity=capacity,
                u_lr=actual_u_lr,
                v_lr=actual_v_lr,
            )
            if bool(getattr(self.args, 'enable_virtual_step_diagnostics', 0)):
                virtual_results = self._virtual_step_diagnostics(
                    model,
                    probe_batch,
                    gradients,
                    actual_u_lr,
                    actual_v_lr,
                )
                rows[0].update(virtual_results)
            return rows
        finally:
            model.train(model_was_training)
            if resnet_clip_aligners is not None:
                restore_aligner_training = (
                    model_was_training
                    if aligners_were_training is None
                    else aligners_were_training
                )
                resnet_clip_aligners.train(restore_aligner_training)

    def _factor_update_statistics(
        self, model, factor_start, current_round, u_lr, v_lr
    ):
        eps = 1e-12
        end_params = dict(model.named_parameters())
        delta_sq = {'u': 0.0, 'v': 0.0}
        start_sq = {'u': 0.0, 'v': 0.0}
        u_subspace_drift_sq = 0.0
        u_subspace_rank_scale = 0.0
        c_u_sq = 0.0
        c_v_sq = 0.0
        c_uv_sq = 0.0
        d_w_sq = 0.0

        with torch.no_grad():
            for name, start_param in factor_start.items():
                if name not in end_params:
                    raise RuntimeError(
                        f"Factor parameter {name} disappeared during local training."
                    )
                end_param = end_params[name].detach()
                if end_param.shape != start_param.shape:
                    raise RuntimeError(
                        f"Factor parameter shape changed for {name}: "
                        f"start={tuple(start_param.shape)}, "
                        f"end={tuple(end_param.shape)}."
                    )
                delta = end_param - start_param
                factor_kind = self._factor_kind(name)
                delta_sq[factor_kind] += float(
                    torch.sum(delta.float() ** 2).item()
                )
                start_sq[factor_kind] += float(
                    torch.sum(start_param.float() ** 2).item()
                )

            for u_name, u_start_param in factor_start.items():
                if self._factor_kind(u_name) != 'u':
                    continue

                v_name = self._paired_v_name(u_name)
                if v_name not in factor_start or v_name not in end_params:
                    raise RuntimeError(
                        f"Missing paired V factor {v_name} for {u_name}."
                    )

                u_start = u_start_param.float()
                v_start = factor_start[v_name].float()
                u_end = end_params[u_name].detach().float()
                v_end = end_params[v_name].detach().float()
                if u_start.shape != u_end.shape or v_start.shape != v_end.shape:
                    raise RuntimeError(
                        f"Factor shape changed for pair {u_name}/{v_name}: "
                        f"U {tuple(u_start.shape)} -> {tuple(u_end.shape)}, "
                        f"V {tuple(v_start.shape)} -> {tuple(v_end.shape)}."
                    )
                if u_start.shape[1] != v_start.shape[0]:
                    raise RuntimeError(
                        f"Incompatible factor pair {u_name}/{v_name}: "
                        f"{tuple(u_start.shape)} @ {tuple(v_start.shape)}."
                    )
                if not all(
                    torch.isfinite(tensor).all().item()
                    for tensor in (u_start, v_start, u_end, v_end)
                ):
                    raise RuntimeError(
                        f"NaN/Inf found in factor pair {u_name}/{v_name}."
                    )

                delta_u = u_end - u_start
                delta_v = v_end - v_start
                layer_subspace_drift_sq = float(
                    self._u_subspace_drift_sq(u_start, u_end).item()
                )
                u_subspace_drift_sq += layer_subspace_drift_sq
                u_subspace_rank_scale += 2.0 * float(u_start.shape[1])
                c_u_sq += float(
                    self._product_frobenius_sq(delta_u, v_start).item()
                )
                c_v_sq += float(
                    self._product_frobenius_sq(u_start, delta_v).item()
                )
                c_uv_sq += float(
                    self._product_frobenius_sq(delta_u, delta_v).item()
                )
                d_w_sq += float(
                    self._weight_update_frobenius_sq(
                        u_start,
                        v_start,
                        u_end,
                        v_end,
                    ).item()
                )

            d_u = delta_sq['u'] ** 0.5
            d_v = delta_sq['v'] ** 0.5
            r_u = d_u / (start_sq['u'] ** 0.5 + eps)
            r_v = d_v / (start_sq['v'] ** 0.5 + eps)
            r_u_over_r_v = r_u / (r_v + eps)
            u_subspace_drift = u_subspace_drift_sq ** 0.5
            u_subspace_drift_norm = (
                u_subspace_drift_sq / (u_subspace_rank_scale + eps)
            ) ** 0.5
            c_u = c_u_sq ** 0.5
            c_v = c_v_sq ** 0.5
            c_uv = c_uv_sq ** 0.5
            c_u_over_c_v = c_u / (c_v + eps)
            d_w = d_w_sq ** 0.5

        return {
            'round': int(current_round),
            'client_id': int(self.id),
            'u_lr': float(u_lr),
            'v_lr': float(v_lr),
            'R_U': float(r_u),
            'R_V': float(r_v),
            'R_U_over_R_V': float(r_u_over_r_v),
            'D_U': float(d_u),
            'D_V': float(d_v),
            'U_subspace_drift': float(u_subspace_drift),
            'U_subspace_drift_norm': float(u_subspace_drift_norm),
            'C_U': float(c_u),
            'C_V': float(c_v),
            'C_UV': float(c_uv),
            'C_U_over_C_V': float(c_u_over_c_v),
            'D_W': float(d_w),
        }

    def train(self, current_round=0):
        trainloader = self.load_train_data()
        model = load_item(self.role, 'model', self.save_folder_name)
        model.to(self.device)
        self.last_ce_anchor_diagnostics = []
        self.last_agg_path_updates = {}
        use_loss_specific_u_scaling = bool(
            getattr(self.args, 'use_loss_specific_u_scaling', 0)
        )
        u_gradient_scales = {
            'ce': float(getattr(self.args, 'u_ce_grad_scale', 1.0)),
            'anchor': float(getattr(self.args, 'u_anchor_grad_scale', 1.0)),
            'reg': float(getattr(self.args, 'u_reg_grad_scale', 1.0)),
        }
        if any(scale < 0.0 for scale in u_gradient_scales.values()):
            raise ValueError(
                "U loss-specific gradient scales must be non-negative, got "
                f"{u_gradient_scales}."
            )
        named_u_parameters = [
            (name, parameter)
            for name, parameter in model.named_parameters()
            if parameter.requires_grad and self._factor_kind(name) == 'u'
        ]
        if use_loss_specific_u_scaling and not named_u_parameters:
            raise RuntimeError(
                "Loss-specific U scaling was enabled, but no U factors exist."
            )
        # ================= 增加模型大小打印 =================
        total_params = sum(p.numel() for p in model.parameters())
        # 为了方便阅读，将其转换为 百万 (Million, M) 级别
        print(f"[{self.role}] 当前模型参数量为: {total_params} ({total_params / 1e6:.3f} M)")

        factor_start = self._snapshot_factor_parameters(model)
        actual_u_lr = self.learning_rate
        actual_v_lr = self.learning_rate
        if bool(getattr(self.args, 'use_asymmetric_lr', 0)):
            u_lr_ratio = float(getattr(self.args, 'u_lr_ratio', 0.1))
            v_lr_ratio = float(getattr(self.args, 'v_lr_ratio', 1.0))
            if u_lr_ratio < 0.0:
                raise ValueError(
                    f"u_lr_ratio must be non-negative, got {u_lr_ratio}."
                )
            if v_lr_ratio < 0.0:
                raise ValueError(
                    f"v_lr_ratio must be non-negative, got {v_lr_ratio}."
                )
            u_lr_warmup_rounds = int(
                getattr(self.args, 'u_lr_warmup_rounds', -1)
            )
            if u_lr_warmup_rounds < -1:
                raise ValueError(
                    "u_lr_warmup_rounds must be -1 or a non-negative "
                    f"integer, got {u_lr_warmup_rounds}."
                )

            u_is_frozen = (
                u_lr_warmup_rounds >= 0
                and current_round >= u_lr_warmup_rounds
            )
            if use_loss_specific_u_scaling:
                effective_u_lr_ratio = 1.0
                u_is_frozen = False
            else:
                effective_u_lr_ratio = 0.0 if u_is_frozen else u_lr_ratio
            actual_u_lr = self.learning_rate * effective_u_lr_ratio
            actual_v_lr = self.learning_rate * v_lr_ratio
            if self.id == 0:
                if use_loss_specific_u_scaling:
                    frozen_suffix = " (loss-specific U scaling)"
                else:
                    frozen_suffix = " (frozen)" if u_is_frozen else ""
                print(
                    f"[Round {current_round + 1:03d}] U/V LR ratio = "
                    f"{effective_u_lr_ratio}/{v_lr_ratio}{frozen_suffix}"
                )
            u_params = []
            v_params = []
            base_lr_params = []
            for name, param in model.named_parameters():
                if not param.requires_grad:
                    continue
                if name.endswith('weight_u') or name.endswith('conv_u'):
                    u_params.append(param)
                elif name.endswith('weight_v') or name.endswith('conv_v'):
                    v_params.append(param)
                else:
                    base_lr_params.append(param)

            param_groups = []
            if base_lr_params:
                param_groups.append({'params': base_lr_params, 'lr': self.learning_rate})
            if u_params:
                param_groups.append({
                    'params': u_params,
                    'lr': actual_u_lr,
                })
            if v_params:
                param_groups.append({
                    'params': v_params,
                    'lr': actual_v_lr,
                })
            optimizer = torch.optim.SGD(param_groups, lr=self.learning_rate)
        else:
            optimizer = torch.optim.SGD(
                (param for param in model.parameters() if param.requires_grad),
                lr=self.learning_rate,
            )
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
        first_epoch_batches = None
        diagnostic_target = self._diagnostic_target(current_round)
        clip_diagnostics_recorded = False
        if diagnostic_target:
            first_epoch_iterator = iter(trainloader)
            prefetched_batches = []
            try:
                prefetched_batches.append(next(first_epoch_iterator))
            except StopIteration:
                pass
            if (
                prefetched_batches
                and bool(
                    getattr(
                        self.args, 'enable_virtual_step_diagnostics', 0
                    )
                )
            ):
                try:
                    prefetched_batches.append(next(first_epoch_iterator))
                except StopIteration:
                    pass
            if prefetched_batches:
                probe_batch = (
                    prefetched_batches[1]
                    if len(prefetched_batches) > 1
                    else None
                )
                self.last_ce_anchor_diagnostics = self._run_loss_diagnostics(
                    model,
                    prefetched_batches[0],
                    probe_batch,
                    current_round,
                    actual_u_lr,
                    actual_v_lr,
                )
                first_epoch_batches = itertools.chain(
                    prefetched_batches, first_epoch_iterator
                )
        for step in range(max_local_epochs):
            batch_source = (
                first_epoch_batches
                if step == 0 and first_epoch_batches is not None
                else trainloader
            )
            for i, (x, y) in enumerate(batch_source):
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
                if use_loss_specific_u_scaling:
                    ce_loss = self.loss(logits, y)
                    regularization_loss = (
                        model.frobenius_decay()
                        if self.args.is_regular == 1
                        else None
                    )
                    regularization_coefficient = (
                        float(self.args.regular_lamda)
                        if self.args.is_regular == 1
                        else 0.0
                    )
                    combined_u_gradients = scaled_u_gradients(
                        ce_loss=ce_loss,
                        anchor_loss=mse_loss,
                        regularization_loss=regularization_loss,
                        named_u_parameters=named_u_parameters,
                        anchor_coefficient=float(self.args.mse_lamda),
                        regularization_coefficient=regularization_coefficient,
                        ce_scale=u_gradient_scales['ce'],
                        anchor_scale=u_gradient_scales['anchor'],
                        reg_scale=u_gradient_scales['reg'],
                    )
                    loss = ce_loss + self.args.mse_lamda * mse_loss
                    if self.args.is_regular == 1:
                        loss += (
                            self.args.regular_lamda
                            * regularization_loss
                        )
                    loss.backward()
                    for name, parameter in named_u_parameters:
                        parameter.grad = combined_u_gradients[name].clone()
                else:
                    loss = self.loss(logits, y) + self.args.mse_lamda * mse_loss
                    if self.args.is_regular==1:
                        loss += self.args.regular_lamda*model.frobenius_decay()
                    loss.backward()
                pre_clip_total_grad_norm = torch.nn.utils.clip_grad_norm_(
                    clip_params, 10.0
                )
                if (
                    diagnostic_target
                    and not clip_diagnostics_recorded
                    and self.last_ce_anchor_diagnostics
                ):
                    clip_values = gradient_clip_diagnostics(
                        pre_clip_total_grad_norm,
                        max_norm=10.0,
                    )
                    for diagnostic_row in self.last_ce_anchor_diagnostics:
                        if diagnostic_row.get('layer') == '__overall__':
                            diagnostic_row.update(clip_values)
                            break
                    clip_diagnostics_recorded = True
                optimizer.step()
        if torch.cuda.is_available() and str(self.device).startswith("cuda"):
            torch.cuda.synchronize(self.device)
        local_train_time = time.time() - start_time
        self.last_factor_update_stats = self._factor_update_statistics(
            model,
            factor_start,
            current_round,
            actual_u_lr,
            actual_v_lr,
        )
        if self._agg_path_diagnostic_target(current_round):
            self.last_agg_path_updates = collect_agg_path_updates(
                factor_start,
                dict(model.named_parameters()),
            )
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
        if model is None:
            raise RuntimeError(
                f"Missing local model shell for {self.role} in "
                f"{self.save_folder_name}. Client initialization did not "
                "complete successfully."
            )
        model = model.to(self.device)
        
        global_model = load_item('Server', 'model', self.save_folder_name)
        if global_model is None:
            raise RuntimeError(
                f"Missing Server_model.pt in {self.save_folder_name}."
            )
        global_model = global_model.to(self.device)
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

    

import math

import torch
import torch.nn.functional as F


FEATURE_AUX_LOSS_MODES = (
    "none",
    "mse",
    "z2",
    "cosine",
    "contrastive",
    "z1",
    "global_dir_l1",
    "global_dir_l2",
    "global_point_l1",
    "global_point_l2",
    "radial_l1",
    "radial_l2",
)

GLOBAL_FEATURE_ANCHOR_MODES = (
    "global_dir_l1",
    "global_dir_l2",
    "global_point_l1",
    "global_point_l2",
)

RADIAL_FEATURE_AUX_MODES = (
    "radial_l1",
    "radial_l2",
)

AUX_GRADIENT_SCALE_FIELDS = [
    "round",
    "client_id",
    "aux_loss_mode",
    "aux_coefficient",
    "ce_loss",
    "aux_loss",
    "ce_grad_norm",
    "aux_grad_norm",
    "aux_to_ce_grad_ratio",
    "weighted_aux_to_ce_grad_ratio",
    "feature_norm",
    "feature_norm_mean",
    "feature_norm_std",
    "target_feature_norm",
    "global_anchor_norm",
]


def resolve_feature_aux_target_norm(reference_anchors, configured_norm=-1.0):
    """Resolve a fixed positive feature radius from config or CLIP anchors."""
    configured_norm = float(configured_norm)
    if configured_norm != -1.0:
        if configured_norm <= 0.0:
            raise ValueError(
                "feature_aux_target_norm must be -1 or positive, got "
                f"{configured_norm}."
            )
        return configured_norm

    if reference_anchors is None or reference_anchors.ndim < 2:
        raise ValueError(
            "Automatic feature_aux_target_norm requires CLIP anchors with "
            "shape [..., classes, dim]."
        )
    norms = torch.linalg.vector_norm(
        reference_anchors.detach().to(device="cpu", dtype=torch.float64),
        dim=-1,
    )
    target_norm = float(norms.mean().item())
    if not math.isfinite(target_norm) or target_norm <= 0.0:
        raise ValueError(
            "The mean CLIP-anchor norm must be finite and positive, got "
            f"{target_norm}."
        )
    return target_norm


def build_global_feature_anchor(
    reference_anchors,
    seed=0,
    target_norm=-1.0,
):
    """Build one deterministic shared anchor without advancing global RNG."""
    if reference_anchors is None or reference_anchors.ndim < 2:
        raise ValueError(
            "Global feature anchor requires CLIP anchors with shape "
            "[..., classes, dim]."
        )
    resolved_norm = resolve_feature_aux_target_norm(
        reference_anchors,
        configured_norm=target_norm,
    )
    feature_dim = int(reference_anchors.shape[-1])
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    anchor_cpu = torch.randn(
        feature_dim,
        generator=generator,
        dtype=torch.float64,
    )
    anchor_cpu = F.normalize(anchor_cpu, p=2, dim=0, eps=1e-12)
    anchor_cpu = anchor_cpu * resolved_norm
    anchor = anchor_cpu.to(
        device=reference_anchors.device,
        dtype=reference_anchors.dtype,
    )
    return anchor, resolved_norm


def _validate_global_anchor(features, global_anchor, mode):
    if global_anchor is None:
        raise ValueError(f"{mode} auxiliary loss requires a global anchor.")
    if global_anchor.ndim != 1:
        raise ValueError(
            f"Global anchor must have shape [dim], got {global_anchor.shape}."
        )
    if features.shape[1] != global_anchor.shape[0]:
        raise ValueError(
            "Feature and global-anchor dimensions differ: "
            f"{features.shape[1]} vs {global_anchor.shape[0]}."
        )
    return global_anchor.to(device=features.device, dtype=features.dtype)


def _target_norm_tensor(features, target_norm, mode):
    if target_norm is None:
        raise ValueError(f"{mode} auxiliary loss requires target_norm.")
    target = torch.as_tensor(
        target_norm,
        device=features.device,
        dtype=features.dtype,
    )
    if target.numel() != 1 or not torch.isfinite(target) or target <= 0:
        raise ValueError(
            f"{mode} target_norm must be finite and positive, got {target_norm}."
        )
    return target


def feature_contrastive_logits(features, class_anchors, temperature=0.1):
    temperature = float(temperature)
    if temperature <= 0.0:
        raise ValueError(
            "feature_contrastive_tau must be positive, got "
            f"{temperature}."
        )
    if class_anchors is None:
        raise ValueError("Contrastive auxiliary loss requires class anchors.")
    if features.ndim != 2 or class_anchors.ndim != 2:
        raise ValueError(
            "Contrastive auxiliary loss expects [batch, dim] features and "
            "[classes, dim] anchors."
        )
    if features.shape[1] != class_anchors.shape[1]:
        raise ValueError(
            "Feature and anchor dimensions differ: "
            f"{features.shape[1]} vs {class_anchors.shape[1]}."
        )

    normalized_features = F.normalize(features, p=2, dim=-1, eps=1e-12)
    normalized_anchors = F.normalize(
        class_anchors.to(device=features.device, dtype=features.dtype),
        p=2,
        dim=-1,
        eps=1e-12,
    )
    return normalized_features @ normalized_anchors.transpose(0, 1) / temperature


def feature_auxiliary_loss(
    features,
    labels,
    class_anchors=None,
    mode="mse",
    mse_fn=None,
    contrastive_temperature=0.1,
    global_anchor=None,
    target_norm=None,
):
    """Compute one auxiliary objective without changing model state."""
    if mode not in FEATURE_AUX_LOSS_MODES:
        raise ValueError(f"Unsupported feature auxiliary loss: {mode}.")
    if features.ndim != 2:
        raise ValueError(
            f"Feature auxiliary loss expects [batch, dim], got {features.shape}."
        )

    # Keep a differentiable zero so diagnostics and loss-specific gradient
    # scaling can use the same code path without touching anchors.
    if mode == "none":
        return features.sum() * 0.0

    mse_fn = torch.nn.MSELoss() if mse_fn is None else mse_fn
    if mode == "z2":
        return mse_fn(features, torch.zeros_like(features))
    if mode == "z1":
        return features.abs().mean()

    if mode in GLOBAL_FEATURE_ANCHOR_MODES:
        anchor = _validate_global_anchor(features, global_anchor, mode)
        if mode.startswith("global_dir_"):
            normalized_features = F.normalize(
                features, p=2, dim=-1, eps=1e-12
            )
            normalized_anchor = F.normalize(
                anchor, p=2, dim=0, eps=1e-12
            )
            difference = normalized_features - normalized_anchor.unsqueeze(0)
        else:
            difference = features - anchor.unsqueeze(0)
        if mode.endswith("_l1"):
            return difference.abs().mean()
        return (difference ** 2).mean()

    if mode in RADIAL_FEATURE_AUX_MODES:
        target = _target_norm_tensor(features, target_norm, mode)
        radial_difference = torch.linalg.vector_norm(
            features, ord=2, dim=1
        ) - target
        if mode == "radial_l1":
            return radial_difference.abs().mean()
        return (radial_difference ** 2).mean()

    if class_anchors is None:
        raise ValueError(f"{mode} auxiliary loss requires class anchors.")
    anchors = class_anchors.to(device=features.device, dtype=features.dtype)
    labels = labels.to(device=features.device, dtype=torch.long)
    if anchors.ndim != 2:
        raise ValueError(
            f"Class anchors must have shape [classes, dim], got {anchors.shape}."
        )
    if features.shape[1] != anchors.shape[1]:
        raise ValueError(
            "Feature and anchor dimensions differ: "
            f"{features.shape[1]} vs {anchors.shape[1]}."
        )

    if mode == "mse":
        return mse_fn(features, anchors[labels])
    if mode == "cosine":
        normalized_features = F.normalize(features, p=2, dim=-1, eps=1e-12)
        normalized_targets = F.normalize(
            anchors[labels], p=2, dim=-1, eps=1e-12
        )
        return (
            1.0 - (normalized_features * normalized_targets).sum(dim=-1)
        ).mean()

    logits = feature_contrastive_logits(
        features,
        anchors,
        temperature=contrastive_temperature,
    )
    return F.cross_entropy(logits, labels)


def _gradient_norm(loss, parameters):
    parameters = list(parameters)
    if not parameters:
        raise ValueError("No feature-extractor parameters found for diagnostics.")
    if loss is None or not getattr(loss, "requires_grad", False):
        return 0.0
    gradients = torch.autograd.grad(
        loss,
        parameters,
        retain_graph=True,
        allow_unused=True,
    )
    norm_sq = 0.0
    for gradient in gradients:
        if gradient is None:
            continue
        norm_sq += float(torch.sum(gradient.detach().double() ** 2).item())
    return math.sqrt(max(norm_sq, 0.0))


def collect_aux_gradient_scale_diagnostic(
    ce_loss,
    aux_loss,
    named_base_parameters,
    round_number,
    client_id,
    aux_loss_mode,
    aux_coefficient,
    features,
    target_feature_norm=None,
    global_anchor=None,
):
    """Measure CE/auxiliary base gradients without writing ``.grad``."""
    parameters = [
        parameter
        for _, parameter in named_base_parameters
        if parameter.requires_grad
    ]
    ce_grad_norm = _gradient_norm(ce_loss, parameters)
    aux_grad_norm = _gradient_norm(aux_loss, parameters)
    ratio = aux_grad_norm / (ce_grad_norm + 1e-12)
    weighted_ratio = abs(float(aux_coefficient)) * ratio
    feature_norms = torch.linalg.vector_norm(
        features.detach().float().reshape(features.shape[0], -1),
        dim=1,
    )
    feature_norm_mean = float(feature_norms.mean().item())
    feature_norm_std = float(feature_norms.std(unbiased=False).item())
    resolved_target_norm = (
        float("nan")
        if target_feature_norm is None
        else float(target_feature_norm)
    )
    global_anchor_norm = (
        float("nan")
        if global_anchor is None
        else float(
            torch.linalg.vector_norm(
                global_anchor.detach().float().reshape(-1), dim=0
            ).item()
        )
    )
    return {
        "round": int(round_number),
        "client_id": int(client_id),
        "aux_loss_mode": str(aux_loss_mode),
        "aux_coefficient": float(aux_coefficient),
        "ce_loss": float(ce_loss.detach().item()),
        "aux_loss": float(aux_loss.detach().item()),
        "ce_grad_norm": float(ce_grad_norm),
        "aux_grad_norm": float(aux_grad_norm),
        "aux_to_ce_grad_ratio": float(ratio),
        "weighted_aux_to_ce_grad_ratio": float(weighted_ratio),
        "feature_norm": feature_norm_mean,
        "feature_norm_mean": feature_norm_mean,
        "feature_norm_std": feature_norm_std,
        "target_feature_norm": resolved_target_norm,
        "global_anchor_norm": global_anchor_norm,
    }

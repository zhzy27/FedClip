import math

import torch
import torch.nn.functional as F


FEATURE_AUX_LOSS_MODES = (
    "none",
    "mse",
    "z2",
    "cosine",
    "contrastive",
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
]


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
    feature_norm = float(
        torch.linalg.vector_norm(
            features.detach().float().reshape(features.shape[0], -1),
            dim=1,
        ).mean().item()
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
        "feature_norm": feature_norm,
    }

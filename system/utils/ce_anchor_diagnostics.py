import math

import torch


_EPS = 1e-12


def _is_u_parameter(name):
    return name.endswith("weight_u") or name.endswith("conv_u")


def _is_v_parameter(name):
    return name.endswith("weight_v") or name.endswith("conv_v")


def _group_metrics(entries, mse_lambda, prefix, empty_as_nan=False):
    if not entries:
        return {
            f"{prefix}_ce_grad_norm": math.nan,
            f"{prefix}_anchor_grad_norm": math.nan,
            f"{prefix}_anchor_to_ce_grad_ratio": math.nan,
            f"{prefix}_ce_anchor_grad_cos": math.nan,
        }

    reference = entries[0][1]
    ce_sq = torch.zeros((), dtype=torch.float64, device=reference.device)
    anchor_sq = torch.zeros_like(ce_sq)
    dot = torch.zeros_like(ce_sq)
    has_gradient = False

    for _, parameter, ce_grad, anchor_grad in entries:
        if ce_grad is not None:
            ce_value = ce_grad.detach().to(dtype=torch.float64)
            ce_sq = ce_sq + torch.sum(ce_value * ce_value)
            has_gradient = True
        else:
            ce_value = None

        if anchor_grad is not None:
            anchor_value = anchor_grad.detach().to(dtype=torch.float64)
            anchor_sq = anchor_sq + torch.sum(anchor_value * anchor_value)
            has_gradient = True
        else:
            anchor_value = None

        if ce_value is not None and anchor_value is not None:
            dot = dot + torch.sum(ce_value * anchor_value)

    if empty_as_nan and not has_gradient:
        return {
            f"{prefix}_ce_grad_norm": math.nan,
            f"{prefix}_anchor_grad_norm": math.nan,
            f"{prefix}_anchor_to_ce_grad_ratio": math.nan,
            f"{prefix}_ce_anchor_grad_cos": math.nan,
        }

    ce_norm = torch.sqrt(torch.clamp(ce_sq, min=0.0))
    anchor_norm = torch.sqrt(torch.clamp(anchor_sq, min=0.0))
    cosine = dot / (ce_norm * anchor_norm + _EPS)
    cosine = torch.clamp(torch.nan_to_num(cosine), -1.0, 1.0)
    weighted_anchor_norm = abs(float(mse_lambda)) * anchor_norm
    ratio = weighted_anchor_norm / (ce_norm + _EPS)

    return {
        f"{prefix}_ce_grad_norm": float(ce_norm.item()),
        f"{prefix}_anchor_grad_norm": float(anchor_norm.item()),
        f"{prefix}_anchor_to_ce_grad_ratio": float(ratio.item()),
        f"{prefix}_ce_anchor_grad_cos": float(cosine.item()),
    }


def collect_ce_anchor_gradient_diagnostics(
    enabled,
    ce_loss=None,
    anchor_loss=None,
    named_shared_parameters=None,
    mse_lambda=1.0,
):
    """Measure CE/anchor gradients without writing parameter ``.grad`` fields."""
    if not enabled:
        return None

    named_parameters = [
        (name, parameter)
        for name, parameter in (named_shared_parameters or [])
        if parameter.requires_grad
    ]
    if not named_parameters:
        raise ValueError("CE-anchor diagnostics require shared parameters.")

    parameters = [parameter for _, parameter in named_parameters]
    ce_grads = torch.autograd.grad(
        ce_loss,
        parameters,
        retain_graph=True,
        allow_unused=True,
    )
    anchor_grads = torch.autograd.grad(
        anchor_loss,
        parameters,
        retain_graph=True,
        allow_unused=True,
    )
    entries = [
        (name, parameter, ce_grad, anchor_grad)
        for (name, parameter), ce_grad, anchor_grad in zip(
            named_parameters, ce_grads, anchor_grads
        )
    ]

    all_metrics = _group_metrics(entries, mse_lambda, "shared")
    result = {
        "ce_grad_norm": all_metrics["shared_ce_grad_norm"],
        "anchor_grad_norm": all_metrics["shared_anchor_grad_norm"],
        "weighted_anchor_grad_norm": (
            abs(float(mse_lambda)) * all_metrics["shared_anchor_grad_norm"]
        ),
        "anchor_to_ce_grad_ratio": all_metrics[
            "shared_anchor_to_ce_grad_ratio"
        ],
        "ce_anchor_grad_cos": all_metrics["shared_ce_anchor_grad_cos"],
    }
    result["gradient_conflict"] = int(result["ce_anchor_grad_cos"] < 0.0)

    u_entries = [entry for entry in entries if _is_u_parameter(entry[0])]
    v_entries = [entry for entry in entries if _is_v_parameter(entry[0])]
    result.update(_group_metrics(u_entries, mse_lambda, "u", empty_as_nan=True))
    result.update(_group_metrics(v_entries, mse_lambda, "v", empty_as_nan=True))
    return result

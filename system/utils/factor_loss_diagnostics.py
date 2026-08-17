import math

import torch


EPS = 1e-12
VIRTUAL_FIELDS = [
    "virtual_ce_to_u_delta_anchor",
    "virtual_ce_to_v_delta_anchor",
    "virtual_anchor_to_u_delta_ce",
    "virtual_anchor_to_v_delta_ce",
    "virtual_ce_to_u_delta_ce",
    "virtual_ce_to_v_delta_ce",
    "virtual_anchor_to_u_delta_anchor",
    "virtual_anchor_to_v_delta_anchor",
]
DIAGNOSTIC_FIELDS = [
    "round",
    "client_id",
    "layer",
    "capacity",
    "anchor_coefficient",
    "regularization_coefficient",
    "u_lr",
    "v_lr",
    "ce_loss",
    "anchor_loss",
    "regularization_loss",
    "u_ce_grad_norm",
    "u_anchor_grad_norm",
    "u_reg_grad_norm",
    "v_ce_grad_norm",
    "v_anchor_grad_norm",
    "v_reg_grad_norm",
    "u_weighted_anchor_to_ce_ratio",
    "v_weighted_anchor_to_ce_ratio",
    "u_ce_anchor_cos",
    "v_ce_anchor_cos",
    "wpath_u_ce_norm",
    "wpath_u_anchor_norm",
    "wpath_v_ce_norm",
    "wpath_v_anchor_norm",
    "wpath_u_ce_anchor_cos",
    "wpath_v_ce_anchor_cos",
    *VIRTUAL_FIELDS,
]


def factor_kind(name):
    if name.endswith("weight_u") or name.endswith("conv_u"):
        return "u"
    if name.endswith("weight_v") or name.endswith("conv_v"):
        return "v"
    return None


def paired_v_name(u_name):
    if u_name.endswith("weight_u"):
        return f"{u_name[:-len('weight_u')]}weight_v"
    if u_name.endswith("conv_u"):
        return f"{u_name[:-len('conv_u')]}conv_v"
    raise ValueError(f"Not a recognized U-factor parameter: {u_name}")


def layer_name_from_u(u_name):
    for suffix in ("weight_u", "conv_u"):
        if u_name.endswith(suffix):
            return u_name[:-len(suffix)].rstrip(".")
    return u_name


def named_factor_parameters(model):
    return [
        (name, parameter)
        for name, parameter in model.named_parameters()
        if parameter.requires_grad and factor_kind(name) is not None
    ]


def _loss_gradients(loss, parameters, retain_graph=True):
    if loss is None or not getattr(loss, "requires_grad", False):
        return [None] * len(parameters)
    return list(
        torch.autograd.grad(
            loss,
            parameters,
            retain_graph=retain_graph,
            allow_unused=True,
        )
    )


def component_gradients(
    ce_loss,
    anchor_loss,
    regularization_loss,
    named_parameters,
):
    """Return detached loss-specific gradients without touching ``.grad``."""
    named_parameters = list(named_parameters)
    parameters = [parameter for _, parameter in named_parameters]
    ce_grads = _loss_gradients(ce_loss, parameters, retain_graph=True)
    anchor_grads = _loss_gradients(
        anchor_loss, parameters, retain_graph=True
    )
    reg_grads = _loss_gradients(
        regularization_loss, parameters, retain_graph=True
    )
    return {
        loss_name: {
            name: None if grad is None else grad.detach().clone()
            for (name, _), grad in zip(named_parameters, grads)
        }
        for loss_name, grads in (
            ("ce", ce_grads),
            ("anchor", anchor_grads),
            ("reg", reg_grads),
        )
    }


def _norm_sq(grads, names):
    total = None
    for name in names:
        grad = grads.get(name)
        if grad is None:
            continue
        value = torch.sum(grad.double() * grad.double())
        total = value if total is None else total + value
    return 0.0 if total is None else float(total.item())


def _dot(grads_a, grads_b, names):
    total = None
    for name in names:
        grad_a = grads_a.get(name)
        grad_b = grads_b.get(name)
        if grad_a is None or grad_b is None:
            continue
        value = torch.sum(grad_a.double() * grad_b.double())
        total = value if total is None else total + value
    return 0.0 if total is None else float(total.item())


def _group_metrics(gradients, names, prefix, anchor_coefficient):
    ce_sq = _norm_sq(gradients["ce"], names)
    anchor_sq = _norm_sq(gradients["anchor"], names)
    reg_sq = _norm_sq(gradients["reg"], names)
    ce_norm = math.sqrt(max(ce_sq, 0.0))
    anchor_norm = math.sqrt(max(anchor_sq, 0.0))
    reg_norm = math.sqrt(max(reg_sq, 0.0))
    cosine = _dot(gradients["ce"], gradients["anchor"], names) / (
        ce_norm * anchor_norm + EPS
    )
    cosine = max(-1.0, min(1.0, cosine)) if math.isfinite(cosine) else 0.0
    return {
        f"{prefix}_ce_grad_norm": ce_norm,
        f"{prefix}_anchor_grad_norm": anchor_norm,
        f"{prefix}_reg_grad_norm": reg_norm,
        f"{prefix}_weighted_anchor_to_ce_ratio": (
            abs(float(anchor_coefficient)) * anchor_norm / (ce_norm + EPS)
        ),
        f"{prefix}_ce_anchor_cos": cosine,
    }


def _product_pair_stats(
    left_a,
    right_a,
    left_b,
    right_b,
    max_chunk_elements=4_000_000,
):
    output_columns = int(right_a.shape[1])
    rows_per_chunk = max(
        1,
        min(
            int(left_a.shape[0]),
            max_chunk_elements // max(output_columns, 1),
        ),
    )
    norm_a_sq = left_a.new_zeros((), dtype=torch.float64)
    norm_b_sq = left_a.new_zeros((), dtype=torch.float64)
    dot = left_a.new_zeros((), dtype=torch.float64)
    for row_start in range(0, int(left_a.shape[0]), rows_per_chunk):
        row_end = row_start + rows_per_chunk
        product_a = left_a[row_start:row_end] @ right_a
        product_b = left_b[row_start:row_end] @ right_b
        product_a = product_a.double()
        product_b = product_b.double()
        norm_a_sq += torch.sum(product_a * product_a)
        norm_b_sq += torch.sum(product_b * product_b)
        dot += torch.sum(product_a * product_b)
    return (
        float(norm_a_sq.item()),
        float(norm_b_sq.item()),
        float(dot.item()),
    )


def _path_metrics(parameter_map, gradients, u_names):
    sums = {
        "u_ce_sq": 0.0,
        "u_anchor_sq": 0.0,
        "u_dot": 0.0,
        "v_ce_sq": 0.0,
        "v_anchor_sq": 0.0,
        "v_dot": 0.0,
    }
    for u_name in u_names:
        v_name = paired_v_name(u_name)
        if v_name not in parameter_map:
            raise RuntimeError(f"Missing paired V parameter {v_name}.")
        u = parameter_map[u_name].detach()
        v = parameter_map[v_name].detach()
        if u.ndim != 2 or v.ndim != 2 or u.shape[1] != v.shape[0]:
            raise ValueError(
                f"W-path diagnostics require 2-D factors, got "
                f"{u_name}={tuple(u.shape)}, {v_name}={tuple(v.shape)}."
            )
        u_ce = gradients["ce"].get(u_name)
        u_anchor = gradients["anchor"].get(u_name)
        v_ce = gradients["ce"].get(v_name)
        v_anchor = gradients["anchor"].get(v_name)
        u_ce = torch.zeros_like(u) if u_ce is None else u_ce
        u_anchor = torch.zeros_like(u) if u_anchor is None else u_anchor
        v_ce = torch.zeros_like(v) if v_ce is None else v_ce
        v_anchor = torch.zeros_like(v) if v_anchor is None else v_anchor

        u_stats = _product_pair_stats(u_ce, v, u_anchor, v)
        v_stats = _product_pair_stats(u, v_ce, u, v_anchor)
        sums["u_ce_sq"] += u_stats[0]
        sums["u_anchor_sq"] += u_stats[1]
        sums["u_dot"] += u_stats[2]
        sums["v_ce_sq"] += v_stats[0]
        sums["v_anchor_sq"] += v_stats[1]
        sums["v_dot"] += v_stats[2]

    u_ce_norm = math.sqrt(max(sums["u_ce_sq"], 0.0))
    u_anchor_norm = math.sqrt(max(sums["u_anchor_sq"], 0.0))
    v_ce_norm = math.sqrt(max(sums["v_ce_sq"], 0.0))
    v_anchor_norm = math.sqrt(max(sums["v_anchor_sq"], 0.0))
    u_cos = sums["u_dot"] / (u_ce_norm * u_anchor_norm + EPS)
    v_cos = sums["v_dot"] / (v_ce_norm * v_anchor_norm + EPS)
    u_cos = u_cos if math.isfinite(u_cos) else 0.0
    v_cos = v_cos if math.isfinite(v_cos) else 0.0
    return {
        "wpath_u_ce_norm": u_ce_norm,
        "wpath_u_anchor_norm": u_anchor_norm,
        "wpath_v_ce_norm": v_ce_norm,
        "wpath_v_anchor_norm": v_anchor_norm,
        "wpath_u_ce_anchor_cos": max(-1.0, min(1.0, u_cos)),
        "wpath_v_ce_anchor_cos": max(-1.0, min(1.0, v_cos)),
    }


def collect_factor_loss_diagnostics(
    model,
    ce_loss,
    anchor_loss,
    regularization_loss,
    anchor_coefficient,
    regularization_coefficient,
    round_number,
    client_id,
    capacity,
    u_lr,
    v_lr,
):
    named_parameters = named_factor_parameters(model)
    if not named_parameters:
        raise ValueError("No trainable U/V factors found for diagnostics.")
    parameter_map = dict(named_parameters)
    gradients = component_gradients(
        ce_loss,
        anchor_loss,
        regularization_loss,
        named_parameters,
    )
    u_names = sorted(
        name for name in parameter_map if factor_kind(name) == "u"
    )
    v_names = sorted(
        name for name in parameter_map if factor_kind(name) == "v"
    )
    common = {
        "round": int(round_number),
        "client_id": int(client_id),
        "capacity": float(capacity),
        "anchor_coefficient": float(anchor_coefficient),
        "regularization_coefficient": float(regularization_coefficient),
        "u_lr": float(u_lr),
        "v_lr": float(v_lr),
        "ce_loss": float(ce_loss.detach().item()),
        "anchor_loss": float(anchor_loss.detach().item()),
        "regularization_loss": (
            0.0
            if regularization_loss is None
            else float(regularization_loss.detach().item())
        ),
        **{field: math.nan for field in VIRTUAL_FIELDS},
    }

    rows = []
    overall = {
        **common,
        "layer": "__overall__",
        **_group_metrics(gradients, u_names, "u", anchor_coefficient),
        **_group_metrics(gradients, v_names, "v", anchor_coefficient),
        **_path_metrics(parameter_map, gradients, u_names),
    }
    rows.append(overall)
    for u_name in u_names:
        v_name = paired_v_name(u_name)
        if v_name not in parameter_map:
            continue
        rows.append(
            {
                **common,
                "layer": layer_name_from_u(u_name),
                **_group_metrics(
                    gradients, [u_name], "u", anchor_coefficient
                ),
                **_group_metrics(
                    gradients, [v_name], "v", anchor_coefficient
                ),
                **_path_metrics(parameter_map, gradients, [u_name]),
            }
        )
    return rows, gradients


def scaled_u_gradients(
    ce_loss,
    anchor_loss,
    regularization_loss,
    named_u_parameters,
    anchor_coefficient,
    regularization_coefficient,
    ce_scale,
    anchor_scale,
    reg_scale,
):
    named_u_parameters = list(named_u_parameters)
    gradients = component_gradients(
        ce_loss,
        anchor_loss,
        regularization_loss,
        named_u_parameters,
    )
    combined = {}
    for name, parameter in named_u_parameters:
        value = torch.zeros_like(parameter)
        ce_grad = gradients["ce"].get(name)
        anchor_grad = gradients["anchor"].get(name)
        reg_grad = gradients["reg"].get(name)
        if ce_grad is not None:
            value.add_(ce_grad, alpha=float(ce_scale))
        if anchor_grad is not None:
            value.add_(
                anchor_grad,
                alpha=float(anchor_coefficient) * float(anchor_scale),
            )
        if reg_grad is not None:
            value.add_(
                reg_grad,
                alpha=float(regularization_coefficient) * float(reg_scale),
            )
        combined[name] = value
    return combined

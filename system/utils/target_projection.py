"""Target-centric projection in the complete parameter space used by FedCLIP Avg."""

import math

import torch


EPS = 1e-12
MODES = ("avg", "target_only", "projection")


def _dot(left, right):
    # Reduce tensor by tensor, in double precision; never flatten the model.
    value = sum(
        (left[name].double() * right[name].double()).sum()
        for name in left
    )
    result = float(value.item())
    if not math.isfinite(result):
        raise ValueError("Non-finite model update encountered during projection.")
    return result


def _delta(parameters, reference):
    if parameters.keys() != reference.keys():
        raise ValueError("Recovered client/global parameter names do not match.")
    result = {}
    for name, initial in reference.items():
        if parameters[name].shape != initial.shape:
            raise ValueError(f"Recovered parameter shape mismatch: {name}")
        result[name] = parameters[name].detach().to(initial) - initial
    return result


def _cosine(dot, norm_sq, target_norm_sq):
    denominator = math.sqrt(norm_sq) * math.sqrt(target_norm_sq)
    # A zero vector has no direction; use a documented finite logging convention.
    return max(-1.0, min(1.0, dot / denominator)) if denominator else 0.0


@torch.no_grad()
def aggregate_target_updates(global_params, target_params, uploads, target_id, mode):
    """Consume (client_id, sample_weight, recovered_parameters) once.

    Inputs are read-only. Parameters must be the SAME named_parameters() set as
    baseline Avg, including the head. Buffers and client-only CLIP aligners are
    intentionally absent. Memory use is independent of the number of clients.
    Diagnostics describe hypothetical Avg and Projection in every mode.
    """
    if mode not in MODES:
        raise ValueError(f"Unknown target projection mode: {mode}")
    if not global_params:
        raise ValueError("Cannot aggregate an empty parameter set.")
    target = _delta(target_params, global_params)
    target_norm_sq = _dot(target, target)
    target_norm = math.sqrt(target_norm_sq)
    avg_delta = {name: torch.zeros_like(value) for name, value in target.items()}
    # Retain baseline's weighted-parameter arithmetic for an exact Avg control.
    avg_params = {name: torch.zeros_like(value) for name, value in target.items()}
    correction = 0.0
    removed_norm_sum = 0.0
    other_norm_sum = 0.0
    weight_sum = 0.0
    seen = set()
    rows = []
    for client_id, weight, parameters in uploads:
        if client_id in seen:
            raise ValueError(f"Duplicate uploaded client: {client_id}")
        if not math.isfinite(weight) or weight < 0:
            raise ValueError("Client sample weights must be finite and non-negative.")
        seen.add(client_id)
        weight_sum += weight
        delta = target if client_id == target_id else _delta(parameters, global_params)
        dot = _dot(delta, target)
        norm = math.sqrt(_dot(delta, delta))
        conflict = client_id != target_id and dot < 0.0
        coefficient = dot / (target_norm_sq + EPS) if conflict else 0.0
        removed_norm = abs(coefficient) * target_norm
        if client_id != target_id:
            other_norm_sum += norm
            removed_norm_sum += removed_norm
        correction += weight * coefficient
        for name in global_params:
            avg_delta[name].add_(delta[name] * weight)
            avg_params[name].add_(parameters[name].detach().to(avg_params[name]) * weight)
        rows.append({
            "client_id": int(client_id),
            "weight": float(weight),
            "is_target": int(client_id == target_id),
            "conflict": int(conflict),
            "dot_before": dot,
            "dot_after_projection": dot - coefficient * target_norm_sq,
            "delta_norm": norm,
            "removed_norm": removed_norm,
        })

    if target_id not in seen:
        raise ValueError(f"Target client {target_id} did not upload an update.")
    if not math.isclose(weight_sum, 1.0, rel_tol=0.0, abs_tol=1e-6):
        raise ValueError(f"Baseline sample weights must sum to one, got {weight_sum}.")

    avg_norm_sq = _dot(avg_delta, avg_delta)
    avg_dot = _dot(avg_delta, target)
    # Linearity: sum_j p_j (delta_j - c_j target) = G_avg - sum_j p_j c_j target.
    # Each c_j uses ONE dot product over the complete model, never layer-wise.
    for name in avg_delta:
        avg_delta[name].add_(target[name], alpha=-correction)
    proj_norm_sq = _dot(avg_delta, avg_delta)
    proj_dot = _dot(avg_delta, target)
    conflicts = sum(row["conflict"] for row in rows)
    metrics = {
        "conflict_client_ratio": conflicts / max(len(seen) - 1, 1),
        "removed_update_ratio": removed_norm_sum / (other_norm_sum + EPS),
        "avg_target_cos": _cosine(avg_dot, avg_norm_sq, target_norm_sq),
        "proj_target_cos": _cosine(proj_dot, proj_norm_sq, target_norm_sq),
        "target_delta_norm": target_norm,
        "avg_update_norm": math.sqrt(avg_norm_sq),
        "proj_update_norm": math.sqrt(proj_norm_sq),
        "conflict_client_count": conflicts,
        "uploaded_client_count": len(seen),
    }
    if mode == "avg":
        result = avg_params
    elif mode == "target_only":
        result = {name: value.detach().clone() for name, value in target_params.items()}
    else:
        result = {name: value.detach() + avg_delta[name] for name, value in global_params.items()}
    return result, metrics, rows

"""Causal full-W APA with fixed target mass and learned helper softmax logits."""

import math

import torch

from utils.target_projection import _delta, _dot


APA_LOGIT_LR = 0.01
TARGET_MASS = 0.05
HELPER_MASS = 0.95


def validate_apa_logit_lr(lr):
    if not math.isfinite(lr) or lr < 0:
        raise ValueError("apa_logit_lr must be finite and non-negative.")


def _vector(value, count):
    result = torch.as_tensor(value, dtype=torch.float64, device="cpu").detach().clone()
    if result.shape != (count,) or not torch.isfinite(result).all():
        raise ValueError("APA-Logit requires finite float64 helper vectors.")
    return result


@torch.no_grad()
def helper_probabilities(logits):
    logits = _vector(logits, len(logits))
    if not len(logits):
        raise ValueError("APA-Logit requires at least one helper.")
    q = torch.softmax(logits, dim=0)  # Subtracts max internally, in float64.
    if not torch.isfinite(q).all() or (q <= 0).any() or not math.isclose(q.sum().item(), 1., abs_tol=1e-14):
        # Never silently floor probabilities, prune helpers or reset to uniform.
        raise ValueError("APA-Logit softmax is non-finite or underflowed to zero.")
    return q


@torch.no_grad()
def centered_logit_gradient(raw_gradient, q):
    raw = _vector(raw_gradient, len(q))
    q = _vector(q, len(q))
    if (q <= 0).any() or not math.isclose(q.sum().item(), 1., rel_tol=0, abs_tol=1e-14):
        raise ValueError("APA-Logit helper probabilities must be positive and sum to one.")
    mean = torch.dot(q, raw)
    centered = raw - mean
    gradient = HELPER_MASS * q * centered
    if not torch.isfinite(centered).all() or not torch.isfinite(gradient).all():
        raise ValueError("Non-finite APA-Logit gradient.")
    return centered, gradient, mean.item()


@torch.no_grad()
def apa_logit_proxy_gradient(server_params, target_post, bases, client_ids, target_id, q):
    """Read only previous bases; return helper raw, centered and logit gradients."""
    residual = _delta(server_params, target_post)
    residual_sq = _dot(residual, residual)
    raw_by_id = {}
    for cid, basis in bases:
        if cid in raw_by_id or cid not in client_ids:
            raise ValueError("APA-Logit basis has duplicate or unexpected client IDs.")
        if basis.keys() != residual.keys() or any(basis[key].shape != residual[key].shape for key in residual):
            raise ValueError("APA-Logit basis must match the full-W parameter layout.")
        raw_by_id[cid] = _dot(basis, residual)
    if raw_by_id.keys() != set(client_ids):
        raise ValueError("APA-Logit requires every previous-round basis.")
    raw = torch.tensor([raw_by_id[cid] for cid in client_ids if cid != target_id], dtype=torch.float64)
    centered, gradient, mean = centered_logit_gradient(raw, q)
    return raw, centered, gradient, mean, .5 * residual_sq, math.sqrt(residual_sq)


@torch.no_grad()
def aggregate_apa_logit(server_params, target_post, uploads, target_id, samples, loop_round,
                        logits=None, bases=None, cache_basis=None, lr=APA_LOGIT_LR):
    validate_apa_logit_lr(lr)
    ids = sorted(samples)
    helpers = [cid for cid in ids if cid != target_id]
    if not helpers or target_id not in samples or loop_round < 0 or not server_params:
        raise ValueError("APA-Logit requires a target, helpers, model and non-negative round.")
    if any(not math.isfinite(p) or p < 0 for p in samples.values()) or not math.isclose(
            math.fsum(samples.values()), 1., rel_tol=0, abs_tol=1e-6):
        raise ValueError("Invalid sample weights.")
    raw = centered = gradient = gradient_mean = None
    if loop_round == 0:
        if logits is not None or bases is not None:
            raise ValueError("APA-Logit round 0 must start without previous state.")
        before = torch.zeros(len(helpers), dtype=torch.float64)
        updated = before.clone()
        q_before = helper_probabilities(before)
        residual = _delta(server_params, target_post)
        residual_sq = _dot(residual, residual)
        loss, norm = .5 * residual_sq, math.sqrt(residual_sq)
    else:
        if logits is None or bases is None:
            raise ValueError("APA-Logit update requires previous logits and bases.")
        before = _vector(logits, len(helpers))
        q_before = helper_probabilities(before)
        raw, centered, gradient, gradient_mean, loss, norm = apa_logit_proxy_gradient(
            server_params, target_post, bases, ids, target_id, q_before)
        updated = before - lr * gradient  # No momentum, clipping, recentering or scheduler.
    q = helper_probabilities(updated)
    helper_weights = HELPER_MASS * q
    if loop_round == 0 and len(helpers) == 19:
        # 0.95 * (1/19) differs from literal .05 by one float64 ULP.
        # Use the exact uniform representation to retain legacy Avg arithmetic.
        helper_weights.fill_(TARGET_MASS)
    if (helper_weights <= 0).any() or not math.isclose(helper_weights.sum().item(), HELPER_MASS, abs_tol=1e-14):
        raise ValueError("Invalid APA-Logit helper mass.")
    weights = {target_id: TARGET_MASS, **dict(zip(helpers, helper_weights.tolist()))}
    if not math.isclose(math.fsum(weights.values()), 1., rel_tol=0, abs_tol=1e-14):
        raise ValueError("Invalid APA-Logit total mass.")
    metrics = dict(apa_logit_lr=lr, apa_logit_momentum=0, apa_logit_update_enabled=int(loop_round > 0),
        apa_basis_loop_round=loop_round - 1 if loop_round else None,
        apa_proxy_loss=loss, apa_residual_norm=norm,
        apa_raw_gradient_norm=math.hypot(*raw.tolist()) if raw is not None else None,
        apa_centered_gradient_norm=math.hypot(*centered.tolist()) if centered is not None else None,
        apa_logit_gradient_norm=math.hypot(*gradient.tolist()) if gradient is not None else None,
        apa_helper_gradient_mean=gradient_mean, apa_logit_update_norm=math.hypot(*(updated - before).tolist()),
        target_weight=TARGET_MASS, helper_total_weight=math.fsum(helper_weights.tolist()),
        min_helper_weight=helper_weights.min().item(), max_helper_weight=helper_weights.max().item(),
        mean_helper_weight=helper_weights.mean().item(), std_helper_weight=helper_weights.std(unbiased=False).item(),
        effective_helper_count=1. / q.square().sum().item(), max_abs_logit=updated.abs().max().item(),
        logit_std=updated.std(unbiased=False).item(), uploaded_client_count=len(ids))
    if any(isinstance(value, float) and not math.isfinite(value) for value in metrics.values()):
        raise ValueError("Non-finite APA-Logit diagnostic; state was not committed.")
    result = {name: torch.zeros_like(value) for name, value in server_params.items()}
    seen = set()
    for cid, sample_weight, post in uploads:
        if cid in seen or cid not in samples or sample_weight != samples[cid]:
            raise ValueError("APA-Logit requires matching full-participation uploads.")
        seen.add(cid)
        if post.keys() != result.keys():
            raise ValueError("APA-Logit upload parameter names do not match.")
        for name in result:
            value = post[name].detach().to(result[name])
            if value.shape != result[name].shape or not torch.isfinite(value).all():
                raise ValueError(f"Invalid APA-Logit uploaded parameter: {name}")
            result[name].add_(value * weights[cid])
        if cache_basis is not None:
            cache_basis(cid, post)
    if seen != samples.keys() or any(not torch.isfinite(value).all() for value in result.values()):
        raise ValueError("Incomplete or non-finite APA-Logit aggregate.")
    index_by_id = {cid: index for index, cid in enumerate(helpers)}
    rows = []
    for cid in ids:
        index = index_by_id.get(cid)
        row = dict(client_id=cid, is_target=int(cid == target_id), sample_weight=samples[cid],
                   aggregation_weight=weights[cid], apa_weight=weights[cid])
        for name, values in (("apa_logit", updated), ("apa_q", q), ("apa_logit_before", before),
                             ("apa_q_before", q_before), ("apa_raw_grad", raw),
                             ("apa_centered_grad", centered), ("apa_logit_grad", gradient)):
            row[name] = values[index].item() if index is not None and values is not None else None
        rows.append(row)
    return result, metrics, rows, updated

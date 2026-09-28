"""Target-only APA surrogate with causal full-W bases and scalar momentum SGD."""

import math

import torch

from utils.target_projection import _delta, _dot


APA_SERVER_LR = 0.01
APA_MOMENTUM = 0.9
APA_SELF_WEIGHT = 0.5


def validate_apa_options(server_lr, momentum, self_weight):
    if not math.isfinite(server_lr) or server_lr < 0:
        raise ValueError("apa_server_lr must be finite and non-negative.")
    if not math.isfinite(momentum) or not 0 <= momentum < 1:
        raise ValueError("apa_momentum must be finite and in [0, 1).")
    if not math.isfinite(self_weight) or not 0 <= self_weight <= 1:
        raise ValueError("apa_self_weight must be finite and in [0, 1].")


def _vector(value, count, name):
    result = torch.as_tensor(value, dtype=torch.float64, device="cpu").detach().clone()
    if result.shape != (count,) or not torch.isfinite(result).all():
        raise ValueError(f"{name} must be a finite vector with {count} entries.")
    return result


def _simplex(weights):
    if (weights < 0).any() or (weights > 1).any() or not math.isclose(
            weights.sum().item(), 1.0, rel_tol=0, abs_tol=1e-6):
        raise ValueError("APA weights must be non-negative and sum to one.")


@torch.no_grad()
def apa_proxy_gradient(server_params, target_post, bases, client_ids):
    """d(0.5||server-target_post||^2)/da_j = <previous basis_j, residual>.

    The server is the pre-decomposition full-W mixture. Low-rank downlink is part
    of the client protocol, not a differentiable operation in this surrogate.
    """
    residual = _delta(server_params, target_post)
    residual_sq = _dot(residual, residual)
    gradients = {}
    for cid, basis in bases:
        if cid in gradients or cid not in client_ids:
            raise ValueError("APA basis has duplicate or unexpected client IDs.")
        if basis.keys() != residual.keys() or any(basis[name].shape != residual[name].shape for name in residual):
            raise ValueError("APA basis must match the recovered full-W parameter layout.")
        gradients[cid] = _dot(basis, residual)
    if gradients.keys() != set(client_ids):
        raise ValueError("APA requires every previous-round basis model.")
    return torch.tensor([gradients[cid] for cid in client_ids], dtype=torch.float64), 0.5 * residual_sq, math.sqrt(residual_sq)


@torch.no_grad()
def apa_weight_step(weights, velocity, gradient, target_index, server_lr=APA_SERVER_LR,
                    momentum=APA_MOMENTUM, self_weight=APA_SELF_WEIGHT):
    """Momentum SGD, then clip -> set pre-normalization self weight -> normalize."""
    validate_apa_options(server_lr, momentum, self_weight)
    count = len(weights)
    if not count or not 0 <= target_index < count:
        raise ValueError("APA target index is outside the weight vector.")
    weights = _vector(weights, count, "apa_weights")
    _simplex(weights)
    velocity = _vector(velocity, count, "apa_velocity")
    gradient = _vector(gradient, count, "apa_gradient")
    next_velocity = momentum * velocity + gradient
    raw_weights = weights - server_lr * next_velocity
    if not torch.isfinite(next_velocity).all() or not torch.isfinite(raw_weights).all():
        raise ValueError("Non-finite APA optimizer update; previous state was not modified.")
    updated = raw_weights.clamp(0.0, 1.0)
    updated[target_index] = self_weight
    total = updated.sum().item()
    fallback = total == 0.0
    if fallback:
        updated.fill_(1.0 / count)
    else:
        updated /= total
    _simplex(updated)
    return updated, next_velocity, raw_weights, fallback


@torch.no_grad()
def aggregate_apa(server_params, target_post, uploads, target_id, samples, loop_round,
                  weights=None, velocity=None, bases=None, cache_basis=None,
                  server_lr=APA_SERVER_LR, momentum=APA_MOMENTUM, self_weight=APA_SELF_WEIGHT):
    """Learn from prior bases before consuming any current upload, then cache new bases.

    cache_basis is server-owned disk storage. Only parameter tensors are cached;
    no additional client messages, training data, gradients or autograd graph.
    """
    validate_apa_options(server_lr, momentum, self_weight)
    ids = sorted(samples)
    if not ids or target_id not in samples or loop_round < 0 or not server_params:
        raise ValueError("APA requires clients, a target, a model and a non-negative round.")
    original = _vector([samples[cid] for cid in ids], len(ids), "sample_weights")
    _simplex(original)
    target_index = ids.index(target_id)
    if loop_round == 0:
        if weights is not None or velocity is not None or bases is not None:
            raise ValueError("APA round 0 must start fresh without a previous basis or optimizer state.")
        before = original
        updated = before.clone()
        next_velocity = torch.zeros_like(before)
        raw_weights, gradient, fallback = before.clone(), None, False
        residual = _delta(server_params, target_post)
        residual_sq = _dot(residual, residual)
        proxy_loss, residual_norm = 0.5 * residual_sq, math.sqrt(residual_sq)
    else:
        if weights is None or velocity is None or bases is None:
            raise ValueError("APA update requires previous-round bases, weights and velocity.")
        before = _vector(weights, len(ids), "apa_weights")
        gradient, proxy_loss, residual_norm = apa_proxy_gradient(server_params, target_post, bases, ids)
        updated, next_velocity, raw_weights, fallback = apa_weight_step(
            before, velocity, gradient, target_index, server_lr, momentum, self_weight)
    by_id = {cid: updated[index].item() for index, cid in enumerate(ids)}
    result = {name: torch.zeros_like(value) for name, value in server_params.items()}
    seen = set()
    # Same weighted-parameter arithmetic/order as legacy Avg, especially in round 0.
    for cid, sample_weight, post in uploads:
        if cid in seen or cid not in samples or sample_weight != samples[cid]:
            raise ValueError("APA upload IDs/sample weights do not match full participation.")
        seen.add(cid)
        if post.keys() != result.keys():
            raise ValueError("APA upload parameter names do not match full-W model.")
        for name in result:
            value = post[name].detach().to(result[name])
            if value.shape != result[name].shape or not torch.isfinite(value).all():
                raise ValueError(f"Invalid APA uploaded parameter: {name}")
            result[name].add_(value * by_id[cid])
        if cache_basis is not None:
            cache_basis(cid, post)
    if seen != samples.keys():
        raise ValueError("APA requires an upload from every client.")
    if any(not torch.isfinite(value).all() for value in result.values()):
        raise ValueError("Non-finite APA aggregate; previous optimizer state was not modified.")
    helpers = [by_id[cid] for cid in ids if cid != target_id]
    mass = math.fsum(helpers)
    mean = mass / len(helpers) if helpers else 0.0
    squared = math.fsum((weight / mass) ** 2 for weight in helpers) if mass else 0.0
    metrics = dict(apa_server_lr=server_lr, apa_momentum=momentum, apa_self_weight=self_weight,
        apa_proxy_loss=proxy_loss, apa_residual_norm=residual_norm,
        apa_grad_norm=gradient.norm().item() if gradient is not None else None,
        apa_weight_update_norm=(updated - before).norm().item(),
        apa_weight_update_enabled=int(loop_round > 0), apa_uniform_fallback_used=int(fallback),
        apa_basis_loop_round=loop_round - 1 if loop_round > 0 else None,
        target_weight=by_id[target_id], helper_total_weight=mass,
        min_helper_weight=min(helpers, default=0.0), max_helper_weight=max(helpers, default=0.0),
        mean_helper_weight=mean,
        std_helper_weight=math.sqrt(math.fsum((weight - mean) ** 2 for weight in helpers) / len(helpers)) if helpers else 0.0,
        effective_helper_count=1.0 / squared if squared else 0.0,
        uploaded_client_count=len(ids))
    rows = [dict(client_id=cid, is_target=int(cid == target_id), sample_weight=samples[cid],
                 apa_weight_before=before[index].item(), apa_raw_weight=raw_weights[index].item(),
                 apa_weight=updated[index].item(), apa_grad=gradient[index].item() if gradient is not None else None,
                 apa_velocity=next_velocity[index].item()) for index, cid in enumerate(ids)]
    return result, metrics, rows, updated, next_velocity

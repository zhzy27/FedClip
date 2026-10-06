"""C0 guidance-distance weighting; optional untouched full-global projection."""

import math
import random

import numpy as np
import torch

from utils.target_projection import EPS, aggregate_target_updates


DWA_MODES = ("dwa_soft", "dwa_soft_projection")
DWA_DISTANCE_EPS = 1e-12
TARGET_MASS = 0.05
HELPER_MASS = 0.95


def validate_distance_epsilon(epsilon):
    if not math.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("dwa_distance_eps must be finite and positive.")


@torch.no_grad()
def squared_parameter_distance(left, right):
    if not left or left.keys() != right.keys():
        raise ValueError("DWA full-W parameter names must match, including the head.")
    parts = []
    for name, value in right.items():
        if left[name].shape != value.shape:
            raise ValueError(f"DWA full-W parameter shape mismatch: {name}")
        delta = left[name].detach().to(device=value.device, dtype=torch.float64) - value.detach().double()
        parts.append(delta.square().sum(dtype=torch.float64).item())
    result = math.fsum(parts)
    if not math.isfinite(result) or result < 0:
        raise ValueError("Non-finite DWA squared distance.")
    return result


def guidance_distance_weights(distances, target_id=0, epsilon=DWA_DISTANCE_EPS):
    """Normalize inverse squared distances without squaring again or Softmax.

    Scale inverses by the smallest denominator. Ratios are at most one and
    remain valid even when 1/epsilon or distance+epsilon would overflow.
    """
    validate_distance_epsilon(epsilon)
    if not distances or target_id in distances:
        raise ValueError("DWA distance normalization requires helpers only.")
    if any(not math.isfinite(d) or d < 0 for d in distances.values()):
        raise ValueError("DWA helper squared distances must be finite and non-negative.")
    minimum = min(distances.values())
    scores = {}
    for cid, distance in distances.items():
        scale = max(distance, epsilon)
        scores[cid] = (minimum / scale + epsilon / scale) / (distance / scale + epsilon / scale)
    if any(not math.isfinite(score) or score <= 0 for score in scores.values()):
        raise ValueError("DWA inverse-distance normalization underflowed; no helper was discarded.")
    total = math.fsum(scores.values())
    q = {cid: score / total for cid, score in scores.items()}
    weights = {target_id: TARGET_MASS, **{cid: HELPER_MASS * prob for cid, prob in q.items()}}
    if any(not math.isfinite(weight) or weight <= 0 for weight in weights.values()):
        raise ValueError("Non-finite or underflowed DWA weight.")
    if not math.isclose(math.fsum(q.values()), 1., rel_tol=0, abs_tol=1e-14) or not math.isclose(
            math.fsum(weights.values()), 1., rel_tol=0, abs_tol=1e-14):
        raise ValueError("Invalid DWA aggregation mass.")
    helpers = [weights[cid] for cid in distances]
    mean = math.fsum(helpers) / len(helpers)
    summary = dict(target_weight=TARGET_MASS, helper_total_weight=math.fsum(helpers),
        min_helper_weight=min(helpers), max_helper_weight=max(helpers), mean_helper_weight=mean,
        std_helper_weight=math.sqrt(math.fsum((w - mean) ** 2 for w in helpers) / len(helpers)),
        effective_helper_count=1. / math.fsum(prob ** 2 for prob in q.values()),
        same_label_helper_weight=math.fsum(weights.get(cid, 0.) for cid in (1, 2, 3) if cid != target_id),
        cross_label_helper_weight=math.fsum(w for cid, w in weights.items() if cid >= 4 and cid != target_id))
    return weights, q, scores, summary


@torch.no_grad()
def aggregate_dwa(global_params, target_post, uploads_factory, guidance, target_id, samples,
                  loop_round, upload_rounds, mode, epsilon=DWA_DISTANCE_EPS):
    if mode not in DWA_MODES:
        raise ValueError(f"Unknown DWA mode: {mode}")
    validate_distance_epsilon(epsilon)
    if target_id not in samples or not global_params or loop_round < 0:
        raise ValueError("DWA requires a target, a model and a valid round.")
    if guidance["client_id"] != target_id or guidance["loop_round"] != loop_round or guidance["source_post_local_round"] != loop_round:
        raise ValueError("DWA guidance does not correspond to this target post-local round.")
    if upload_rounds.keys() != samples.keys() or any(round_id != loop_round for round_id in upload_rounds.values()):
        raise ValueError("DWA uploads do not correspond to the guidance round.")
    if any(not math.isfinite(p) or not 0 <= p <= 1 for p in samples.values()) or not math.isclose(
            math.fsum(samples.values()), 1., rel_tol=0, abs_tol=1e-6):
        raise ValueError("Invalid DWA sample-count weights.")
    guide = guidance["parameters"]
    guide_target_distance = squared_parameter_distance(guide, target_post)
    squared_parameter_distance(global_params, target_post)  # Validate full-W scope.
    distances, seen = {}, set()
    for cid, p, post in uploads_factory():
        if cid in seen or cid not in samples or p != samples[cid]:
            raise ValueError("DWA requires matching full-participation uploads.")
        seen.add(cid)
        if post.keys() != global_params.keys():
            raise ValueError("DWA uploaded full-W parameter names differ from server.")
        distance = squared_parameter_distance(guide, post)
        if cid != target_id:
            distances[cid] = distance
    if seen != samples.keys():
        raise ValueError("DWA is missing uploads for guidance scoring.")
    weights, q, scores, summary = guidance_distance_weights(distances, target_id, epsilon)
    # Reread existing server-side uploads, without advancing the normal RNG flow.
    seen = set()
    def weighted_uploads():
        for cid, p, post in uploads_factory():
            if cid in seen or cid not in samples or p != samples[cid]:
                raise ValueError("DWA upload IDs/weights changed between scoring and aggregation.")
            seen.add(cid)
            if post.keys() != global_params.keys() or any(post[name].shape != global_params[name].shape for name in global_params):
                raise ValueError("DWA upload layout changed between reads.")
            if any(not torch.isfinite(value).all() for value in post.values()):
                raise ValueError("Non-finite DWA upload.")
            yield cid, weights[cid], post
        if seen != samples.keys():
            raise ValueError("DWA is missing uploads for aggregation.")
    python_rng, numpy_rng = random.getstate(), np.random.get_state()
    try:
        with torch.random.fork_rng():
            if mode == "dwa_soft_projection":
                # Exactly the legacy global-delta kernel, with guidance weights.
                result, metrics, projection_rows = aggregate_target_updates(
                    global_params, target_post, weighted_uploads(), target_id, "projection")
                projection_lookup = {row["client_id"]: row for row in projection_rows}
            else:
                result = {name: torch.zeros_like(value) for name, value in global_params.items()}
                for cid, weight, post in weighted_uploads():
                    for name in result:
                        result[name].add_(post[name].detach().to(result[name]) * weight)
                metrics, projection_lookup = {}, {}
    finally:
        random.setstate(python_rng)
        np.random.set_state(numpy_rng)
    if any(not torch.isfinite(value).all() for value in result.values()):
        raise ValueError("Non-finite DWA aggregate.")
    metrics.update(summary, dwa_distance_eps=epsilon, projection_enabled=int(mode == "dwa_soft_projection"),
                   guidance_target_post_squared_distance=guide_target_distance)
    if mode == "dwa_soft_projection":
        metrics["projection_epsilon"] = EPS
    rows = []
    for cid in sorted(samples):
        entry = dict(client_id=cid, is_target=int(cid == target_id), sample_weight=samples[cid],
                     guidance_squared_distance=distances.get(cid), dwa_q=q.get(cid),
                     inverse_distance_scaled=scores.get(cid), aggregation_weight=weights[cid])
        if projection_lookup:
            p = projection_lookup[cid]
            entry.update(conflict=p["conflict"], dot_before_projection=p["dot_before"],
                         dot_after_projection=p["dot_after_projection"], delta_norm=p["delta_norm"],
                         removed_component_norm=p["removed_norm"])
        rows.append(entry)
    return result, metrics, rows

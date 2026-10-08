"""DWA ablations normalizing guidance inverse distances over target and helpers."""

import math
import random

import numpy as np
import torch

from utils.dwa_aggregation import DWA_DISTANCE_EPS, squared_parameter_distance, validate_distance_epsilon
from utils.target_projection import EPS, aggregate_target_updates


DWA_ADAPTIVE_MODES = ("dwa_adaptive_self", "dwa_adaptive_self_projection")


def adaptive_guidance_weights(distances, target_id=0, epsilon=DWA_DISTANCE_EPS):
    """All-client inverse squared distance; never derive helper mass as 1-alpha0."""
    validate_distance_epsilon(epsilon)
    if target_id not in distances or len(distances) < 2:
        raise ValueError("Adaptive DWA requires target and helper distances.")
    if any(not math.isfinite(d) or d < 0 for d in distances.values()):
        raise ValueError("Adaptive DWA squared distances must be finite and non-negative.")
    minimum = min(distances.values())
    scores = {}
    for cid, distance in distances.items():
        # Same inverse-distance rule/stability as fixed DWA, now including C0.
        scale = max(distance, epsilon)
        scores[cid] = (minimum / scale + epsilon / scale) / (distance / scale + epsilon / scale)
    if any(not math.isfinite(s) or s <= 0 for s in scores.values()):
        raise ValueError("Adaptive DWA inverse distance underflowed; no client was discarded.")
    denominator = math.fsum(scores.values())
    weights = {cid: score / denominator for cid, score in scores.items()}
    if any(not math.isfinite(w) or w <= 0 for w in weights.values()) or not math.isclose(
            math.fsum(weights.values()), 1., rel_tol=0, abs_tol=1e-14):
        raise ValueError("Invalid adaptive DWA aggregation weights.")
    helpers = {cid: w for cid, w in weights.items() if cid != target_id}
    mass = math.fsum(helpers.values())
    helper_q = {cid: w / mass for cid, w in helpers.items()}
    mean = mass / len(helpers)
    same = math.fsum(w for cid, w in helpers.items() if cid in (1, 2, 3))
    cross = math.fsum(w for cid, w in helpers.items() if cid >= 4)
    summary = dict(target_weight=weights[target_id], target_squared_distance=distances[target_id],
        helper_total_weight=mass, min_helper_weight=min(helpers.values()), max_helper_weight=max(helpers.values()),
        mean_helper_weight=mean,
        std_helper_weight=math.hypot(*(w - mean for w in helpers.values())) / math.sqrt(len(helpers)),
        effective_helper_count=1. / math.fsum(q ** 2 for q in helper_q.values()),
        effective_all_client_count=1. / math.fsum(w ** 2 for w in weights.values()),
        same_label_helper_weight=same, cross_label_helper_weight=cross,
        same_label_helper_share=same / mass, cross_label_helper_share=cross / mass)
    if any(not math.isfinite(v) for v in summary.values()):
        raise ValueError("Non-finite adaptive DWA diagnostic.")
    return weights, helper_q, scores, summary


def target_weight_phase(loop_round, global_rounds):
    planned = global_rounds + 1
    if planned < 1 or not 0 <= loop_round < planned:
        raise ValueError("Invalid adaptive DWA round for phase diagnostics.")
    return ("early", "mid", "late")[min(2, loop_round * 3 // planned)]


def target_weight_history_summary(history, global_rounds):
    """History-only diagnostics, with phases anchored to the planned inclusive run."""
    rows = [row for row in history if row.get("target_weight") is not None]
    summary = {}
    for prefix, selected in [("", rows)] + [
            (phase + "_", [row for row in rows if target_weight_phase(row["loop_round"], global_rounds) == phase])
            for phase in ("early", "mid", "late")]:
        values = [row["target_weight"] for row in selected]
        summary.update({prefix + "target_weight_count": len(values),
                        prefix + "target_weight_mean": math.fsum(values) / len(values) if values else None,
                        prefix + "target_weight_min": min(values) if values else None,
                        prefix + "target_weight_max": max(values) if values else None})
    for threshold, label in ((.5, "0_5"), (.8, "0_8"), (.95, "0_95")):
        summary[f"target_weight_gt_{label}_count"] = sum(row["target_weight"] > threshold for row in rows)
    return summary


@torch.no_grad()
def aggregate_dwa_adaptive(global_params, target_post, uploads_factory, guidance, target_id, samples,
                           loop_round, upload_rounds, mode, epsilon=DWA_DISTANCE_EPS):
    if mode not in DWA_ADAPTIVE_MODES:
        raise ValueError(f"Unknown adaptive DWA mode: {mode}")
    validate_distance_epsilon(epsilon)
    if target_id not in samples or not global_params or loop_round < 0:
        raise ValueError("Adaptive DWA requires a target, model and valid round.")
    if guidance["client_id"] != target_id or guidance["loop_round"] != loop_round or guidance["source_post_local_round"] != loop_round:
        raise ValueError("Adaptive DWA guidance does not correspond to this target post-local round.")
    if upload_rounds.keys() != samples.keys() or any(r != loop_round for r in upload_rounds.values()):
        raise ValueError("Adaptive DWA uploads do not correspond to the guidance round.")
    if any(not math.isfinite(p) or not 0 <= p <= 1 for p in samples.values()) or not math.isclose(
            math.fsum(samples.values()), 1., rel_tol=0, abs_tol=1e-6):
        raise ValueError("Invalid adaptive DWA sample-count weights.")
    guide = guidance["parameters"]
    target_distance = squared_parameter_distance(guide, target_post)
    squared_parameter_distance(global_params, target_post)
    distances, seen = {}, set()
    for cid, p, post in uploads_factory():
        if cid in seen or cid not in samples or p != samples[cid]:
            raise ValueError("Adaptive DWA requires matching full-participation uploads.")
        seen.add(cid)
        if post.keys() != global_params.keys():
            raise ValueError("Adaptive DWA uploaded full-W names differ from server.")
        distances[cid] = squared_parameter_distance(guide, post)
        if cid == target_id and squared_parameter_distance(target_post, post) != 0.:
            raise ValueError("Adaptive DWA target upload differs from the ordinary target post-model.")
    if seen != samples.keys():
        raise ValueError("Adaptive DWA is missing uploads for scoring.")
    weights, helper_q, scores, summary = adaptive_guidance_weights(distances, target_id, epsilon)
    seen = set()
    def weighted_uploads():
        for cid, p, post in uploads_factory():
            if cid in seen or cid not in samples or p != samples[cid]:
                raise ValueError("Adaptive DWA upload IDs/weights changed between reads.")
            seen.add(cid)
            if post.keys() != global_params.keys() or any(post[n].shape != global_params[n].shape for n in global_params):
                raise ValueError("Adaptive DWA upload layout changed between reads.")
            if any(not torch.isfinite(value).all() for value in post.values()):
                raise ValueError("Non-finite adaptive DWA upload.")
            yield cid, weights[cid], post
        if seen != samples.keys():
            raise ValueError("Adaptive DWA is missing uploads for aggregation.")
    python_rng, numpy_rng = random.getstate(), np.random.get_state()
    projected = mode == "dwa_adaptive_self_projection"
    try:
        with torch.random.fork_rng():
            if projected:
                result, metrics, projection_rows = aggregate_target_updates(
                    global_params, target_post, weighted_uploads(), target_id, "projection")
                lookup = {row["client_id"]: row for row in projection_rows}
            else:
                result = {name: torch.zeros_like(value) for name, value in global_params.items()}
                for cid, weight, post in weighted_uploads():
                    for name in result:
                        result[name].add_(post[name].detach().to(result[name]) * weight)
                metrics, lookup = {}, {}
    finally:
        random.setstate(python_rng)
        np.random.set_state(numpy_rng)
    if any(not torch.isfinite(value).all() for value in result.values()):
        raise ValueError("Non-finite adaptive DWA aggregate.")
    metrics.update(summary, dwa_distance_eps=epsilon, projection_enabled=int(projected),
                   guidance_target_post_squared_distance=target_distance)
    if projected:
        metrics["projection_epsilon"] = EPS
    rows = []
    for cid in sorted(samples):
        entry = dict(client_id=cid, is_target=int(cid == target_id), sample_weight=samples[cid],
            guidance_squared_distance=distances[cid], dwa_q=helper_q.get(cid), helper_q=helper_q.get(cid),
            all_client_q=weights[cid], inverse_distance_scaled=scores[cid], aggregation_weight=weights[cid])
        if lookup:
            p = lookup[cid]
            entry.update(conflict=p["conflict"], dot_before_projection=p["dot_before"],
                dot_after_projection=p["dot_after_projection"], delta_norm=p["delta_norm"],
                removed_component_norm=p["removed_norm"])
        rows.append(entry)
    return result, metrics, rows

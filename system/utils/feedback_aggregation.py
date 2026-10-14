"""First-stage full-W feedback aggregation; no change to any old kernel."""

import math

import torch

from utils.feedback_options import FEEDBACK_MODES, FEEDBACK_SELF_WEIGHT
from utils.target_projection import EPS, _cosine, _delta, _dot, aggregate_target_updates


def feedback_weights(samples, target_id, mode):
    if mode not in FEEDBACK_MODES or target_id not in samples:
        raise ValueError("Feedback requires a known mode and the ordinary target upload.")
    if (any(not math.isfinite(p) or not 0 <= p <= 1 for p in samples.values())
            or not math.isclose(math.fsum(samples.values()), 1., abs_tol=1e-6, rel_tol=0)):
        raise ValueError("Ordinary sample fractions must be finite, nonnegative and sum to one.")
    if mode == "feedback_avg":
        return dict(samples)  # Exact original Avg weights and accumulation order.
    helper_mass = math.fsum(p for cid, p in samples.items() if cid != target_id)
    if helper_mass <= 0:
        raise ValueError("Fixed .875 helper mass requires positive helper training samples.")
    return {cid: FEEDBACK_SELF_WEIGHT if cid == target_id else
            (1. - FEEDBACK_SELF_WEIGHT) * (p / helper_mass) for cid, p in samples.items()}


@torch.no_grad()
def aggregate_feedback_updates(global_params, target_post, uploads, samples, target_id, mode):
    """Reuse legacy weighted-parameter Avg or per-helper global-delta Projection.

    Uploads and reference are immutable *first-stage* recovered parameters. The
    legacy Projection computes each helper's own negative-dot coefficient before
    summing weighted corrections; it never projects a combined helper vector.
    """
    weights = feedback_weights(samples, target_id, mode)
    reference = _delta(target_post, global_params)
    reference_sq = _dot(reference, reference)
    del reference
    seen = set()
    def weighted_uploads():
        for cid, sample_weight, post in uploads:
            if cid in seen or cid not in samples or sample_weight != samples[cid]:
                raise ValueError("Feedback upload IDs/sample fractions do not match this round.")
            seen.add(cid)
            yield cid, weights[cid], post
        if seen != samples.keys():
            raise ValueError("Missing ordinary first-stage feedback upload.")
    enabled = mode == "feedback_projection_self"
    result, metrics, original_rows = aggregate_target_updates(global_params, target_post, weighted_uploads(),
                                                             target_id, "projection" if enabled else "avg")
    clients = []
    for row in original_rows:
        cid = row["client_id"]
        dot = row["dot_before"]
        coefficient = dot / (reference_sq + EPS) if enabled and row["conflict"] else 0.
        clients.append(dict(client_id=cid, is_target=row["is_target"], sample_weight=samples[cid],
            aggregation_weight=weights[cid], raw_conflict=row["conflict"], conflict=row["conflict"],
            projection_applied=int(enabled and row["conflict"]), projection_coefficient=coefficient,
            delta_norm=row["delta_norm"], dot_before_projection=dot,
            cosine_before_projection=_cosine(dot, row["delta_norm"] ** 2, reference_sq),
            dot_after_projection=row["dot_after_projection"] if enabled else dot,
            removed_component_norm=row["removed_norm"] if enabled else 0.))
    # Avg's existing kernel reports hypothetical Projection. Keep raw conflicts,
    # but expose actual removal separately and explicitly zero it in Avg modes.
    if not enabled:
        metrics.update(removed_update_ratio=0., proj_target_cos=metrics["avg_target_cos"],
                       proj_update_norm=metrics["avg_update_norm"])
    helpers = [weights[cid] for cid in weights if cid != target_id]
    mass = math.fsum(helpers)
    metrics.update(projection_enabled=int(enabled), target_weight=weights[target_id], helper_total_weight=mass,
        self_weight_is_fixed=int(mode != "feedback_avg"), aggregation_weight_sum=math.fsum(weights.values()),
        effective_all_client_count=1. / math.fsum(w ** 2 for w in weights.values()),
        effective_helper_count=1. / math.fsum((w / mass) ** 2 for w in helpers) if mass else 0.,
        min_helper_weight=min(helpers, default=0.), max_helper_weight=max(helpers, default=0.),
        projected_client_count=sum(row["projection_applied"] for row in clients))
    return result, metrics, sorted(clients, key=lambda row: row["client_id"])


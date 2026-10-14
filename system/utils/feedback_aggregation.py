"""Helper-only full-W Avg and optional immutable server-anchor retention."""

import math

import torch

from utils.feedback_options import feedback_anchor_weight
from utils.target_projection import _delta, _dot


def feedback_weights(helper_samples, target_id, mode, anchor_weight=None):
    """C0 has no upload here: keys contain only helpers, values are train counts."""
    anchor = feedback_anchor_weight(mode, anchor_weight)
    if target_id in helper_samples or not helper_samples:
        raise ValueError("Feedback aggregation requires only helper sample counts, never a C0 upload.")
    if any(not math.isfinite(n) or n < 0 for n in helper_samples.values()):
        raise ValueError("Helper training sample counts must be finite and nonnegative.")
    total = math.fsum(helper_samples.values())
    if total <= 0 or not math.isfinite(total):
        raise ValueError("Helpers need positive finite total training sample mass.")
    q = {cid: n / total for cid, n in helper_samples.items()}
    weights = {cid: (1. - anchor) * share for cid, share in q.items()}
    return weights, q, anchor


@torch.no_grad()
def aggregate_feedback_updates(server_start, uploads, helper_samples, target_id, mode, anchor_weight=None):
    """H = alpha*S_t + (1-alpha)*Avg(helpers); no target reference or Projection.

    Use weighted-parameter Avg in upload order. Explicit alpha=0 follows exactly
    the pure-Avg path; zero helper deltas return the immutable starting parameters.
    Uploads are (helper_id, raw_train_count, recovered_full_W_named_parameters).
    """
    weights, q, anchor = feedback_weights(helper_samples, target_id, mode, anchor_weight)
    if not server_start:
        raise ValueError("Feedback requires nonempty starting server parameters.")
    _dot(server_start, server_start)
    helper_average = {name: torch.zeros_like(p) for name, p in server_start.items()}
    rows, seen, all_zero = [], set(), True
    for cid, count, post in uploads:
        if cid in seen or cid not in helper_samples or count != helper_samples[cid]:
            raise ValueError("Feedback helper IDs/sample counts do not match this round.")
        seen.add(cid)
        delta = _delta(post, server_start)
        norm_sq = _dot(delta, delta)
        all_zero = all_zero and norm_sq == 0
        for name, value in helper_average.items():
            value.add_(post[name].detach().to(value) * q[cid])
        rows.append(dict(client_id=cid, train_samples=count, helper_sample_weight=q[cid],
            aggregation_weight=weights[cid], delta_norm=math.sqrt(norm_sq),
            projection_applied=0, projection_coefficient=0., removed_component_norm=0.))
    if seen != helper_samples.keys():
        raise ValueError("Missing helper upload; C0 must not be read or uploaded in this stage.")
    if all_zero or anchor == 1.:
        result = {name: p.detach().clone() for name, p in server_start.items()}
    elif anchor == 0.:
        result = helper_average
    else:
        result = {name: server_start[name].detach() * anchor + p * (1. - anchor)
                  for name, p in helper_average.items()}
    helper_delta = _delta(helper_average, server_start)
    intermediate_delta = _delta(result, server_start)
    mass = math.fsum(weights.values())
    metrics = dict(projection_enabled=0, server_anchor_weight=anchor,
        helper_total_weight=mass, aggregation_weight_sum=anchor + mass,
        helper_upload_count=len(rows), target_upload_in_helper_stage=0,
        helper_average_vs_server_start_delta_norm=0. if all_zero else math.sqrt(_dot(helper_delta, helper_delta)),
        intermediate_vs_server_start_delta_norm=math.sqrt(_dot(intermediate_delta, intermediate_delta)),
        effective_helper_count=1. / math.fsum((w / mass) ** 2 for w in weights.values()) if mass else 0.,
        effective_aggregate_component_count=1. / (anchor ** 2 + math.fsum(w ** 2 for w in weights.values())),
        min_helper_weight=min(weights.values()), max_helper_weight=max(weights.values()))
    return result, metrics, sorted(rows, key=lambda row: row["client_id"])

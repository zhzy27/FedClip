"""Positive-only continuous weighting that preserves each layer's helper mass."""

import math
import random

import numpy as np
import torch

from utils.layer_mask import aggregate_layer_mask
from utils.target_projection import _delta


WEIGHTING_MODES = ("layer_softmax", "layer_relu")
SOFTMAX_TAU = 0.2


def positive_helper_weights(client_rows, layer_rows, layer_names, target_id, mode):
    """Return scalar weights only; the target is excluded from every denominator.

    ReLU uses exact mass normalization rather than adding epsilon to a positive
    denominator, which would shrink P. Scaling scores first handles tiny positive
    cosines without losing mass. Empty/zero-mass sets map to all-zero weights.
    """
    if mode not in WEIGHTING_MODES:
        raise ValueError(f"Unknown continuous weighting mode: {mode}")
    samples = {row["client_id"]: row["weight"] for row in client_rows}
    lookup = {(row["client_id"], row["layer"]): row for row in layer_rows}
    weights = {(cid, layer): 0.0 for cid in samples for layer in layer_names}
    summaries = []
    for layer in layer_names:
        positive = [cid for cid in samples if cid != target_id
                    and lookup[cid, layer]["layer_local_delta_cos"] > 0.0]
        cosines = {cid: lookup[cid, layer]["layer_local_delta_cos"] for cid in positive}
        mass = math.fsum(samples[cid] for cid in positive)
        if mass > 0.0:
            maximum = max(cosines.values())
            if mode == "layer_softmax":
                scores = {cid: samples[cid] * math.exp((cosines[cid] - maximum) / SOFTMAX_TAU)
                          for cid in positive}
            else:
                scores = {cid: samples[cid] * (cosines[cid] / maximum) for cid in positive}
            denominator = math.fsum(scores.values())
            for cid in positive:
                weights[cid, layer] = mass * (scores[cid] / denominator)
        positive_weights = [weights[cid, layer] for cid in positive if weights[cid, layer] > 0.0]
        squared_shares = math.fsum((weight / mass) ** 2 for weight in positive_weights) if mass else 0.0
        final_mass = math.fsum(weights[cid, layer] for cid in samples)
        summaries.append({
            "layer": layer, "positive_clients": len(positive),
            "original_weight_mass": mass, "final_weight_mass": final_mass,
            "mass_error": final_mass - mass,
            "min_positive_cos": min(cosines.values(), default=0.0),
            "max_positive_cos": max(cosines.values(), default=0.0),
            "mean_positive_cos": math.fsum(cosines.values()) / len(positive) if positive else 0.0,
            "max_final_weight": max(positive_weights, default=0.0),
            "min_positive_final_weight": min(positive_weights, default=0.0),
            "effective_helper_count": 1.0 / squared_shares if squared_shares else 0.0,
        })
    return weights, summaries


@torch.no_grad()
def aggregate_layer_weighting(global_params, target_post, target_pre, uploads_factory, target_id, groups, mode):
    """Two server-side reads; no extra client training/communication or budget.

    First reuse the original mask's diagnostics verbatim. After scalar weights are
    known, reread existing checkpoints one at a time instead of retaining all full
    models. Protect RNG during the extra recovery pass to preserve training streams.
    """
    if mode not in WEIGHTING_MODES:
        raise ValueError(f"Unknown continuous weighting mode: {mode}")
    result, metrics, clients, layers, matrices = aggregate_layer_mask(
        global_params, target_post, target_pre, uploads_factory(), target_id, groups,
    )
    weights, summaries = positive_helper_weights(clients, layers, list(groups), target_id, mode)
    for name, value in result.items():
        value.copy_(target_post[name])
    expected_samples = {row["client_id"]: row["weight"] for row in clients}
    seen = set()
    python_rng, numpy_rng = random.getstate(), np.random.get_state()
    try:
        with torch.random.fork_rng():
            for cid, weight, post, pre in uploads_factory():
                if cid in seen or cid not in expected_samples or weight != expected_samples[cid]:
                    raise ValueError("Continuous weighting upload IDs/weights changed between server reads.")
                seen.add(cid)
                if cid == target_id:
                    continue
                local = _delta(post, pre)
                for layer, names in groups.items():
                    alpha = weights[cid, layer]
                    if alpha > 0.0:
                        for name in names:
                            result[name].add_(local[name].to(result[name]), alpha=alpha)
    finally:
        random.setstate(python_rng)
        np.random.set_state(numpy_rng)
    if seen != set(expected_samples):
        raise ValueError("Continuous weighting is missing uploads on the second server read.")
    for row in layers:
        row["final_weight"] = weights[row["client_id"], row["layer"]]
        row["anchor_coefficient"] = int(row["client_id"] == target_id)
        row["aggregation_role"] = "anchor" if row["client_id"] == target_id else "helper"
    metrics.update({
        "mean_effective_helper_count": math.fsum(row["effective_helper_count"] for row in summaries) / len(summaries),
        "max_weight_mass_error": max(abs(row["mass_error"]) for row in summaries),
    })
    matrices.update({
        "weighting_mode": mode,
        "weighting_normalization": "preserve_positive_helper_sample_weight_mass",
        "final_weight_matrix": [["anchor" if cid == target_id else weights[cid, layer]
                                 for layer in groups] for cid in matrices["client_ids"]],
        "helper_weight_matrix": [[weights[cid, layer] for layer in groups] for cid in matrices["client_ids"]],
        "original_weight_mass": [row["original_weight_mass"] for row in summaries],
        "effective_helper_count": [row["effective_helper_count"] for row in summaries],
        "weight_summary": summaries,
    })
    if mode == "layer_softmax":
        matrices["softmax_tau"] = SOFTMAX_TAU
    return result, metrics, clients, layers, matrices


def print_layer_weighting_diagnostics(round_number, matrices):
    label = "LayerSoftmax" if matrices["weighting_mode"] == "layer_softmax" else "LayerReLU"
    prefix = f"[{label}][Round {round_number}]"
    width = max(14, max(map(len, matrices["layer_names"])) + 2)
    print(f"\n{prefix} Aggregation weight matrix")
    print(" " * 12 + "".join(f"{layer:>{width}}" for layer in matrices["layer_names"]))
    for cid, values in zip(matrices["client_ids"], matrices["final_weight_matrix"]):
        cells = [value if isinstance(value, str) else f"{value:.6e}" for value in values]
        print(f"{'Client ' + str(cid):<12}" + "".join(f"{value:>{width}}" for value in cells))
    print(f"\n{prefix} Positive-helper weight summary")
    for row in matrices["weight_summary"]:
        print(f"{row['layer']}: " + " ".join(
            f"{key}={value:.8g}" if isinstance(value, float) else f"{key}={value}"
            for key, value in row.items() if key != "layer"
        ))

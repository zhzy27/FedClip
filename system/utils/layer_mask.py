"""Target-anchored hard masks using pure local updates in recovered full-W space."""

import math

import torch

from utils.target_projection import EPS, _delta, _dot


def logical_layer_groups(parameters):
    """Group each recovered module's weight/bias together, without alias duplication."""
    groups = {}
    for name in parameters:
        prefix = name.rsplit(".", 1)[0] if "." in name else "<root>"
        groups.setdefault(prefix, []).append(name)
    if not groups:
        raise ValueError("Cannot mask an empty parameter set.")
    return groups


def _cosine(dot, norm, target_norm):
    zero_norm = norm < EPS or target_norm < EPS
    cosine = 0.0 if zero_norm else dot / (norm * target_norm + EPS)
    return max(-1.0, min(1.0, cosine)), zero_norm


def _norm(parameters):
    return math.sqrt(_dot(parameters, parameters))


@torch.no_grad()
def aggregate_layer_mask(global_params, target_post, target_pre, uploads, target_id, groups):
    """Consume (client_id, weight, post_full_parameters, pre_full_parameters).

    No data or client checkpoint is modified. The target post-local model is the
    unweighted anchor; only positive-cosine helper layers contribute p_i * D_i.
    Diagnostics use all clients in the weighted masked-norm denominator, including
    the target, whose mask is always one even when its local norm is zero.
    """
    if groups != logical_layer_groups(global_params):
        raise ValueError("Layer groups must cover the recovered model by module prefix.")
    target_local = _delta(target_post, target_pre)
    target_old = _delta(target_post, global_params)
    target_norm = _norm(target_local)
    target_old_norm = _norm(target_old)
    target_layers = {layer: {name: target_local[name] for name in names}
                     for layer, names in groups.items()}
    target_layer_norms = {layer: _norm(values) for layer, values in target_layers.items()}
    result = {name: value.detach().clone().to(global_params[name])
              for name, value in target_post.items()}
    client_rows, layer_rows = [], []
    seen, weight_sum = set(), 0.0
    layer_totals = {layer: {"negative_count": 0, "masked_count": 0, "zero_norm_count": 0,
                            "weighted_norm": 0.0, "weighted_masked_norm": 0.0}
                    for layer in groups}

    for client_id, weight, post, pre in uploads:
        if client_id in seen:
            raise ValueError(f"Duplicate uploaded client: {client_id}")
        if not math.isfinite(weight) or weight < 0:
            raise ValueError("Client sample weights must be finite and non-negative.")
        seen.add(client_id)
        weight_sum += weight
        local = _delta(post, pre)
        old = _delta(post, global_params)
        local_norm, old_norm = _norm(local), _norm(old)
        full_cosine, full_zero = _cosine(_dot(local, target_local), local_norm, target_norm)
        old_cosine, old_zero = _cosine(_dot(old, target_old), old_norm, target_old_norm)
        client_rows.append({
            "client_id": int(client_id), "weight": float(weight),
            "is_target": int(client_id == target_id),
            "old_global_delta_cos": old_cosine, "full_local_delta_cos": full_cosine,
            "local_update_norm": local_norm, "old_global_delta_norm": old_norm,
            "full_local_zero_norm": full_zero, "old_global_zero_norm": old_zero,
        })
        for layer, names in groups.items():
            values = {name: local[name] for name in names}
            norm = _norm(values)
            dot = _dot(values, target_layers[layer])
            cosine, zero_norm = _cosine(dot, norm, target_layer_norms[layer])
            is_target = client_id == target_id
            if is_target:
                cosine = 1.0
            mask = int(is_target or cosine > 0.0)
            if not is_target and mask:
                for name in names:
                    result[name].add_(local[name].to(result[name]), alpha=weight)
            total = layer_totals[layer]
            total["negative_count"] += int(not is_target and cosine < 0.0)
            total["masked_count"] += int(not is_target and not mask)
            total["zero_norm_count"] += int(not is_target and zero_norm)
            total["weighted_norm"] += weight * norm
            total["weighted_masked_norm"] += weight * (1 - mask) * norm
            layer_rows.append({
                "client_id": int(client_id), "layer": layer, "weight": float(weight),
                "layer_local_delta_cos": cosine, "mask": mask, "zero_norm": zero_norm,
                "local_update_norm": norm, "target_local_update_norm": target_layer_norms[layer],
                "local_dot": dot, "weighted_masked_norm": weight * (1 - mask) * norm,
            })

    if target_id not in seen:
        raise ValueError(f"Target client {target_id} did not upload an update.")
    if not math.isclose(weight_sum, 1.0, rel_tol=0.0, abs_tol=1e-6):
        raise ValueError(f"Baseline sample weights must sum to one, got {weight_sum}.")
    client_rows.sort(key=lambda row: row["client_id"])
    client_ids, layer_names = sorted(seen), list(groups)
    layer_lookup = {(row["client_id"], row["layer"]): row for row in layer_rows}
    layer_rows = [layer_lookup[cid, layer] for cid in client_ids for layer in layer_names]
    summaries = []
    helpers = len(seen) - 1
    for layer, total in layer_totals.items():
        summaries.append({
            "layer": layer, **total, "helper_count": helpers,
            "conflict_ratio": total["negative_count"] / max(helpers, 1),
            "masked_entry_ratio": total["masked_count"] / max(helpers, 1),
            "masked_norm_ratio": total["weighted_masked_norm"] / (total["weighted_norm"] + EPS),
        })
    negative_entries = sum(row["negative_count"] for row in summaries)
    masked_entries = sum(row["masked_count"] for row in summaries)
    entries = helpers * len(groups)
    metrics = {
        "total_negative_entries": negative_entries,
        "total_helper_layer_entries": entries,
        "layer_conflict_ratio": negative_entries / max(entries, 1),
        "total_masked_entries": masked_entries,
        "masked_entry_ratio": masked_entries / max(entries, 1),
        "masked_update_ratio": sum(row["weighted_masked_norm"] for row in summaries) /
                               (sum(row["weighted_norm"] for row in summaries) + EPS),
        "target_local_update_norm": target_norm,
        "uploaded_client_count": len(seen), "layer_count": len(groups),
    }
    matrices = {
        "target_client_id": int(target_id), "client_ids": client_ids,
        "layer_names": layer_names, "layer_groups": groups,
        "cosine_matrix": [[layer_lookup[cid, layer]["layer_local_delta_cos"]
                           for layer in layer_names] for cid in client_ids],
        "mask_matrix": [[layer_lookup[cid, layer]["mask"]
                         for layer in layer_names] for cid in client_ids],
        "layer_update_norm_matrix": [[layer_lookup[cid, layer]["local_update_norm"]
                                      for layer in layer_names] for cid in client_ids],
        "zero_norm_matrix": [[layer_lookup[cid, layer]["zero_norm"]
                              for layer in layer_names] for cid in client_ids],
        "full_local_cosine": [row["full_local_delta_cos"] for row in client_rows],
        "old_global_cosine": [row["old_global_delta_cos"] for row in client_rows],
        "local_update_norm": [row["local_update_norm"] for row in client_rows],
        "layer_conflict_ratio": [row["conflict_ratio"] for row in summaries],
        "overall_layer_conflict_ratio": metrics["layer_conflict_ratio"],
        "masked_update_ratio": metrics["masked_update_ratio"],
        "layer_summary": summaries,
    }
    return result, metrics, client_rows, layer_rows, matrices


def print_layer_mask_diagnostics(round_number, matrices, client_rows, metrics):
    """Print every client/layer cell, without NumPy/Pandas truncation."""
    prefix = f"[LayerMask][Round {round_number}]"
    layers = matrices["layer_names"]
    width = max(14, max(map(len, layers)) + 2)

    def print_matrix(title, key, formatter):
        print(f"\n{prefix} {title}")
        print(" " * 12 + "".join(f"{layer:>{width}}" for layer in layers))
        for cid, values in zip(matrices["client_ids"], matrices[key]):
            print(f"{'Client ' + str(cid):<12}" + "".join(f"{formatter(v):>{width}}" for v in values))

    print_matrix(f"Local-update cosine matrix vs Client {matrices['target_client_id']}",
                 "cosine_matrix", lambda v: f"{v:.4f}")
    print_matrix("Mask matrix", "mask_matrix", str)
    print_matrix("Layer local-update norm matrix", "layer_update_norm_matrix", lambda v: f"{v:.6e}")
    print_matrix("Zero-norm matrix (either helper or target norm < eps)", "zero_norm_matrix", str)
    print(f"\n{prefix} Full-model diagnostics")
    for row in client_rows:
        print(f"Client {row['client_id']}: old_global_delta_cos={row['old_global_delta_cos']:.6f} "
              f"full_local_delta_cos={row['full_local_delta_cos']:.6f} "
              f"local_update_norm={row['local_update_norm']:.6e} "
              f"zero_norm={row['full_local_zero_norm']}")
    print(f"\n{prefix} Conflict summary (negative entries; zeros are separately masked)")
    for row in matrices["layer_summary"]:
        print(f"{row['layer']}: conflict={row['negative_count']}/{row['helper_count']} "
              f"ratio={row['conflict_ratio']:.4f} masked={row['masked_count']}/{row['helper_count']} "
              f"zero_norm={row['zero_norm_count']} masked_norm_ratio={row['masked_norm_ratio']:.6f}")
    print(f"total_negative_entries = {metrics['total_negative_entries']} / {metrics['total_helper_layer_entries']}")
    print(f"layer_conflict_ratio = {metrics['layer_conflict_ratio']:.6f}")
    print(f"masked_update_ratio = {metrics['masked_update_ratio']:.6f}")

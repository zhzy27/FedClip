"""Independent Avg-based projection ablations; legacy projection is unchanged."""

import math
import random

import numpy as np
import torch

from utils.target_projection import EPS, _cosine, _delta, _dot, aggregate_target_updates


LOCAL_PROJECTION_MODES = ("projection_local", "layer_projection_local")
LAYER_PROJECTION_MODES = ("layer_projection_global", "layer_projection_local")
SOURCE_MODES = ("projection_same_label", "projection_cross_label")
PROJECTION_WEIGHTING_MODES = ("projection_softmax", "projection_relu")
PROJECTION_SOFTMAX_TAU = 0.2
PROJECTION_VARIANT_MODES = ("projection_local", *LAYER_PROJECTION_MODES, *SOURCE_MODES,
                            *PROJECTION_WEIGHTING_MODES)


def projection_similarity_weights(rows, target_id, mode):
    """Pure similarity over all helpers, with fixed target and helper mass.

    Rows contain original sample_weight and cosine_before_projection. ReLU's
    all-nonpositive fallback returns the original sample weights verbatim.
    """
    if mode not in PROJECTION_WEIGHTING_MODES:
        raise ValueError(f"Unknown projection weighting mode: {mode}")
    samples = {row["client_id"]: row["sample_weight"] for row in rows}
    if len(samples) != len(rows) or target_id not in samples:
        raise ValueError("Unique upload IDs and a target upload are required.")
    if any(not math.isfinite(p) or not 0 <= p <= 1 for p in samples.values()) or not math.isclose(
            math.fsum(samples.values()), 1.0, rel_tol=0, abs_tol=1e-6):
        raise ValueError("Sample weights must be finite, non-negative and sum to one.")
    cosines = {row["client_id"]: row["cosine_before_projection"] for row in rows
               if row["client_id"] != target_id}
    if any(not math.isfinite(cosine) or not -1 <= cosine <= 1 for cosine in cosines.values()):
        raise ValueError("Helper cosines must be finite and in [-1, 1].")
    mass = 1.0 - samples[target_id]
    shift = max(cosines.values(), default=0.0)
    if mode == "projection_softmax":
        scores = {cid: math.exp((cosine - shift) / PROJECTION_SOFTMAX_TAU)
                  for cid, cosine in cosines.items()}
    else:
        scores = {cid: max(cosine, 0.0) for cid, cosine in cosines.items()}
    maximum = max(scores.values(), default=0.0)
    fallback = mode == "projection_relu" and maximum == 0.0
    if fallback:
        weights = dict(samples)
    else:
        # Scaling also protects ReLU normalization for tiny positive cosines.
        scaled = {cid: score / maximum for cid, score in scores.items()} if maximum else {}
        denominator = math.fsum(scaled.values())
        weights = {cid: mass * (score / denominator) for cid, score in scaled.items()}
        weights[target_id] = samples[target_id]
    helpers = [weights[cid] for cid in cosines]
    squared_shares = math.fsum((weight / mass) ** 2 for weight in helpers) if mass else 0.0
    summary = dict(target_weight=weights[target_id], helper_total_weight=math.fsum(helpers),
                   min_helper_weight=min(helpers, default=0.0), max_helper_weight=max(helpers, default=0.0),
                   effective_helper_count=1.0 / squared_shares if squared_shares else 0.0,
                   helper_weight_mass_error=math.fsum(helpers) - mass)
    if mode == "projection_softmax":
        summary.update(temperature=PROJECTION_SOFTMAX_TAU, softmax_shift_max=shift)
    else:
        summary["relu_fallback_used"] = int(fallback)
    return weights, scores, summary


@torch.no_grad()
def aggregate_projection_weighting(global_params, target_post, uploads_factory, target_id, mode):
    """Reuse the untouched legacy projection kernel, changing only its weights.

    First obtain pre-projection diagnostics with original sample weights. Then
    reread existing server-side uploads to aggregate with similarity weights.
    The second recovery pass restores RNG and never requests extra communication.
    """
    if mode not in PROJECTION_WEIGHTING_MODES:
        raise ValueError(f"Unknown projection weighting mode: {mode}")
    preliminary, _, original_rows = aggregate_target_updates(
        global_params, target_post, uploads_factory(), target_id, "projection")
    del preliminary
    target = _delta(target_post, global_params)
    target_sq = _dot(target, target)
    for row in original_rows:
        row["sample_weight"] = row["weight"]
        row["cosine_before_projection"] = _cosine(row["dot_before"], row["delta_norm"] ** 2, target_sq)
    weights, scores, summary = projection_similarity_weights(original_rows, target_id, mode)
    lookup = {row["client_id"]: row for row in original_rows}
    projected_norms, seen = {}, set()

    def weighted_uploads():
        for cid, sample_weight, post in uploads_factory():
            if cid in seen or cid not in lookup or sample_weight != lookup[cid]["sample_weight"]:
                raise ValueError("Projection weighting upload IDs/weights changed between reads.")
            seen.add(cid)
            # Diagnostic norm in double precision, without changing aggregation.
            row = lookup[cid]
            coefficient = row["dot_before"] / (target_sq + EPS) if row["conflict"] else 0.0
            delta = _delta(post, global_params)
            norm_sq = math.fsum(float((delta[name].double() - coefficient * target[name].double()).square().sum())
                               for name in target)
            projected_norms[cid] = math.sqrt(norm_sq)
            del delta
            yield cid, weights[cid], post
        if seen != lookup.keys():
            raise ValueError("Projection weighting is missing uploads on the second read.")

    python_rng, numpy_rng = random.getstate(), np.random.get_state()
    try:
        with torch.random.fork_rng():
            # The original kernel projects raw global deltas in the original target
            # direction, then uses alpha_i for both delta and correction sums.
            result, metrics, projected_rows = aggregate_target_updates(
                global_params, target_post, weighted_uploads(), target_id, "projection")
    finally:
        random.setstate(python_rng)
        np.random.set_state(numpy_rng)
    clients = []
    for row in projected_rows:
        cid = row["client_id"]
        original = lookup[cid]
        entry = dict(client_id=cid, is_target=row["is_target"], sample_weight=original["sample_weight"],
                     cosine_before_projection=original["cosine_before_projection"],
                     dot_before_projection=row["dot_before"], conflict=row["conflict"],
                     projection_coefficient=row["dot_before"] / (target_sq + EPS) if row["conflict"] else 0.0,
                     removed_component_norm=row["removed_norm"], projected_update_norm=projected_norms[cid],
                     raw_similarity_score=scores.get(cid, 0.0), aggregation_weight=weights[cid])
        if mode == "projection_softmax":
            entry.update(temperature=PROJECTION_SOFTMAX_TAU, softmax_shift_max=summary["softmax_shift_max"])
        else:
            entry.update(relu_score=scores.get(cid, 0.0), relu_fallback_used=summary["relu_fallback_used"])
        clients.append(entry)
    clients.sort(key=lambda row: row["client_id"])
    metrics.update(summary)
    matrices = dict(delta_scope="global", projection_scope="full_model", weighting_mode=mode,
                    weighting_scope="helper_similarity_only_before_projection", weighting_summary=summary,
                    client_ids=[row["client_id"] for row in clients], clients=clients)
    return result, metrics, clients, [], matrices


def validate_source_config(args):
    if args.target_proj_mode not in SOURCE_MODES:
        return
    expected = dict(dataset="Cifar100", partition="pat", class_per_client=20,
                    target_client_id=0, num_clients=20)
    if any(getattr(args, key, None) != value for key, value in expected.items()):
        raise ValueError(f"Source projection requires {expected}; helper IDs are fixed partition metadata.")


def source_helper_ids(mode):
    if mode not in SOURCE_MODES:
        raise ValueError(f"Unknown source mode: {mode}")
    return set(range(1, 4)) if mode == "projection_same_label" else set(range(4, 20))


def source_weights(samples, target_id, selected):
    """Keep q_target=p_target and redistribute only the original helper mass."""
    if target_id not in samples or target_id in selected or not selected <= samples.keys():
        raise ValueError("Invalid source helper IDs or missing target.")
    if any(not math.isfinite(p) or p < 0 for p in samples.values()):
        raise ValueError("Sample weights must be finite and non-negative.")
    if not math.isclose(math.fsum(samples.values()), 1.0, abs_tol=1e-6, rel_tol=0):
        raise ValueError("Sample weights must sum to one.")
    mass = 1.0 - samples[target_id]
    selected_mass = math.fsum(samples[cid] for cid in selected)
    if selected_mass <= 0 and mass > 0:
        raise ValueError("Selected helpers have zero sample mass.")
    # This control must reproduce legacy projection exactly when all helpers are selected.
    if selected == samples.keys() - {target_id}:
        return dict(samples)
    return {cid: (p if cid == target_id else mass * (p / selected_mass)
                  if cid in selected and selected_mass else 0.0) for cid, p in samples.items()}


@torch.no_grad()
def aggregate_source_projection(global_params, target_post, uploads, target_id, samples, mode,
                                selected_helpers=None):
    selected = source_helper_ids(mode) if selected_helpers is None else set(selected_helpers)
    effective = source_weights(samples, target_id, selected)
    excluded_rows, seen = [], set()
    target = _delta(target_post, global_params)
    target_sq = _dot(target, target)

    def selected_uploads():
        for cid, weight, post in uploads:
            if cid in seen or cid not in samples or weight != samples[cid]:
                raise ValueError("Source upload IDs/weights do not match sample weights.")
            seen.add(cid)
            if cid == target_id or cid in selected:
                yield cid, effective[cid], post
            else:
                delta = _delta(post, global_params)
                dot, norm_sq = _dot(delta, target), _dot(delta, delta)
                excluded_rows.append(dict(client_id=cid, weight=0.0, is_target=0,
                    conflict=int(dot < 0), dot_before=dot, dot_after_projection=dot,
                    delta_norm=math.sqrt(norm_sq), removed_norm=0.0))

    # Reuse the exact original full-model/global-delta arithmetic on selected uploads.
    result, metrics, rows = aggregate_target_updates(
        global_params, target_post, selected_uploads(), target_id, "projection")
    if seen != samples.keys():
        raise ValueError("Source projection requires all clients to upload.")
    rows.extend(excluded_rows)
    rows.sort(key=lambda row: row["client_id"])
    for row in rows:
        cid = row["client_id"]
        row.update(selected_as_helper=int(cid in selected), original_sample_weight=samples[cid],
                   effective_aggregation_weight=effective[cid],
                   cosine=_cosine(row["dot_before"], row["delta_norm"] ** 2, target_sq),
                   removed_component_norm=row["removed_norm"])
    summary = dict(selected_helper_ids=sorted(selected), selected_helper_count=len(selected),
                   target_weight=effective[target_id],
                   selected_helper_total_weight=math.fsum(effective[cid] for cid in selected),
                   excluded_helper_count=len(samples) - len(selected) - 1)
    metrics.update({key: value for key, value in summary.items() if key != "selected_helper_ids"})
    metrics["uploaded_client_count"] = len(samples)
    matrices = dict(delta_scope="global", projection_scope="full_model", source_summary=summary,
                    client_ids=[row["client_id"] for row in rows], clients=rows)
    return result, metrics, rows, [], matrices


@torch.no_grad()
def aggregate_projection_variant(global_params, target_post, uploads, target_id, mode,
                                 target_pre=None, groups=None):
    """Uploads are (id, p, post, pre-or-None). Avg post parameters + correction.

    No conflict leaves the exact weighted-parameter Avg arithmetic intact, including
    heterogeneous clients' pre-local structural differences. Buffers are excluded.
    """
    if mode not in ("projection_local", *LAYER_PROJECTION_MODES):
        raise ValueError(f"Unknown projection variant: {mode}")
    local = mode in LOCAL_PROJECTION_MODES
    if local and target_pre is None:
        raise ValueError("Pure-local projection requires pre-local snapshots.")
    target = _delta(target_post, target_pre if local else global_params)
    if mode == "projection_local":
        groups = {"full_model": list(global_params)}
    if not groups or sorted(name for names in groups.values() for name in names) != sorted(global_params):
        raise ValueError("Projection groups must partition all model parameters exactly once.")
    target_layers = {layer: {name: target[name] for name in names} for layer, names in groups.items()}
    target_sq = {layer: _dot(values, values) for layer, values in target_layers.items()}
    full_target_sq = _dot(target, target)
    correction = dict.fromkeys(groups, 0.0)
    result = {name: torch.zeros_like(value) for name, value in global_params.items()}
    clients, layers, seen = [], [], set()
    weight_sum = 0.0
    for cid, weight, post, pre in uploads:
        if cid in seen or not math.isfinite(weight) or weight < 0:
            raise ValueError("Invalid upload IDs or sample weights.")
        if local and pre is None:
            raise ValueError("Pure-local projection requires every client's snapshot.")
        seen.add(cid)
        weight_sum += weight
        delta = _delta(post, pre if local else global_params)
        removed_sq, conflicts = 0.0, 0
        for layer, names in groups.items():
            values = {name: delta[name] for name in names}
            dot, norm_sq = _dot(values, target_layers[layer]), _dot(values, values)
            conflict = cid != target_id and dot < 0
            coefficient = dot / (target_sq[layer] + EPS) if conflict else 0.0
            removed = abs(coefficient) * math.sqrt(target_sq[layer])
            norm = math.sqrt(norm_sq)
            correction[layer] += weight * coefficient
            removed_sq += removed ** 2
            conflicts += int(conflict)
            layers.append(dict(client_id=cid, layer=layer, sample_weight=weight,
                dot=dot, cosine=_cosine(dot, norm_sq, target_sq[layer]), conflict=int(conflict),
                projection_coefficient=coefficient, update_norm=norm, removed_component_norm=removed,
                removed_component_ratio=removed / (norm + EPS),
                dot_after_projection=dot - coefficient * target_sq[layer]))
        for name in result:
            result[name].add_(post[name].detach().to(result[name]) * weight)
        full_dot, full_sq = _dot(delta, target), _dot(delta, delta)
        norm, removed = math.sqrt(full_sq), math.sqrt(removed_sq)
        row = dict(client_id=cid, sample_weight=weight, conflict=int(conflicts > 0),
                   cosine=_cosine(full_dot, full_sq, full_target_sq), delta_norm=norm,
                   removed_component_norm=removed, removed_component_ratio=removed / (norm + EPS))
        if mode == "projection_local":
            row.update(full_local_dot=full_dot, full_local_cos=row["cosine"],
                       projection_coefficient=layers[-1]["projection_coefficient"])
        clients.append(row)
    if target_id not in seen or not math.isclose(weight_sum, 1.0, rel_tol=0, abs_tol=1e-6):
        raise ValueError("Target upload required and sample weights must sum to one.")
    for layer, names in groups.items():
        if correction[layer] != 0.0:
            for name in names:
                result[name].add_(target[name], alpha=-correction[layer])
    clients.sort(key=lambda row: row["client_id"])
    helpers = [row for row in clients if row["client_id"] != target_id]
    layer_summary = []
    for layer in groups:
        entries = [row for row in layers if row["layer"] == layer and row["client_id"] != target_id]
        n = max(len(entries), 1)
        layer_summary.append(dict(layer=layer, conflict_count=sum(row["conflict"] for row in entries),
            conflict_ratio=sum(row["conflict"] for row in entries) / n,
            mean_cosine=math.fsum(row["cosine"] for row in entries) / n,
            mean_removed_ratio=math.fsum(row["removed_component_ratio"] for row in entries) / n,
            weighted_removed_norm=math.fsum(row["sample_weight"] * row["removed_component_norm"] for row in entries)))
    entries = [row for row in layers if row["client_id"] != target_id]
    metrics = dict(conflict_client_count=sum(row["conflict"] for row in helpers),
        conflict_client_ratio=sum(row["conflict"] for row in helpers) / max(len(helpers), 1),
        removed_update_ratio=math.fsum(row["removed_component_norm"] for row in helpers) /
                             (math.fsum(row["delta_norm"] for row in helpers) + EPS),
        total_conflict_entries=sum(row["conflict"] for row in entries),
        overall_layer_conflict_ratio=sum(row["conflict"] for row in entries) / max(len(entries), 1),
        overall_removed_update_ratio=math.fsum(row["sample_weight"] * row["removed_component_norm"] for row in entries) /
            (math.fsum(row["sample_weight"] * row["update_norm"] for row in entries) + EPS),
        uploaded_client_count=len(seen))
    ids = sorted(seen)
    lookup = {(row["client_id"], row["layer"]): row for row in layers}
    matrices = dict(delta_scope="local" if local else "global",
                    projection_scope="full_model" if mode == "projection_local" else "layer",
                    client_ids=ids, layer_names=list(groups), layer_groups=groups, layer_summary=layer_summary)
    for key in ("cosine", "conflict", "projection_coefficient", "update_norm", "removed_component_norm"):
        matrices[f"{key}_matrix"] = [[lookup[cid, layer][key] for layer in groups] for cid in ids]
    return result, metrics, clients, layers, matrices


def print_projection_variant(round_number, mode, matrices, clients):
    prefix = f"[{mode}][Round {round_number}]"
    if "source_summary" in matrices:
        print(prefix + " " + " ".join(f"{key}={value}" for key, value in matrices["source_summary"].items()))
    elif "weighting_summary" in matrices:
        print(prefix + " " + " ".join(f"{key}={value}" for key, value in matrices["weighting_summary"].items()))
    else:
        for key in ("cosine", "conflict"):
            print(f"{prefix} {key} matrix")
            print(" " * 12 + "".join(f"{name:>15}" for name in matrices["layer_names"]))
            for cid, values in zip(matrices["client_ids"], matrices[f"{key}_matrix"]):
                print(f"{'Client ' + str(cid):<12}" + "".join(f"{value:15.7g}" for value in values))
        for row in matrices["layer_summary"]:
            print(prefix + " " + " ".join(f"{key}={value}" for key, value in row.items()))
    for row in clients:
        print(prefix + " " + " ".join(f"{key}={value}" for key, value in row.items()))

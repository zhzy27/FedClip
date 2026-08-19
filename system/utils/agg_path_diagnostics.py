import csv
import math
import os

import torch


AGG_PATH_FIELDS = [
    "round",
    "layer",
    "S_U",
    "S_V",
    "same_rank_u_cos",
    "cross_rank_u_cos",
    "same_rank_v_cos",
    "cross_rank_v_cos",
    "same_rank_pair_count",
    "cross_rank_pair_count",
    "mean_u_path_norm",
    "mean_v_path_norm",
    "agg_u_path_norm",
    "agg_v_path_norm",
]

GLOBAL_TRUNCATION_FIELDS = [
    "round",
    "layer",
    "rank",
    "retained_energy",
    "relative_truncation_error",
]

PRELOCAL_DOWNLOAD_FIELDS = [
    "round",
    "send_round",
    "source_aggregation_round",
    "client_id",
    "capacity",
    "test_samples",
    "download_acc",
]


def parse_rounds(value):
    if value is None:
        return None
    if isinstance(value, (list, tuple, set)):
        return {int(item) for item in value}
    text = str(value).strip()
    if not text:
        return None
    return {int(item.strip()) for item in text.split(",") if item.strip()}


def diagnostic_round_selected(current_round, configured_rounds):
    rounds = parse_rounds(configured_rounds)
    return rounds is None or int(current_round) in rounds


def resolve_diagnostic_output_dir(explicit_dir, fallback_dir):
    explicit_dir = str(explicit_dir or "").strip()
    if explicit_dir:
        return os.path.abspath(os.path.expanduser(explicit_dir))

    train_log_dir = os.environ.get("FEDCLIP_TRAIN_LOG_DIR", "").strip()
    if train_log_dir:
        return os.path.abspath(os.path.expanduser(train_log_dir))

    stdout_link = "/proc/self/fd/1"
    try:
        target = os.readlink(stdout_link)
    except (AttributeError, OSError):
        target = ""
    if target and not target.startswith(("pipe:", "socket:", "/dev/")):
        target = target.removesuffix(" (deleted)")
        return os.path.dirname(os.path.abspath(target))

    return os.path.abspath(os.path.expanduser(str(fallback_dir)))


def append_csv_rows(path, fieldnames, rows):
    if not rows:
        return
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    write_header = not os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=fieldnames,
            extrasaction="ignore",
        )
        if write_header:
            writer.writeheader()
        writer.writerows(rows)


def _factor_layer_info(u_name):
    if u_name.endswith("weight_u"):
        prefix = u_name[: -len("weight_u")]
        return prefix.rstrip("."), f"{prefix}weight_v", f"{prefix}weight"
    if u_name.endswith("conv_u"):
        prefix = u_name[: -len("conv_u")]
        return prefix.rstrip("."), f"{prefix}conv_v", f"{prefix}weight"
    raise ValueError(f"Not a recognized U-factor parameter: {u_name}")


def collect_agg_path_updates(factor_start, end_named_parameters):
    updates = {}
    with torch.no_grad():
        for u_name in sorted(factor_start):
            if not u_name.endswith(("weight_u", "conv_u")):
                continue
            layer_name, v_name, weight_name = _factor_layer_info(u_name)
            if v_name not in factor_start:
                raise RuntimeError(f"Missing start V factor for {u_name}.")
            if u_name not in end_named_parameters or v_name not in end_named_parameters:
                raise RuntimeError(f"Missing end factor pair for {layer_name}.")

            u_start = factor_start[u_name].detach().to("cpu", torch.float32)
            v_start = factor_start[v_name].detach().to("cpu", torch.float32)
            u_end = end_named_parameters[u_name].detach().to("cpu", torch.float32)
            v_end = end_named_parameters[v_name].detach().to("cpu", torch.float32)
            if u_start.shape != u_end.shape or v_start.shape != v_end.shape:
                raise RuntimeError(
                    f"Start/end factor shape mismatch for {layer_name}: "
                    f"U={tuple(u_start.shape)}/{tuple(u_end.shape)}, "
                    f"V={tuple(v_start.shape)}/{tuple(v_end.shape)}."
                )
            if u_start.ndim != 2 or v_start.ndim != 2:
                raise RuntimeError(
                    f"Aggregation-path factors must be matrices for {layer_name}."
                )
            if u_start.shape[1] != v_start.shape[0]:
                raise RuntimeError(
                    f"Incompatible factor shapes for {layer_name}: "
                    f"{tuple(u_start.shape)} @ {tuple(v_start.shape)}."
                )

            delta_u = u_end - u_start
            delta_v = v_end - v_start
            updates[layer_name] = {
                "u_path": (delta_u @ v_start).detach(),
                "v_path": (u_start @ delta_v).detach(),
                "rank": int(u_start.shape[1]),
                "weight_name": weight_name,
            }
    return updates


def _safe_cosine_from_dot(dot, norm_a, norm_b, eps):
    denominator = float(norm_a) * float(norm_b)
    if denominator <= eps:
        return 0.0
    return float(dot) / denominator


def _pairwise_grouped_cosines(u_paths, v_paths, rank_keys, weights, eps):
    totals = {
        "same_u": 0.0,
        "same_v": 0.0,
        "same_weight": 0.0,
        "same_count": 0,
        "cross_u": 0.0,
        "cross_v": 0.0,
        "cross_weight": 0.0,
        "cross_count": 0,
    }
    u_norms = [float(torch.linalg.vector_norm(path).item()) for path in u_paths]
    v_norms = [float(torch.linalg.vector_norm(path).item()) for path in v_paths]
    for left in range(len(u_paths)):
        for right in range(left + 1, len(u_paths)):
            pair_weight = float(weights[left]) * float(weights[right])
            u_dot = float(torch.sum(u_paths[left] * u_paths[right]).item())
            v_dot = float(torch.sum(v_paths[left] * v_paths[right]).item())
            u_cos = _safe_cosine_from_dot(
                u_dot, u_norms[left], u_norms[right], eps
            )
            v_cos = _safe_cosine_from_dot(
                v_dot, v_norms[left], v_norms[right], eps
            )
            group = "same" if rank_keys[left] == rank_keys[right] else "cross"
            totals[f"{group}_u"] += pair_weight * u_cos
            totals[f"{group}_v"] += pair_weight * v_cos
            totals[f"{group}_weight"] += pair_weight
            totals[f"{group}_count"] += 1

    def weighted_value(group, path_kind):
        weight = totals[f"{group}_weight"]
        if weight <= eps:
            return float("nan")
        return totals[f"{group}_{path_kind}"] / weight

    return {
        "same_rank_u_cos": weighted_value("same", "u"),
        "cross_rank_u_cos": weighted_value("cross", "u"),
        "same_rank_v_cos": weighted_value("same", "v"),
        "cross_rank_v_cos": weighted_value("cross", "v"),
        "same_rank_pair_count": totals["same_count"],
        "cross_rank_pair_count": totals["cross_count"],
    }


def _single_layer_consistency(round_index, layer, entries, weights, eps):
    u_paths = [entry["u_path"] for entry in entries]
    v_paths = [entry["v_path"] for entry in entries]
    ranks = [int(entry["rank"]) for entry in entries]
    reference_shape = tuple(u_paths[0].shape)
    if any(tuple(path.shape) != reference_shape for path in u_paths + v_paths):
        raise RuntimeError(f"Full-W path shapes differ across clients for {layer}.")

    agg_u = torch.zeros_like(u_paths[0])
    agg_v = torch.zeros_like(v_paths[0])
    mean_u_norm = 0.0
    mean_v_norm = 0.0
    for weight, u_path, v_path in zip(weights, u_paths, v_paths):
        agg_u.add_(u_path, alpha=float(weight))
        agg_v.add_(v_path, alpha=float(weight))
        mean_u_norm += float(weight) * float(torch.linalg.vector_norm(u_path).item())
        mean_v_norm += float(weight) * float(torch.linalg.vector_norm(v_path).item())
    agg_u_norm = float(torch.linalg.vector_norm(agg_u).item())
    agg_v_norm = float(torch.linalg.vector_norm(agg_v).item())

    row = {
        "round": int(round_index),
        "layer": layer,
        "S_U": agg_u_norm / (mean_u_norm + eps),
        "S_V": agg_v_norm / (mean_v_norm + eps),
        "mean_u_path_norm": mean_u_norm,
        "mean_v_path_norm": mean_v_norm,
        "agg_u_path_norm": agg_u_norm,
        "agg_v_path_norm": agg_v_norm,
    }
    row.update(_pairwise_grouped_cosines(u_paths, v_paths, ranks, weights, eps))
    return row


def _overall_consistency(round_index, ordered_updates, layer_names, weights, eps):
    client_u_norm_sq = [0.0] * len(ordered_updates)
    client_v_norm_sq = [0.0] * len(ordered_updates)
    rank_signatures = []
    aggregate_u_norm_sq = 0.0
    aggregate_v_norm_sq = 0.0

    for client_index, updates in enumerate(ordered_updates):
        rank_signatures.append(
            tuple(int(updates[layer]["rank"]) for layer in layer_names)
        )
        for layer in layer_names:
            u_path = updates[layer]["u_path"]
            v_path = updates[layer]["v_path"]
            client_u_norm_sq[client_index] += float(torch.sum(u_path * u_path).item())
            client_v_norm_sq[client_index] += float(torch.sum(v_path * v_path).item())

    for layer in layer_names:
        aggregate_u = torch.zeros_like(ordered_updates[0][layer]["u_path"])
        aggregate_v = torch.zeros_like(ordered_updates[0][layer]["v_path"])
        for weight, updates in zip(weights, ordered_updates):
            aggregate_u.add_(updates[layer]["u_path"], alpha=float(weight))
            aggregate_v.add_(updates[layer]["v_path"], alpha=float(weight))
        aggregate_u_norm_sq += float(torch.sum(aggregate_u * aggregate_u).item())
        aggregate_v_norm_sq += float(torch.sum(aggregate_v * aggregate_v).item())

    mean_u_norm = sum(
        float(weight) * math.sqrt(value)
        for weight, value in zip(weights, client_u_norm_sq)
    )
    mean_v_norm = sum(
        float(weight) * math.sqrt(value)
        for weight, value in zip(weights, client_v_norm_sq)
    )
    agg_u_norm = math.sqrt(max(aggregate_u_norm_sq, 0.0))
    agg_v_norm = math.sqrt(max(aggregate_v_norm_sq, 0.0))

    pair_totals = {
        "same_u": 0.0,
        "same_v": 0.0,
        "same_weight": 0.0,
        "same_count": 0,
        "cross_u": 0.0,
        "cross_v": 0.0,
        "cross_weight": 0.0,
        "cross_count": 0,
    }
    for left in range(len(ordered_updates)):
        for right in range(left + 1, len(ordered_updates)):
            u_dot = 0.0
            v_dot = 0.0
            for layer in layer_names:
                u_dot += float(
                    torch.sum(
                        ordered_updates[left][layer]["u_path"]
                        * ordered_updates[right][layer]["u_path"]
                    ).item()
                )
                v_dot += float(
                    torch.sum(
                        ordered_updates[left][layer]["v_path"]
                        * ordered_updates[right][layer]["v_path"]
                    ).item()
                )
            u_cos = _safe_cosine_from_dot(
                u_dot,
                math.sqrt(client_u_norm_sq[left]),
                math.sqrt(client_u_norm_sq[right]),
                eps,
            )
            v_cos = _safe_cosine_from_dot(
                v_dot,
                math.sqrt(client_v_norm_sq[left]),
                math.sqrt(client_v_norm_sq[right]),
                eps,
            )
            group = (
                "same"
                if rank_signatures[left] == rank_signatures[right]
                else "cross"
            )
            pair_weight = float(weights[left]) * float(weights[right])
            pair_totals[f"{group}_u"] += pair_weight * u_cos
            pair_totals[f"{group}_v"] += pair_weight * v_cos
            pair_totals[f"{group}_weight"] += pair_weight
            pair_totals[f"{group}_count"] += 1

    def overall_pair_value(group, kind):
        pair_weight = pair_totals[f"{group}_weight"]
        if pair_weight <= eps:
            return float("nan")
        return pair_totals[f"{group}_{kind}"] / pair_weight

    row = {
        "round": int(round_index),
        "layer": "__overall__",
        "S_U": agg_u_norm / (mean_u_norm + eps),
        "S_V": agg_v_norm / (mean_v_norm + eps),
        "mean_u_path_norm": mean_u_norm,
        "mean_v_path_norm": mean_v_norm,
        "agg_u_path_norm": agg_u_norm,
        "agg_v_path_norm": agg_v_norm,
        "same_rank_u_cos": overall_pair_value("same", "u"),
        "cross_rank_u_cos": overall_pair_value("cross", "u"),
        "same_rank_v_cos": overall_pair_value("same", "v"),
        "cross_rank_v_cos": overall_pair_value("cross", "v"),
        "same_rank_pair_count": pair_totals["same_count"],
        "cross_rank_pair_count": pair_totals["cross_count"],
    }
    return row


def aggregation_path_consistency_rows(
    round_index,
    client_updates,
    client_ids,
    weights,
    eps=1e-12,
):
    if not client_ids:
        raise ValueError("At least one uploaded client is required.")
    if len(client_ids) != len(weights):
        raise ValueError("client_ids and weights must have the same length.")
    ordered_updates = [client_updates[int(client_id)] for client_id in client_ids]
    layer_names = sorted(ordered_updates[0])
    if not layer_names:
        raise ValueError("No low-rank aggregation path updates were provided.")
    for client_id, updates in zip(client_ids, ordered_updates):
        if sorted(updates) != layer_names:
            raise RuntimeError(
                f"Client_{client_id} aggregation-path layers do not match."
            )

    rows = []
    rank_metadata = {}
    for layer in layer_names:
        entries = [updates[layer] for updates in ordered_updates]
        rows.append(
            _single_layer_consistency(
                round_index,
                layer,
                entries,
                weights,
                eps,
            )
        )
        weight_names = {entry["weight_name"] for entry in entries}
        if len(weight_names) != 1:
            raise RuntimeError(f"Full-rank weight name mismatch for {layer}.")
        rank_metadata[layer] = {
            "weight_name": next(iter(weight_names)),
            "ranks": sorted({int(entry["rank"]) for entry in entries}),
        }
    rows.append(
        _overall_consistency(
            round_index,
            ordered_updates,
            layer_names,
            weights,
            eps,
        )
    )
    return rows, rank_metadata


def weight_to_svd_matrix(weight):
    if weight.ndim == 2:
        return weight
    if weight.ndim == 4:
        out_channels, _, kernel_height, kernel_width = weight.shape
        return weight.permute(0, 2, 1, 3).reshape(
            out_channels * kernel_height,
            weight.shape[1] * kernel_width,
        )
    raise ValueError(
        f"Only linear and convolution weights are supported, got {tuple(weight.shape)}."
    )


def global_truncation_rows(round_index, named_parameters, rank_metadata, eps=1e-12):
    rows = []
    with torch.no_grad():
        for layer in sorted(rank_metadata):
            metadata = rank_metadata[layer]
            weight_name = metadata["weight_name"]
            if weight_name not in named_parameters:
                raise RuntimeError(
                    f"Aggregated full-rank parameter {weight_name} is missing."
                )
            matrix = weight_to_svd_matrix(named_parameters[weight_name].detach())
            singular_values = torch.linalg.svdvals(matrix)
            energy = singular_values.square()
            total_energy = float(energy.sum().item())
            cumulative = energy.cumsum(dim=0)
            for rank in metadata["ranks"]:
                effective_rank = min(max(int(rank), 0), int(energy.numel()))
                if total_energy <= eps:
                    retained = 1.0
                elif effective_rank == 0:
                    retained = 0.0
                else:
                    retained = float(
                        (cumulative[effective_rank - 1] / total_energy).item()
                    )
                retained = min(max(retained, 0.0), 1.0)
                rows.append(
                    {
                        "round": int(round_index),
                        "layer": layer,
                        "rank": int(rank),
                        "retained_energy": retained,
                        "relative_truncation_error": math.sqrt(
                            max(1.0 - retained, 0.0)
                        ),
                    }
                )
    return rows

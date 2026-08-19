import hashlib
import itertools
import math

import torch
import torch.nn.functional as F

from utils.agg_path_diagnostics import diagnostic_round_selected


ANCHOR_MODES = (
    "none",
    "clip",
    "shared_random",
    "client_random",
    "shuffled_clip",
)

SEMANTIC_PROTOTYPE_CLIENT_FIELDS = [
    "round",
    "stage",
    "client_id",
    "class_id",
    "capacity",
    "sample_count",
    "prototype_norm",
    "train_anchor_cos",
    "train_anchor_l2",
    "true_clip_anchor_cos",
]

SEMANTIC_PROTOTYPE_SUMMARY_FIELDS = [
    "round",
    "stage",
    "overall_same_class_cos",
    "same_capacity_same_class_cos",
    "cross_capacity_same_class_cos",
    "overall_pair_count",
    "same_capacity_pair_count",
    "cross_capacity_pair_count",
    "mean_train_anchor_cos",
    "mean_true_clip_anchor_cos",
]

SEMANTIC_PROTOTYPE_CLASS_SUMMARY_FIELDS = [
    "round",
    "stage",
    "class_id",
    "overall_same_class_cos",
    "same_capacity_cos",
    "cross_capacity_cos",
    "overall_pair_count",
    "same_capacity_pair_count",
    "cross_capacity_pair_count",
]

PROTOTYPE_LOCAL_DRIFT_FIELDS = [
    "round",
    "record_type",
    "client_id",
    "class_id",
    "capacity",
    "sample_count",
    "local_proto_drift",
    "pre_train_anchor_cos",
    "post_train_anchor_cos",
    "pre_true_clip_cos",
    "post_true_clip_cos",
]


def prototype_human_round(loop_round):
    return int(loop_round) + 1


def prototype_round_selected(loop_round, configured_rounds):
    return diagnostic_round_selected(
        prototype_human_round(loop_round), configured_rounds
    )


def _configuration_hash(mode, seed, client_id, anchors, permutation):
    digest = hashlib.sha256()
    digest.update(str(mode).encode("utf-8"))
    digest.update(str(int(seed)).encode("utf-8"))
    digest.update(str(client_id).encode("utf-8"))
    digest.update(anchors.detach().cpu().contiguous().numpy().tobytes())
    if permutation is not None:
        digest.update(
            permutation.detach().cpu().contiguous().numpy().tobytes()
        )
    return digest.hexdigest()


def build_anchor_configuration(
    true_clip_anchors,
    mode="clip",
    seed=2026,
    client_id=0,
):
    """Build deterministic training anchors without touching global RNG state."""
    if mode not in ANCHOR_MODES:
        raise ValueError(f"Unsupported anchor mode: {mode}.")
    if true_clip_anchors.ndim < 2:
        raise ValueError("CLIP anchors must have shape [..., classes, dim].")

    true_anchors = true_clip_anchors
    num_classes = int(true_anchors.shape[-2])
    permutation = None
    effective_seed = int(seed)

    if mode in ("none", "clip"):
        training_anchors = true_anchors
    elif mode in ("shared_random", "client_random"):
        if mode == "client_random":
            effective_seed += int(client_id) * 1_000_003
        generator = torch.Generator(device="cpu")
        generator.manual_seed(effective_seed)
        true_cpu = true_anchors.detach().to(device="cpu", dtype=torch.float32)
        random_cpu = torch.randn(
            true_cpu.shape,
            generator=generator,
            dtype=true_cpu.dtype,
        )
        random_cpu = F.normalize(random_cpu, p=2, dim=-1, eps=1e-12)
        training_cpu = random_cpu * torch.linalg.vector_norm(
            true_cpu, dim=-1, keepdim=True
        )
        training_anchors = training_cpu.to(
            device=true_anchors.device,
            dtype=true_anchors.dtype,
        )
    else:
        if num_classes < 2:
            raise ValueError(
                "shuffled_clip requires at least two classes for a derangement."
            )
        generator = torch.Generator(device="cpu")
        generator.manual_seed(effective_seed)
        identity = torch.arange(num_classes)
        while True:
            permutation = torch.randperm(num_classes, generator=generator)
            if torch.all(permutation != identity):
                break
        permutation_device = permutation.to(true_anchors.device)
        training_anchors = true_anchors.index_select(-2, permutation_device)

    config_hash = _configuration_hash(
        mode,
        seed,
        client_id if mode == "client_random" else "shared",
        training_anchors,
        permutation,
    )
    return {
        "anchors": training_anchors,
        "permutation": permutation,
        "hash": config_hash,
        "effective_seed": effective_seed,
    }


def collect_model_class_prototypes(model, data_loader, device, num_classes):
    """Collect class means from model.base without changing model state."""
    was_training = bool(model.training)
    sums = {}
    counts = {}
    try:
        model.eval()
        with torch.no_grad():
            for x, y in data_loader:
                if isinstance(x, list):
                    x = x[0]
                x = x.to(device)
                y = y.to(device)
                features = model.base(x)
                if not torch.is_tensor(features):
                    raise TypeError("model.base(x) must return a tensor.")
                features = features.reshape(features.shape[0], -1)
                for class_id in torch.unique(y).tolist():
                    class_id = int(class_id)
                    if class_id < 0 or class_id >= int(num_classes):
                        raise ValueError(f"Invalid class id {class_id}.")
                    mask = y == class_id
                    class_sum = features[mask].sum(dim=0).detach().cpu()
                    sums[class_id] = sums.get(class_id, 0) + class_sum
                    counts[class_id] = counts.get(class_id, 0) + int(mask.sum())
    finally:
        model.train(was_training)

    return {
        class_id: {
            "prototype": sums[class_id] / counts[class_id],
            "sample_count": counts[class_id],
        }
        for class_id in sorted(sums)
        if counts[class_id] > 0
    }


def _cosine(left, right):
    return float(
        F.cosine_similarity(
            left.reshape(1, -1).float(),
            right.reshape(1, -1).float(),
            dim=1,
            eps=1e-12,
        ).item()
    )


def _weighted_mean(values):
    finite_values = [
        (float(value), float(weight))
        for value, weight in values
        if math.isfinite(float(value)) and float(weight) > 0.0
    ]
    if not finite_values:
        return float("nan")
    total_weight = sum(weight for _, weight in finite_values)
    return sum(value * weight for value, weight in finite_values) / total_weight


def prototype_client_rows(round_number, stage, client_results):
    rows = []
    for client_id in sorted(client_results):
        result = client_results[client_id]
        training_anchors = result.get("training_anchors")
        true_anchors = result["true_clip_anchors"]
        for class_id, entry in sorted(result["prototypes"].items()):
            prototype = entry["prototype"].float()
            true_anchor = true_anchors[class_id].detach().cpu().float()
            if training_anchors is None:
                train_cos = float("nan")
                train_l2 = float("nan")
            else:
                train_anchor = training_anchors[class_id].detach().cpu().float()
                train_cos = _cosine(prototype, train_anchor)
                train_l2 = float(torch.linalg.vector_norm(prototype - train_anchor))
            rows.append(
                {
                    "round": int(round_number),
                    "stage": stage,
                    "client_id": int(client_id),
                    "class_id": int(class_id),
                    "capacity": float(result["capacity"]),
                    "sample_count": int(entry["sample_count"]),
                    "prototype_norm": float(torch.linalg.vector_norm(prototype)),
                    "train_anchor_cos": train_cos,
                    "train_anchor_l2": train_l2,
                    "true_clip_anchor_cos": _cosine(prototype, true_anchor),
                }
            )
    return rows


def prototype_summary_row(round_number, stage, client_results, client_rows):
    overall_pairs = []
    same_capacity_pairs = []
    cross_capacity_pairs = []
    class_entries = {}
    for client_id, result in client_results.items():
        for class_id, entry in result["prototypes"].items():
            class_entries.setdefault(class_id, []).append(
                (int(client_id), result, entry)
            )

    for entries in class_entries.values():
        for (_, left_result, left), (_, right_result, right) in itertools.combinations(
            entries, 2
        ):
            cosine = _cosine(left["prototype"], right["prototype"])
            weight = int(left["sample_count"]) * int(right["sample_count"])
            pair = (cosine, weight)
            overall_pairs.append(pair)
            same_capacity = math.isclose(
                float(left_result["capacity"]),
                float(right_result["capacity"]),
                rel_tol=0.0,
                abs_tol=1e-12,
            )
            (
                same_capacity_pairs
                if same_capacity
                else cross_capacity_pairs
            ).append(pair)

    train_anchor_values = [
        (row["train_anchor_cos"], row["sample_count"]) for row in client_rows
    ]
    true_anchor_values = [
        (row["true_clip_anchor_cos"], row["sample_count"])
        for row in client_rows
    ]
    return {
        "round": int(round_number),
        "stage": stage,
        "overall_same_class_cos": _weighted_mean(overall_pairs),
        "same_capacity_same_class_cos": _weighted_mean(same_capacity_pairs),
        "cross_capacity_same_class_cos": _weighted_mean(cross_capacity_pairs),
        "overall_pair_count": len(overall_pairs),
        "same_capacity_pair_count": len(same_capacity_pairs),
        "cross_capacity_pair_count": len(cross_capacity_pairs),
        "mean_train_anchor_cos": _weighted_mean(train_anchor_values),
        "mean_true_clip_anchor_cos": _weighted_mean(true_anchor_values),
    }


def prototype_class_summary_rows(round_number, stage, client_results):
    class_entries = {}
    for client_id, result in client_results.items():
        for class_id, entry in result["prototypes"].items():
            class_entries.setdefault(int(class_id), []).append(
                (int(client_id), result, entry)
            )

    rows = []
    for class_id, entries in sorted(class_entries.items()):
        overall_pairs = []
        same_capacity_pairs = []
        cross_capacity_pairs = []
        for (_, left_result, left), (_, right_result, right) in itertools.combinations(
            entries, 2
        ):
            cosine = _cosine(left["prototype"], right["prototype"])
            weight = int(left["sample_count"]) * int(right["sample_count"])
            pair = (cosine, weight)
            overall_pairs.append(pair)
            same_capacity = math.isclose(
                float(left_result["capacity"]),
                float(right_result["capacity"]),
                rel_tol=0.0,
                abs_tol=1e-12,
            )
            (
                same_capacity_pairs
                if same_capacity
                else cross_capacity_pairs
            ).append(pair)
        rows.append(
            {
                "round": int(round_number),
                "stage": stage,
                "class_id": int(class_id),
                "overall_same_class_cos": _weighted_mean(overall_pairs),
                "same_capacity_cos": _weighted_mean(same_capacity_pairs),
                "cross_capacity_cos": _weighted_mean(cross_capacity_pairs),
                "overall_pair_count": len(overall_pairs),
                "same_capacity_pair_count": len(same_capacity_pairs),
                "cross_capacity_pair_count": len(cross_capacity_pairs),
            }
        )
    return rows


def prototype_local_drift_rows(round_number, prelocal, postlocal):
    detail_rows = []
    for client_id in sorted(set(prelocal) & set(postlocal)):
        pre = prelocal[client_id]
        post = postlocal[client_id]
        common_classes = set(pre["prototypes"]) & set(post["prototypes"])
        pre_rows = {
            row["class_id"]: row
            for row in prototype_client_rows(round_number, "prelocal", {client_id: pre})
        }
        post_rows = {
            row["class_id"]: row
            for row in prototype_client_rows(round_number, "postlocal", {client_id: post})
        }
        for class_id in sorted(common_classes):
            pre_entry = pre["prototypes"][class_id]
            post_entry = post["prototypes"][class_id]
            detail_rows.append(
                {
                    "round": int(round_number),
                    "record_type": "client_class",
                    "client_id": int(client_id),
                    "class_id": int(class_id),
                    "capacity": float(pre["capacity"]),
                    "sample_count": min(
                        int(pre_entry["sample_count"]),
                        int(post_entry["sample_count"]),
                    ),
                    "local_proto_drift": 1.0
                    - _cosine(pre_entry["prototype"], post_entry["prototype"]),
                    "pre_train_anchor_cos": pre_rows[class_id]["train_anchor_cos"],
                    "post_train_anchor_cos": post_rows[class_id]["train_anchor_cos"],
                    "pre_true_clip_cos": pre_rows[class_id]["true_clip_anchor_cos"],
                    "post_true_clip_cos": post_rows[class_id]["true_clip_anchor_cos"],
                }
            )

    summaries = []
    groups = {"__overall__": detail_rows}
    for row in detail_rows:
        groups.setdefault(float(row["capacity"]), []).append(row)
    metric_fields = (
        "local_proto_drift",
        "pre_train_anchor_cos",
        "post_train_anchor_cos",
        "pre_true_clip_cos",
        "post_true_clip_cos",
    )
    for capacity, rows in groups.items():
        if not rows:
            continue
        summary = {
            "round": int(round_number),
            "record_type": "overall_summary"
            if capacity == "__overall__"
            else "capacity_summary",
            "client_id": "",
            "class_id": "",
            "capacity": capacity,
            "sample_count": sum(int(row["sample_count"]) for row in rows),
        }
        for field in metric_fields:
            summary[field] = _weighted_mean(
                (row[field], row["sample_count"]) for row in rows
            )
        summaries.append(summary)
    return detail_rows + summaries

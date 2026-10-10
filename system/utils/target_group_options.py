"""Lightweight group CLI validation; launchers need no training dependencies."""

import math
import numbers

GROUP_MODES = ("group_avg_all", "group_avg_only", "group_projection_softmax")
DEFAULT_GROUP_IDS = (0, 1, 2, 3, 4, 6, 7)
GROUP_SOFTMAX_TAU = .2


def normalize_group_ids(value, num_clients):
    if isinstance(value, str):
        fields = value.split(",")
        if any(not field.strip() for field in fields):
            raise ValueError("target_group_ids must be a nonempty comma-separated list.")
        try:
            ids = [int(field.strip()) for field in fields]
        except ValueError as error:
            raise ValueError("target_group_ids must contain integer IDs.") from error
    else:
        try:
            ids = list(value) if value is not None else list(DEFAULT_GROUP_IDS)
        except TypeError as error:
            raise ValueError("target_group_ids must be an ID list.") from error
    if not ids or any(isinstance(cid, bool) or not isinstance(cid, numbers.Integral)
                      or not 0 <= cid < num_clients for cid in ids):
        raise ValueError("Target group must be nonempty with IDs in [0,num_clients).")
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate target group member.")
    return sorted(int(cid) for cid in ids)


def validate_group_options(ids, weight, mode, num_clients, meta_split="", collect=False):
    if weight is not None and (mode != "group_projection_softmax" or not math.isfinite(weight)
                               or not 0 <= weight <= 1):
        raise ValueError("target_group_weight is finite in [0,1] and only supported by group_projection_softmax.")
    if mode not in GROUP_MODES:
        if ids is not None:
            raise ValueError("Explicit target_group_ids requires a group mode.")
        return None
    if meta_split or collect:
        raise ValueError("Group experiments use the original full training set, without the meta holdout protocol.")
    return normalize_group_ids(ids, num_clients)


def group_tag(ids):
    return "group" + "_".join(map(str, ids))

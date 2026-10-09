"""CLI-safe validation/naming for the ProjectionSoftmax self-mass ablation."""

import math


def validate_projection_self_weight(value, mode, meta_split=""):
    if value is None:
        return
    if mode != "projection_softmax":
        raise ValueError("projection_self_weight is supported only by projection_softmax.")
    if not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError("projection_self_weight must be finite and in [0, 1].")
    if meta_split:
        raise ValueError("The self-weight ablation uses the full training set; do not combine it with meta_c0_split.")


def self_weight_tag(value):
    label = f"{value:.2f}" if round(value, 2) == value else repr(float(value))
    return "self" + label.replace(".", "p")

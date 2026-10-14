"""CLI-safe fixed-budget C0 feedback experiment options."""

import os
import math

FEEDBACK_MODES = ("feedback_avg", "feedback_avg_self")
DEFERRED_FEEDBACK_MODES = ("feedback_projection_self",)
FEEDBACK_ANCHOR_WEIGHT = .125


def feedback_anchor_weight(mode, value=None):
    if mode in DEFERRED_FEEDBACK_MODES:
        raise ValueError("feedback_projection_self is deferred: the helper-first flow has no current-round C0 update; Projection reference is not defined.")
    if mode not in FEEDBACK_MODES:
        raise ValueError(f"Unknown feedback mode: {mode}")
    if value is not None and (mode != "feedback_avg_self" or not math.isfinite(value) or not 0 <= value <= 1):
        raise ValueError("feedback_anchor_weight must be finite in [0,1] and is supported only by feedback_avg_self.")
    return 0. if mode == "feedback_avg" else FEEDBACK_ANCHOR_WEIGHT if value is None else float(value)


def feedback_tag(mode, anchor_weight=None):
    anchor = feedback_anchor_weight(mode, anchor_weight)
    return mode + ("_helpers_only_local5" if mode == "feedback_avg" else
                   "_anchor" + repr(anchor).replace(".", "p") + "_local5")


def ensure_feedback_names(args):
    if args.target_proj_mode not in FEEDBACK_MODES:
        return
    tag = feedback_tag(args.target_proj_mode, getattr(args, "feedback_anchor_weight", None))
    if hasattr(args, "exp_name") and tag not in args.exp_name:
        args.exp_name += "_" + tag
    for key in ("save_folder_name", "h5_result_root", "final_model_root", "training_log_root", "run_log_dir"):
        value = getattr(args, key, "")
        if value and tag not in value:
            setattr(args, key, os.path.join(value, tag))
    if hasattr(args, "run_log_dir") and not args.run_log_dir:
        external = os.environ.get("FEDCLIP_TRAIN_LOG_DIR", "")
        if external and tag not in external:
            args.run_log_dir = os.path.join(external, tag)


def validate_feedback_options(args):
    value = getattr(args, "feedback_anchor_weight", None)
    if args.target_proj_mode in DEFERRED_FEEDBACK_MODES:
        feedback_anchor_weight(args.target_proj_mode, value)
    if value is not None and args.target_proj_mode not in FEEDBACK_MODES:
        raise ValueError("feedback_anchor_weight is supported only by feedback_avg_self.")
    if args.target_proj_mode not in FEEDBACK_MODES:
        return
    feedback_anchor_weight(args.target_proj_mode, value)
    expected = dict(algorithm="FedTargetProj", target_client_id=0, num_clients=20,
                    local_epochs=5, eval_gap=1)
    if any(getattr(args, key, None) != value for key, value in expected.items()):
        raise ValueError(f"Feedback experiments require {expected}; each client trains once per round.")
    if (getattr(args, "auto_break", False) or getattr(args, "few_shot", 0)
            or getattr(args, "train_slow_rate", 0) or getattr(args, "target_prox_mu", 0)
            or getattr(args, "target_prox_experiment", False)
            or getattr(args, "meta_c0_split", "") or getattr(args, "meta_collect_snapshots", False)
            or getattr(args, "meta_weight_file", "")):
        raise ValueError("Feedback requires complete normal epochs/data, no early stop, Prox or meta protocol.")
    if getattr(args, "projection_self_weight", None) is not None:
        raise ValueError("Feedback retains S_t via feedback_anchor_weight; do not use projection_self_weight.")


"""CLI-safe fixed-budget C0 feedback experiment options."""

import os

FEEDBACK_MODES = ("feedback_avg", "feedback_avg_self", "feedback_projection_self")
FEEDBACK_SELF_WEIGHT = .125


def feedback_tag(mode):
    if mode not in FEEDBACK_MODES:
        raise ValueError(f"Unknown feedback mode: {mode}")
    return mode + ("_sample" if mode == "feedback_avg" else "_self0p125")


def ensure_feedback_names(args):
    if args.target_proj_mode not in FEEDBACK_MODES:
        return
    tag = feedback_tag(args.target_proj_mode)
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
    if args.target_proj_mode not in FEEDBACK_MODES:
        return
    expected = dict(algorithm="FedTargetProj", target_client_id=0, num_clients=20,
                    local_epochs=5, eval_gap=1)
    if any(getattr(args, key, None) != value for key, value in expected.items()):
        raise ValueError(f"Feedback experiments require {expected}; local_epochs is per stage.")
    if (getattr(args, "auto_break", False) or getattr(args, "few_shot", 0)
            or getattr(args, "train_slow_rate", 0) or getattr(args, "target_prox_mu", 0)
            or getattr(args, "target_prox_experiment", False)
            or getattr(args, "meta_c0_split", "") or getattr(args, "meta_collect_snapshots", False)
            or getattr(args, "meta_weight_file", "")):
        raise ValueError("Feedback requires complete normal epochs/data, no early stop, Prox or meta protocol.")
    if getattr(args, "projection_self_weight", None) is not None:
        raise ValueError("Feedback self mass is fixed internally at .125; do not use projection_self_weight.")


"""Narrow FedRep-style head5/backbone5 ablation; joint is the legacy default."""

import os


def fedrep_schedule(args):
    return getattr(args, "target_local_schedule", "joint") == "fedrep"


def validate_local_schedule(args):
    schedule = getattr(args, "target_local_schedule", "joint")
    if schedule not in ("joint", "fedrep"):
        raise ValueError("target_local_schedule must be joint or fedrep.")
    if type(getattr(args, "fedrep_head_epochs", 5)) is not int:
        raise ValueError("fedrep_head_epochs must be an integer.")
    if schedule == "joint":
        if getattr(args, "fedrep_head_epochs", 5) != 5:
            raise ValueError("fedrep_head_epochs requires target_local_schedule=fedrep.")
        return
    if (getattr(args, "algorithm", "") != "FedTargetProj" or getattr(args, "target_proj_mode", "") != "avg"
            or getattr(args, "target_head_mode", "shared") != "local"):
        raise ValueError("FedRep-style training supports only FedTargetProj/avg with a local fc3.")
    if (getattr(args, "model_family", "") != "Decom_CNN-5-512" or getattr(args, "num_classes", 100) != 100
            or getattr(args, "target_client_id", 0) != 0):
        raise ValueError("FedRep-style training requires Decom_CNN-5-512, 100 classes and C0.")
    if (getattr(args, "target_prox_mu", 0.) != 0 or getattr(args, "meta_c0_split", "")
            or getattr(args, "meta_collect_snapshots", False) or getattr(args, "meta_snapshot_dir", "")
            or getattr(args, "meta_weight_file", "") or getattr(args, "head_diag_collect", False)
            or getattr(args, "head_diag_dir", "") or getattr(args, "learning_rate_decay", False)
            or getattr(args, "models_folder_name", "") or getattr(args, "resume", False)
            or getattr(args, "auto_break", False) or getattr(args, "few_shot", 0) or getattr(args, "train_slow_rate", 0)
            or getattr(args, "target_group_ids", None) is not None or getattr(args, "target_group_weight", None) is not None
            or getattr(args, "feedback_anchor_weight", None) is not None
            or getattr(args, "projection_self_weight", None) is not None):
        raise ValueError("FedRep-style training forbids Prox, Projection/group/feedback/DWA/meta, head_diag collection, decay and truncated/pretrained/resumed runs.")
    for key, expected in (("fedrep_head_epochs", 5), ("local_epochs", 5), ("batch_size", 16),
                          ("local_learning_rate", .005), ("num_clients", 20), ("eval_gap", 1),
                          ("is_regular", 1), ("regular_lamda", .001)):
        if getattr(args, key, expected) != expected:
            raise ValueError(f"This FedRep-style ablation requires {key}={expected}; total data epochs=10.")


def schedule_tag(args):
    return f"fedrep_head{getattr(args, 'fedrep_head_epochs', 5)}_backbone{getattr(args, 'local_epochs', 5)}"


def ensure_schedule_names(args):
    if not fedrep_schedule(args):
        return
    tag = schedule_tag(args)
    if tag not in getattr(args, "exp_name", ""):
        args.exp_name = getattr(args, "exp_name", "") + "_" + tag
    for key in ("save_folder_name", "final_model_root", "h5_result_root", "training_log_root", "run_log_dir", "training_log_dir"):
        value = getattr(args, key, "")
        if value and tag not in value:
            setattr(args, key, os.path.join(value, tag))
    if not getattr(args, "run_log_dir", ""):
        external = os.environ.get("FEDCLIP_TRAIN_LOG_DIR", "")
        if external and tag not in external:
            args.run_log_dir = os.path.join(external, tag)

"""Independent post-local head-diagnostic collection options; no meta protocol."""

import os


def diagnostic_rounds(args):
    raw = str(getattr(args, "head_diag_rounds", "10,20,30")).split(",")
    try:
        rounds = [int(value.strip()) for value in raw]
    except ValueError as error:
        raise ValueError("head_diag_rounds must be comma-separated completed-round integers.") from error
    total = getattr(args, "global_rounds", getattr(args, "rounds", 100)) + 1
    if not rounds or len(set(rounds)) != len(rounds) or any(r < 1 or r > total for r in rounds):
        raise ValueError(f"head_diag_rounds must be unique completed rounds in [1,{total}].")
    return tuple(sorted(rounds))


def validate_head_diagnostic(args):
    if not getattr(args, "head_diag_collect", False):
        if getattr(args, "head_diag_dir", ""):
            raise ValueError("head_diag_dir requires head_diag_collect; collection is off by default.")
        return
    if (getattr(args, "algorithm", "") != "FedTargetProj"
            or getattr(args, "target_proj_mode", "") not in ("avg", "projection_softmax")
            or getattr(args, "target_head_mode", "shared") not in ("shared", "local")):
        raise ValueError("Head diagnostics support only synchronous FedTargetProj avg/projection_softmax and shared/local heads.")
    if (getattr(args, "model_family", "") != "Decom_CNN-5-512"
            or getattr(args, "num_classes", 100) != 100 or getattr(args, "target_client_id", 0) != 0):
        raise ValueError("Head diagnostics require Decom_CNN-5-512, 100 classes and C0.")
    if args.target_proj_mode == "projection_softmax" and getattr(args, "projection_self_weight", None) != .125:
        raise ValueError("Head diagnostics require projection_softmax self weight .125.")
    if (getattr(args, "target_prox_mu", 0.) != 0 or getattr(args, "meta_c0_split", "")
            or getattr(args, "meta_collect_snapshots", False) or getattr(args, "meta_weight_file", "")
            or getattr(args, "meta_snapshot_dir", "") or getattr(args, "learning_rate_decay", False)
            or getattr(args, "models_folder_name", "") or getattr(args, "resume", False)
            or getattr(args, "auto_break", False) or getattr(args, "few_shot", 0) or getattr(args, "train_slow_rate", 0)):
        raise ValueError("Head diagnostics use fresh full-data ordinary training, no Prox/meta/decay/early stop/slow truncation.")
    for name, default in (("local_epochs", 5), ("batch_size", 16), ("local_learning_rate", .005),
                          ("eval_gap", 1), ("num_clients", 20), ("is_regular", 1), ("regular_lamda", .001)):
        if getattr(args, name, default) != default:
            raise ValueError(f"Head diagnostic controls require {name}={default}.")
    diagnostic_rounds(args)


def ensure_head_diagnostic_names(args):
    if not getattr(args, "head_diag_collect", False):
        return
    tag = f"head_diag_{args.target_proj_mode}_{getattr(args, 'target_head_mode', 'shared')}"
    if "head_diag" not in getattr(args, "exp_name", ""):
        args.exp_name = getattr(args, "exp_name", "") + "_" + tag
    for key in ("save_folder_name", "final_model_root", "h5_result_root", "training_log_root", "run_log_dir", "training_log_dir"):
        value = getattr(args, key, "")
        if value and "head_diag" not in value:
            setattr(args, key, os.path.join(value, tag))
    if not getattr(args, "run_log_dir", ""):
        external = os.environ.get("FEDCLIP_TRAIN_LOG_DIR", "")
        if external and "head_diag" not in external:
            args.run_log_dir = os.path.join(external, tag)

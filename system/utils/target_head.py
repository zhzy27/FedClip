"""Options and alias-aware parameter scope for the local-fc3 ablation."""

from contextlib import contextmanager
import os
import random

HEAD_MODES = ("shared", "local")
CLASSIFIER_NAMES = ("fc3.weight", "fc3.bias")


def local_head(args):
    return getattr(args, "target_head_mode", "shared") == "local"


def validate_target_head(args):
    mode = getattr(args, "target_head_mode", "shared")
    if mode not in HEAD_MODES:
        raise ValueError("target_head_mode must be shared or local.")
    if mode == "shared":
        return
    if (getattr(args, "algorithm", "") != "FedTargetProj"
            or getattr(args, "target_proj_mode", "") not in ("avg", "projection_softmax")):
        raise ValueError("Local head supports only synchronous FedTargetProj avg or projection_softmax.")
    if (getattr(args, "model_family", "") != "Decom_CNN-5-512"
            or getattr(args, "num_classes", 100) != 100 or getattr(args, "target_client_id", 0) != 0):
        raise ValueError("Local head requires Decom_CNN-5-512, a 100-class fc3 and target C0.")
    if args.target_proj_mode == "projection_softmax" and getattr(args, "projection_self_weight", None) != .125:
        raise ValueError("Local-head projection_softmax requires projection_self_weight=0.125.")
    if (getattr(args, "target_prox_mu", 0.) != 0 or getattr(args, "meta_c0_split", "")
            or getattr(args, "meta_collect_snapshots", False) or getattr(args, "meta_weight_file", "")
            or getattr(args, "meta_snapshot_dir", "") or getattr(args, "target_group_ids", None) is not None
            or getattr(args, "target_group_weight", None) is not None
            or getattr(args, "feedback_anchor_weight", None) is not None):
        raise ValueError("Local head does not support Prox, meta, group or feedback protocols.")
    if (getattr(args, "models_folder_name", "") or getattr(args, "resume", False)
            or getattr(args, "auto_break", False) or getattr(args, "few_shot", 0)
            or getattr(args, "train_slow_rate", 0) or getattr(args, "eval_gap", 1) != 1):
        raise ValueError("Local-head controls require fresh models, complete local training and every-round evaluation.")


def ensure_head_names(args):
    if not local_head(args):
        return
    tag = f"{args.target_proj_mode}_local_head"
    if "local_head" not in getattr(args, "exp_name", ""):
        args.exp_name = getattr(args, "exp_name", "") + "_" + tag
    defaults = dict(save_folder_name="temp", final_model_root="final_models", h5_result_root="h5_results",
                    training_log_root="training_logs")
    for key in (*defaults, "run_log_dir"):
        value = getattr(args, key, defaults.get(key, ""))
        if value and "local_head" not in value:
            setattr(args, key, os.path.join(value, tag))
    if not getattr(args, "run_log_dir", ""):
        external = os.environ.get("FEDCLIP_TRAIN_LOG_DIR", "")
        if external and "local_head" not in external:
            args.run_log_dir = os.path.join(external, tag)


def classifier_names(model):
    """Resolve by object identity, then assert the actual canonical fc3 names.

    Hyper_CNN_512 registers fc3 before head.1; named_parameters deduplicates
    these aliases. Prefix filtering on head.* alone would miss both parameters.
    """
    import torch
    classifier = getattr(model, "fc3", None)
    if not isinstance(classifier, torch.nn.Linear) or (classifier.in_features, classifier.out_features) != (512, 100):
        raise ValueError("Local head requires exactly the trainable 512->100 fc3 weight and bias.")
    if classifier.bias is None or not all(p.requires_grad for p in classifier.parameters()):
        raise ValueError("Local fc3 must include an unfrozen weight and bias.")
    ids = {id(p) for p in classifier.parameters()}
    names = tuple(name for name, p in model.named_parameters() if id(p) in ids)
    if names != CLASSIFIER_NAMES or {id(p) for p in model.head.parameters()} != ids:
        raise ValueError("Unexpected fc3/head aliases; refusing an ambiguous classifier parameter scope.")
    return names


def backbone_parameters(model):
    excluded = classifier_names(model)
    parameters = {n: p for n, p in model.named_parameters() if n not in excluded}
    if not parameters or not any(n.startswith("fc1.") for n in parameters) or not any(n.startswith("fc2.") for n in parameters):
        raise ValueError("Local-head backbone must retain fc1 and fc2.")
    return parameters


@contextmanager
def preserved_head_rng():
    import numpy as np
    import torch
    python_rng, numpy_rng = random.getstate(), np.random.get_state()
    try:
        with torch.random.fork_rng():
            yield
    finally:
        random.setstate(python_rng)
        np.random.set_state(numpy_rng)

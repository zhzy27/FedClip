"""Independent proximal experiment options; never consult legacy args.mu."""

import math
import os

PROX_SCOPES = ("helpers", "all")


def validate_target_prox(mu, scope, algorithm="FedTargetProj", mode="projection_softmax",
                         target_id=0, self_weight=.125, meta_split="", collect=False):
    if not math.isfinite(mu) or mu < 0:
        raise ValueError("target_prox_mu must be finite and non-negative.")
    if scope not in PROX_SCOPES:
        raise ValueError("target_prox_scope must be helpers or all.")
    if mu and (algorithm != "FedTargetProj" or mode != "projection_softmax"):
        raise ValueError("Nonzero target_prox_mu requires FedTargetProj/projection_softmax.")
    if mu and (target_id != 0 or self_weight != .125):
        raise ValueError("This proximal experiment requires C0 and projection_self_weight=0.125.")
    if mu and (meta_split or collect):
        raise ValueError("Target proximal experiments use the original full training data, without meta protocols.")


def prox_experiment(args):
    return bool(getattr(args, "target_prox_experiment", False) or getattr(args, "target_prox_mu", 0.) > 0)


def effective_prox_mu(args, client_id):
    mu = getattr(args, "target_prox_mu", 0.)
    return mu if mu and (getattr(args, "target_prox_scope", "helpers") == "all"
                        or client_id != getattr(args, "target_client_id", 0)) else 0.


def prox_tag(mu, scope):
    return f"prox_{scope}_mu{repr(float(mu)).replace('.', 'p').replace('-', 'm').replace('+', '')}"


def ensure_prox_names(args):
    if not prox_experiment(args):
        return
    tag = prox_tag(getattr(args, "target_prox_mu", 0.), getattr(args, "target_prox_scope", "helpers"))
    if hasattr(args, "exp_name") and tag not in args.exp_name:
        args.exp_name += "_" + tag
    for key in ("save_folder_name", "final_model_root", "h5_result_root", "training_log_root", "run_log_dir"):
        value = getattr(args, key, "")
        if value and tag not in value:
            setattr(args, key, os.path.join(value, tag))
    # Shell runners may point stdout at a generic folder. The internal captured
    # log tree must still identify mu/scope and stay separate from model storage.
    if hasattr(args, "run_log_dir") and not args.run_log_dir:
        external = os.environ.get("FEDCLIP_TRAIN_LOG_DIR", "")
        if external and tag not in external:
            args.run_log_dir = os.path.join(external, tag)

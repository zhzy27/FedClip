"""Resolve the three controlled factors for the ResNet-18 ablation branch."""

import torch


TARGET_MODEL_FAMILY = "Decom_resnet18_5"


def is_resnet_3factor_target(args):
    return (
        getattr(args, "algorithm", None) == "FedCLIP"
        and getattr(args, "model_family", None) == TARGET_MODEL_FAMILY
    )


def use_resnet_personalized_aggregation(args):
    return bool(getattr(args, "resnet_personalized_agg", 1))


def effective_resnet_rho(args):
    """Return the effective U/base learning-rate ratio for the target ResNet."""
    if not bool(getattr(args, "use_asymmetric_lr", 1)):
        return 1.0
    return 0.1 if bool(getattr(args, "resnet_legacy_rho", 1)) else 0.3


def effective_resnet_rank_dropout_mode(args):
    return "original" if bool(getattr(args, "resnet_ordered_dropout", 1)) else "none"


def resnet_aggregation_method_name(args):
    if is_resnet_3factor_target(args) and not use_resnet_personalized_aggregation(args):
        return "aggregate_parameters_avg"
    if "resnet" in getattr(args, "model_family", "").lower():
        return "aggregate_parameters_v_svd_res"
    return "aggregate_parameters_v_svd"


def sample_weighted_average_parameter_dicts_(target_params, source_params, weights):
    """Write a sample-weighted average into an existing parameter dictionary."""
    if not source_params:
        raise ValueError("At least one source model is required for Avg aggregation.")
    if len(source_params) != len(weights):
        raise ValueError("The number of source models and Avg weights must match.")

    target_names = tuple(target_params.keys())
    for source_idx, params in enumerate(source_params):
        if tuple(params.keys()) != target_names:
            raise RuntimeError(f"Source model {source_idx} is incompatible with the Avg target.")

    with torch.no_grad():
        for target_param in target_params.values():
            target_param.zero_()
        for source_idx, weight in enumerate(weights):
            for name, target_param in target_params.items():
                source_param = source_params[source_idx][name]
                if source_param.shape != target_param.shape:
                    raise RuntimeError(
                        f"Avg shape mismatch for {name}: target={tuple(target_param.shape)}, "
                        f"source={tuple(source_param.shape)}"
                    )
                target_param.add_(
                    source_param.to(device=target_param.device, dtype=target_param.dtype),
                    alpha=float(weight),
                )


def print_resnet_3factor_summary(args):
    if not is_resnet_3factor_target(args):
        return

    personalized = use_resnet_personalized_aggregation(args)
    rho = effective_resnet_rho(args)
    dropout_mode = effective_resnet_rank_dropout_mode(args)
    base_lr = float(getattr(args, "local_learning_rate", 0.005))
    asymmetric = bool(getattr(args, "use_asymmetric_lr", 1))
    u_lr = base_lr * rho if asymmetric else base_lr

    print("===== ResNet 3-Factor Ablation =====")
    print(f"Personalized aggregation: {'ON' if personalized else 'OFF'}")
    print(f"Effective rho: {rho}")
    print(f"Ordered rank dropout: {'ON' if dropout_mode == 'original' else 'OFF'}")
    print(f"Effective U LR: {u_lr:.6f}")
    print(f"Effective V LR: {base_lr:.6f}")
    print(
        "Aggregation function: "
        + resnet_aggregation_method_name(args)
    )
    print(f"Rank dropout mode: {dropout_mode}")
    print("===================================")

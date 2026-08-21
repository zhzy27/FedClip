import copy
import math

import torch


def validate_factor_continuation_mode(
    factor_continuation,
    homogeneous_capacity,
):
    if factor_continuation and not homogeneous_capacity:
        raise ValueError(
            "factor_continuation=1 requires homogeneous_capacity=1 "
            "because all clients must have identical U/V shapes."
        )


def resolve_capacity_ratios(
    default_ratios,
    homogeneous_capacity=False,
    homogeneous_ratio=0.35,
):
    """Return the original capacity list or one shared client capacity."""
    ratios = [float(ratio) for ratio in default_ratios]
    if not homogeneous_capacity:
        return ratios

    ratio = float(homogeneous_ratio)
    if not math.isfinite(ratio) or ratio <= 0.0 or ratio > 1.0:
        raise ValueError(
            "homogeneous_ratio must be finite and in (0, 1], got "
            f"{homogeneous_ratio}."
        )
    return [ratio]


def factor_kind(parameter_name):
    if parameter_name.endswith("weight_u") or parameter_name.endswith("conv_u"):
        return "u"
    if parameter_name.endswith("weight_v") or parameter_name.endswith("conv_v"):
        return "v"
    return None


def factor_rank_signature(model):
    """Return each U factor's actual retained rank."""
    signature = []
    for name, parameter in model.named_parameters():
        if factor_kind(name) != "u":
            continue
        if parameter.ndim != 2:
            raise RuntimeError(
                f"Factor continuation expects matrix U factors, got "
                f"{name} with shape {tuple(parameter.shape)}."
            )
        signature.append((name, int(parameter.shape[1])))
    return tuple(signature)


def sample_weighted_factor_average(models, weights):
    """Average matching factorized models without recovering or re-SVD."""
    if not models:
        raise ValueError("Factor continuation requires at least one client model.")
    if len(models) != len(weights):
        raise ValueError(
            f"Expected one weight per model, got {len(weights)} weights for "
            f"{len(models)} models."
        )

    numeric_weights = [float(weight) for weight in weights]
    if any(not math.isfinite(weight) or weight < 0.0 for weight in numeric_weights):
        raise ValueError(f"Invalid factor aggregation weights: {numeric_weights}.")
    if not math.isclose(sum(numeric_weights), 1.0, rel_tol=1e-6, abs_tol=1e-6):
        raise ValueError(
            "Factor aggregation weights must sum to one, got "
            f"{sum(numeric_weights):.12g}."
        )

    averaged_model = copy.deepcopy(models[0])
    averaged_parameters = dict(averaged_model.named_parameters())
    reference_names = tuple(averaged_parameters)
    source_parameters = []
    for model_index, model in enumerate(models):
        parameters = dict(model.named_parameters())
        if tuple(parameters) != reference_names:
            raise RuntimeError(
                "Factor continuation parameter names differ for client model "
                f"index {model_index}."
            )
        for name, reference in averaged_parameters.items():
            if parameters[name].shape != reference.shape:
                raise RuntimeError(
                    "Factor continuation shape mismatch for "
                    f"{name}: reference={tuple(reference.shape)}, "
                    f"client_{model_index}={tuple(parameters[name].shape)}."
                )
        source_parameters.append(parameters)

    with torch.no_grad():
        for parameter in averaged_parameters.values():
            parameter.zero_()
        for weight, parameters in zip(numeric_weights, source_parameters):
            for name, target in averaged_parameters.items():
                target.add_(parameters[name].to(target.device), alpha=weight)

    return averaged_model

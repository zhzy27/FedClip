"""Fixed beta=1 target-anchored budget on the sum of masked helper updates."""

import math

import torch

from utils.layer_mask import aggregate_layer_mask, _norm
from utils.target_projection import EPS, _delta


BUDGET_BETA = 1.0


@torch.no_grad()
def clip_helper_budget(helper, target_local, groups):
    """Clip each whole logical layer once, preserving its direction and inputs.

    Target-near-zero takes precedence when both norms are tiny and H is nonzero.
    An exactly zero helper always has scale=1. Layers within budget are unchanged.
    """
    clipped = {name: value.detach().clone() for name, value in helper.items()}
    rows = []
    for layer, names in groups.items():
        target_norm = _norm({name: target_local[name] for name in names})
        helper_norm = _norm({name: helper[name] for name in names})
        if target_norm < EPS and helper_norm > 0.0:
            scale = 0.0
        elif helper_norm < EPS or helper_norm <= BUDGET_BETA * target_norm:
            scale = 1.0
        else:
            scale = min(1.0, BUDGET_BETA * target_norm / (helper_norm + EPS))
        if scale < 1.0:
            for name in names:
                clipped[name].mul_(scale)
        rows.append({
            "layer": layer,
            "target_local_norm": target_norm,
            "raw_helper_norm": helper_norm,
            "helper_target_ratio": helper_norm / (target_norm + EPS),
            "budget_scale": scale,
            "clipped_helper_norm": _norm({name: clipped[name] for name in names}),
            "budget_active": scale < 1.0,
        })
    metrics = {
        "budget_active_layers": sum(row["budget_active"] for row in rows),
        "max_helper_target_ratio": max(row["helper_target_ratio"] for row in rows),
        "mean_helper_target_ratio": sum(row["helper_target_ratio"] for row in rows) / len(rows),
        "raw_helper_norm": math.sqrt(sum(row["raw_helper_norm"] ** 2 for row in rows)),
        "clipped_helper_norm": math.sqrt(sum(row["clipped_helper_norm"] ** 2 for row in rows)),
    }
    return clipped, rows, metrics


@torch.no_grad()
def aggregate_layer_mask_budget(global_params, target_post, target_pre, uploads, target_id, groups):
    # Accumulate H separately, never infer it by subtracting a large anchor from
    # rounded model weights. Consume each upload only once, using the existing mask.
    helper = {name: torch.zeros_like(value) for name, value in global_params.items()}
    result, metrics, clients, layers, matrices = aggregate_layer_mask(
        global_params, target_post, target_pre, uploads, target_id, groups,
        helper_accumulator=helper,
    )
    clipped, budget_rows, budget_metrics = clip_helper_budget(
        helper, _delta(target_post, target_pre), groups,
    )
    for row in budget_rows:
        if row["budget_active"]:
            for name in groups[row["layer"]]:
                result[name] = target_post[name].detach().to(result[name]) + clipped[name]
        # Inactive layers retain the exact original LayerMask floating-point
        # addition order, guaranteeing bitwise equality when no budget is active.
    metrics.update(budget_metrics)
    matrices.update({"budget_beta": BUDGET_BETA, "budget_layers": budget_rows,
                     "budget_summary": budget_metrics})
    return result, metrics, clients, layers, matrices


def print_layer_budget_diagnostics(round_number, matrices):
    print(f"\n[LayerBudget][Round {round_number}] beta={BUDGET_BETA:g}")
    print(f"{'Layer':<16} {'target_norm':>15} {'raw_helper_norm':>17} {'helper/target':>15} "
          f"{'budget_scale':>15} {'clipped_helper_norm':>20} {'budget_active':>14}")
    for row in matrices["budget_layers"]:
        print(f"{row['layer']:<16} {row['target_local_norm']:>15.6e} {row['raw_helper_norm']:>17.6e} "
              f"{row['helper_target_ratio']:>15.6e} {row['budget_scale']:>15.8g} "
              f"{row['clipped_helper_norm']:>20.6e} {str(row['budget_active']):>14}")
    metrics = matrices["budget_summary"]
    print(f"budget_active_layers = {metrics['budget_active_layers']} / {len(matrices['budget_layers'])}")
    for key in ("max_helper_target_ratio", "mean_helper_target_ratio", "raw_helper_norm", "clipped_helper_norm"):
        print(f"{key} = {metrics[key]:.8e}")

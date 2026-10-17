"""Independent global-delta aggregation for one fixed target group."""

import math
from contextlib import contextmanager
import random

import numpy as np
import torch

from utils.target_projection import EPS, _cosine, _delta, _dot

from utils.target_group_options import GROUP_MODES, GROUP_SOFTMAX_MODES, DEFAULT_GROUP_IDS, GROUP_SOFTMAX_TAU, normalize_group_ids, validate_group_options, group_tag


@contextmanager
def preserved_group_rng():
    """Extra group observation never advances normal training's random streams."""
    python_rng, numpy_rng = random.getstate(), np.random.get_state()
    try:
        with torch.random.fork_rng():
            yield
    finally:
        random.setstate(python_rng)
        np.random.set_state(numpy_rng)


def effective_count(weights):
    """Conditional effective count; exactly zero for an empty/zero-mass role."""
    mass = math.fsum(weights)
    return 1. / math.fsum((w / mass) ** 2 for w in weights) if mass else 0.


@torch.no_grad()
def aggregate_group_updates(global_params, uploads_factory, samples, group_ids, mode, group_weight=None):
    """Read existing full-W uploads; never modify inputs or request new uploads.

    Three bounded GPU passes build the reference, score raw updates, and sum.
    Only group deltas are temporarily retained on CPU for pair diagnostics.
    All reads/recovery/diagnostics preserve Python/NumPy/Torch CPU/CUDA RNG.
    """
    ids = validate_group_options(group_ids, group_weight, mode, len(samples))
    if mode not in GROUP_MODES or not global_params or not callable(uploads_factory):
        raise ValueError("Require a group mode, full-W parameters and replayable uploads.")
    if set(samples) != set(range(len(samples))) or any(
            not math.isfinite(w) or not 0 <= w <= 1 for w in samples.values()) or not math.isclose(
                math.fsum(samples.values()), 1., abs_tol=1e-6, rel_tol=0):
        raise ValueError("Sample weights must cover all clients and sum to one.")
    members = set(ids)
    outside = sorted(set(samples) - members)
    group_sample_mass = math.fsum(samples[cid] for cid in ids)
    if group_sample_mass <= 0:
        raise ValueError("The group needs positive total training sample mass.")
    a = {cid: samples[cid] / group_sample_mass for cid in ids}
    # This is n_group/n_all, including the all-members case exactly equal to 1.
    # Normalized sample fractions can sum to 1 +/- a last-bit rounding error.
    beta = (group_sample_mass / math.fsum(samples.values()) if group_weight is None else group_weight)
    if mode == "group_avg_only":
        beta = 1.
    if mode in GROUP_SOFTMAX_MODES and not outside and 1. - beta > 0:
        raise ValueError("Positive outside mass requires at least one outside client.")
    order = []
    def checked(pass_number):
        seen, actual_order = set(), []
        for cid, weight, post in uploads_factory():
            if cid not in samples or cid in seen or weight != samples[cid]:
                raise ValueError("Upload IDs/sample weights changed or are invalid.")
            seen.add(cid)
            actual_order.append(cid)
            yield cid, post
        if seen != set(samples):
            raise ValueError("Every client must upload in every group mode.")
        if pass_number == 0:
            order.extend(actual_order)
        elif actual_order != order:
            raise ValueError("Upload summation order changed between reads.")

    with preserved_group_rng():
        reference = {n: torch.zeros_like(p) for n, p in global_params.items()}
        member_deltas = {}
        for cid, post in checked(0):
            delta = _delta(post, global_params)
            _dot(delta, delta)  # Finite/parameter-set check for every ordinary upload.
            if cid in members:
                for name in reference:
                    reference[name].add_(delta[name] * a[cid])
                member_deltas[cid] = {n: p.cpu().clone() for n, p in delta.items()}
            del post, delta
        ref_sq = _dot(reference, reference)
        ref_norm = math.sqrt(ref_sq)
        pair_cosines, pair_conflicts = [], []
        member_norm_sq = {cid: _dot(d, d) for cid, d in member_deltas.items()}
        for left in ids:
            cosines, conflicts = [], []
            for right in ids:
                dot = _dot(member_deltas[left], member_deltas[right])
                cosines.append(_cosine(dot, member_norm_sq[left], member_norm_sq[right]))
                conflicts.append(int(left != right and dot < 0))
            pair_cosines.append(cosines)
            pair_conflicts.append(conflicts)
        del member_deltas
        rows = []
        for cid, post in checked(1):
            delta = _delta(post, global_params)
            dot, norm = _dot(delta, reference), math.sqrt(_dot(delta, delta))
            projected = mode == "group_projection_softmax" and cid not in members and dot < 0
            coefficient = dot / (ref_sq + EPS) if projected else 0.
            removed = abs(coefficient) * ref_norm
            rows.append(dict(client_id=cid, in_target_group=int(cid in members),
                original_sample_weight=float(samples[cid]), group_sample_share=a.get(cid, 0.),
                cosine_before_projection=_cosine(dot, norm ** 2, ref_sq),
                dot_before_projection=dot, raw_conflict_with_reference=int(dot < 0),
                conflict=int(projected), projection_coefficient=coefficient,
                delta_norm=norm, removed_component_norm=removed,
                dot_after_projection=dot - coefficient * ref_sq))
            del delta, post
        lookup = {row["client_id"]: row for row in rows}
        shift = max((lookup[cid]["cosine_before_projection"] for cid in outside), default=0.)
        scores = {cid: math.exp((lookup[cid]["cosine_before_projection"] - shift) / GROUP_SOFTMAX_TAU)
                  for cid in outside} if mode in GROUP_SOFTMAX_MODES else {}
        denominator = math.fsum(scores.values())
        q = {cid: score / denominator for cid, score in scores.items()}
        if mode == "group_avg_all":
            weights = dict(samples)
        elif mode == "group_avg_only":
            weights = {cid: a.get(cid, 0.) for cid in samples}
        else:
            weights = {cid: beta * a[cid] if cid in members else (1. - beta) * q[cid]
                       for cid in samples}
        if any(not math.isfinite(w) or w < 0 for w in weights.values()) or not math.isclose(
                math.fsum(weights.values()), 1., abs_tol=1e-6, rel_tol=0):
            raise ValueError("Invalid group aggregation mass.")
        accumulated = {n: torch.zeros_like(p) for n, p in global_params.items()}
        correction = 0.
        for cid, post in checked(2):
            if mode in GROUP_SOFTMAX_MODES:
                values = _delta(post, global_params)
                if mode == "group_projection_softmax":
                    correction += weights[cid] * lookup[cid]["projection_coefficient"]
            else:
                # Exact legacy Avg parameter arithmetic/order for group_avg_all.
                values = post
            for name in accumulated:
                accumulated[name].add_(values[name].detach().to(accumulated[name]) * weights[cid])
            del values, post
        if mode in GROUP_SOFTMAX_MODES:
            if mode == "group_projection_softmax":
                for name in accumulated:
                    accumulated[name].add_(reference[name], alpha=-correction)
            result = {n: global_params[n].detach() + p for n, p in accumulated.items()}
        else:
            result = accumulated
        group_mass = math.fsum(weights[cid] for cid in ids)
        outside_mass = math.fsum(weights[cid] for cid in outside)
        for row in rows:
            cid = row["client_id"]
            row.update(aggregation_weight=weights[cid], raw_softmax_score=scores.get(cid, 0.),
                       outside_q=q.get(cid, 0.),
                       weighted_removed_norm=weights[cid] * row["removed_component_norm"])
        metrics = dict(group_member_count=len(ids), outside_client_count=len(outside),
            configured_group_beta=beta, target_group_weight_explicit=int(group_weight is not None),
            original_group_sample_mass=group_sample_mass, actual_group_mass=group_mass,
            actual_outside_mass=outside_mass, aggregation_mass=math.fsum(weights.values()),
            group_reference_norm=ref_norm, reference_is_zero=int(ref_sq == 0),
            reference_epsilon_dominated=int(ref_sq <= EPS),
            outside_conflict_count=sum(lookup[cid]["conflict"] for cid in outside),
            weighted_removed_norm=math.fsum(row["weighted_removed_norm"] for row in rows),
            effective_all_client_count=effective_count(list(weights.values())),
            effective_group_count=effective_count([weights[cid] for cid in ids]),
            effective_outside_count=effective_count([weights[cid] for cid in outside]),
            min_group_weight=min(weights[cid] for cid in ids),
            max_group_weight=max(weights[cid] for cid in ids),
            min_outside_weight=min((weights[cid] for cid in outside), default=0.),
            max_outside_weight=max((weights[cid] for cid in outside), default=0.))
        matrices = dict(target_group_ids=ids, group_pair_cosine_matrix=pair_cosines,
            group_pair_conflict_matrix=pair_conflicts,
            group_member_update_norms=[math.sqrt(member_norm_sq[cid]) for cid in ids],
            reference_scope="sample_weighted_group_global_delta",
            parameter_scope="recovered_full_W_named_parameters_including_head_no_buffers",
            projection_scope="full_model_outside_only" if mode == "group_projection_softmax" else "none",
            weighting_scope="outside_raw_cosine_softmax" if mode in GROUP_SOFTMAX_MODES else "sample_count_only",
            softmax_temperature=GROUP_SOFTMAX_TAU, softmax_shift_max=shift,
            projection_epsilon=EPS, aggregation_order=order,
            zero_mass_convention="effective role count=0; role min/max weights=0")
        if mode == "group_softmax_only":
            # Keep conflict's existing meaning (projection actually performed).
            # Raw negative dots remain visible independently of that field.
            metrics.update(projection_enabled=0,
                raw_outside_conflict_count=sum(lookup[cid]["raw_conflict_with_reference"] for cid in outside),
                projected_client_count=0)
            matrices.update(projection_enabled=0,
                conflict_field_semantics="conflict=projection_performed; raw_conflict_with_reference=negative_raw_dot")
        return result, metrics, sorted(rows, key=lambda row: row["client_id"]), matrices


def accuracy_statistics(members, key):
    values = [row[key] for row in members]
    if not values or any(not math.isfinite(v) or not 0 <= v <= 1 for v in values):
        raise ValueError("Group accuracy needs finite per-member fractions.")
    counts = [row["test_samples"] for row in members]
    if any(n <= 0 for n in counts):
        raise ValueError("Every member needs test samples.")
    mean = math.fsum(values) / len(values)
    weighted = math.fsum(v * n for v, n in zip(values, counts)) / sum(counts)
    correct_key = key.removesuffix("_acc") + "_correct"
    if any(correct_key in row for row in members):
        if any(correct_key not in row or not 0 <= row[correct_key] <= row["test_samples"]
               or not math.isclose(row[key], row[correct_key] / row["test_samples"], abs_tol=1e-15)
               for row in members):
            raise ValueError("Member accuracy and correct/sample counts disagree.")
        weighted = sum(row[correct_key] for row in members) / sum(counts)
    return dict(macro=mean, sample_weighted=weighted,
                worst=min(values), std=math.sqrt(math.fsum((v - mean) ** 2 for v in values) / len(values)))


def group_accuracy_summary(history):
    if not history:
        return {}
    weighted_primary = (history[0].get("mode") == "group_softmax_only" or
                        history[0].get('primary_average') == 'test_sample_weighted_average')
    def summary(rows, key):
        best = max(rows, key=lambda row: row[key])  # First occurrence resolves ties.
        tail = rows[-10:]
        result = dict(final=rows[-1][key], last10=math.fsum(row[key] for row in tail) / len(tail),
                      last10_count=len(tail), best=best[key], best_round=best["round"])
        if weighted_primary:
            tail20 = rows[-20:]
            result.update(last20=math.fsum(row[key] for row in tail20) / len(tail20), last20_count=len(tail20))
        return result
    primary_key = "group_post_local_sample_weighted_acc" if weighted_primary else "group_post_local_acc"
    result = dict(group_post_local=summary(history, primary_key), members={})
    if weighted_primary:
        result["group_post_local_macro"] = summary(history, "group_post_local_acc")
    for member in history[0]["members"]:
        cid = member["client_id"]
        rows = [dict(round=row["round"], **next(m for m in row["members"] if m["client_id"] == cid))
                for row in history]
        result["members"][str(cid)] = summary(rows, "post_local_acc")
    return result

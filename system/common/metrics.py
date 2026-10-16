"""Pure metrics over collected records. Never evaluate models or create loaders."""
import math
from numbers import Integral

import numpy as np


def sample_weighted_accuracy(correct, samples, *, client_ids=None, subset_ids=None):
    """Return a FRACTION, sum(correct)/sum(samples), in the supplied order.

    Require nonempty, finite counts, positive per-client sample counts and
    explicit valid IDs for a subset. No zero/empty/nonfinite fallback is used.
    Builtin sum intentionally preserves the legacy reduction order.
    """
    correct, samples = list(correct), list(samples)
    if len(correct) != len(samples) or not samples:
        raise ValueError("correct/samples must have the same nonzero length")
    ids = list(range(len(samples))) if client_ids is None else list(client_ids)
    if len(ids) != len(samples) or len(set(ids)) != len(ids):
        raise ValueError("client_ids must be unique and match count lengths")
    for c, n in zip(correct, samples):
        if not math.isfinite(c) or not math.isfinite(n) or n <= 0 or c < 0 or c > n:
            raise ValueError("Require finite 0 <= correct <= samples and positive samples")
    if subset_ids is not None:
        if client_ids is None:
            raise ValueError("Explicit client_ids are required for subset selection")
        subset = list(subset_ids)
        if not subset or len(set(subset)) != len(subset) or not set(subset).issubset(ids):
            raise ValueError("subset_ids must be nonempty, unique and present in client_ids")
        wanted = set(subset)
        selected = [i for i, cid in enumerate(ids) if cid in wanted]
    else:
        selected = range(len(ids))
    return sum(correct[i] for i in selected) * 1.0 / sum(samples[i] for i in selected)


def summarize_evaluations(points, *, value_key, unit, evaluation_object):
    """Summary in the EXPLICIT input unit ('fraction' or 'percent').

    Each chronological point supplies loop_round and completed_train_count.
    Tail windows count observed evaluation points, never implicit training rounds.
    Declared percent inputs stay percent throughout the reduction, avoiding a
    divide/multiply round trip that would change legacy exported tail means.
    """
    points = list(points)
    if not points or unit not in {"fraction", "percent"} or not evaluation_object:
        raise ValueError("Require nonempty evaluation points, explicit unit and evaluation object")
    values = []
    previous = -1
    for point in points:
        loop, completed, value = point["loop_round"], point["completed_train_count"], point[value_key]
        if not isinstance(loop, Integral) or loop < 0 or loop <= previous or not isinstance(completed, Integral) or completed < 1:
            raise ValueError("Require chronological nonnegative loop_round and positive completed_train_count")
        if not math.isfinite(value) or not 0 <= value <= (1 if unit == "fraction" else 100):
            raise ValueError(f"Nonfinite/out-of-range accuracy for unit={unit}")
        if point.get("evaluation_object", evaluation_object) != evaluation_object:
            raise ValueError("Mixed evaluation objects in one summary")
        previous = loop
        values.append(value)
    best = max(values)
    idx = values.index(best)
    return dict(unit=unit, evaluation_object=evaluation_object, averaging_unit="evaluation points",
                final=values[-1], best=best, best_loop_round=points[idx]["loop_round"],
                best_completed_train_count=points[idx]["completed_train_count"],
                last10=float(np.mean(values[-10:])), last20=float(np.mean(values[-20:])),
                last10_count=min(10, len(values)), last20_count=min(20, len(values)), point_count=len(values))

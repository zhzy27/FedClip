"""Low-rank-coordinate proximal loss and read-only CPU observation of local SGD."""

from contextlib import contextmanager
import math
from pathlib import Path
import random

import numpy as np
import torch

from utils.target_prox_options import effective_prox_mu


def trainable_parameters(model):
    return {name: p for name, p in model.named_parameters() if p.requires_grad}


def make_prox_anchor(model):
    parameters = trainable_parameters(model)
    if not parameters:
        raise ValueError("Target proximal training needs trainable parameters.")
    return {name: p.detach().clone() for name, p in parameters.items()}


def proximal_penalty(parameters, anchor, mu):
    if parameters.keys() != anchor.keys() or any(parameters[n].shape != a.shape
            or parameters[n].device != a.device or parameters[n].dtype != a.dtype or a.requires_grad
            for n, a in anchor.items()):
        raise ValueError("Proximal anchor names/shapes/dtype/device or gradient flags do not match.")
    return (mu / 2.) * sum((p - anchor[name]).square().sum() for name, p in parameters.items())


@contextmanager
def protected_observation():
    python_rng, numpy_rng = random.getstate(), np.random.get_state()
    try:
        with torch.random.fork_rng():
            yield
    finally:
        random.setstate(python_rng)
        np.random.set_state(numpy_rng)


def _cpu_local_parameters(client):
    # Observation does not allocate CUDA model copies or recover/decompose it.
    # These are the actual, saved, low-rank parameters and include the head.
    path = Path(client.save_folder_name) / f"{client.role}_model.pt"
    model = torch.load(path, map_location="cpu", weights_only=False)
    return {n: p.detach().clone() for n, p in trainable_parameters(model).items()}


def observe_prox_local_train(client, loop_round):
    with protected_observation():
        reference = _cpu_local_parameters(client)
    client.train(current_round=loop_round)
    with protected_observation(), torch.no_grad():
        final = _cpu_local_parameters(client)
        if final.keys() != reference.keys() or any(final[n].shape != p.shape for n, p in reference.items()):
            raise ValueError("Ordinary low-rank parameter set changed during local training.")
        square = math.fsum(float((final[n].double() - p.double()).square().sum()) for n, p in reference.items())
    stats = client.last_normal_train_stats
    active_mu = effective_prox_mu(client.args, client.id)
    row = dict(round=loop_round + 1, loop_round=loop_round, client_id=client.id,
        target_prox_mu=getattr(client.args, "target_prox_mu", 0.),
        target_prox_scope=getattr(client.args, "target_prox_scope", "helpers"),
        prox_enabled=int(active_mu > 0), effective_prox_mu=active_mu,
        ce_loss=stats["ce_loss"], frobenius_regularization_loss=stats["regularization_loss"],
        proximal_loss=stats["proximal_loss"], low_rank_start_squared_distance=square,
        low_rank_start_distance=math.sqrt(square), end_proximal_loss=.5 * active_mu * square,
        ordinary_train_seconds=stats["train_seconds"], actual_local_epochs=stats["local_epochs"],
        processed_train_samples=stats["processed_train_samples"], train_batches=stats["train_batches"],
        prox_parameter_count=sum(p.numel() for p in reference.values()))
    if any(not math.isfinite(value) for value in row.values() if isinstance(value, float)):
        raise ValueError("Non-finite proximal local diagnostic.")
    print("[TargetProxLocal] " + " ".join(f"{key}={value}" for key, value in row.items()))
    return row

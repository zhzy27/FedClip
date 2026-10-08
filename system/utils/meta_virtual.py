"""Exact functional C0 adaptation: native SVD, full SGD unroll, validation CE."""

from contextlib import contextmanager
import copy
import random
import time

import numpy as np
import torch
from torch import nn
from torch.func import functional_call

from utils.target_projection import EPS, _delta, _dot


@contextmanager
def preserved_rng(seed=None):
    python_state, numpy_state = random.getstate(), np.random.get_state()
    try:
        with torch.random.fork_rng():
            if seed is not None:
                random.seed(seed)
                np.random.seed(seed)
                torch.manual_seed(seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(seed)
            yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)


def sync(device):
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize(device)


@contextmanager
def snapshot_backend(profile):
    previous = dict(cudnn_benchmark=torch.backends.cudnn.benchmark,
        cudnn_deterministic=torch.backends.cudnn.deterministic,
        cudnn_allow_tf32=torch.backends.cudnn.allow_tf32, matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32)
    def apply(values):
        torch.backends.cudnn.benchmark = values["cudnn_benchmark"]
        torch.backends.cudnn.deterministic = values["cudnn_deterministic"]
        torch.backends.cudnn.allow_tf32 = values["cudnn_allow_tf32"]
        torch.backends.cuda.matmul.allow_tf32 = values["matmul_allow_tf32"]
    try:
        if profile:
            apply(profile)
        yield
    finally:
        apply(previous)


@torch.no_grad()
def projection_coefficients(global_params, target_post, uploads):
    target = _delta(target_post, global_params)
    square = _dot(target, target)
    coefficients = {}
    for cid, post in uploads:
        dot = _dot(_delta(post, global_params), target)
        coefficients[cid] = dot / (square + EPS) if cid != 0 and dot < 0 else 0.
    return coefficients


def differentiable_projection(global_params, uploads, weights, coefficients, client_ids, aggregation_order=None):
    """Snapshots/coefficients are constants; only weights carry a gradient."""
    if list(client_ids) != sorted(client_ids) or len(weights) != len(client_ids):
        raise ValueError("Projection client IDs must be sorted and match weights.")
    initial = {n: p.detach() for n, p in global_params.items()}
    target_post = uploads(0)
    target = _delta(target_post, initial)
    mean = {n: torch.zeros_like(p) for n, p in initial.items()}
    correction = weights.new_zeros(())
    order = aggregation_order or client_ids
    if sorted(order) != list(client_ids):
        raise ValueError("Invalid snapshot aggregation order.")
    for cid in order:
        index = client_ids.index(cid)
        post = target_post if cid == 0 else uploads(cid)
        delta = _delta(post, initial)
        alpha = weights[index]
        for name in mean:
            mean[name] = mean[name] + delta[name] * alpha.to(mean[name])
        correction = correction + alpha * coefficients[cid]
    return {n: initial[n] + (mean[n] - target[n] * correction.to(mean[n])) for n in initial}


def differentiable_decompose(full_params, template):
    """Match FactorizedConv/Linear rank, flattening and sqrt(S) allocation."""
    parameters, diagnostics = {}, []
    factor_names = set()
    for prefix, module in template.named_modules():
        if hasattr(module, "conv_u") and hasattr(module, "conv_v"):
            k = module.kernel_size
            matrix = full_params[prefix + ".weight"].permute(0, 2, 1, 3).reshape(module.out_channels * k, module.in_channels * k)
            names = prefix + ".conv_u", prefix + ".conv_v"
        elif hasattr(module, "weight_u") and hasattr(module, "weight_v"):
            matrix = full_params[prefix + ".weight"]
            names = prefix + ".weight_u", prefix + ".weight_v"
        else:
            continue
        rank = module.rank
        expected = max(1, round(module.rank_rate * min(matrix.shape)))
        if rank != expected:
            raise ValueError(f"Unsupported rank rule in {prefix}.")
        try:
            u, s, vh = torch.linalg.svd(matrix, full_matrices=False)
        except RuntimeError as error:
            raise RuntimeError(f"Native SVD failed for {prefix}, matrix={tuple(matrix.shape)}: {error}") from error
        root = torch.sqrt(s[:rank])
        parameters[names[0]] = u[:, :rank] @ torch.diag(root)
        parameters[names[1]] = torch.diag(root) @ vh[:rank, :]
        factor_names.update(names)
        values = s.detach().double()
        gaps = (values[:-1] - values[1:]).abs()
        diagnostics.append(dict(layer=prefix, rank=rank, matrix_shape=list(matrix.shape),
            max_singular=values.max().item(), min_singular=values.min().item(),
            min_retained_singular=values[rank - 1].item(),
            min_relative_singular_gap=(gaps.min() / values.max()).item() if len(gaps) and values.max() else 0.,
            cutoff_gap=(values[rank - 1] - values[rank]).item() if rank < len(values) else None))
    for name, reference in template.named_parameters():
        if name not in factor_names:
            if name not in full_params or full_params[name].shape != reference.shape:
                raise ValueError(f"Unsupported full-W to template mapping: {name}")
            parameters[name] = full_params[name]
    if parameters.keys() != dict(template.named_parameters()).keys():
        raise ValueError("Functional decomposition parameter set differs from C0.")
    if any(not torch.isfinite(p).all() for p in parameters.values()):
        raise ValueError("Non-finite native SVD factors; no gradient approximation was applied.")
    return {name: parameters[name] for name, _ in template.named_parameters()}, diagnostics


class _Objective(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x, y, regularization):
        ce = nn.functional.cross_entropy(self.model(x), y)
        reg = regularization * self.model.frobenius_decay() if regularization else ce.new_zeros(())
        return ce + reg


def functional_loss(wrapper, parameters, buffers, x, y, regularization):
    state = {"model." + n: p for n, p in parameters.items()}
    state.update({"model." + n: p for n, p in buffers.items()})
    return functional_call(wrapper, state, (x, y, regularization), tie_weights=True, strict=True)


def clipped_functional_step(parameters, gradients, lr, clip_norm=10.):
    norm = torch.linalg.vector_norm(torch.stack([torch.linalg.vector_norm(g, 2) for g in gradients]), 2)
    scale = torch.clamp(clip_norm / (norm + 1e-6), max=1.)
    return {name: value.add(gradient * scale, alpha=-lr)
            for (name, value), gradient in zip(parameters.items(), gradients)}, norm


class _ExactSGDSegment(torch.autograd.Function):
    """Exact segment VJP recomputation, including all internal loss Hessians.

    PyTorch 2.0 non-reentrant checkpoint cannot nest the internal autograd.grad.
    Save original inputs (never detach them), recompute the same segment and
    return its full Jacobian-vector product to every preceding segment/SVD.
    """
    @staticmethod
    def forward(ctx, function, *inputs):
        ctx.function = function
        ctx.save_for_backward(*inputs)
        ctx.python_rng, ctx.numpy_rng = random.getstate(), np.random.get_state()
        ctx.cpu_rng = torch.get_rng_state()
        ctx.cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
        with torch.enable_grad():
            return function(*inputs)

    @staticmethod
    def backward(ctx, *grad_outputs):
        higher_order = torch.is_grad_enabled()
        with preserved_rng(), torch.enable_grad():
            random.setstate(ctx.python_rng)
            np.random.set_state(ctx.numpy_rng)
            torch.set_rng_state(ctx.cpu_rng)
            if ctx.cuda_rng:
                torch.cuda.set_rng_state_all(ctx.cuda_rng)
            inputs = ctx.saved_tensors
            outputs = ctx.function(*inputs)
            gradients = torch.autograd.grad(outputs, inputs, grad_outputs=grad_outputs,
                                            create_graph=higher_order)
        return (None, *gradients)


def functional_train(template, parameters, batches, lr=.005, regularization=.001, checkpoint_steps=0):
    wrapper = _Objective(template)
    wrapper.train()
    buffers = {n: b.detach().clone() for n, b in template.named_buffers()}
    if checkpoint_steps and buffers:
        raise ValueError("Segment recomputation currently requires a buffer-free CNN; it will not silently replay mutable buffers.")
    names = list(parameters)
    clipped = []
    def segment(*values, segment_batches):
        current = dict(zip(names, values))
        modes = [(module, module.training) for module in wrapper.modules()]
        try:
            wrapper.train()
            for x, y in segment_batches:
                loss = functional_loss(wrapper, current, buffers, x, y, regularization)
                gradients = torch.autograd.grad(loss, tuple(current.values()), create_graph=True)
                current, norm = clipped_functional_step(current, gradients, lr)
                if not checkpoint_steps:
                    clipped.append(norm.detach().item() > 10.)
        finally:
            for module, training in modes:
                module.training = training
        return tuple(current.values())
    size = checkpoint_steps or len(batches)
    if not size:
        raise ValueError("Virtual training requires a complete non-empty epoch plan.")
    for start in range(0, len(batches), size):
        block = batches[start:start + size]
        if checkpoint_steps:
            # Bind this block: backward recomputes the same segment, with no detach.
            def run(*values, block=block):
                return segment(*values, segment_batches=block)
            values = _ExactSGDSegment.apply(run, *parameters.values())
        else:
            values = segment(*parameters.values(), segment_batches=block)
        parameters = dict(zip(names, values))
    return wrapper, parameters, buffers, dict(actual_train_batches=len(batches),
                                             clipped_step_count=sum(clipped) if not checkpoint_steps else None)


def adapted_validation_loss(template, full_params, train_batches, validation_batches, lr=.005,
                            regularization=.001, seed=777, checkpoint_steps=0, return_parameters=False):
    device = next(iter(full_params.values())).device
    with preserved_rng(seed):
        model = copy.deepcopy(template).to(device)
        if any(isinstance(module, nn.modules.batchnorm._BatchNorm) for module in model.modules()):
            raise ValueError("Full meta adaptation is audited for the buffer-free low-rank CNN; mutable BatchNorm running-stat derivatives are unsupported.")
        sync(device)
        start = time.perf_counter()
        parameters, svd = differentiable_decompose(full_params, model)
        sync(device)
        decomposition_seconds = time.perf_counter() - start
        start = time.perf_counter()
        wrapper, parameters, buffers, details = functional_train(
            model, parameters, train_batches, lr, regularization, checkpoint_steps)
        sync(device)
        train_seconds = time.perf_counter() - start
        wrapper.eval()
        count, total = 0, None
        start = time.perf_counter()
        for x, y in validation_batches:
            ce = functional_loss(wrapper, parameters, buffers, x, y, 0.)
            total = ce * len(y) if total is None else total + ce * len(y)
            count += len(y)
        if not count:
            raise ValueError("Validation CE requires held-out C0 samples.")
        loss = total / count
        if not torch.isfinite(loss):
            raise ValueError("Non-finite adapted validation CE.")
        sync(device)
        details.update(decomposition_seconds=decomposition_seconds, virtual_train_seconds=train_seconds,
                       validation_seconds=time.perf_counter() - start, validation_samples=count,
                       svd=svd, checkpoint_steps=checkpoint_steps, gradient_path="native_svd_full_sgd_unroll")
        return (loss, details, parameters) if return_parameters else (loss, details)

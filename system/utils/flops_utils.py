import math
from collections import defaultdict

import torch
import torch.nn as nn


def _as_pair(value):
    if isinstance(value, tuple):
        return value
    return (value, value)


def _as_scalar(value):
    if isinstance(value, tuple):
        return value[0]
    return value


def _conv2d_flops(module, inputs, output):
    x = inputs[0]
    if not torch.is_tensor(x) or not torch.is_tensor(output):
        return 0
    if output.dim() != 4:
        return 0

    batch_size = output.shape[0]
    out_channels = output.shape[1]
    out_h = output.shape[2]
    out_w = output.shape[3]
    kernel_h, kernel_w = _as_pair(module.kernel_size)
    in_channels = module.in_channels
    groups = module.groups
    macs = batch_size * out_channels * out_h * out_w * (in_channels // groups) * kernel_h * kernel_w
    return 2 * macs


def _linear_flops(module, inputs, output):
    x = inputs[0]
    if not torch.is_tensor(x):
        return 0
    batch_items = x.numel() // max(1, module.in_features)
    macs = batch_items * module.in_features * module.out_features
    return 2 * macs


def _factorized_conv_flops(module, inputs, output):
    x = inputs[0]
    if not torch.is_tensor(x) or not torch.is_tensor(output):
        return 0
    if x.dim() != 4 or output.dim() != 4:
        return 0

    batch_size, in_channels, in_h, in_w = x.shape
    out_channels = getattr(module, "out_channels", output.shape[1])
    rank = getattr(module, "rank", None)
    if rank is None:
        return 0

    kernel_size = _as_scalar(getattr(module, "kernel_size", 1))
    stride = _as_scalar(getattr(module, "stride", 1))
    padding = _as_scalar(getattr(module, "padding", 0))

    first_h = in_h
    first_w = math.floor((in_w + 2 * padding - kernel_size) / stride + 1)
    out_h = output.shape[2]
    out_w = output.shape[3]

    first_macs = batch_size * rank * first_h * first_w * in_channels * kernel_size
    second_macs = batch_size * out_channels * out_h * out_w * rank * kernel_size
    return 2 * (first_macs + second_macs)


def _factorized_linear_flops(module, inputs, output):
    x = inputs[0]
    if not torch.is_tensor(x):
        return 0
    weight_v = getattr(module, "weight_v", None)
    weight_u = getattr(module, "weight_u", None)
    if weight_v is None or weight_u is None:
        return 0

    in_features = weight_v.shape[1]
    rank = weight_v.shape[0]
    out_features = weight_u.shape[0]
    batch_items = x.numel() // max(1, in_features)
    macs = batch_items * (in_features * rank + rank * out_features)
    return 2 * macs


def _factorized_conv_frobenius_flops(module):
    conv_v = getattr(module, "conv_v", None)
    conv_u = getattr(module, "conv_u", None)
    if conv_v is None or conv_u is None:
        return 0

    dim1, rank = conv_u.shape
    rank_v, dim2 = conv_v.shape
    if rank != rank_v:
        return 0

    matmul_flops = 2 * dim1 * rank * dim2
    square_and_sum_flops = 2 * dim1 * dim2
    return int(matmul_flops + square_and_sum_flops)


def _factorized_linear_frobenius_flops(module):
    weight_v = getattr(module, "weight_v", None)
    weight_u = getattr(module, "weight_u", None)
    if weight_v is None or weight_u is None:
        return 0

    out_features, rank = weight_u.shape
    rank_v, in_features = weight_v.shape
    if rank != rank_v:
        return 0

    matmul_flops = 2 * out_features * rank * in_features
    square_and_sum_flops = 2 * out_features * in_features
    return int(matmul_flops + square_and_sum_flops)


def _module_flops(module, inputs, output):
    if isinstance(module, nn.Conv2d):
        return _conv2d_flops(module, inputs, output)
    if isinstance(module, nn.Linear):
        return _linear_flops(module, inputs, output)
    if hasattr(module, "conv_v") and hasattr(module, "conv_u"):
        return _factorized_conv_flops(module, inputs, output)
    if hasattr(module, "weight_v") and hasattr(module, "weight_u"):
        return _factorized_linear_flops(module, inputs, output)
    return 0


def estimate_low_rank_frobenius_flops(model):
    """
    Estimate one call to model.frobenius_decay() for repo low-rank layers.
    This counts the explicit W = U @ V reconstruction plus the W**2 reduction.
    """
    totals = defaultdict(int)

    for module in model.modules():
        if module is model:
            continue
        if hasattr(module, "conv_v") and hasattr(module, "conv_u"):
            flops = _factorized_conv_frobenius_flops(module)
        elif hasattr(module, "weight_v") and hasattr(module, "weight_u"):
            flops = _factorized_linear_frobenius_flops(module)
        else:
            flops = 0
        if flops:
            totals[module.__class__.__name__] += int(flops)

    return int(sum(totals.values())), dict(totals)


def estimate_forward_flops(model, sample_input, device):
    """
    Estimate one forward pass FLOPs for Conv2d/Linear and the repo low-rank layers.
    One MAC is counted as two FLOPs.
    """
    totals = defaultdict(int)
    handles = []

    def hook(module, inputs, output):
        flops = _module_flops(module, inputs, output)
        if flops:
            totals[module.__class__.__name__] += int(flops)

    for module in model.modules():
        if module is model:
            continue
        should_hook = (
            isinstance(module, (nn.Conv2d, nn.Linear))
            or (hasattr(module, "conv_v") and hasattr(module, "conv_u"))
            or (hasattr(module, "weight_v") and hasattr(module, "weight_u"))
        )
        if should_hook:
            handles.append(module.register_forward_hook(hook))

    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            if isinstance(sample_input, (list, tuple)):
                model_input = sample_input[0].to(device)
            else:
                model_input = sample_input.to(device)
            try:
                model(model_input)
            except TypeError as exc:
                totals.clear()
                if hasattr(model, "base") and hasattr(model, "head"):
                    rep = model.base(model_input)
                    model.head(rep)
                else:
                    raise exc
    finally:
        for handle in handles:
            handle.remove()
        model.train(was_training)

    return int(sum(totals.values())), dict(totals)

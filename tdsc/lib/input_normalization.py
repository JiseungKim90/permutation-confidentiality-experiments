"""Exact input-normalization rewrite for ordinary zero-padded Conv2d.

This module is a normal-inference reference. It is not wired into the recorded
integer simulator or any recovery entry point. The legacy evidence therefore
retains its original model semantics.
"""

import torch
from torch import nn
from torch.nn import functional as F


def fold_normalized_conv2d(conv, mean, std, input_hw):
    """Return scaled weights and a spatial bias for a fixed input geometry.

For valid kernel positions, W * (x - mean) / std contributes a centering
term. Padded positions contribute zero in the normalized domain and must not
be subtracted. An all-one support mask computes exactly these valid terms.
Only ordinary, ungrouped, zero-padded convolutions are supported.
"""
    if not isinstance(conv, nn.Conv2d):
        raise TypeError("conv must be a Conv2d")
    if conv.groups != 1 or conv.padding_mode != "zeros":
        raise ValueError("only ungrouped zero-padding is supported")
    if isinstance(conv.padding, str):
        raise ValueError("padding must be an explicit integer pair")
    if len(input_hw) != 2 or any(int(n) != n or n <= 0 for n in input_hw):
        raise ValueError("input_hw must contain two positive integers")
    weight = conv.weight.detach()
    mean = torch.as_tensor(mean, dtype=weight.dtype, device=weight.device)
    std = torch.as_tensor(std, dtype=weight.dtype, device=weight.device)
    if mean.shape != (conv.in_channels,) or std.shape != mean.shape:
        raise ValueError("normalization vectors must match the input channels")
    if not torch.isfinite(mean).all() or not torch.isfinite(std).all() or not (std > 0).all():
        raise ValueError("mean must be finite and std finite and positive")
    mean = mean.reshape(1, -1, 1, 1)
    std = std.reshape(1, -1, 1, 1)
    support = torch.ones((1, conv.in_channels, int(input_hw[0]), int(input_hw[1])),
                         dtype=weight.dtype, device=weight.device)
    shift = F.conv2d(support * mean / std, weight, None, conv.stride,
                     conv.padding, conv.dilation)
    bias = torch.zeros(conv.out_channels, dtype=weight.dtype, device=weight.device)
    if conv.bias is not None:
        bias = conv.bias.detach()
    return weight / std, bias.reshape(1, -1, 1, 1) - shift


class NormalizedInputConv2d(nn.Module):
    """Ordinary inference on raw inputs, equivalent to normalize-then-convolve.

The bias map is tied to input_hw. A different shape is rejected instead of
silently applying a correction for the wrong image boundary. This is an
inference-only rewrite: the source convolution's parameters are copied.
"""

    def __init__(self, conv, mean, std, input_hw):
        super().__init__()
        weight, bias_map = fold_normalized_conv2d(conv, mean, std, input_hw)
        self.register_buffer("weight", weight.clone())
        self.register_buffer("bias_map", bias_map.clone())
        self.input_hw = tuple(int(n) for n in input_hw)
        self.in_channels = conv.in_channels
        self.stride = conv.stride
        self.padding = conv.padding
        self.dilation = conv.dilation

    def forward(self, raw):
        if raw.ndim != 4 or raw.shape[1] != self.in_channels or tuple(raw.shape[-2:]) != self.input_hw:
            raise ValueError("input shape does not match the validated convolution geometry")
        return F.conv2d(raw, self.weight, None, self.stride, self.padding,
                        self.dilation) + self.bias_map

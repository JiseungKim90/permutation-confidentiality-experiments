"""Independent scalar-oracle tests for an ordinary convolution rewrite."""
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lib.input_normalization import NormalizedInputConv2d


def scalar_convolution(x, weight, bias, stride, padding, dilation):
    n, cin, height, width = x.shape
    cout, _, kh, kw = weight.shape
    oh = (height + 2 * padding[0] - dilation[0] * (kh - 1) - 1) // stride[0] + 1
    ow = (width + 2 * padding[1] - dilation[1] * (kw - 1) - 1) // stride[1] + 1
    out = np.zeros((n, cout, oh, ow), dtype=np.float64)
    for batch in range(n):
        for co in range(cout):
            for row in range(oh):
                for col in range(ow):
                    value = float(bias[co])
                    for ci in range(cin):
                        for kr in range(kh):
                            for kc in range(kw):
                                ri = row * stride[0] - padding[0] + kr * dilation[0]
                                cj = col * stride[1] - padding[1] + kc * dilation[1]
                                if 0 <= ri < height and 0 <= cj < width:
                                    value += weight[co, ci, kr, kc] * x[batch, ci, ri, cj]
                    out[batch, co, row, col] = value
    return out


torch.set_num_threads(1)
rng = np.random.default_rng(20261002)
geometries = [
    ((3, 3), (3, 3), (1, 1), (1, 1), (1, 1)),
    ((5, 7), (3, 3), (2, 2), (1, 1), (1, 1)),
    ((7, 6), (3, 2), (1, 2), (2, 1), (2, 1)),
    ((5, 5), (1, 1), (1, 1), (0, 0), (1, 1)),
    ((1, 1), (3, 3), (1, 1), (1, 1), (1, 1)),
    ((6, 7), (3, 3), (1, 1), (0, 0), (1, 1)),
]
records = []
for hw, kernel, stride, padding, dilation in geometries:
    for has_bias in (False, True):
        conv = nn.Conv2d(3, 2, kernel, stride=stride, padding=padding,
                         dilation=dilation, bias=has_bias).double()
        with torch.no_grad():
            conv.weight.copy_(torch.from_numpy(rng.normal(size=tuple(conv.weight.shape))))
            if has_bias:
                conv.bias.copy_(torch.from_numpy(rng.normal(size=2)))
        mean, std = rng.normal(size=3), rng.uniform(0.1, 2.0, size=3)
        x = rng.uniform(-1.0, 1.0, size=(2, 3, hw[0], hw[1]))
        normalized = (x - mean[None, :, None, None]) / std[None, :, None, None]
        bias = conv.bias.detach().numpy() if has_bias else np.zeros(2)
        expected = scalar_convolution(normalized, conv.weight.detach().numpy(), bias,
                                      stride, padding, dilation)
        rewritten = NormalizedInputConv2d(conv, mean, std, hw)
        got = rewritten(torch.from_numpy(x)).detach().numpy()
        error = float(np.max(np.abs(got - expected)))
        assert error < 1e-10, (hw, kernel, error)
        records.append({"input_hw": hw, "kernel": kernel, "stride": stride,
                        "padding": padding, "dilation": dilation,
                        "bias": has_bias, "max_abs_error": error})
rejections = []
conv = nn.Conv2d(3, 2, 3, padding=1).double()
for label, thunk in [
    ("zero_std", lambda: NormalizedInputConv2d(conv, [0] * 3, [1, 0, 1], (3, 3))),
    ("nonfinite_mean", lambda: NormalizedInputConv2d(conv, [0, float("nan"), 0], [1] * 3, (3, 3))),
    ("wrong_channels", lambda: NormalizedInputConv2d(conv, [0] * 2, [1] * 2, (3, 3))),
    ("wrong_geometry", lambda: NormalizedInputConv2d(conv, [0] * 3, [1] * 3, (3, 3))(torch.zeros(1, 3, 4, 4, dtype=torch.float64))),
]:
    try:
        thunk()
    except ValueError:
        rejections.append(label)
    else:
        raise AssertionError("invalid input accepted: " + label)
print(json.dumps({"success": True, "seed": 20261002,
                  "scalar_oracle_cases": records, "invalid_inputs_rejected": rejections}, indent=2))

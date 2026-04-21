"""Shared attack functions: round-then-sort recovery and helpers."""
import numpy as np
import torch
import torch.nn as nn


def round_then_sort(observed, gamma):
    """Round each entry to nearest multiple of gamma, then sort."""
    return np.sort(np.round(observed / gamma) * gamma)


def simulate_query(true_values, p, noise_fn=None):
    """
    Simulate one Safhire query: permute + bounded noise.
    true_values: 1D array (one column's W_{:,i}+b after quantization)
    p: precision (quantization step = 1/p, noise bound = 1/(2p))
    noise_fn: callable(n) -> noise array, or None for default Gaussian
    Returns: observed array (permuted + noisy)
    """
    m = len(true_values)
    gamma = 1.0 / p
    Delta = 1.0 / (2 * p)
    B = Delta - 1e-12  # strict bound

    perm = np.random.permutation(m)
    if noise_fn is None:
        noise = np.random.normal(0, Delta / 3, m).clip(-B, B)
    else:
        noise = noise_fn(m)
    return true_values[perm] + noise


def attack_layer(W_mat, b_vec, p, noise_fn=None):
    """
    Full round-then-sort attack on one layer.
    W_mat: (C_out, d) weight matrix
    b_vec: (C_out,) bias vector
    p: precision parameter
    Returns: (max_error, list of recovered sorted columns)
    """
    gamma = 1.0 / p
    C_out, d = W_mat.shape

    # Quantize
    W_q = np.round(W_mat * p) / p
    b_q = np.round(b_vec * p) / p

    max_err = 0.0
    recovered_cols = []

    for i in range(d):
        true_col = W_q[:, i] + b_q
        true_sorted = np.sort(true_col)

        observed = simulate_query(true_col, p, noise_fn)
        recovered = round_then_sort(observed, gamma)

        err = np.max(np.abs(recovered - true_sorted))
        max_err = max(max_err, err)
        recovered_cols.append(recovered)

    return max_err, recovered_cols


def get_linear_layers(model):
    """Extract all Linear layers as (name, W_mat, b_vec, has_bias)."""
    layers = []
    for name, mod in model.named_modules():
        if isinstance(mod, nn.Linear):
            W = mod.weight.detach().numpy()
            b = mod.bias.detach().numpy() if mod.bias is not None else np.zeros(W.shape[0])
            layers.append((name, W, b, mod.bias is not None))
    return layers


def _get_parent_module(model, module_name):
    modules = dict(model.named_modules())
    parent_name = module_name.rsplit(".", 1)[0] if "." in module_name else ""
    return modules[parent_name]


def _following_batchnorm2d(model, conv_name):
    """Return the immediate sibling BatchNorm2d after conv_name, if present."""
    parent = _get_parent_module(model, conv_name)
    child_name = conv_name.rsplit(".", 1)[-1]
    children = list(parent.named_children())
    for idx, (name, _child) in enumerate(children[:-1]):
        if name == child_name and isinstance(children[idx + 1][1], nn.BatchNorm2d):
            return children[idx + 1][1]
    return None


def _fold_conv_batchnorm(conv, bn):
    """Fold an eval-mode BatchNorm2d into a Conv2d weight/bias pair."""
    W = conv.weight.detach().cpu()
    if conv.bias is None:
        b = torch.zeros(W.shape[0], dtype=W.dtype)
    else:
        b = conv.bias.detach().cpu()

    scale = bn.weight.detach().cpu() / torch.sqrt(
        bn.running_var.detach().cpu() + bn.eps
    )
    folded_W = W * scale.reshape(-1, 1, 1, 1)
    folded_b = (b - bn.running_mean.detach().cpu()) * scale + bn.bias.detach().cpu()
    return folded_W.numpy(), folded_b.numpy()


def get_conv_layers(model, fold_batchnorm=True):
    """Extract all Conv2d layers as (name, W_mat, b_vec, has_bias, C_out, d).

    When fold_batchnorm is true, an immediate following BatchNorm2d sibling is
    folded into the convolution, matching the affine layer exposed at inference.
    """
    layers = []
    for name, mod in model.named_modules():
        if isinstance(mod, nn.Conv2d):
            bn = _following_batchnorm2d(model, name) if fold_batchnorm else None
            if bn is not None:
                W, b = _fold_conv_batchnorm(mod, bn)
                has_bias = True
            else:
                W = mod.weight.detach().cpu().numpy()
                b = (
                    mod.bias.detach().cpu().numpy()
                    if mod.bias is not None
                    else np.zeros(W.shape[0])
                )
                has_bias = mod.bias is not None
            C_out = W.shape[0]
            d = int(np.prod(W.shape[1:]))
            W_mat = W.reshape(C_out, d)
            layers.append((name, W_mat, b, has_bias, C_out, d))
    return layers


def fingerprint_from_layer(W_mat, b_vec, p):
    """Compute sorted column-norm fingerprint via attack."""
    _, recovered_cols = attack_layer(W_mat, b_vec, p)
    norms = np.sort([np.linalg.norm(c) for c in recovered_cols])
    return norms


def noise_fns_bounded(Delta):
    """Return dict of noise functions, all with |noise| < Delta (strict)."""
    B = Delta - 1e-12
    return {
        "Zero":       lambda n, B=B: np.zeros(n),
        "Gaussian":   lambda n, B=B, D=Delta: np.random.normal(0, D/3, n).clip(-B, B),
        "Uniform":    lambda n, B=B: np.random.uniform(-B, B, n),
        "Triangular": lambda n, B=B: np.random.triangular(-B, 0, B, n),
        "Laplace":    lambda n, B=B, D=Delta: np.random.laplace(0, D/2, n).clip(-B, B),
        "Worst-case": lambda n, B=B: np.full(n, B) * np.where(np.arange(n)%2==0, 1, -1),
    }

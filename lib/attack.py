"""Shared attack functions: round-then-sort recovery and helpers."""
import numpy as np
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


def get_conv_layers(model):
    """Extract all Conv2d layers as (name, W_mat, b_vec, has_bias, C_out, d)."""
    layers = []
    for name, mod in model.named_modules():
        if isinstance(mod, nn.Conv2d):
            W = mod.weight.detach().numpy()
            C_out = W.shape[0]
            d = int(np.prod(W.shape[1:]))
            W_mat = W.reshape(C_out, d)
            b = mod.bias.detach().numpy() if mod.bias is not None else np.zeros(C_out)
            layers.append((name, W_mat, b, mod.bias is not None, C_out, d))
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

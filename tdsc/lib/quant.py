"""Per-layer symmetric quantisation and the integer forward pass (plan Section 1).

For layer r with float weights W_r and bias b_r:
    s_r     = max|W_r| / (2^(w-1) - 1)
    W_int_r = round(W_r / s_r)
    b_int_r = round(b_r / (s_r * s_a_r))        s_a_r = input activation scale
    u_r     = W_int_r x_r + b_int_r             x_r in {0..A}^d_r, A = 2^a - 1
    x_{r+1} = clip(round(ReLU(u_r) / eta_r), 0, A)
    s_a_{r+1} = s_r * s_a_r * eta_r
eta_r is a public per-layer scalar calibrated once so that the largest activation
observed on a calibration set maps to A.  The last layer emits integer logits and
has no eta.
"""
import numpy as np


class QuantChain:
    """An integer chain of R linear layers.

    slots = 1 is the MLP frame model (d_{r+1} = m_r); slots = 9 is the conv frame
    model (d_{r+1} = 9 m_r, channel-major flattening index = channel * 9 + slot,
    matching the PyTorch (C_out, C_in, kh, kw) layout).  The chain is evaluated on
    spatially constant feature maps, which is exactly what a 3x3 convolution with
    `same` padding computes at an interior pixel of a constant map: the m_r output
    channels are replicated into all 9 slots of the next layer's input patch.
    """

    def __init__(self, W_int, b_int, eta, A, slots):
        self.W = [np.ascontiguousarray(w, dtype=np.int64) for w in W_int]
        self.b = [np.ascontiguousarray(bb, dtype=np.int64) for bb in b_int]
        self.eta = list(eta)
        self.A = int(A)
        self.slots = int(slots)
        self.R = len(self.W)

    def shapes(self):
        return [(int(w.shape[0]), int(w.shape[1])) for w in self.W]

    def forward_int(self, X, return_acc_max=False):
        """X: (N, d_1) integer array in {0..A}.  Returns integer logits (N, m_R)."""
        X = np.asarray(X, dtype=np.int64)
        acc = 0
        for r in range(self.R):
            U = X @ self.W[r].T + self.b[r]
            acc = max(acc, int(np.abs(U).max()) if U.size else 0)
            if r == self.R - 1:
                return (U, acc) if return_acc_max else U
            a = np.clip(np.rint(np.maximum(U, 0) / self.eta[r]), 0, self.A)
            a = a.astype(np.int64)
            X = np.repeat(a, self.slots, axis=1) if self.slots > 1 else a
        raise RuntimeError("unreachable")


def build_quant_chain(Wf_list, bf_list, w_bits, a_bits, slots, calib_X,
                      s_a1=None):
    """Quantise a float chain and calibrate the requantisation scales.

    calib_X: (N, d_1) integer calibration inputs in {0..A}.
    Returns (QuantChain, info dict).
    """
    A = 2 ** a_bits - 1
    qmax = 2 ** (w_bits - 1) - 1
    R = len(Wf_list)
    if s_a1 is None:
        s_a1 = 1.0 / A
    s_w, W_int = [], []
    for W in Wf_list:
        W = np.asarray(W, dtype=np.float64)
        mx = float(np.abs(W).max())
        s = mx / qmax if mx > 0 else 1.0
        s_w.append(s)
        W_int.append(np.rint(W / s).astype(np.int64))
    b_int = [None] * R
    eta = [None] * (R - 1)
    s_a = [0.0] * (R + 1)
    s_a[0] = s_a1
    X = np.asarray(calib_X, dtype=np.int64)
    acc_max = 0
    dead = []
    for r in range(R):
        b_int[r] = np.rint(np.asarray(bf_list[r], dtype=np.float64)
                           / (s_w[r] * s_a[r])).astype(np.int64)
        U = X @ W_int[r].T + b_int[r]
        acc_max = max(acc_max, int(np.abs(U).max()) if U.size else 0)
        if r == R - 1:
            break
        pos = np.maximum(U, 0)
        mx = int(pos.max()) if pos.size else 0
        if mx == 0:
            eta[r] = 1.0
            dead.append(r)
        else:
            eta[r] = mx / A
        a = np.clip(np.rint(pos / eta[r]), 0, A).astype(np.int64)
        s_a[r + 1] = s_w[r] * s_a[r] * eta[r]
        X = np.repeat(a, slots, axis=1) if slots > 1 else a
    chain = QuantChain(W_int, b_int, eta, A, slots)
    info = {
        "w_bits": int(w_bits), "a_bits": int(a_bits), "A": int(A),
        "weight_scales": [float(s) for s in s_w],
        "requant_scales_eta": [None if e is None else float(e) for e in eta],
        "activation_scales": [float(v) for v in s_a[:R]],
        "W_int_absmax": [int(np.abs(w).max()) for w in W_int],
        "b_int_absmax": [int(np.abs(bb).max()) for bb in b_int],
        "accumulator_absmax_calibration": int(acc_max),
        "accumulator_bits_calibration": int(acc_max).bit_length() + 1,
        "zero_weight_fraction": [float(np.mean(w == 0)) for w in W_int],
        "dead_layers_eta_set_to_1": dead,
        "calibration_inputs": int(X.shape[0]) if X.ndim == 2 else 0,
    }
    return chain, info


def accumulator_absmax_for_queries(W_int, b_int, A):
    """Worst-case |W x + b| over x in {0..A}^d (attained by a dense query)."""
    W = np.asarray(W_int, dtype=np.int64)
    b = np.asarray(b_int, dtype=np.int64)
    pos = np.sum(np.maximum(W, 0), axis=1) * A + b
    neg = np.sum(np.minimum(W, 0), axis=1) * A + b
    return int(max(int(np.abs(pos).max()), int(np.abs(neg).max())))

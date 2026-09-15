"""Quantised ResNet-20 with explicit residual structure, and the round graphs of
the three residual instantiations R2/R3/R4 (step 4, sub-problem A).

New module; it does not touch any existing lib entry point.

Single-patch interface, as in steps 2b and 3: a 3x3 convolution consumes a patch of
9 * C_in coordinates (flat index channel * 9 + slot) and the chain is composed on
spatially constant feature maps, so a layer's m outputs are replicated into the 9
slots of the next layer's patch.  A 1x1 shortcut and the final linear layer consume
one slot, i.e. C_in coordinates.

Quantisation.  Each convolution has its own symmetric weight scale.  The residual
addition happens in the F branch's accumulator, so the shortcut is quantised onto
that scale:  S_int = round(W_s * s_x / s_z) with s_z = s_2 * s_a1.  This keeps the
addition exact in integers with only public scalars, at the price of a shortcut
matrix whose entries are not bounded by 2^(w-1) - 1; the attained range is reported.
"""
import numpy as np

from .checkpoint import load_verified_checkpoint
from .graph_oracle import RoundSpec
from .models import fold_bn


CIFAR10_RESNET20_SHA256 = "4118986f0df73003d572b0e397f0ac7b3f60af1f31aff3d2da164536e36f6ec8"


def _flat(W):
    """(C_out, C_in, kh, kw) -> (C_out, C_in * kh * kw), channel-major."""
    W = np.asarray(W, dtype=np.float64)
    return W.reshape(W.shape[0], -1)


def resnet20_float_params(model):
    """Fold BatchNorm and return the float parameters of the whole network."""
    mods = dict(model.named_modules())
    out = {"blocks": []}
    w0 = mods["conv1"].weight.detach().numpy()
    W, b = fold_bn(model, "conv1", _flat(w0), np.zeros(w0.shape[0]))
    out["stem"] = {"name": "conv1", "W": W, "b": b,
                   "C_in": 3, "C_out": W.shape[0]}
    for stage in (1, 2, 3):
        for idx in range(3):
            pre = "layer%d.%d" % (stage, idx)
            w1 = mods[pre + ".conv1"].weight.detach().numpy()
            W1, b1 = fold_bn(model, pre + ".conv1", _flat(w1),
                             np.zeros(w1.shape[0]))
            w2 = mods[pre + ".conv2"].weight.detach().numpy()
            W2, b2 = fold_bn(model, pre + ".conv2", _flat(w2),
                             np.zeros(w2.shape[0]))
            blk = {"name": pre, "W1": W1, "b1": b1,
                   "W2": W2, "b2": b2,
                   "C_in": w1.shape[1], "C_out": w2.shape[0]}
            sc = pre + ".shortcut.0"
            if sc in mods:
                ws = mods[sc].weight.detach().numpy()
                Ws, bs = fold_bn(model, sc, _flat(ws), np.zeros(ws.shape[0]))
                blk["shortcut"] = "conv"
                blk["Ws"] = Ws                 # (C_out, C_in), 1x1 kernel
                blk["bs"] = bs
            else:
                blk["shortcut"] = "identity"
                blk["Ws"] = np.eye(blk["C_out"], blk["C_in"], dtype=np.float64)
                blk["bs"] = np.zeros(blk["C_out"])
            out["blocks"].append(blk)
    fc = mods["fc"]
    out["fc"] = {"name": "fc", "W": fc.weight.detach().numpy().astype(np.float64),
                 "b": fc.bias.detach().numpy().astype(np.float64)}
    return out


def _rep(a, slots):
    return np.repeat(a, slots, axis=1) if slots > 1 else a


class IntResNet20:
    """Integer ResNet-20 on the single-patch interface."""

    def __init__(self, stem, blocks, fc, A):
        self.stem = stem          # {"W","b","eta"}
        self.blocks = blocks      # list of {"W1","b1","eta1","W2","b2","S","bs","eta"}
        self.fc = fc              # {"W","b"}
        self.A = int(A)

    def forward_int(self, X, return_acc_max=False):
        """X: (N, 9*C_in0) integer patches in {0..A}.  Returns integer logits."""
        X = np.asarray(X, dtype=np.int64)
        acc = 0
        u = X @ self.stem["W"].T + self.stem["b"]
        acc = max(acc, int(np.abs(u).max()))
        a = np.clip(np.rint(np.maximum(u, 0) / self.stem["eta"]), 0, self.A).astype(np.int64)
        for blk in self.blocks:
            u1 = _rep(a, 9) @ blk["W1"].T + blk["b1"]
            acc = max(acc, int(np.abs(u1).max()))
            a1 = np.clip(np.rint(np.maximum(u1, 0) / blk["eta1"]), 0,
                         self.A).astype(np.int64)
            u = (_rep(a1, 9) @ blk["W2"].T + blk["b2"]
                 + a @ blk["S"].T + blk["bs"])
            acc = max(acc, int(np.abs(u).max()))
            a = np.clip(np.rint(np.maximum(u, 0) / blk["eta"]), 0,
                        self.A).astype(np.int64)
        out = a @ self.fc["W"].T + self.fc["b"]
        acc = max(acc, int(np.abs(out).max()))
        return (out, acc) if return_acc_max else out


def quantise_resnet20(model, w_bits, a_bits, calib_X, s_in=None, normalise=None):
    """Quantise the folded network and calibrate the requantisation scalars.

    normalise = (mean, std) folds a per-channel input normalisation into the stem.
    """
    A = 2 ** a_bits - 1
    qmax = 2 ** (w_bits - 1) - 1
    fp = resnet20_float_params(model)
    if normalise is not None:
        fp = fold_input_normalisation(fp, normalise[0], normalise[1])
    if s_in is None:
        s_in = 1.0 / A
    info = {"w_bits": w_bits, "a_bits": a_bits, "A": A, "weight_scales": {},
            "W_int_absmax": {}, "shortcut_int_absmax": {}, "requant_scales": {},
            "zero_weight_fraction": {}, "accumulator_absmax_calibration": 0,
            "dead_requant": []}
    X = np.asarray(calib_X, dtype=np.int64)
    acc = 0

    def q(W):
        mx = float(np.abs(W).max())
        s = mx / qmax if mx > 0 else 1.0
        return np.rint(W / s).astype(np.int64), s

    Ws, ss = q(fp["stem"]["W"])
    bs_int = np.rint(fp["stem"]["b"] / (ss * s_in)).astype(np.int64)
    u = X @ Ws.T + bs_int
    acc = max(acc, int(np.abs(u).max()))
    pos = np.maximum(u, 0)
    mx = int(pos.max())
    eta = mx / A if mx > 0 else 1.0
    if mx == 0:
        info["dead_requant"].append("stem")
    stem = {"W": Ws, "b": bs_int, "eta": eta}
    info["weight_scales"]["conv1"] = ss
    info["W_int_absmax"]["conv1"] = int(np.abs(Ws).max())
    info["zero_weight_fraction"]["conv1"] = float(np.mean(Ws == 0))
    info["requant_scales"]["conv1"] = eta
    a = np.clip(np.rint(pos / eta), 0, A).astype(np.int64)
    s_a = ss * s_in * eta

    blocks = []
    for blk in fp["blocks"]:
        nm = blk["name"]
        W1, s1 = q(blk["W1"])
        b1 = np.rint(blk["b1"] / (s1 * s_a)).astype(np.int64)
        u1 = _rep(a, 9) @ W1.T + b1
        acc = max(acc, int(np.abs(u1).max()))
        pos1 = np.maximum(u1, 0)
        m1 = int(pos1.max())
        eta1 = m1 / A if m1 > 0 else 1.0
        if m1 == 0:
            info["dead_requant"].append(nm + ".conv1")
        a1 = np.clip(np.rint(pos1 / eta1), 0, A).astype(np.int64)
        s_a1 = s1 * s_a * eta1
        W2, s2 = q(blk["W2"])
        b2 = np.rint(blk["b2"] / (s2 * s_a1)).astype(np.int64)
        s_z = s2 * s_a1
        S = np.rint(blk["Ws"] * s_a / s_z).astype(np.int64)
        bsc = np.rint(blk["bs"] / s_z).astype(np.int64)
        u = _rep(a1, 9) @ W2.T + b2 + a @ S.T + bsc
        acc = max(acc, int(np.abs(u).max()))
        pos = np.maximum(u, 0)
        mx = int(pos.max())
        etab = mx / A if mx > 0 else 1.0
        if mx == 0:
            info["dead_requant"].append(nm + ".block")
        blocks.append({"name": nm, "W1": W1, "b1": b1, "eta1": eta1,
                       "W2": W2, "b2": b2, "S": S, "bs": bsc, "eta": etab,
                       "shortcut": blk["shortcut"],
                       "C_in": blk["C_in"], "C_out": blk["C_out"]})
        info["weight_scales"][nm + ".conv1"] = s1
        info["weight_scales"][nm + ".conv2"] = s2
        info["W_int_absmax"][nm + ".conv1"] = int(np.abs(W1).max())
        info["W_int_absmax"][nm + ".conv2"] = int(np.abs(W2).max())
        info["zero_weight_fraction"][nm + ".conv1"] = float(np.mean(W1 == 0))
        info["zero_weight_fraction"][nm + ".conv2"] = float(np.mean(W2 == 0))
        info["shortcut_int_absmax"][nm] = int(np.abs(S).max())
        info["requant_scales"][nm + ".conv1"] = eta1
        info["requant_scales"][nm + ".block"] = etab
        a = np.clip(np.rint(pos / etab), 0, A).astype(np.int64)
        s_a = s_z * etab

    Wf, sf = q(fp["fc"]["W"])
    bf = np.rint(fp["fc"]["b"] / (sf * s_a)).astype(np.int64)
    out = a @ Wf.T + bf
    acc = max(acc, int(np.abs(out).max()))
    info["weight_scales"]["fc"] = sf
    info["W_int_absmax"]["fc"] = int(np.abs(Wf).max())
    info["zero_weight_fraction"]["fc"] = float(np.mean(Wf == 0))
    info["accumulator_absmax_calibration"] = int(acc)
    info["accumulator_bits_calibration"] = int(acc).bit_length() + 1
    net = IntResNet20(stem, blocks, {"W": Wf, "b": bf}, A)
    return net, info


# ---------------------------------------------------------------- round graphs
def build_rounds(net, mode):
    """Round graph for residual mode "R2", "R3" or "R4".

    Returns (rounds, meta) where meta lists, per logical layer, the round index,
    the input index to vary, the observing round, and the group whose frame its
    query lives in.
    """
    rounds = []
    meta = []
    g = 0
    stem_group = g
    rounds.append(RoundSpec(
        "conv1", stem_group, net.stem["W"].shape[0], net.stem["b"],
        [{"kind": "new", "W": net.stem["W"], "d": net.stem["W"].shape[1],
          "frame": -1, "slots": 9}], role="conv"))
    meta.append({"layer": "conv1", "round": 0, "vary": 0, "observe": 0,
                 "kind": "plain", "group": stem_group})
    in_group = stem_group
    for bi, blk in enumerate(net.blocks):
        m = blk["C_out"]
        g += 1
        g1 = g
        r1 = len(rounds)
        rounds.append(RoundSpec(
            blk["name"] + ".conv1", g1, m, blk["b1"],
            [{"kind": "new", "W": blk["W1"], "d": blk["W1"].shape[1],
              "frame": in_group, "slots": 9}], role="conv"))
        meta.append({"layer": blk["name"] + ".conv1", "round": r1, "vary": 0,
                     "observe": r1, "kind": "plain", "group": g1})
        g += 1
        g2 = g
        r2 = len(rounds)
        if mode == "R2":
            rounds.append(RoundSpec(
                blk["name"] + ".conv2", g2, m, blk["b2"],
                [{"kind": "new", "W": blk["W2"], "d": blk["W2"].shape[1],
                  "frame": g1, "slots": 9}], role="conv"))
            rs = len(rounds)
            rounds.append(RoundSpec(
                blk["name"] + ".shortcut", g2, m, blk["bs"],
                [{"kind": "new", "W": blk["S"], "d": blk["S"].shape[1],
                  "frame": in_group, "slots": 1}], role="shortcut"))
            meta.append({"layer": blk["name"] + ".conv2", "round": r2, "vary": 0,
                         "observe": r2, "kind": "plain", "group": g2})
            meta.append({"layer": blk["name"] + ".shortcut", "round": rs, "vary": 0,
                         "observe": rs, "kind": "labelled", "group": g2,
                         "probe_round": r2})
        elif mode == "R3":
            rounds.append(RoundSpec(
                blk["name"] + ".conv2+shortcut", g2, m, blk["b2"] + blk["bs"],
                [{"kind": "new", "W": blk["W2"], "d": blk["W2"].shape[1],
                  "frame": g1, "slots": 9},
                 {"kind": "ret", "W": _embed_centre(blk["S"], blk["C_in"]),
                  "d": 9 * blk["C_in"], "src": (r1, 0)}], role="sum"))
            meta.append({"layer": blk["name"] + ".conv2", "round": r2, "vary": 0,
                         "observe": r2, "kind": "sum_primary", "group": g2})
            meta.append({"layer": blk["name"] + ".shortcut", "round": r1, "vary": 0,
                         "observe": r2, "kind": "sum_secondary", "group": g2,
                         "primary_round": r2, "centre_only": True,
                         "d_eff": blk["C_in"]})
        elif mode == "R4":
            rounds.append(RoundSpec(
                blk["name"] + ".conv2+shortcut", g2, m, blk["b2"] + blk["bs"],
                [{"kind": "new", "W": blk["W2"], "d": blk["W2"].shape[1],
                  "frame": g1, "slots": 9},
                 {"kind": "new", "W": blk["S"], "d": blk["S"].shape[1],
                  "frame": in_group, "slots": 1}], role="sum"))
            meta.append({"layer": blk["name"] + ".conv2", "round": r2, "vary": 0,
                         "observe": r2, "kind": "sum_primary", "group": g2})
            meta.append({"layer": blk["name"] + ".shortcut", "round": r2, "vary": 1,
                         "observe": r2, "kind": "sum_secondary", "group": g2,
                         "primary_round": r2, "d_eff": blk["S"].shape[1]})
        else:
            raise ValueError("unknown residual mode %r" % mode)
        in_group = g2
    g += 1
    rf = len(rounds)
    rounds.append(RoundSpec(
        "fc", g, net.fc["W"].shape[0], net.fc["b"],
        [{"kind": "new", "W": net.fc["W"], "d": net.fc["W"].shape[1],
          "frame": in_group, "slots": 1}], unshuffled=True, role="fc"))
    meta.append({"layer": "fc", "round": rf, "vary": 0, "observe": rf,
                 "kind": "plain", "group": g, "unshuffled": True})
    return rounds, meta


def _embed_centre(S, C_in):
    """Place a 1x1 matrix (m, C_in) into the centre slot of a 9-slot patch."""
    m = S.shape[0]
    out = np.zeros((m, 9 * C_in), dtype=np.int64)
    out[:, np.arange(C_in) * 9 + 4] = S
    return out


def load_resnet20_cifar10(path, expected_sha256=CIFAR10_RESNET20_SHA256):
    """Load a chenyaofo/pytorch-cifar-models ResNet-20 checkpoint into
    lib.models._ResNetCIFAR(n=3).  The only naming difference is
    `downsample` versus `shortcut`."""
    from .models import _ResNetCIFAR
    sd = load_verified_checkpoint(path, expected_sha256)
    if isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]
    remap = {k.replace(".downsample.", ".shortcut."): v for k, v in sd.items()}
    model = _ResNetCIFAR(n=3)
    missing, unexpected = model.load_state_dict(remap, strict=False)
    model.eval()
    return model, {"missing_keys": list(missing), "unexpected_keys": list(unexpected)}


def fold_input_normalisation(fp, mean, std):
    """Fold a per-channel input normalisation (x - mean)/std into the stem, so
    that the network input is the raw image in [0, 1] and stays non-negative."""
    W = fp["stem"]["W"].copy()
    b = fp["stem"]["b"].copy()
    C_in = fp["stem"]["C_in"]
    slots = W.shape[1] // C_in
    mean_e = np.repeat(np.asarray(mean, dtype=np.float64), slots)
    std_e = np.repeat(np.asarray(std, dtype=np.float64), slots)
    b = b - W @ (mean_e / std_e)
    W = W / std_e[None, :]
    fp["stem"] = dict(fp["stem"])
    fp["stem"]["W"] = W
    fp["stem"]["b"] = b
    return fp

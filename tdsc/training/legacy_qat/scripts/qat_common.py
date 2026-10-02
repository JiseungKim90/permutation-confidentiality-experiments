"""Shared code for the QAT item of PLAN_qat.md.

Everything is imported from `lib_qat`, a `git archive` snapshot of `lib/` taken
at commit 22b2843, so that a concurrent edit of `lib/` by another agent cannot
change any number reported by this item.

Three groups of code live here.

1. An LSQ-style quantisation-aware-training implementation (learned per-layer
   step sizes, straight-through estimator, no new packages).  BatchNorm is
   folded into the convolution *inside* the forward pass with frozen statistics,
   so the object that carries the learned step size is exactly the BN-folded
   matrix the attack sees.  Training an unfolded weight and folding afterwards
   would take the weights off the 4-bit lattice and defeat the purpose.

2. Export of a trained model to the integer form used by every previous step of
   this programme (`W_int`, `b_int`, activation alphabet {0..A}), plus the
   lattice-occupancy histogram and the post-hoc quantiser of
   `lib_qat/quant.py` for the side-by-side comparison.

3. The accumulator-constrained anchor machinery of `scripts/suite_b_common.py`,
   copied verbatim with its imports pointed at `lib_qat`, so that the frontier
   numbers are produced by the same code that produced `REPORT_suite_b.md`
   Section 3 and are not merely "the same method".
"""
import math
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

DATA = os.path.join(ROOT, "data")
RESULTS = os.path.join(ROOT, "results")
LOGS = os.path.join(ROOT, "logs")
for _d in (DATA, RESULTS, LOGS):
    os.makedirs(_d, exist_ok=True)

from lib_qat.lta import Channel, noise_fns_bounded            # noqa: E402
from lib_qat.lta_run import run_lta                           # noqa: E402
from lib_qat.quant import QuantChain                          # noqa: E402
from lib_qat.quant import accumulator_absmax_for_queries      # noqa: E402
from lib_qat.checkpoint import load_verified_checkpoint       # noqa: E402

LIB_SNAPSHOT = "lib_qat (git archive of lib/ at commit 22b2843)"
N_WORKERS = int(os.environ.get("QAT_WORKERS", "9"))
RECURSION_LIMIT = 60000
THREAD_STACK_BYTES = 512 * 1024 * 1024

CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD = (0.2023, 0.1994, 0.2010)   # reproduces both checkpoints
# (0.2470, 0.2435, 0.2616), the constant in lib_qat/cifar10.py, loses 0.33 to
# 0.48 top-1 on these two checkpoints, so they were trained with the constants above.
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


# ==========================================================================
# 1. LSQ quantisation-aware training
# ==========================================================================
def _round_pass(x):
    return (torch.round(x) - x).detach() + x


def _grad_scale(x, s):
    return (x - x * s).detach() + x * s


class LsqQuant(nn.Module):
    """Learned-step-size quantiser (Esser et al., LSQ), implemented directly.

    signed=True gives the symmetric weight lattice {-qmax..qmax} with
    qmax = 2^(b-1) - 1, which is the lattice of `lib_qat/quant.py`.
    signed=False gives the unsigned activation alphabet {0..2^b - 1} = {0..A}.
    The step is a trained parameter; its gradient is scaled by
    1/sqrt(numel * qmax) as in the LSQ paper.
    """

    def __init__(self, n_bits, signed):
        super().__init__()
        self.n_bits = int(n_bits)
        self.signed = bool(signed)
        self.identity = self.n_bits >= 32
        if signed:
            self.qmax = 2 ** (n_bits - 1) - 1
            self.qmin = -self.qmax
        else:
            self.qmax = 2 ** n_bits - 1
            self.qmin = 0
        self.step = nn.Parameter(torch.tensor(1.0))
        self.register_buffer("initialised", torch.zeros((), dtype=torch.uint8))

    @torch.no_grad()
    def init_from(self, x):
        s = 2.0 * x.detach().abs().mean() / math.sqrt(max(1.0, float(self.qmax)))
        if not torch.isfinite(s) or float(s) <= 0:
            s = torch.tensor(1e-4)
        self.step.copy_(s.clamp(min=1e-8))
        self.initialised.fill_(1)

    def forward(self, x):
        if self.identity:
            return x, torch.ones((), dtype=x.dtype)
        if int(self.initialised.item()) == 0:
            self.init_from(x)
        g = 1.0 / math.sqrt(max(1, x.numel()) * self.qmax)
        s = _grad_scale(self.step.abs().clamp(min=1e-8), g)
        xq = torch.clamp(_round_pass(x / s), self.qmin, self.qmax)
        return xq * s, s


class QLayer(nn.Module):
    """A convolution or a linear layer with BN folded in and both operands
    quantised by learned step sizes.

    `folded()` returns the float BN-folded (W, b); the weight quantiser acts on
    that, so `export_layer` recovers integers on the 4-bit lattice with no
    further requantisation.  BatchNorm statistics are frozen at the values of
    the float checkpoint (the standard BN-freezing recipe), which makes the fold
    exact and the training stable at 4 bits.
    """

    def __init__(self, mod, bn, w_bits, a_bits, name, quant_in=True,
                 a_signed=False, kind="conv"):
        super().__init__()
        self.name = name
        self.kind = kind
        self.weight = nn.Parameter(mod.weight.detach().clone().float())
        C = self.weight.shape[0]
        if kind == "conv":
            self.stride = mod.stride
            self.padding = mod.padding
            self.dilation = mod.dilation
            self.groups = mod.groups
        self.has_bn = bn is not None
        cb = (mod.bias.detach().clone().float() if mod.bias is not None
              else torch.zeros(C))
        if self.has_bn:
            self.gamma = nn.Parameter(bn.weight.detach().clone().float())
            self.beta = nn.Parameter(bn.bias.detach().clone().float())
            self.register_buffer("run_mean", bn.running_mean.detach().clone().float())
            self.register_buffer("run_var", bn.running_var.detach().clone().float())
            self.eps = float(bn.eps)
            self.register_buffer("conv_bias", cb)
        else:
            self.beta = nn.Parameter(cb)
        self.wq = LsqQuant(w_bits, True)
        self.aq = LsqQuant(a_bits, a_signed) if quant_in else None
        self.acc_int_absmax = 0.0
        self._acc_term = None

    def folded(self):
        if not self.has_bn:
            return self.weight, self.beta
        sig = torch.sqrt(self.run_var + self.eps)
        sc = self.gamma / sig
        shape = [-1] + [1] * (self.weight.dim() - 1)
        Wf = self.weight * sc.view(*shape)
        bf = self.beta - self.gamma * self.run_mean / sig + sc * self.conv_bias
        return Wf, bf

    def forward(self, x):
        if self.aq is not None:
            xq, s_a = self.aq(x)
        else:
            xq, s_a = x, None
        Wf, bf = self.folded()
        Wq, s_w = self.wq(Wf)
        if self.kind == "conv":
            y = F.conv2d(xq, Wq, bf, self.stride, self.padding, self.dilation,
                         self.groups)
        else:
            y = F.linear(xq, Wq, bf)
        if s_a is not None and not (self.wq.identity or self.aq.identity):
            acc = y.abs().amax() / (s_w * s_a)
            self._acc_term = acc
            self.acc_int_absmax = float(acc.detach())
        else:
            self._acc_term = None
        return y


def q_layers(model):
    return [m for m in model.modules() if isinstance(m, QLayer)]


def accumulator_penalty(model, limit):
    """Hinge penalty on the honest integer accumulator magnitude.

    `limit` is the largest magnitude a signed budget allows (8191 for 14 bits).
    The term is differentiable in the weights and in both step sizes, so the
    optimiser can satisfy it either by shrinking the accumulator or by
    coarsening a step.  It is reported as attained, not assumed.
    """
    total = None
    for ly in q_layers(model):
        if ly._acc_term is None:
            continue
        t = torch.relu(ly._acc_term / float(limit) - 1.0)
        total = t if total is None else total + t
    return total if total is not None else torch.zeros(())


# ==========================================================================
# CIFAR ResNet (lib_qat/models.py naming) in QAT form
# ==========================================================================
class QBasicBlockCIFAR(nn.Module):
    def __init__(self, blk, w_bits, a_bits, pre):
        super().__init__()
        self.conv1 = QLayer(blk.conv1, blk.bn1, w_bits, a_bits, pre + ".conv1")
        self.conv2 = QLayer(blk.conv2, blk.bn2, w_bits, a_bits, pre + ".conv2")
        self.has_short = len(list(blk.shortcut.children())) > 0
        if self.has_short:
            self.short = QLayer(blk.shortcut[0], blk.shortcut[1], w_bits, a_bits,
                                pre + ".shortcut.0")

    def forward(self, x):
        out = torch.relu(self.conv1(x))
        out = self.conv2(out)
        out = out + (self.short(x) if self.has_short else x)
        return torch.relu(out)


class QResNetCIFAR(nn.Module):
    """QAT wrapper around lib_qat.models._ResNetCIFAR with identical names."""

    def __init__(self, float_model, n, w_bits, a_bits, n_classes=10):
        super().__init__()
        self.n = n
        self.register_buffer("mean", torch.tensor(CIFAR_MEAN).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(CIFAR_STD).view(1, 3, 1, 1))
        self.inq = LsqQuant(a_bits, True)          # input, signed (normalised)
        self.conv1 = QLayer(float_model.conv1, float_model.bn1, w_bits, a_bits,
                            "conv1", quant_in=False)
        stages = []
        for si, stage in enumerate((float_model.layer1, float_model.layer2,
                                    float_model.layer3), start=1):
            blocks = nn.ModuleList([
                QBasicBlockCIFAR(b, w_bits, a_bits, "layer%d.%d" % (si, bi))
                for bi, b in enumerate(stage)])
            stages.append(blocks)
        self.layer1, self.layer2, self.layer3 = stages
        self.fc = QLayer(float_model.fc, None, w_bits, a_bits, "fc", kind="linear")

    def forward(self, x):
        x = (x - self.mean) / self.std
        x, _ = self.inq(x)
        out = torch.relu(self.conv1(x))
        for stage in (self.layer1, self.layer2, self.layer3):
            for blk in stage:
                out = blk(out)
        out = out.mean(dim=[2, 3])
        return self.fc(out)


# ==========================================================================
# ImageNet ResNet-18 / ResNet-34 (torchvision naming) in QAT form
# ==========================================================================
class _TVBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inp, out, stride=1, down=False):
        super().__init__()
        self.conv1 = nn.Conv2d(inp, out, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out)
        self.conv2 = nn.Conv2d(out, out, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out)
        if down:
            self.downsample = nn.Sequential(
                nn.Conv2d(inp, out, 1, stride, bias=False), nn.BatchNorm2d(out))
        else:
            self.downsample = None

    def forward(self, x):
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + (self.downsample(x) if self.downsample is not None else x)
        return torch.relu(out)


class TVResNet(nn.Module):
    """torchvision-layout ResNet-18/34 skeleton, written out so that the plain
    state dicts at download.pytorch.org load without torchvision."""

    def __init__(self, layers, n_classes=1000):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 7, 2, 3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.inp = 64
        self.layer1 = self._make(64, layers[0], 1)
        self.layer2 = self._make(128, layers[1], 2)
        self.layer3 = self._make(256, layers[2], 2)
        self.layer4 = self._make(512, layers[3], 2)
        self.fc = nn.Linear(512, n_classes)

    def _make(self, out, n, stride):
        down = stride != 1 or self.inp != out
        blocks = [_TVBasicBlock(self.inp, out, stride, down)]
        self.inp = out
        for _ in range(1, n):
            blocks.append(_TVBasicBlock(out, out, 1, False))
        return nn.Sequential(*blocks)

    def forward(self, x):
        out = torch.relu(self.bn1(self.conv1(x)))
        out = F.max_pool2d(out, 3, 2, 1)
        for s in (self.layer1, self.layer2, self.layer3, self.layer4):
            out = s(out)
        return self.fc(out.mean(dim=[2, 3]))


@torch.no_grad()
def reestimate_bn(float_model, batches, mean, std):
    """Re-estimate every BatchNorm's running statistics on the target data.

    The ImageNet checkpoints carry statistics measured at 224x224 on ImageNet.
    Feeding 64x64 Tiny-ImageNet images leaves the frozen affine transform
    mis-scaled -- measured on `resnet18-f37072fd.pth`, the actual per-layer
    variances at 64x64 are about twice the stored ones and the means drift by up
    to 0.3 -- which a 4-bit activation quantiser then clips.  Resetting the
    statistics and accumulating an exact cumulative average (momentum=None) over
    `batches` forward passes is the standard remedy and uses no labels.
    """
    mean_t = torch.tensor(mean).view(1, 3, 1, 1)
    std_t = torch.tensor(std).view(1, 3, 1, 1)
    bns = [m for m in float_model.modules() if isinstance(m, nn.BatchNorm2d)]
    for bn in bns:
        bn.reset_running_stats()
        bn.momentum = None
    float_model.train()
    n = 0
    for xb in batches:
        float_model((torch.as_tensor(xb) - mean_t) / std_t)
        n += 1
    float_model.eval()
    return {"bn_layers": len(bns), "batches": n,
            "method": "reset_running_stats + cumulative average (momentum=None)"}


class QBasicBlockTV(nn.Module):
    def __init__(self, blk, w_bits, a_bits, pre):
        super().__init__()
        self.conv1 = QLayer(blk.conv1, blk.bn1, w_bits, a_bits, pre + ".conv1")
        self.conv2 = QLayer(blk.conv2, blk.bn2, w_bits, a_bits, pre + ".conv2")
        self.has_down = blk.downsample is not None
        if self.has_down:
            self.down = QLayer(blk.downsample[0], blk.downsample[1], w_bits,
                               a_bits, pre + ".downsample.0")

    def forward(self, x):
        out = torch.relu(self.conv1(x))
        out = self.conv2(out)
        out = out + (self.down(x) if self.has_down else x)
        return torch.relu(out)


class QResNetTV(nn.Module):
    def __init__(self, float_model, w_bits, a_bits, n_classes):
        super().__init__()
        self.register_buffer("mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1))
        self.inq = LsqQuant(a_bits, True)
        self.conv1 = QLayer(float_model.conv1, float_model.bn1, w_bits, a_bits,
                            "conv1", quant_in=False)
        stages = []
        for si in (1, 2, 3, 4):
            stage = getattr(float_model, "layer%d" % si)
            stages.append(nn.ModuleList([
                QBasicBlockTV(b, w_bits, a_bits, "layer%d.%d" % (si, bi))
                for bi, b in enumerate(stage)]))
        self.layer1, self.layer2, self.layer3, self.layer4 = stages
        fc = nn.Linear(512, n_classes)
        self.fc = QLayer(fc, None, w_bits, a_bits, "fc", kind="linear")

    def forward(self, x):
        x = (x - self.mean) / self.std
        x, _ = self.inq(x)
        out = torch.relu(self.conv1(x))
        out = F.max_pool2d(out, 3, 2, 1)
        for stage in (self.layer1, self.layer2, self.layer3, self.layer4):
            for blk in stage:
                out = blk(out)
        out = out.mean(dim=[2, 3])
        return self.fc(out)


def load_tv_resnet(path, expected_sha256, layers, n_classes=1000):
    net = TVResNet(layers, n_classes)
    sd = load_verified_checkpoint(path, expected_sha256)
    if isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]
    missing, unexpected = net.load_state_dict(sd, strict=False)
    net.eval()
    return net, {"missing_keys": list(missing), "unexpected_keys": list(unexpected)}


def load_cifar_float(path, expected_sha256, n):
    """lib_qat.models._ResNetCIFAR(n) from either checkpoint layout."""
    from lib_qat.models import _ResNetCIFAR
    ck = load_verified_checkpoint(path, expected_sha256)
    sd = ck["model_state_dict"] if (isinstance(ck, dict)
                                    and "model_state_dict" in ck) else ck
    if isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]
    sd = {k.replace(".downsample.", ".shortcut."): v for k, v in sd.items()}
    model = _ResNetCIFAR(n=n)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    model.eval()
    meta = {"missing_keys": list(missing), "unexpected_keys": list(unexpected)}
    if isinstance(ck, dict) and "test_accuracy" in ck:
        meta["checkpoint_test_accuracy"] = float(ck["test_accuracy"])
    return model, meta


# ==========================================================================
# 2. Export, lattice occupancy, post-hoc baseline
# ==========================================================================
@torch.no_grad()
def export_layer(ly):
    """Integer form of one trained QAT layer.

    W_int is on the symmetric lattice {-qmax..qmax} of the learned step and is
    exactly the matrix the fake-quantised forward pass used; b_int follows the
    step-3 convention b_int = round(b_fold / (s_w s_a)).
    """
    Wf, bf = ly.folded()
    s_w = float(ly.wq.step.abs().clamp(min=1e-8))
    qmax = ly.wq.qmax
    s_a = float(ly.aq.step.abs().clamp(min=1e-8)) if ly.aq is not None else 1.0
    W_int = torch.clamp(torch.round(Wf / s_w), -qmax, qmax).to(torch.int64)
    b_int = torch.round(bf / (s_w * s_a)).to(torch.int64)
    m = W_int.shape[0]
    return {
        "name": ly.name,
        "kind": ly.kind,
        "shape": list(W_int.shape),
        "W_int": W_int.reshape(m, -1).numpy().astype(np.int64),
        "b_int": b_int.numpy().astype(np.int64),
        "W_float_folded": Wf.reshape(m, -1).numpy().astype(np.float64),
        "b_float_folded": bf.numpy().astype(np.float64),
        "s_w": s_w,
        "s_a_in": s_a,
        "qmax": int(qmax),
        "a_bits": int(ly.aq.n_bits) if ly.aq is not None else None,
        "accumulator_int_absmax_last_batch": float(ly.acc_int_absmax),
    }


@torch.no_grad()
def export_model(model):
    return [export_layer(ly) for ly in q_layers(model)]


def posthoc_quantise(W_float, w_bits):
    """The per-layer symmetric post-hoc quantiser of lib_qat/quant.py:
    s = max|W| / (2^(w-1) - 1), W_int = round(W/s)."""
    qmax = 2 ** (w_bits - 1) - 1
    W = np.asarray(W_float, dtype=np.float64)
    mx = float(np.abs(W).max())
    s = mx / qmax if mx > 0 else 1.0
    return np.rint(W / s).astype(np.int64), s


def lattice_histogram(W_int, qmax):
    """Fraction of entries at each of the 2*qmax+1 lattice levels."""
    W = np.asarray(W_int, dtype=np.int64).ravel()
    W = np.clip(W, -qmax, qmax)
    cnt = np.bincount(W + qmax, minlength=2 * qmax + 1).astype(np.float64)
    return (cnt / max(1, W.size)).tolist()


def occupancy_stats(W_int, qmax):
    W = np.asarray(W_int, dtype=np.int64)
    hist = np.asarray(lattice_histogram(W, qmax))
    nz = hist[hist > 0]
    return {
        "levels": list(range(-qmax, qmax + 1)),
        "histogram": hist.tolist(),
        "zero_fraction": float(hist[qmax]),
        "levels_occupied": int(np.sum(hist > 0)),
        "entropy_bits": float(-np.sum(nz * np.log2(nz))),
        "abs_max_level": int(np.abs(W).max()) if W.size else 0,
        "n_entries": int(W.size),
    }


def bn_folded_float_layers_cifar(model, names):
    """BN-folded float (W, b) of the named convolutions of a float
    lib_qat.models._ResNetCIFAR, exactly as step 3 and suite B build them."""
    from lib_qat.models import get_conv_layers, fold_bn
    convs = {n: (W, b) for n, W, b, hb, m, d in get_conv_layers(model)}
    out = []
    for nm in names:
        W_mat, b_vec = convs[nm]
        Wf, bf = fold_bn(model, nm, W_mat, b_vec)
        out.append((nm, Wf, bf))
    return out


# ==========================================================================
# 3. Anchor-family machinery, copied from scripts/suite_b_common.py
#    (imports repointed at lib_qat; no change of semantics)
# ==========================================================================
def run_with_big_stack(fn, *a, **kw):
    import threading
    out = {}

    def _t():
        sys.setrecursionlimit(RECURSION_LIMIT)
        try:
            out["r"] = fn(*a, **kw)
        except BaseException as exc:            # noqa: BLE001 - re-raised below
            out["e"] = exc

    old = threading.stack_size(THREAD_STACK_BYTES)
    try:
        th = threading.Thread(target=_t)
        th.start()
        th.join()
    finally:
        threading.stack_size(old)
    if "e" in out:
        raise out["e"]
    return out["r"]


def worker_setup():
    os.environ["OMP_NUM_THREADS"] = "1"
    try:
        torch.set_num_threads(1)
    except Exception:
        pass
    sys.setrecursionlimit(RECURSION_LIMIT)


def strip_private(rec):
    return {k: v for k, v in rec.items() if not k.startswith("_")}


def bits_signed(v):
    return int(abs(int(v))).bit_length() + 1


def budget_absmax(bits):
    return 2 ** (int(bits) - 1) - 1


class TrackedChannel(Channel):
    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.acc_absmax = 0

    def _observe(self, true_vals, x):
        if true_vals.size:
            v = int(np.abs(true_vals).max())
            if v > self.acc_absmax:
                self.acc_absmax = v
        return super()._observe(true_vals, x)


class AnchorRng:
    def __init__(self, rng, sampler, d):
        self._rng = rng
        self._sampler = sampler
        self._d = int(d)
        self.n_anchor_draws = 0

    def integers(self, low, high=None, size=None, **kw):
        if size == self._d and low == 0 and not kw:
            self.n_anchor_draws += 1
            return np.asarray(self._sampler(), dtype=np.int64)
        return self._rng.integers(low, high, size=size, **kw)

    def __getattr__(self, name):
        return getattr(self._rng, name)


def make_sampler(rng, d, support_idx, cap):
    support_idx = np.asarray(support_idx, dtype=np.int64)
    cap = int(cap)

    def sample():
        x = np.zeros(d, dtype=np.int64)
        if support_idx.size and cap > 0:
            x[support_idx] = rng.integers(0, cap + 1, size=support_idx.size)
        return x

    return sample


def acc_bounds(absW_row_l1_on_support, absb, T, w_absmax, cap):
    base = int(np.max(cap * absW_row_l1_on_support + absb)) if absb.size else 0
    step = int(T) * int(w_absmax)
    return base, base + step, base + 2 * step


def _fits(cumL1, absb, cap, T, w_absmax, limit, s):
    base = int(np.max(cap * cumL1[:, s] + absb))
    return base + T * w_absmax <= limit, base


def plan_anchor(W_int, b_int, A, T, limit, family, rng):
    m, d = W_int.shape
    absW = np.abs(W_int)
    absb = np.abs(b_int).astype(np.int64)
    w_absmax = int(absW.max()) if absW.size else 0
    hi = A - 2 * T
    out = {"family": family, "cap_max": int(hi), "w_int_absmax": w_absmax}

    if family == "unconstrained":
        cap, s = hi, d
        order = np.arange(d, dtype=np.int64)
    elif family == "dense_capped":
        order = np.arange(d, dtype=np.int64)
        s = d
        l1 = absW.sum(axis=1)
        cap = None
        for c in range(hi, -1, -1):
            if int(np.max(c * l1 + absb)) + T * w_absmax <= limit:
                cap = c
                break
        if cap is None:
            out.update({"feasible": False, "cap": 0, "support_size": 0,
                        "reason": "even the all-zero anchor overflows the budget"})
            return out
    elif family in ("sparse", "sparse_capped"):
        order = rng.permutation(d).astype(np.int64)
        cumL1 = np.zeros((m, d + 1), dtype=np.int64)
        np.cumsum(absW[:, order], axis=1, out=cumL1[:, 1:])
        if family == "sparse":
            cap = hi
            s = 0
            for ss in range(d, 0, -1):
                ok, _ = _fits(cumL1, absb, cap, T, w_absmax, limit, ss)
                if ok:
                    s = ss
                    break
            if s == 0:
                out.update({"feasible": False, "cap": int(cap), "support_size": 0,
                            "reason": "no single coordinate fits the budget at the "
                                      "full alphabet"})
                return out
        else:
            best = None
            for ss in range(1, d + 1):
                lo, hi2, bc = 1, hi, 0
                while lo <= hi2:
                    mid = (lo + hi2) // 2
                    ok, _ = _fits(cumL1, absb, mid, T, w_absmax, limit, ss)
                    if ok:
                        bc = mid
                        lo = mid + 1
                    else:
                        hi2 = mid - 1
                if bc == 0:
                    break
                score = ss * np.log2(bc + 1.0)
                if best is None or score > best[0] + 1e-12:
                    best = (score, ss, bc)
            if best is None:
                out.update({"feasible": False, "cap": 0, "support_size": 0,
                            "reason": "no (s, cap) with cap >= 1 fits the budget"})
                return out
            _, s, cap = best
    else:
        raise ValueError("unknown family %r" % family)

    support = np.sort(order[:s])
    l1s = absW[:, support].sum(axis=1) if support.size else np.zeros(m, dtype=np.int64)
    base, p1, p2 = acc_bounds(l1s, absb, T, w_absmax, cap)
    out.update({
        "feasible": True, "cap": int(cap), "support_size": int(s),
        "support": support,
        "anchor_bits_entropy": float(s * np.log2(cap + 1.0)),
        "acc_absmax_bound_anchor": base,
        "acc_absmax_bound_pass1": p1,
        "acc_absmax_bound_pass2": p2,
        "acc_bits_bound_pass1": bits_signed(p1),
        "acc_bits_bound_pass2": bits_signed(p2),
    })
    return out


def lta_with_family(W_int, b_int, A, T, seed, family, limit, deadline=None,
                    noise_law="Gaussian", max_anchor_tries=5):
    m, d = W_int.shape
    rng = np.random.default_rng(list(seed))
    rng_anchor = np.random.default_rng(list(seed) + [7])
    plan = plan_anchor(W_int, b_int, A, T, limit, family, rng_anchor)
    if not plan["feasible"]:
        rec = {"m": int(m), "d": int(d), "T": int(T), "A": int(A),
               "status": "anchor_infeasible", "error": plan.get("reason", ""),
               "exact_up_to_row_perm": False, "n_queries": 0}
        rec.update({k: v for k, v in plan.items() if k != "support"})
        return rec
    fns = noise_fns_bounded(rng)
    ch = TrackedChannel(W_int, b_int, A, fns[noise_law], rng)
    sampler = make_sampler(rng_anchor, d, plan["support"], plan["cap"])
    arng = AnchorRng(rng, sampler, d)
    tries = 1 if plan["cap"] == 0 else max_anchor_tries
    rec = strip_private(run_lta(W_int, b_int, A, T, arng, fns[noise_law],
                                channel=ch, deadline=deadline,
                                max_anchor_tries=tries))
    rec.update({k: v for k, v in plan.items() if k != "support"})
    rec["acc_absmax_measured"] = int(ch.acc_absmax)
    rec["acc_bits_measured"] = bits_signed(ch.acc_absmax)
    rec["budget_absmax_limit"] = None if limit is None else int(limit)
    rec["budget_respected_measured"] = (
        True if limit is None else bool(ch.acc_absmax <= limit))
    rec["noise_law"] = noise_law
    return rec


def median(xs):
    xs = sorted(float(x) for x in xs)
    if not xs:
        return None
    n = len(xs)
    return xs[n // 2] if n % 2 else 0.5 * (xs[n // 2 - 1] + xs[n // 2])


def env_info():
    from lib_qat.models import env_info as _ei
    info = _ei()
    info["lib_snapshot"] = LIB_SNAPSHOT
    return info

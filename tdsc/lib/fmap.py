"""Feature-map interface and probe packing (step 4, sub-problem B).

New module.  A layer consumes C_in x H x W and produces C_out x Ho x Wo with a
3x3 (or 1x1) kernel, stride 1 or 2 and `same` zero padding.  Output positions
whose receptive fields are disjoint are independent probes, so one feature-map
query carries several independent LTA probes.

Channel-only permutation leaves the spatial index public, so the client knows
which reply entries belong to which probe; the probes are therefore exactly the
single-patch responses of steps 2b/3, which `equivalence_error` checks against a
real convolution on the packed map for every geometry used.
"""
import numpy as np


def out_size(H, stride):
    return (H + stride - 1) // stride


def lattice(H_in, stride, k=3, spacing=None):
    """Output positions whose full k x k receptive field lies inside the image and
    whose receptive fields are pairwise disjoint."""
    Ho = out_size(H_in, stride)
    r = k // 2
    lo = int(np.ceil(r / stride)) if stride > 0 else 0
    hi = (H_in - 1 - r) // stride
    if k == 1:
        lo, hi = 0, Ho - 1
        step = 1
    else:
        step = int(np.ceil(k / stride))
    ys = list(range(lo, hi + 1, step))
    return ys, Ho


def lattice_count(H_in, W_in, stride, k=3):
    ys, _ = lattice(H_in, stride, k)
    xs, _ = lattice(W_in, stride, k)
    return len(ys) * len(xs)


def lattice_positions(H_in, W_in, stride, k=3):
    ys, _ = lattice(H_in, stride, k)
    xs, _ = lattice(W_in, stride, k)
    return [(y, x) for y in ys for x in xs]


def pack_map(C_in, H_in, W_in, stride, k, patches, positions):
    """Build an input map carrying one patch per output position.

    patches[i] is a flat vector of length C_in * k * k, channel-major
    (index = c * k*k + slot), matching the conv weight flattening.
    """
    xmap = np.zeros((C_in, H_in, W_in), dtype=np.int64)
    r = k // 2
    kk = k * k
    for patch, (y, x) in zip(patches, positions):
        p = np.asarray(patch, dtype=np.int64).reshape(C_in, k, k)
        for kh in range(k):
            for kw in range(k):
                iy = y * stride - r + kh
                ix = x * stride - r + kw
                if 0 <= iy < H_in and 0 <= ix < W_in:
                    xmap[:, iy, ix] = p[:, kh, kw]
    return xmap


def conv_int(xmap, Wflat, b, stride, k=3):
    """Exact integer convolution with `same` zero padding.

    Wflat is (C_out, C_in * k * k) channel-major, as everywhere else here.
    """
    xmap = np.asarray(xmap, dtype=np.int64)
    C_in, H, W = xmap.shape
    C_out = Wflat.shape[0]
    Ho, Wo = out_size(H, stride), out_size(W, stride)
    r = k // 2
    pad = np.zeros((C_in, H + 2 * r, W + 2 * r), dtype=np.int64)
    pad[:, r:r + H, r:r + W] = xmap
    cols = np.empty((C_in * k * k, Ho * Wo), dtype=np.int64)
    ys = np.arange(Ho) * stride
    xs = np.arange(Wo) * stride
    for kh in range(k):
        for kw in range(k):
            blk = pad[:, kh:kh + 0 + (ys[-1] + 1), kw:kw + (xs[-1] + 1)] \
                if False else pad[np.ix_(np.arange(C_in), ys + kh, xs + kw)]
            cols[np.arange(C_in) * (k * k) + (kh * k + kw), :] = blk.reshape(C_in, -1)
    out = Wflat @ cols + np.asarray(b, dtype=np.int64)[:, None]
    return out.reshape(C_out, Ho, Wo)


def equivalence_error(C_in, H_in, W_in, stride, k, Wflat, b, patches, positions,
                      rng=None):
    """Max |real convolution on the packed map at the probe positions
       - (Wflat @ patch + b)| over the probes.  Zero means the packed query is
    exactly a bundle of independent single-patch queries."""
    xmap = pack_map(C_in, H_in, W_in, stride, k, patches, positions)
    full = conv_int(xmap, Wflat, b, stride, k)
    err = 0
    for patch, (y, x) in zip(patches, positions):
        want = Wflat @ np.asarray(patch, dtype=np.int64) + np.asarray(b, dtype=np.int64)
        err = max(err, int(np.abs(full[:, y, x] - want).max()))
    return err, xmap, full


class PackedChannel:
    """lib.lta.Channel-compatible adapter that packs the T d step queries of one
    LTA pass into ceil(T d / (P - 1)) feature-map sessions.  One lattice cell per
    session carries a zero patch whose reply must equal b exactly; mismatches are
    counted."""

    def __init__(self, W_eff, b_eff, A, attack, vary_round, vary_input, observe,
                 schedule, n_probes, embed=None):
        self.W = np.ascontiguousarray(W_eff, dtype=np.int64)
        self.b = np.ascontiguousarray(b_eff, dtype=np.int64)
        self.m, self.d = self.W.shape
        self.A = int(A)
        self.attack = attack
        self.vary_round = int(vary_round)
        self.vary_input = int(vary_input)
        self.observe = int(observe)
        self.schedule = {k: list(v) for k, v in schedule.items()}
        self.embed = embed
        self.n_probes_per_session = max(1, int(n_probes) - 1)
        self.n_queries = 0            # LTA probes
        self.packed_sessions = 0
        self.zero_check_mismatches = 0
        self.zero_checks = 0
        self.inadmissible_query_entries = 0
        self.backend = None
        self.anchor_responses = []
        self._base = None
        self._x0 = None
        self._cache = {}
        self._T = None

    # -- plumbing
    def exact(self, x):
        return self.W @ np.asarray(x, dtype=np.int64) + self.b

    def _check_admissible(self, x):
        self.inadmissible_query_entries += int(np.sum((x < 0) | (x > self.A)))

    def _one(self, xc):
        sched = {k: list(v) for k, v in self.schedule.items()}
        plans = list(sched[self.vary_round])
        plans[self.vary_input] = ("canon", xc if self.embed is None else self.embed(xc))
        sched[self.vary_round] = plans
        return self.attack.run_session(sched, self.observe)

    def _packed(self, xs):
        """One session carrying len(xs) probes plus a zero-patch check."""
        vecs = [x if self.embed is None else self.embed(x) for x in xs]
        zero = np.zeros(self.d, dtype=np.int64)
        vecs.append(zero if self.embed is None else self.embed(zero))
        sched = {k: list(v) for k, v in self.schedule.items()}
        outs = self.attack.run_session_multi(sched, self.observe, self.vary_round,
                                             self.vary_input, vecs)
        self.packed_sessions += 1
        self.zero_checks += 1
        if not np.array_equal(np.sort(outs[-1]), np.sort(self.b)):
            self.zero_check_mismatches += 1
        return outs[:-1]

    def query(self, x):
        x = np.asarray(x, dtype=np.int64)
        self._check_admissible(x)
        self.n_queries += 1
        return self._one(x)

    def prepare_anchor(self, x0):
        x0 = np.asarray(x0, dtype=np.int64)
        self._x0 = x0
        self._base = self.exact(x0)
        self._cache = {}

    def set_anchor(self, x0):
        x0 = np.asarray(x0, dtype=np.int64)
        self._check_admissible(x0)
        self.prepare_anchor(x0)
        y = self.query(x0)
        self.anchor_responses.append((x0.copy(), y.copy()))
        return y

    def set_T(self, T):
        self._T = int(T)

    def _prefetch(self):
        T = self._T
        keys = [(j, t) for j in range(self.d) for t in range(1, T + 1)]
        xs = []
        for j, t in keys:
            x = self._x0.copy()
            x[j] += t
            self._check_admissible(x)
            xs.append(x)
        n = self.n_probes_per_session
        for s in range(0, len(keys), n):
            batch = xs[s:s + n]
            outs = self._packed(batch)
            for kk, yy in zip(keys[s:s + n], outs):
                self._cache[kk] = yy
        self.n_queries += len(keys)

    def query_step(self, j, t):
        if not self._cache:
            self._prefetch()
        return self._cache[(int(j), int(t))]

    @property
    def rounding_exact(self):
        return self.attack.o.rounding_exact

    @property
    def n_rounding_failures(self):
        return self.attack.o.n_rounding_failures


# ----------------------------------------------------------- feature-map net
def block_stride(name):
    return 2 if name in ("layer2.0", "layer3.0") else 1


def quantise_resnet20_fmap(model, w_bits, a_bits, calib_images, normalise=None):
    """Quantise ResNet-20 with the requantisation scalars calibrated on real
    feature maps (the same weight scales as the single-patch quantiser; only the
    activation statistics, hence the eta and the biases, differ)."""
    from .resnet20 import resnet20_float_params, fold_input_normalisation, IntResNet20
    A = 2 ** a_bits - 1
    qmax = 2 ** (w_bits - 1) - 1
    fp = resnet20_float_params(model)
    if normalise is not None:
        fp = fold_input_normalisation(fp, normalise[0], normalise[1])
    s_in = 1.0 / A
    X = np.clip(np.rint(np.asarray(calib_images, dtype=np.float64) * A), 0,
                A).astype(np.int64)
    info = {"w_bits": w_bits, "a_bits": a_bits, "A": A, "weight_scales": {},
            "W_int_absmax": {}, "shortcut_int_absmax": {}, "requant_scales": {},
            "accumulator_absmax_calibration": 0, "calibration_images": int(len(X))}
    acc = 0

    def q(W):
        mx = float(np.abs(W).max())
        s = mx / qmax if mx > 0 else 1.0
        return np.rint(W / s).astype(np.int64), s

    Ws, ss = q(fp["stem"]["W"])
    b0 = np.rint(fp["stem"]["b"] / (ss * s_in)).astype(np.int64)
    maps = [conv_int(x, Ws, b0, 1, 3) for x in X]
    acc = max(acc, max(int(np.abs(u).max()) for u in maps))
    mx = max(int(np.maximum(u, 0).max()) for u in maps)
    eta = mx / A if mx > 0 else 1.0
    acts = [np.clip(np.rint(np.maximum(u, 0) / eta), 0, A).astype(np.int64)
            for u in maps]
    s_a = ss * s_in * eta
    stem = {"W": Ws, "b": b0, "eta": eta}
    info["weight_scales"]["conv1"] = ss
    info["W_int_absmax"]["conv1"] = int(np.abs(Ws).max())
    info["requant_scales"]["conv1"] = eta
    blocks = []
    for blk in fp["blocks"]:
        nm = blk["name"]
        st = block_stride(nm)
        W1, s1 = q(blk["W1"])
        b1 = np.rint(blk["b1"] / (s1 * s_a)).astype(np.int64)
        u1 = [conv_int(a, W1, b1, st, 3) for a in acts]
        acc = max(acc, max(int(np.abs(u).max()) for u in u1))
        m1 = max(int(np.maximum(u, 0).max()) for u in u1)
        eta1 = m1 / A if m1 > 0 else 1.0
        a1 = [np.clip(np.rint(np.maximum(u, 0) / eta1), 0, A).astype(np.int64)
              for u in u1]
        s_a1 = s1 * s_a * eta1
        W2, s2 = q(blk["W2"])
        b2 = np.rint(blk["b2"] / (s2 * s_a1)).astype(np.int64)
        s_z = s2 * s_a1
        S = np.rint(blk["Ws"] * s_a / s_z).astype(np.int64)
        bsc = np.rint(blk["bs"] / s_z).astype(np.int64)
        u = [conv_int(z, W2, b2, 1, 3) + conv_int(a, S, bsc, st, 1)
             for z, a in zip(a1, acts)]
        acc = max(acc, max(int(np.abs(v).max()) for v in u))
        mx = max(int(np.maximum(v, 0).max()) for v in u)
        etab = mx / A if mx > 0 else 1.0
        acts = [np.clip(np.rint(np.maximum(v, 0) / etab), 0, A).astype(np.int64)
                for v in u]
        s_a = s_z * etab
        blocks.append({"name": nm, "W1": W1, "b1": b1, "eta1": eta1, "W2": W2,
                       "b2": b2, "S": S, "bs": bsc, "eta": etab,
                       "shortcut": blk["shortcut"], "stride": st,
                       "C_in": blk["C_in"], "C_out": blk["C_out"]})
        info["weight_scales"][nm + ".conv1"] = s1
        info["weight_scales"][nm + ".conv2"] = s2
        info["W_int_absmax"][nm + ".conv1"] = int(np.abs(W1).max())
        info["W_int_absmax"][nm + ".conv2"] = int(np.abs(W2).max())
        info["shortcut_int_absmax"][nm] = int(np.abs(S).max())
        info["requant_scales"][nm + ".conv1"] = eta1
        info["requant_scales"][nm + ".block"] = etab
    pooled = np.stack([a.mean(axis=(1, 2)) for a in acts])
    Wf, sf = q(fp["fc"]["W"])
    bf = np.rint(fp["fc"]["b"] / (sf * s_a)).astype(np.int64)
    out = np.rint(pooled) @ Wf.T + bf
    acc = max(acc, int(np.abs(out).max()))
    info["weight_scales"]["fc"] = sf
    info["W_int_absmax"]["fc"] = int(np.abs(Wf).max())
    info["accumulator_absmax_calibration"] = int(acc)
    info["accumulator_bits_calibration"] = int(acc).bit_length() + 1
    net = IntResNet20(stem, blocks, {"W": Wf, "b": bf}, A)
    return net, info


def forward_fmap(net, X, batch=200):
    """Real convolutional forward pass of an IntResNet20 on images in [0,1]."""
    A = net.A
    outs = []
    Xi = np.clip(np.rint(np.asarray(X, dtype=np.float64) * A), 0, A).astype(np.int64)
    for s in range(0, len(Xi), batch):
        chunk = Xi[s:s + batch]
        acts = []
        for x in chunk:
            u = conv_int(x, net.stem["W"], net.stem["b"], 1, 3)
            a = np.clip(np.rint(np.maximum(u, 0) / net.stem["eta"]), 0, A).astype(np.int64)
            for blk in net.blocks:
                st = blk.get("stride", block_stride(blk["name"]))
                u1 = conv_int(a, blk["W1"], blk["b1"], st, 3)
                a1 = np.clip(np.rint(np.maximum(u1, 0) / blk["eta1"]), 0,
                             A).astype(np.int64)
                v = (conv_int(a1, blk["W2"], blk["b2"], 1, 3)
                     + conv_int(a, blk["S"], blk["bs"], st, 1))
                a = np.clip(np.rint(np.maximum(v, 0) / blk["eta"]), 0,
                            A).astype(np.int64)
            acts.append(a.mean(axis=(1, 2)))
        pooled = np.rint(np.stack(acts))
        outs.append(pooled @ net.fc["W"].T + net.fc["b"])
    return np.concatenate(outs).astype(np.int64)

"""Locate-then-control under full-coordinate permutation (step 5).

New module.  Nothing here changes the behaviour of any existing lib entry point;
it reuses lib/fmap.py (conv_int, out_size, block_stride, the feature-map
quantiser), lib/lta_run.py (run_lta), lib/session.py (unapply_perm,
find_separating_anchor), lib/resnet20.py (IntResNet20) and lib/verify.py.

Why this module exists.  `REPORT_step4.md` Section 5 concludes that a permutation
of all C_out * H * W output coordinates breaks the sequential attack, because no
input map makes all C_out * H * W reply values distinct, so a recovered layer
cannot serve as a frame probe.  That measures whether the FULL frame can be
resolved.  The attack does not need the full frame: it needs enough located
coordinates to write the next round's query, and it may choose which ones.

The locate step.  The client sends a map that is zero except at a few pixels.  It
knows the layer's matrix in its own canonical row order, so it can predict the
whole true output vector up to that row relabelling; every coordinate whose
predicted value is unique in the predicted multiset is located, i.e. the client
learns which reply position holds it.  With a sparse map almost every coordinate
carries the bias, so only the few informative coordinates are candidates and the
birthday condition applies to `n = (informative coordinates)`, not to
`N = C_out * H * W`.

The control step.  sigma acts on the whole reply, so placing a value at a located
reply position puts it at exactly the intended true coordinate, and the unlocated
positions carry a constant (zero), whose placement is therefore irrelevant.

Frame algebra.  A round with input frame `g` receives the client-frame vector `x`
and forms the true-frame input `xt` with `xt[sigma_g[i]] = x[i]`; the reply of the
round that produced `g` is `y[i] = v[sigma_g[i]]`.  If canonical channel `c` of
that layer is true channel `Pchan[c]`, the canonical coordinate `(c, q)` is the
true coordinate `(Pchan[c], q)` -- the spatial index is not relabelled, only the
channels are, because the client's ambiguity is a row permutation of the matrix.
Hence `pred[c, q] = v_true[Pchan[c], q]` and matching a unique predicted value
against the reply gives `i = sigma_g^{-1}(Pchan[c], q)`, which is precisely the
reply position the client must write to put a value at canonical `(c, q)`.
"""
import time
import numpy as np

from .fmap import conv_int, out_size, block_stride
from .session import unapply_perm


# --------------------------------------------------------------- geometry
def placements(H, W=None):
    """The placement lattice PL(H, W).

    Two properties make this set the right one to carry along a chain:

    * every p in PL is reachable from itself under a stride-1 3x3 convolution, so
      a single-pixel probe at p locates p again in the next layer's frame;
    * the stride-2 pullback of PL(H, W) is exactly PL(H/2, W/2), so the set
      survives the two stride-2 transitions of ResNet-20 without growing.

    It also contains, for a stride-1 3x3 kernel, one interior placement (1, 1)
    where all nine taps respond and the four corners, whose tap-membership
    patterns are pairwise distinct (see `tap_membership_patterns`), and, for a
    stride-2 kernel, pixels of all four parities, which is what covers the nine
    taps there.
    """
    W = H if W is None else W
    cand = [(0, 0), (0, 1), (1, 0), (1, 1), (0, W - 1), (1, W - 1),
            (H - 1, 0), (H - 1, 1), (H - 1, W - 1)]
    out = []
    for p in cand:
        if 0 <= p[0] < H and 0 <= p[1] < W and p not in out:
            out.append(p)
    return out


def reach(p, stride, k, Ho, Wo):
    """[(y, x, tap)] output positions, with the kernel tap, that input pixel p
    feeds under a stride-`stride` k x k convolution with `same` zero padding."""
    r, s = int(p[0]), int(p[1])
    rad = k // 2
    out = []
    for a in range(k):
        na = r + rad - a
        if na % stride:
            continue
        y = na // stride
        if not (0 <= y < Ho):
            continue
        for b in range(k):
            nb = s + rad - b
            if nb % stride:
                continue
            x = nb // stride
            if 0 <= x < Wo:
                out.append((y, x, a * k + b))
    return out


def taps_of(p, stride, k, Ho, Wo):
    return sorted(set(t for (_, _, t) in reach(p, stride, k, Ho, Wo)))


def tap_membership_patterns(H, W, stride, k, pls=None):
    """tap -> tuple of the placements in which it responds."""
    Ho, Wo = out_size(H, stride), out_size(W, stride)
    pls = placements(H, W) if pls is None else pls
    pat = {}
    for t in range(k * k):
        pat[t] = tuple(i for i, p in enumerate(pls)
                       if t in taps_of(p, stride, k, Ho, Wo))
    return pat


def sparse_map(C_in, H, W, pixels):
    """pixels: {(y, x): value vector of length C_in}."""
    x = np.zeros((C_in, H, W), dtype=np.int64)
    for (y, xx), v in pixels.items():
        x[:, int(y), int(xx)] = np.asarray(v, dtype=np.int64)
    return x


# --------------------------------------------------------------- locate
def locate(pred, reply):
    """Match the client's predicted canonical values against the ordered reply.

    Returns (loc, multiset_ok, n_unique) where loc[j] is the reply position of
    canonical coordinate j, or -1 when the predicted value of j is not unique in
    the predicted multiset.  The located coordinates are exactly those with a
    unique predicted value; for them the match is forced whenever the reply
    multiset equals the predicted multiset.
    """
    pred = np.asarray(pred, dtype=np.int64)
    reply = np.asarray(reply, dtype=np.int64)
    uniq, inv, cnt = np.unique(pred, return_inverse=True, return_counts=True)
    mask = cnt[inv] == 1
    order = np.argsort(reply, kind="stable")
    ys = reply[order]
    loc = np.full(pred.size, -1, dtype=np.int64)
    if mask.any():
        idx = np.searchsorted(ys, pred[mask])
        good = idx < ys.size
        idx2 = np.clip(idx, 0, max(0, ys.size - 1))
        good = good & (ys[idx2] == pred[mask])
        sel = np.nonzero(mask)[0][good]
        loc[sel] = order[idx2[good]]
    ok = bool(np.array_equal(np.sort(pred), ys))
    return loc, ok, int(mask.sum())


def multiset_counts(vals):
    u, c = np.unique(np.asarray(vals, dtype=np.int64), return_counts=True)
    return u, c


def subtract_multiset(y, bg_vals, bg_cnts):
    """Remove the multiset (bg_vals, bg_cnts) from y.  Returns (kept, leftover),
    leftover being the number of background entries that were not found."""
    y = np.asarray(y, dtype=np.int64)
    uy, cy = np.unique(y, return_counts=True)
    idx = np.searchsorted(uy, bg_vals)
    idx2 = np.clip(idx, 0, max(0, uy.size - 1))
    match = (idx < uy.size) & (uy[idx2] == bg_vals)
    take = np.zeros(uy.size, dtype=np.int64)
    np.add.at(take, idx2[match], bg_cnts[match])
    take = np.minimum(take, cy)
    leftover = int(bg_cnts.sum() - take.sum())
    kept = np.repeat(uy, cy - take)
    return kept, leftover


# --------------------------------------------------------------- round graph
class FRound:
    """One server round of the feature-map protocol.

    inputs[i]: {"kind": "new"/"ret", "W", "in_shape", "stride", "k",
                "frame": group id or -1, "src": (round, input index),
                "pool": bool}
    The reply is a permutation of the flattened C_out x Ho x Wo output (or of the
    10 logits for the final round), under one sigma per group per session.
    """

    def __init__(self, name, group, out_shape, b, inputs, unshuffled=False,
                 role=""):
        self.name = name
        self.group = int(group)
        self.out_shape = tuple(int(v) for v in out_shape)
        self.b = np.asarray(b, dtype=np.int64)
        self.inputs = inputs
        self.unshuffled = bool(unshuffled)
        self.role = role
        self.m = int(np.prod(self.out_shape))
        self.n_new = sum(1 for s in inputs if s["kind"] == "new")

    def bias_flat(self):
        if len(self.out_shape) == 1:
            return self.b.copy()
        return np.repeat(self.b, self.out_shape[1] * self.out_shape[2])


def apply_input(spec, xmap, out_shape):
    if spec.get("pool"):
        pooled = np.rint(np.asarray(xmap, dtype=np.float64).mean(axis=(1, 2)))
        return spec["W"] @ pooled.astype(np.int64)
    C_out = out_shape[0]
    return conv_int(xmap, spec["W"], np.zeros(C_out, dtype=np.int64),
                    spec["stride"], spec["k"]).reshape(-1)


def build_r3_rounds(net, H0=32):
    """The R3 round graph of a quantised ResNet-20 on the feature-map interface.

    R3 is the hardest residual mode of step 4: the server returns
    sigma(F(x) + S(x)) in one round and retains x from the block's first round.
    """
    rounds = []
    meta = []
    g = 0
    C = net.stem["W"].shape[0]
    rounds.append(FRound("conv1", g, (C, H0, H0), net.stem["b"],
                         [{"kind": "new", "W": net.stem["W"],
                           "in_shape": (net.stem["W"].shape[1] // 9, H0, H0),
                           "stride": 1, "k": 3, "frame": -1}], role="conv"))
    meta.append({"layer": "conv1", "round": 0, "group": g, "kind": "conv",
                 "eta": net.stem["eta"]})
    in_group, H = g, H0
    for blk in net.blocks:
        st = block_stride(blk["name"])
        Ho = out_size(H, st)
        g += 1
        g1 = g
        ra = len(rounds)
        rounds.append(FRound(blk["name"] + ".conv1", g1,
                             (blk["C_out"], Ho, Ho), blk["b1"],
                             [{"kind": "new", "W": blk["W1"],
                               "in_shape": (blk["C_in"], H, H),
                               "stride": st, "k": 3, "frame": in_group}],
                             role="conv"))
        meta.append({"layer": blk["name"] + ".conv1", "round": ra, "group": g1,
                     "kind": "conv", "eta": blk["eta1"]})
        g += 1
        g2 = g
        rb = len(rounds)
        rounds.append(FRound(blk["name"] + ".conv2+shortcut", g2,
                             (blk["C_out"], Ho, Ho), blk["b2"] + blk["bs"],
                             [{"kind": "new", "W": blk["W2"],
                               "in_shape": (blk["C_out"], Ho, Ho),
                               "stride": 1, "k": 3, "frame": g1},
                              {"kind": "ret", "W": blk["S"],
                               "in_shape": (blk["C_in"], H, H),
                               "stride": st, "k": 1, "src": (ra, 0)}],
                             role="sum"))
        meta.append({"layer": blk["name"] + ".shortcut", "round": rb,
                     "group": g2, "kind": "shortcut", "vary_round": ra,
                     "primary_round": rb})
        meta.append({"layer": blk["name"] + ".conv2", "round": rb, "group": g2,
                     "kind": "conv2", "retained_round": ra,
                     "eta": blk["eta"]})
        in_group, H = g2, Ho
    g += 1
    rf = len(rounds)
    rounds.append(FRound("fc", g, (net.fc["W"].shape[0],), net.fc["b"],
                         [{"kind": "new", "W": net.fc["W"],
                           "in_shape": (net.fc["W"].shape[1], H, H),
                           "stride": 1, "k": 1, "frame": in_group,
                           "pool": True}], unshuffled=True, role="fc"))
    meta.append({"layer": "fc", "round": rf, "group": g, "kind": "fc"})
    return rounds, meta


class PartialOracle:
    """Session oracle with a fresh uniform permutation of ALL output coordinates
    per group per session.  Replies are ordered; one query per round per session.

    The deterministic map (true-frame inputs) -> output is memoised on a digest of
    the inputs; the session randomness (the permutations and the noise) is drawn
    outside the cache, so the cache cannot change the distribution of any reply.
    Replies the client never reads are evaluated (and counted) but not permuted.
    """

    def __init__(self, rounds, A, noise_fn, rng, longdouble=True):
        self.rounds = rounds
        self.A = int(A)
        self.noise_fn = noise_fn
        self.rng = rng
        self.R = len(rounds)
        self.longdouble = bool(longdouble)
        self.groups = sorted(set(r.group for r in rounds))
        self.group_m = {r.group: r.m for r in rounds}
        self.group_unshuffled = {}
        for r in rounds:
            self.group_unshuffled[r.group] = (
                self.group_unshuffled.get(r.group, True) and r.unshuffled)
        self.n_sessions = 0
        self.n_round_evaluations = 0
        self.n_replies_read = 0
        self.n_rounding_failures = 0
        self.inadmissible_entries = 0
        self.cache_hits = 0
        self._cache = [dict() for _ in range(self.R)]
        self._seen = [dict() for _ in range(self.R)]

    @property
    def rounding_exact(self):
        return self.n_rounding_failures == 0

    def _digest(self, true_maps):
        import hashlib
        h = hashlib.blake2b(digest_size=16)
        for v in true_maps:
            h.update(np.ascontiguousarray(v).tobytes())
            h.update(b"|")
        return h.digest()

    def eval_round(self, r, true_maps):
        key = self._digest(true_maps)
        hit = self._cache[r].get(key)
        if hit is not None:
            self.cache_hits += 1
            return hit
        spec = self.rounds[r]
        v = spec.bias_flat()
        for s, xm in zip(spec.inputs, true_maps):
            if not np.any(xm):
                continue
            v = v + apply_input(s, xm, spec.out_shape)
        seen = self._seen[r]
        if seen.pop(key, 0):
            self._cache[r][key] = v
        else:
            if len(seen) > 4096:
                seen.clear()
            seen[key] = 1
        return v

    def session(self):
        self.n_sessions += 1
        return PartialSession(self)


class PartialSession:
    def __init__(self, oracle):
        self.o = oracle
        self.sigmas = {}
        self.r = 0
        self.stored = {}

    def _sigma(self, g):
        s = self.sigmas.get(g)
        if s is None:
            m = self.o.group_m[g]
            s = (np.arange(m, dtype=np.int64) if self.o.group_unshuffled[g]
                 else self.o.rng.permutation(m))
            self.sigmas[g] = s
        return s

    def round(self, vectors, read=True):
        """vectors: one client-frame FLAT vector per "new" input."""
        o = self.o
        r = self.r
        if r >= o.R:
            raise RuntimeError("session already used all %d rounds" % o.R)
        spec = o.rounds[r]
        vectors = list(vectors)
        if len(vectors) != spec.n_new:
            raise ValueError("round %d expects %d new inputs, got %d"
                             % (r, spec.n_new, len(vectors)))
        true_maps = []
        k = 0
        for i, s in enumerate(spec.inputs):
            if s["kind"] == "new":
                x = np.asarray(vectors[k], dtype=np.int64).ravel()
                k += 1
                o.inadmissible_entries += int(np.sum((x < 0) | (x > o.A)))
                gfr = s["frame"]
                xt = x if gfr < 0 else unapply_perm(self._sigma(gfr), x)
                xm = xt.reshape(s["in_shape"])
                self.stored[(r, i)] = xm
                true_maps.append(xm)
            else:
                true_maps.append(self.stored[s["src"]])
        v = o.eval_round(r, true_maps)
        o.n_round_evaluations += 1
        self.r += 1
        if not read:
            return None
        o.n_replies_read += 1
        permuted = v[self._sigma(spec.group)]
        dt = np.longdouble if o.longdouble else np.float64
        obs = permuted.astype(dt) + o.noise_fn(permuted.size).astype(dt)
        y = np.rint(obs).astype(np.int64)
        if not np.array_equal(y, permuted):
            o.n_rounding_failures += 1
        return y


# --------------------------------------------------------------- attack
class PartialAttack:
    """Frame-tracked attack with a PARTIAL frame.

    A probe installed at round r is a sparse client map together with the whole
    predicted canonical output vector.  In every session the probe's reply is
    matched against that prediction; the coordinates whose predicted value is
    unique are located, and a later round whose input lives in that group may
    write to those coordinates only.  Unlocated positions carry zero.
    """

    def __init__(self, oracle, A, rng):
        self.o = oracle
        self.A = int(A)
        self.rng = rng
        self.probes = {}
        self.probe_rounds = 0
        self.probe_multiset_failures = 0
        self.frame_unavailable = 0
        self.missing_coordinate_events = 0
        self.missing_coordinates = 0
        self.located_samples = []

    # -- writing a client-frame vector
    def _client_vector(self, spec, plan, loc):
        d = int(np.prod(spec["in_shape"]))
        if plan[0] == "zero":
            return np.zeros(d, dtype=np.int64)
        if plan[0] == "const":
            return np.full(d, int(plan[1]), dtype=np.int64)
        if plan[0] == "pixels":
            cmap = sparse_map(spec["in_shape"][0], spec["in_shape"][1],
                              spec["in_shape"][2], plan[1])
        elif plan[0] == "canonmap":
            cmap = np.asarray(plan[1], dtype=np.int64).reshape(spec["in_shape"])
        else:
            raise ValueError("unknown plan %r" % (plan,))
        flat = cmap.reshape(-1)
        g = spec["frame"]
        if g < 0:
            return flat
        if loc is None or g not in loc:
            self.frame_unavailable += 1
            return np.zeros(d, dtype=np.int64)
        L = loc[g]
        out = np.zeros(d, dtype=np.int64)
        need = np.nonzero(flat != 0)[0]
        if need.size:
            pos = L[need]
            ok = pos >= 0
            out[pos[ok]] = flat[need[ok]]
            nb = int(np.sum(~ok))
            if nb:
                self.missing_coordinate_events += 1
                self.missing_coordinates += nb
        return out

    def run_session(self, plans, observe):
        """One session.  `plans` maps a round index to a list of input plans; a
        round with no plan sends its installed probe when it precedes `observe`
        and zeros otherwise.  Returns the reply of round `observe`."""
        sess = self.o.session()
        loc = {}
        out = None
        for q in range(self.o.R):
            spec = self.o.rounds[q]
            pl = plans.get(q)
            use_probe = False
            if pl is None:
                if q in self.probes and q < observe:
                    pl = self.probes[q]["plans"]
                    use_probe = True
                else:
                    pl = [("zero",)] * spec.n_new
            news = [s for s in spec.inputs if s["kind"] == "new"]
            vecs = [self._client_vector(s, p, loc) for s, p in zip(news, pl)]
            read = use_probe or (q == observe)
            y = sess.round(vecs, read=read)
            if q == observe:
                out = y
            if use_probe:
                self.probe_rounds += 1
                pr = self.probes[q]
                L, ok, nuniq = locate(pr["pred"], y)
                if not ok:
                    self.probe_multiset_failures += 1
                loc[spec.group] = L
        return out

    def install_probe(self, r, plans, pred):
        pred = np.asarray(pred, dtype=np.int64)
        uniq, inv, cnt = np.unique(pred, return_inverse=True,
                                  return_counts=True)
        mask = cnt[inv] == 1
        self.probes[r] = {"plans": plans, "pred": pred, "locatable": mask,
                          "n_locatable": int(mask.sum())}
        return self.probes[r]


class IsolatedChannel:
    """lib.lta.Channel-compatible adapter.

    One LTA query is one session in which input `vary_input` of round
    `vary_round` carries the client's canonical map built from the query vector,
    and round `observe`'s reply is reduced to the informative multiset by
    subtracting the predicted background multiset.
    """

    def __init__(self, W_stack, b_stack, A, attack, vary_round, vary_input,
                 observe, plans, embed, bg_vals, bg_cnts, n_expected):
        self.W = np.ascontiguousarray(W_stack, dtype=np.int64)
        self.b = np.ascontiguousarray(b_stack, dtype=np.int64)
        self.m, self.d = self.W.shape
        self.A = int(A)
        self.attack = attack
        self.vary_round = int(vary_round)
        self.vary_input = int(vary_input)
        self.observe = int(observe)
        self.plans = {k: list(v) for k, v in plans.items()}
        self.embed = embed
        self.bg_vals = bg_vals
        self.bg_cnts = bg_cnts
        self.n_expected = int(n_expected)
        self.n_queries = 0
        self.inadmissible_query_entries = 0
        self.isolation_failures = 0
        self.isolation_leftover = 0
        self.backend = None
        self.anchor_responses = []
        self._x0 = None
        self._base = None

    def exact(self, x):
        return self.W @ np.asarray(x, dtype=np.int64) + self.b

    def raw_query(self, x):
        x = np.asarray(x, dtype=np.int64)
        self.inadmissible_query_entries += int(np.sum((x < 0) | (x > self.A)))
        pl = {k: list(v) for k, v in self.plans.items()}
        row = list(pl[self.vary_round])
        row[self.vary_input] = self.embed(x)
        pl[self.vary_round] = row
        return self.attack.run_session(pl, self.observe)

    def query(self, x):
        self.n_queries += 1
        y = self.raw_query(x)
        kept, leftover = subtract_multiset(y, self.bg_vals, self.bg_cnts)
        if leftover or kept.size != self.n_expected:
            self.isolation_failures += 1
            self.isolation_leftover += int(leftover)
        if kept.size != self.n_expected:
            if kept.size > self.n_expected:
                kept = kept[:self.n_expected]
            else:
                kept = np.concatenate(
                    [kept, np.zeros(self.n_expected - kept.size, np.int64)])
        return kept

    def prepare_anchor(self, x0):
        self._x0 = np.asarray(x0, dtype=np.int64)
        self._base = self.exact(self._x0)

    def set_anchor(self, x0):
        self.prepare_anchor(x0)
        y = self.query(x0)
        self.anchor_responses.append((np.asarray(x0).copy(), y.copy()))
        return y

    def query_step(self, j, t):
        x = self._x0.copy()
        x[j] += t
        return self.query(x)

    @property
    def rounding_exact(self):
        return self.attack.o.rounding_exact

    @property
    def n_rounding_failures(self):
        return self.attack.o.n_rounding_failures


# --------------------------------------------------------------- helpers
def separating_values(predict, n_pix, C, A, rng, tries=60):
    """Client-side search (no oracle query) for pixel value vectors that make
    `predict(values)` as injective as possible on the coordinates it returns."""
    best, best_bad = None, None
    for _ in range(tries):
        vals = [rng.integers(0, A + 1, size=C).astype(np.int64)
                for _ in range(n_pix)]
        key = np.asarray(predict(vals), dtype=np.int64)
        bad = key.size - int(np.unique(key).size)
        if best_bad is None or bad < best_bad:
            best, best_bad = vals, bad
        if bad == 0:
            break
    return best, int(best_bad)


def required_placements(Ho, Wo, stride, k, pool=False):
    """The placements a consumer of this frame actually needs.

    A stride-1 3x3 consumer needs one interior placement, where all nine taps
    respond, and the four corners, whose tap-membership patterns already separate
    the nine taps; a stride-2 3x3 consumer needs all nine positions of PL, because
    there the taps split by the parity of the probe pixel; a 1x1 consumer needs
    one position.  Asking for less is what keeps the locate step's birthday
    condition small.
    """
    pl = placements(Ho, Wo)
    five = [p for p in [(1, 1), (0, 0), (0, Wo - 1), (Ho - 1, 0),
                        (Ho - 1, Wo - 1), (0, 1), (1, 0)] if p in pl]
    if pool:
        return five
    if k == 1:
        return [(0, 0)]
    if stride == 1:
        return five
    return pl


def linkage_pairs(p, p2, stride, k, Ho, Wo):
    """Output positions that BOTH input pixels feed, with the pair of kernel taps
    that meets there.  A reply value at such a position is
    slope_{t1} . u + slope_{t2} . v + b, so observing it certifies that the two
    recovered rows belong to the SAME output channel -- which is the only signal
    available when two channels share a bias."""
    r1 = {(y, x): t for (y, x, t) in reach(p, stride, k, Ho, Wo)}
    r2 = {(y, x): t for (y, x, t) in reach(p2, stride, k, Ho, Wo)}
    both = sorted(set(r1) & set(r2))
    return [(r1[q], r2[q]) for q in both], both, sorted(set(r1) | set(r2))


def linkage_plan(p, pls, stride, k, Ho, Wo):
    """Partner pixels whose tap pairings connect the taps of one channel."""
    kk = k * k
    parent = list(range(kk))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    seen = set(t for (_, _, t) in reach(p, stride, k, Ho, Wo))
    chosen = []
    for p2 in pls:
        if tuple(p2) == tuple(p):
            continue
        pairs, both, inf = linkage_pairs(p, p2, stride, k, Ho, Wo)
        gain = sum(1 for (t1, t2) in pairs if find(t1) != find(t2))
        if gain <= 0:
            continue
        for (t1, t2) in pairs:
            ra, rb = find(t1), find(t2)
            if ra != rb:
                parent[ra] = rb
        seen |= set(t for (_, _, t) in reach(p2, stride, k, Ho, Wo))
        chosen.append(tuple(p2))
    connected = bool(len(set(find(t) for t in seen)) == 1)
    return chosen, connected


def pullback_pixels(need, H, Win, stride, k, Ho, Wo, pls=None):
    """A smallest greedy set of input pixels of PL(H, Win) whose output reaches
    cover every needed output position."""
    pls = placements(H, Win) if pls is None else pls
    cov = {p: set((y, x) for (y, x, _) in reach(p, stride, k, Ho, Wo))
           for p in pls}
    out, done = [], set()
    for q in need:
        q = tuple(q)
        if q in done:
            continue
        for p in pls:
            if q in cov[p]:
                out.append(p)
                done |= cov[p]
                break
    return out


def membership_collisions(u, cands, slopes, bgmap, pls, stride, k, Ho, Wo):
    """Rows whose membership value at some placement coincides with another
    row's.  Soundness of the membership test needs this to be zero: a value that
    is present must belong to the row that predicted it.  Equal values INSIDE one
    row are harmless, they merely fail to narrow."""
    allt = set(range(k * k))
    bad = 0
    for pm in pls:
        rch = reach(pm, stride, k, Ho, Wo)
        taps_m = set(t for _, _, t in rch)
        if not taps_m or taps_m == allt:
            continue
        posof = {t: (y, x) for y, x, t in rch}
        buckets = {}
        for i in range(len(cands)):
            su = int(np.dot(slopes[i], u))
            for (ci, t) in cands[i]:
                if t not in taps_m:
                    continue
                y, x = posof[t]
                buckets.setdefault(su + int(bgmap[ci, y, x]), set()).add(i)
        bad += sum(len(v) - 1 for v in buckets.values() if len(v) > 1)
    return int(bad)


def best_membership_anchor(cands, slopes, bgmap, pls, stride, k, Ho, Wo, C, A,
                           rng, tries=40):
    best, best_bad = None, None
    for _ in range(tries):
        u = rng.integers(0, A + 1, size=C).astype(np.int64)
        bad = membership_collisions(u, cands, slopes, bgmap, pls, stride, k,
                                    Ho, Wo)
        if best_bad is None or bad < best_bad:
            best, best_bad = u, bad
        if bad == 0:
            break
    return best, int(best_bad)


def coverage_placements(H, W, stride, k, pls=None):
    """A smallest greedy subset of PL whose taps cover all k*k taps."""
    Ho, Wo = out_size(H, stride), out_size(W, stride)
    pls = placements(H, W) if pls is None else pls
    need = set(range(k * k))
    chosen = []
    while need:
        best, gain = None, -1
        for p in pls:
            ts = set(taps_of(p, stride, k, Ho, Wo))
            gn = len(ts & need)
            if gn > gain:
                best, gain = p, gn
        if gain <= 0:
            break
        chosen.append(best)
        need -= set(taps_of(best, stride, k, Ho, Wo))
    return chosen, sorted(need)


def _match_rows_to_taps(cand, n_taps):
    """Bipartite matching of rows (candidate tap sets) to taps; returns a list of
    tap indices or None."""
    n = len(cand)
    assign = [-1] * n
    used = {}

    def try_row(i, seen):
        for t in cand[i]:
            if t in seen:
                continue
            seen.add(t)
            if t not in used or try_row(used[t], seen):
                used[t] = i
                assign[i] = t
                return True
        return False

    order = sorted(range(n), key=lambda i: len(cand[i]))
    for i in order:
        if not try_row(i, set()):
            return None
    out = [-1] * n
    for t, i in used.items():
        out[i] = t
    return out


def refine_candidates(cands, slopes, bgmap, pls, stride, k, Ho, Wo, us,
                      do_query):
    """Narrow (channel, tap) candidates by membership queries.

    For every placement whose tap set is a proper non-empty subset of the kernel,
    one query per probe value in `us`.  If an in-set candidate of a row predicts a
    value that is present in the isolated reply for EVERY probe value, the row
    responds at that placement, so its tap lies in that placement's tap set and
    among the surviving candidates; if no in-set candidate survives, the row does
    not respond and its tap lies outside.

    Several probe values are used because the test is only sound when a present
    value belongs to the row that predicted it.  At the widest layers the value
    sets are large enough that a single probe value leaves a handful of
    cross-row coincidences (measured), and a coincidence that survives all of
    several independent probe values is vanishingly unlikely.
    """
    allt = set(range(k * k))
    nq = 0
    us = [np.asarray(u, dtype=np.int64) for u in
          (us if isinstance(us, (list, tuple)) else [us])]
    for pm in pls:
        if all(len(c) <= 1 for c in cands):
            break
        rch = reach(pm, stride, k, Ho, Wo)
        taps_m = set(t for _, _, t in rch)
        if not taps_m or taps_m == allt:
            continue
        posof = {t: (y, x) for y, x, t in rch}
        obs = []
        for u in us:
            kept = do_query(pm, sorted(taps_m), posof, u)
            nq += 1
            obs.append((u, set(int(v) for v in np.asarray(kept).ravel()),
                        [int(np.dot(sl, u)) for sl in slopes]))
        for i in range(len(cands)):
            if len(cands[i]) <= 1:
                continue
            inset = [(ci, t) for (ci, t) in cands[i] if t in taps_m]
            hit = []
            thr = max(1, len(obs) - 1)
            for (ci, t) in inset:
                y, x = posof[t]
                votes = sum(1 for (_u, pres, su) in obs
                            if su[i] + int(bgmap[ci, y, x]) in pres)
                if votes >= thr:
                    hit.append((ci, t))
            if hit:
                cands[i] = hit
            else:
                rest = [(ci, t) for (ci, t) in cands[i] if t not in taps_m]
                if rest:
                    cands[i] = rest
    return cands, nq


def assemble_from_candidates(cands, slopes, intercepts, C_out, C_in, k):
    """Assign (channel, tap) to every recovered row and build the layer.

    Rows are grouped by their candidate channel; inside a channel the rows are
    matched to taps by augmenting paths.  Rows whose slope vectors are equal can
    be exchanged without changing the matrix, so an arbitrary choice among them
    is harmless and is counted separately from a genuine failure.
    """
    kk = k * k
    W = np.zeros((C_out, C_in * kk), dtype=np.int64)
    b = np.zeros(C_out, dtype=np.int64)
    info = {"rows": len(cands), "rows_with_unique_candidate": 0,
            "channels_resolved": 0, "matching_failures": 0,
            "rows_with_no_candidate": 0, "channel_ambiguous_rows": 0}
    bychan = {}
    for i, cd in enumerate(cands):
        if not cd:
            info["rows_with_no_candidate"] += 1
            continue
        if len(cd) == 1:
            info["rows_with_unique_candidate"] += 1
        chs = set(ci for (ci, _) in cd)
        if len(chs) > 1:
            info["channel_ambiguous_rows"] += 1
        ci = sorted(chs)[0]
        bychan.setdefault(ci, []).append(i)
    for ci, idxs in bychan.items():
        if ci >= C_out:
            info["matching_failures"] += 1
            continue
        cand_t = [sorted(set(t for (c2, t) in cands[i] if c2 == ci))
                  or list(range(kk)) for i in idxs]
        assign = _match_rows_to_taps(cand_t, kk)
        if assign is None:
            info["matching_failures"] += 1
            assign, used = [], set()
            for ct in cand_t:
                pick = next((t for t in ct if t not in used), None)
                if pick is None:
                    pick = next((t for t in range(kk) if t not in used), 0)
                used.add(pick)
                assign.append(pick)
        else:
            info["channels_resolved"] += 1
        for j, i in enumerate(idxs):
            t = assign[j]
            W[ci, np.arange(C_in) * kk + t] = slopes[i]
        b[ci] = intercepts[idxs[0]] if idxs else 0
    info["channels_seen"] = len(bychan)
    return W, b, info


# --------------------------------------------------------------- the chain
def _strip(rec):
    return {k: v for k, v in rec.items() if not k.startswith("_")}


def extract_chain_partial(net, T=3, seed=0, noise_law="Gaussian", H0=32,
                          log=print, deadline=None, longdouble=True,
                          diagnostics=True, probe_tries=80, sep_tries=80,
                          repair_attempts=0):
    """Locate-then-control extraction of a quantised ResNet-20 under a fresh
    uniform permutation of ALL output coordinates per group per session, on the
    feature-map interface, in residual mode R3.

    Order inside a block.  The shortcut S is recovered BEFORE conv2, which is the
    opposite of lib/extract_graph.py.  The reason is specific to this threat
    model: the isolation step needs the client to predict the whole background of
    the observed reply, and the retained input of the sum round contributes S
    times the preceding round's probe map to that background.  Recovering S first
    with a frame-invariant (zero) primary input makes that contribution known
    before conv2 is attacked.  The shortcut LTA needs no frame for the primary
    input and only the preceding group's frame for the retained one.
    """
    from .lta import noise_fns_bounded
    from .lta_run import run_lta, require_lta_run_certified
    from .verify import (canonicalise, row_exactness, row_permutation,
                         frame_check, canonical_column_matrix)
    from .resnet20 import IntResNet20

    A = net.A
    rounds, meta = build_r3_rounds(net, H0)
    rng_o = np.random.default_rng([seed, 1])
    rng_a = np.random.default_rng([seed, 2])
    noise_fn = noise_fns_bounded(rng_o)[noise_law]
    oracle = PartialOracle(rounds, A, noise_fn, rng_o, longdouble=longdouble)
    attack = PartialAttack(oracle, A, rng_a)
    Pchan = {-1: None}
    live_map = {-1: None}
    group_need = {}
    for _q, _rr in enumerate(rounds):
        for _sp in _rr.inputs:
            if _sp["kind"] != "new" or _sp["frame"] < 0:
                continue
            _C, _Hi, _Wi = _sp["in_shape"]
            _cur = group_need.setdefault(_sp["frame"], [])
            for _p in required_placements(_Hi, _Wi, _sp["stride"], _sp["k"],
                                          bool(_sp.get("pool"))):
                if _p not in _cur:
                    _cur.append(_p)
    recovered = {}
    probe_canon = {}
    records = []
    stopped = ""
    repairs = 0
    t0 = time.time()

    def zero_reply(r):
        pl = {q: [("zero",)] * rounds[q].n_new for q in range(r + 1)}
        return attack.run_session(pl, r)

    def bias_multiset(r):
        y = zero_reply(r)
        u, c = np.unique(y, return_counts=True)
        sh = rounds[r].out_shape
        hw = 1 if len(sh) == 1 else sh[1] * sh[2]
        ok = bool(np.all(c % hw == 0))
        return u, (c // hw).astype(np.int64), ok

    # ---------------------------------------------------------------- layers
    def do_conv(item):
        r, g = item["round"], item["group"]
        spec = rounds[r].inputs[0]
        C_in, H, Win = spec["in_shape"]
        st, k = spec["stride"], spec["k"]
        C_out, Ho, Wo = rounds[r].out_shape
        gfr = spec["frame"]
        Sc = canonical_column_matrix(spec["W"], Pchan.get(gfr), k * k)
        b_t = np.asarray(rounds[r].b, dtype=np.int64)
        s0 = oracle.n_sessions
        rec = {"layer": item["layer"], "round": r, "group": g, "kind": "conv",
               "C_out": C_out, "C_in": C_in, "H_in": H, "H_out": Ho,
               "stride": st, "kernel": k,
               "coordinates_nominal": int(C_out * Ho * Wo)}
        # A channel of the preceding layer whose recovered row is identically zero
        # is constant in every reply, so no query can locate its coordinates.  Its
        # activation is the public constant clip(rint(ReLU(b)/eta)); when that is
        # zero the channel feeds zero and its columns are irrelevant to the
        # function, which the client certifies from its own recovered matrix.
        kk = k * k
        live = live_map.get(gfr)
        live_idx = (np.arange(C_in, dtype=np.int64) if live is None
                    else np.asarray(live, dtype=np.int64))
        C_live = int(live_idx.size)
        cols = (live_idx[:, None] * kk + np.arange(kk)).ravel()
        dead_cols = np.setdiff1d(np.arange(C_in * kk), cols)
        Sc_cmp = np.array(Sc, copy=True)
        Sc_cmp[:, dead_cols] = 0
        Sc_live = np.ascontiguousarray(Sc[:, cols])
        rec["input_channels_live"] = C_live
        rec["input_channels_dead"] = int(C_in - C_live)
        rec["d_lta"] = C_live

        def _emb(xq, _p, _C=C_in, _li=live_idx):
            full = np.zeros(_C, dtype=np.int64)
            full[_li] = np.asarray(xq, dtype=np.int64)
            return ("pixels", {_p: full})
        bv, bc, bok = bias_multiset(r)
        rec["bias_multiset_recovered"] = bok
        rec["distinct_bias_values"] = int(bv.size)
        rec["bias_collisions"] = int(C_out - bv.size)
        pls_avail = (required_placements(H, Win, st, k) if gfr < 0
                     else group_need.get(gfr, placements(H, Win)))
        rec["addressable_placements"] = [list(p) for p in pls_avail]
        cov, missing = coverage_placements(H, Win, st, k, pls_avail)
        rec["coverage_placements"] = [list(p) for p in cov]
        rec["taps_not_covered_by_coverage_set"] = missing
        slopes, intercepts, cands, lta_recs = [], [], [], []
        iso_fail = 0
        for p in cov:
            taps = taps_of(p, st, k, Ho, Wo)
            nr = len(reach(p, st, k, Ho, Wo))
            Wst = np.concatenate(
                [Sc_live.reshape(C_out, C_live, kk)[:, :, t] for t in taps],
                axis=0)
            bst = np.concatenate([b_t for _ in taps])
            ch = IsolatedChannel(Wst, bst, A, attack, r, 0, r,
                                 {r: [("zero",)]},
                                 lambda xq, _p=p: _emb(xq, _p),
                                 bv, bc * (Ho * Wo - nr), nr * C_out)
            out = run_lta(Wst, bst, A, T, rng_a, None, channel=ch,
                          deadline=deadline, diagnostics=diagnostics)
            require_lta_run_certified(
                out, "%s placement %s" % (item["layer"], p))
            Wr, br = out.pop("_W_rec"), out.pop("_b_rec")
            out.pop("_x0")
            iso_fail += ch.isolation_failures
            lr = _strip(out)
            lr.update({"placement": list(p), "taps": taps,
                       "isolation_failures": int(ch.isolation_failures),
                       "informative_entries": int(nr * C_out)})
            lta_recs.append(lr)
            for i in range(Wr.shape[0]):
                slopes.append(Wr[i])
                intercepts.append(int(br[i]))
                cands.append(set(taps))
        rec["lta_runs"] = lta_recs
        rec["isolation_failures"] = int(iso_fail)
        rec["informative_entries_per_query"] = int(
            len(reach(cov[0], st, k, Ho, Wo)) * C_out) if cov else 0
        # Channels from the shared intercept.  Every one of a channel's kk rows
        # has intercept b[c], so a group of kk rows is one channel; a group of
        # 2 kk rows means two channels share a bias, and then the only way to
        # split them is to observe two of their taps ADDING at one output
        # coordinate, which is what the linkage query does.
        groups = {}
        for i, bb in enumerate(intercepts):
            groups.setdefault(int(bb), []).append(i)
        rec["bias_groups"] = len(groups)
        rec["channels_from_bias"] = len(groups)
        rec["bias_groups_oversized"] = int(
            sum(1 for v in groups.values() if len(v) != kk))
        link_q, link_fail, partners, conn = 0, 0, [], None
        if rec["bias_groups_oversized"]:
            partners, conn = linkage_plan(cov[0], pls_avail, st, k, Ho, Wo)
            rec["linkage_partners"] = [list(q) for q in partners]
            rec["linkage_tap_graph_connected"] = conn

        def split_group(idxs, bb, n_draw=3, thr=2):
            nq = 0
            for _att in range(4):
                votes = {}
                for p2 in partners:
                    pairs, both, inf = linkage_pairs(cov[0], p2, st, k, Ho, Wo)
                    if not pairs:
                        continue
                    for _d in range(n_draw):
                        uu = rng_a.integers(0, A + 1,
                                            size=C_live).astype(np.int64)
                        vv = rng_a.integers(0, A + 1,
                                            size=C_live).astype(np.int64)
                        f1 = np.zeros(C_in, dtype=np.int64)
                        f1[live_idx] = uu
                        f2 = np.zeros(C_in, dtype=np.int64)
                        f2[live_idx] = vv
                        y = attack.run_session(
                            {r: [("pixels", {cov[0]: f1, p2: f2})]}, r)
                        kept, lo = subtract_multiset(
                            y, bv, bc * (Ho * Wo - len(inf)))
                        nq += 1
                        obs = set(int(z) for z in kept)
                        su = {i: int(np.dot(slopes[i], uu)) for i in idxs}
                        sv = {j: int(np.dot(slopes[j], vv)) for j in idxs}
                        for i in idxs:
                            for j in idxs:
                                if i == j:
                                    continue
                                if su[i] + sv[j] + bb in obs:
                                    kk2 = (min(i, j), max(i, j))
                                    votes[kk2] = votes.get(kk2, 0) + 1
                adj = {i: set() for i in idxs}
                for (i, j), v in votes.items():
                    if v >= thr:
                        adj[i].add(j)
                        adj[j].add(i)
                comps, seen = [], set()
                for i in idxs:
                    if i in seen:
                        continue
                    st2, comp = [i], []
                    seen.add(i)
                    while st2:
                        a = st2.pop()
                        comp.append(a)
                        for nb in adj[a]:
                            if nb not in seen:
                                seen.add(nb)
                                st2.append(nb)
                    comps.append(sorted(comp))
                if all(len(c) == kk for c in comps):
                    return comps, nq
            return None, nq

        chan_of, chan_bias = {}, []
        for bb in sorted(groups):
            idxs = groups[bb]
            if len(idxs) == kk:
                chan_of.update({i: len(chan_bias) for i in idxs})
                chan_bias.append(bb)
                continue
            comps, nq = split_group(idxs, bb)
            link_q += nq
            if comps is None:
                link_fail += 1
                comps = [idxs[z:z + kk] for z in range(0, len(idxs), kk)]
            for c in comps:
                chan_of.update({i: len(chan_bias) for i in c})
                chan_bias.append(bb)
        rec["linkage_queries"] = link_q
        rec["linkage_failures"] = link_fail
        rec["channels_after_linkage"] = len(chan_bias)
        cand2 = [[(chan_of[i], t) for t in sorted(cands[i])]
                 for i in range(len(cands))]
        bgmap = np.zeros((max(C_out, len(chan_bias)), Ho, Wo), dtype=np.int64)
        for i, v in enumerate(chan_bias):
            bgmap[i, :, :] = v
        Rw = np.stack(slopes) if slopes else np.zeros((0, C_live), np.int64)
        Rb = np.asarray(intercepts, dtype=np.int64)
        cand0 = [list(c) for c in cand2]
        nmq, attempt, ainfo = 0, 0, None
        while attempt < 4:
            attempt += 1
            us, ubad = [], 0
            for _u in range(3):
                uu, bad = best_membership_anchor(cand0, slopes, bgmap,
                                                pls_avail, st, k, Ho, Wo,
                                                C_live, A, rng_a,
                                                tries=sep_tries)
                us.append(uu)
                ubad = max(ubad, bad)
                if bad == 0:
                    break

            def do_q(pm, taps_m, posof, _u):
                nr = len(posof)
                y = attack.run_session({r: [_emb(_u, pm)]}, r)
                kept, lo = subtract_multiset(y, bv, bc * (Ho * Wo - nr))
                return kept

            cand2 = [list(c) for c in cand0]
            cand2, nq2 = refine_candidates(cand2, slopes, bgmap,
                                           pls_avail, st, k, Ho, Wo, us,
                                           do_q)
            nmq += nq2
            W_live, b_can, ainfo = assemble_from_candidates(
                cand2, slopes, intercepts, C_out, C_live, k)
            if not ainfo["matching_failures"]:
                break
        rec["membership_anchor_collisions"] = ubad
        rec["membership_queries"] = nmq
        rec["membership_attempts"] = attempt
        rec["rows_still_ambiguous"] = int(sum(1 for c in cand2 if len(c) > 1))
        rec["assembly"] = ainfo
        W_can = np.zeros((C_out, C_in * kk), dtype=np.int64)
        W_can[:, cols] = W_live
        W_can, b_can = canonicalise(W_can, b_can)
        max_err, n_exact = row_exactness(W_can, b_can, Sc_cmp, b_t)
        P, n_matched = row_permutation(W_can, b_can, Sc_cmp, b_t)
        fok, ferr = frame_check(W_can, b_can, Sc_cmp, b_t, P)
        rec.update({"max_abs_error_up_to_row_perm": max_err,
                    "exact_up_to_row_perm": bool(max_err == 0),
                    "rows_recovered_exactly": n_exact,
                    "rows_matched_to_true_rows": n_matched,
                    "frame_check_exact": fok,
                    "frame_check_max_abs_error": ferr})
        Pchan[g] = P
        recovered[item["layer"]] = {"W": W_can, "b": b_can}
        live_map[g] = np.array([c for c in range(C_out) if W_can[c].any()],
                               dtype=np.int64)
        dead_ch = [c for c in range(C_out) if not W_can[c].any()]
        rec["output_channels_dead"] = len(dead_ch)
        if dead_ch:
            eta = float(item.get("eta", 1.0))
            act = [int(np.clip(np.rint(max(int(b_can[c]), 0) / eta), 0, A))
                   for c in dead_ch]
            rec["dead_output_channels"] = dead_ch
            rec["dead_output_channel_constant_activation"] = act
            rec["dead_output_channels_feed_zero"] = bool(all(a == 0 for a in act))
        rec.update(install_conv_probe(r, spec, W_can, b_can, C_in, H, Win, st, k,
                                     C_out, Ho, Wo, None, live_map[g],
                                     pls_avail, live_idx))
        rec["sessions_used"] = int(oracle.n_sessions - s0)
        return rec

    def install_conv_probe(r, spec, W_can, b_can, C_in, H, Win, st, k,
                           C_out, Ho, Wo, extra, live_out=None,
                           avail_in=None, live_in=None):
        """Install round r's frame probe: pixels at PL(H, Win), values searched so
        that the coordinates the next round needs are uniquely valued.  Only the
        live output channels need locating; a dead channel is constant in every
        reply and feeds a public constant."""
        g_out = rounds[r].group
        need = group_need.get(g_out, placements(Ho, Wo))
        chans = (list(range(C_out)) if live_out is None
                 else [int(c) for c in live_out])
        need_idx = np.array([c * (Ho * Wo) + y * Wo + x
                             for c in chans for (y, x) in need],
                            dtype=np.int64)
        ppls = pullback_pixels(need, H, Win, st, k, Ho, Wo, avail_in)
        if not ppls:
            ppls = placements(H, Win)
        li = (np.arange(C_in, dtype=np.int64) if live_in is None
              else np.asarray(live_in, dtype=np.int64))

        def _draw():
            v = np.zeros(C_in, dtype=np.int64)
            v[li] = rng_a.integers(0, A + 1, size=li.size).astype(np.int64)
            return v

        def _score(vals):
            cmap = sparse_map(C_in, H, Win, dict(zip(ppls, vals)))
            pred = conv_int(cmap, W_can, b_can, st, k).reshape(-1)
            if extra is not None:
                pred = pred + extra.reshape(-1)
            uu, inv, cc = np.unique(pred, return_inverse=True,
                                    return_counts=True)
            mask = cc[inv] == 1
            return int(mask[need_idx].sum()), pred, mask, cmap

        best = None
        for _ in range(probe_tries):
            vals = [_draw() for _ in ppls]
            sc, pred, mask, cmap = _score(vals)
            if best is None or sc > best[0]:
                best = (sc, [v.copy() for v in vals], pred, mask, cmap)
            if sc == need_idx.size:
                break
        # hill climbing on one pixel at a time; no oracle query is involved
        steps = 0
        while best[0] < need_idx.size and steps < 20 * len(ppls):
            steps += 1
            vals = [v.copy() for v in best[1]]
            j = int(rng_a.integers(0, len(ppls)))
            vals[j] = _draw()
            sc, pred, mask, cmap = _score(vals)
            if sc > best[0]:
                best = (sc, vals, pred, mask, cmap)
        score, vals, pred, mask, cmap = best
        rec_hill = steps
        plans = [("pixels", dict(zip(ppls, vals)))]
        attack.install_probe(r, plans, pred)
        probe_canon[r] = cmap
        y = attack.run_session({r: plans}, r)
        _ = rec_hill
        bgf = np.repeat(b_can, Ho * Wo)
        if extra is not None:
            bgf = bgf + extra.reshape(-1)
        nz = int(np.sum(pred != bgf))
        return {"probe_pixels": len(ppls),
                "probe_pixel_positions": [list(q) for q in ppls],
                "probe_needed_placements": [list(q) for q in need],
                "probe_hill_climb_steps": int(rec_hill),
                "probe_needed_coordinates": int(need_idx.size),
                "probe_located_needed": int(score),
                "probe_needed_all_located": bool(score == need_idx.size),
                "probe_locatable_coordinates": int(mask.sum()),
                "probe_locatable_fraction": round(
                    float(mask.sum()) / max(1, pred.size), 6),
                "probe_informative_coordinates": nz,
                "probe_verified": bool(np.array_equal(np.sort(y),
                                                      np.sort(pred)))}

    def do_shortcut(item):
        rb, ra, g2 = item["round"], item["vary_round"], item["group"]
        sspec = rounds[rb].inputs[1]
        C_in, H, Win = sspec["in_shape"]
        st = sspec["stride"]
        C_out, Ho, Wo = rounds[rb].out_shape
        gfr = rounds[ra].inputs[0]["frame"]
        Sc = canonical_column_matrix(sspec["W"], Pchan.get(gfr), 1)
        b_t = np.asarray(rounds[rb].b, dtype=np.int64)
        s0 = oracle.n_sessions
        rec = {"layer": item["layer"], "round": rb, "group": g2,
               "kind": "shortcut", "C_out": C_out, "C_in": C_in,
               "stride": st, "coordinates_nominal": int(C_out * Ho * Wo)}
        live = live_map.get(gfr)
        live_idx = (np.arange(C_in, dtype=np.int64) if live is None
                    else np.asarray(live, dtype=np.int64))
        C_live = int(live_idx.size)
        dead_cols = np.setdiff1d(np.arange(C_in), live_idx)
        Sc_cmp = np.array(Sc, copy=True)
        Sc_cmp[:, dead_cols] = 0
        Sc_live = np.ascontiguousarray(Sc[:, live_idx])
        rec["input_channels_live"] = C_live
        rec["input_channels_dead"] = int(C_in - C_live)
        rec["d_lta"] = C_live

        def _emb(xq, _p, _C=C_in, _li=live_idx):
            full = np.zeros(_C, dtype=np.int64)
            full[_li] = np.asarray(xq, dtype=np.int64)
            return ("pixels", {_p: full})
        bv, bc, bok = bias_multiset(rb)
        rec["bias_multiset_recovered"] = bok
        rec["distinct_bias_values"] = int(bv.size)
        rec["bias_collisions"] = int(C_out - bv.size)
        p0 = (0, 0)
        nr = len(reach(p0, st, 1, Ho, Wo))
        rec["informative_entries_per_query"] = int(nr * C_out)
        ch = IsolatedChannel(Sc_live, b_t, A, attack, ra, 0, rb,
                             {ra: [("zero",)], rb: [("zero",)]},
                             lambda xq: _emb(xq, p0),
                             bv, bc * (Ho * Wo - nr), nr * C_out)
        out = run_lta(Sc_live, b_t, A, T, rng_a, None, channel=ch,
                      deadline=deadline, diagnostics=diagnostics)
        require_lta_run_certified(out, item["layer"])
        Wr, br = out.pop("_W_rec"), out.pop("_b_rec")
        out.pop("_x0")
        rec["lta"] = _strip(out)
        rec["isolation_failures"] = int(ch.isolation_failures)
        Wfull = np.zeros((C_out, C_in), dtype=np.int64)
        Wfull[:, live_idx] = Wr
        S_can, b_block = canonicalise(Wfull, br)
        max_err, n_exact = row_exactness(S_can, b_block, Sc_cmp, b_t)
        P, n_matched = row_permutation(S_can, b_block, Sc_cmp, b_t)
        fok, ferr = frame_check(S_can, b_block, Sc_cmp, b_t, P)
        rec.update({"max_abs_error_up_to_row_perm": max_err,
                    "exact_up_to_row_perm": bool(max_err == 0),
                    "rows_recovered_exactly": n_exact,
                    "rows_matched_to_true_rows": n_matched,
                    "frame_check_exact": fok,
                    "frame_check_max_abs_error": ferr,
                    "sessions_used": int(oracle.n_sessions - s0)})
        Pchan[g2] = P
        recovered[item["layer"]] = {"W": S_can, "b": b_block}
        return rec

    def do_conv2(item):
        rb, ra, g2 = item["round"], item["retained_round"], item["group"]
        zspec = rounds[rb].inputs[0]
        sspec = rounds[rb].inputs[1]
        C_z, Ho, Wo = zspec["in_shape"]
        g1 = zspec["frame"]
        st = sspec["stride"]
        C_out = rounds[rb].out_shape[0]
        nm = item["layer"][: -len(".conv2")]
        gfr = rounds[ra].inputs[0]["frame"]
        s0 = oracle.n_sessions
        rec = {"layer": item["layer"], "round": rb, "group": g2,
               "kind": "conv2", "C_out": C_out, "C_in": C_z, "H_out": Ho,
               "coordinates_nominal": int(C_out * Ho * Wo)}
        Sc2 = canonical_column_matrix(zspec["W"], Pchan.get(g1), 9)
        b_t = np.asarray(rounds[rb].b, dtype=np.int64)
        P2 = np.asarray(Pchan[g2], dtype=np.int64)
        live = live_map.get(g1)
        live_idx = (np.arange(C_z, dtype=np.int64) if live is None
                    else np.asarray(live, dtype=np.int64))
        C_live = int(live_idx.size)
        cols = (live_idx[:, None] * 9 + np.arange(9)).ravel()
        dead_cols = np.setdiff1d(np.arange(C_z * 9), cols)
        Sc2[:, dead_cols] = 0
        Sc2_can = Sc2[P2]
        Sc2_live = np.ascontiguousarray(Sc2_can[:, cols])
        rec["input_channels_live"] = C_live
        rec["input_channels_dead"] = int(C_z - C_live)
        rec["d_lta"] = C_live

        def _emb(xq, _p, _C=C_z, _li=live_idx):
            full = np.zeros(_C, dtype=np.int64)
            full[_li] = np.asarray(xq, dtype=np.int64)
            return ("pixels", {_p: full})
        Sc_sc = canonical_column_matrix(sspec["W"], Pchan.get(gfr), 1)
        xprobe = probe_canon[ra]
        S_can = recovered[nm + ".shortcut"]["W"]
        b_block_can = recovered[nm + ".shortcut"]["b"]
        zc = np.zeros(C_out, dtype=np.int64)
        Scon = conv_int(xprobe, S_can, zc, st, 1)
        bgmap = b_block_can[:, None, None] + Scon
        Scon_t = conv_int(xprobe, Sc_sc[P2], zc, st, 1)
        bgmap_t = b_t[P2][:, None, None] + Scon_t
        rec["background_matches_truth"] = bool(np.array_equal(bgmap, bgmap_t))
        pls_avail = group_need.get(g1, placements(Ho, Wo))
        rec["addressable_placements"] = [list(q) for q in pls_avail]
        cov2, miss2 = coverage_placements(Ho, Wo, 1, 3, pls_avail)
        p = cov2[0] if cov2 else (1, 1)
        rch = reach(p, 1, 3, Ho, Wo)
        taps = sorted(t for _, _, t in rch)
        posof = {t: (y, x) for y, x, t in rch}
        infpos = set((y, x) for y, x, _ in rch)
        rec["informative_entries_per_query"] = int(len(rch) * C_out)

        def bg_for(positions):
            mask = np.ones((C_out, Ho, Wo), dtype=bool)
            for (y, x) in positions:
                mask[:, y, x] = False
            return multiset_counts(bgmap[mask])

        bv, bc = bg_for(infpos)
        Wst = np.concatenate([Sc2_live.reshape(C_out, C_live, 9)[:, :, t]
                              for t in taps], axis=0)
        bst = np.concatenate([bgmap_t[:, posof[t][0], posof[t][1]]
                              for t in taps])
        ch = IsolatedChannel(Wst, bst, A, attack, rb, 0, rb,
                             {rb: [("zero",)]},
                             lambda xq, _p=p: _emb(xq, _p),
                             bv, bc, len(rch) * C_out)
        out = run_lta(Wst, bst, A, T, rng_a, None, channel=ch,
                      deadline=deadline, diagnostics=diagnostics)
        require_lta_run_certified(out, item["layer"])
        Wr, br = out.pop("_W_rec"), out.pop("_b_rec")
        out.pop("_x0")
        rec["lta"] = _strip(out)
        rec["isolation_failures"] = int(ch.isolation_failures)
        exp = {}
        for ci in range(C_out):
            for t in taps:
                y, x = posof[t]
                exp.setdefault(int(bgmap[ci, y, x]), []).append((ci, t))
        slopes = [Wr[i] for i in range(Wr.shape[0])]
        cands = [list(exp.get(int(br[i]), [])) for i in range(Wr.shape[0])]
        rec["rows_without_intercept_match"] = int(sum(1 for c in cands if not c))
        rec["rows_fixed_by_intercept"] = int(sum(1 for c in cands if len(c) == 1))
        cands0 = [list(c) for c in cands]
        nmq, attempt, ainfo = 0, 0, None
        while attempt < 4:
            attempt += 1
            us, ubad = [], 0
            for _u in range(3):
                uu, bad = best_membership_anchor(cands0, slopes, bgmap,
                                                 pls_avail, 1, 3, Ho, Wo,
                                                 C_live, A, rng_a,
                                                 tries=sep_tries)
                us.append(uu)
                ubad = max(ubad, bad)
                if bad == 0:
                    break

            def do_q(pm, taps_m, posof_m, _u):
                bv2, bc2 = bg_for(set(posof_m.values()))
                y = attack.run_session({rb: [_emb(_u, pm)]}, rb)
                kept, lo = subtract_multiset(y, bv2, bc2)
                return kept

            cands = [list(c) for c in cands0]
            cands, nq2 = refine_candidates(cands, slopes, bgmap,
                                           pls_avail, 1, 3, Ho, Wo, us,
                                           do_q)
            nmq += nq2
            W_live, b_unused, ainfo = assemble_from_candidates(
                cands, slopes, [int(v) for v in br], C_out, C_live, 3)
            if not ainfo["matching_failures"]:
                break
        rec["membership_anchor_collisions"] = ubad
        rec["membership_queries"] = nmq
        rec["membership_attempts"] = attempt
        rec["rows_still_ambiguous"] = int(sum(1 for c in cands if len(c) > 1))
        rec["assembly"] = ainfo
        W_can = np.zeros((C_out, C_z * 9), dtype=np.int64)
        W_can[:, cols] = W_live
        max_err, n_exact = row_exactness(W_can, b_block_can, Sc2_can,
                                         b_t[P2])
        fok = bool(np.array_equal(W_can, Sc2_can)
                   and np.array_equal(b_block_can, b_t[P2]))
        ferr = int(np.abs(W_can - Sc2_can).max()) if W_can.size else 0
        rec.update({"max_abs_error_up_to_row_perm": max_err,
                    "exact_up_to_row_perm": bool(max_err == 0),
                    "rows_recovered_exactly": int(np.sum(
                        np.all(W_can == Sc2_can, axis=1))),
                    "frame_check_exact": fok,
                    "frame_check_max_abs_error": ferr})
        recovered[item["layer"]] = {"W": W_can, "b": b_block_can}
        Srec = recovered[nm + ".shortcut"]["W"]
        live_map[g2] = np.array([c for c in range(C_out)
                                 if W_can[c].any() or Srec[c].any()],
                                dtype=np.int64)
        dead_ch = [c for c in range(C_out)
                   if not (W_can[c].any() or Srec[c].any())]
        rec["output_channels_dead"] = len(dead_ch)
        if dead_ch:
            eta = float(item.get("eta", 1.0))
            act = [int(np.clip(np.rint(max(int(b_block_can[c]), 0) / eta), 0, A))
                   for c in dead_ch]
            rec["dead_output_channels"] = dead_ch
            rec["dead_output_channel_constant_activation"] = act
            rec["dead_output_channels_feed_zero"] = bool(all(a == 0 for a in act))
        rec.update(install_conv_probe(rb, zspec, W_can, b_block_can, C_z, Ho,
                                      Wo, 1, 3, C_out, Ho, Wo, Scon,
                                      live_map[g2], pls_avail, live_idx))
        rec["sessions_used"] = int(oracle.n_sessions - s0)
        return rec

    def do_fc(item):
        rf, g = item["round"], item["group"]
        spec = rounds[rf].inputs[0]
        C, H, Win = spec["in_shape"]
        gfr = spec["frame"]
        m = rounds[rf].out_shape[0]
        s0 = oracle.n_sessions
        rec = {"layer": "fc", "round": rf, "group": g, "kind": "fc",
               "C_in": C, "H_in": H, "m": m}
        src = max(r for r in attack.probes if rounds[r].group == gfr)
        mask = attack.probes[src]["locatable"].reshape(C, H, Win)
        live = live_map.get(gfr)
        live_idx = (np.arange(C, dtype=np.int64) if live is None
                    else np.asarray(live, dtype=np.int64))
        rec["input_channels_live"] = int(live_idx.size)
        rec["input_channels_dead"] = int(C - live_idx.size)
        Q = [(y, x) for y in range(H) for x in range(Win)
             if bool(mask[live_idx, y, x].all())]
        hw = H * Win
        nq = len(Q)
        rec["pooling_positions_located"] = nq
        rec["pooling_positions_total"] = hw
        tmap, A_eff = {}, -1
        for uv in range(0, A + 1):
            t = int(round(uv * hw / nq)) if nq else 0
            if t > A:
                break
            if nq and int(np.rint(nq * t / hw)) == uv:
                tmap[uv] = t
                A_eff = uv
            else:
                break
        rec["pooled_alphabet_max"] = int(A_eff)
        Sc = canonical_column_matrix(spec["W"], Pchan.get(gfr), 1)
        b_t = np.asarray(rounds[rf].b, dtype=np.int64)
        if A_eff < 1:
            rec.update({"status": "pooled alphabet empty",
                        "exact_up_to_row_perm": False,
                        "sessions_used": int(oracle.n_sessions - s0)})
            recovered["fc"] = {"W": np.zeros((m, C), np.int64),
                               "b": np.zeros(m, np.int64)}
            return rec

        def fcq(u):
            pix = {q: np.array([tmap[int(v)] for v in u], dtype=np.int64)
                   for q in Q}
            return attack.run_session({rf: [("pixels", pix)]}, rf)

        y0 = fcq(np.zeros(C, dtype=np.int64))
        W_fc = np.zeros((m, C), dtype=np.int64)
        for c in live_idx:
            uu = np.zeros(C, dtype=np.int64)
            uu[int(c)] = 1
            W_fc[:, int(c)] = fcq(uu) - y0
        b_fc = np.asarray(y0, dtype=np.int64)
        rec["readout"] = ("unshuffled final round, labelled read, "
                          "live channels + 1 sessions")
        Sc = np.array(Sc, copy=True)
        Sc[:, np.setdiff1d(np.arange(C), live_idx)] = 0
        err = int(np.abs(W_fc - Sc).max()) if W_fc.size else 0
        errb = int(np.abs(b_fc - b_t).max())
        rec.update({"max_abs_error_up_to_row_perm": max(err, errb),
                    "exact_up_to_row_perm": bool(max(err, errb) == 0),
                    "rows_recovered_exactly": int(np.sum(
                        np.all(W_fc == Sc, axis=1) & (b_fc == b_t))),
                    "frame_check_exact": bool(max(err, errb) == 0),
                    "frame_check_max_abs_error": max(err, errb),
                    "sessions_used": int(oracle.n_sessions - s0)})
        recovered["fc"] = {"W": W_fc, "b": b_fc}
        return rec

    handlers = {"conv": do_conv, "shortcut": do_shortcut, "conv2": do_conv2,
                "fc": do_fc}
    for item in meta:
        if deadline is not None and time.time() > deadline:
            stopped = "time box reached before %s" % item["layer"]
            break
        gin = None
        for sp in rounds[item["round"]].inputs:
            if sp["kind"] == "new":
                gin = sp["frame"]
                break
        lv = live_map.get(gin, None)
        if lv is not None and len(lv) == 0:
            stopped = ("no live input channel remains at %s: the preceding "
                       "frame collapsed" % item["layer"])
            break
        rec = handlers[item["kind"]](item)
        # Detect and retry, as in step 4: the probe is installed with a fresh
        # anchor, so its verification is a genuine prediction about the recovered
        # matrix and a failure is the client's own signal to redo the layer.
        tries = 0
        while (repair_attempts and rec.get("probe_verified") is False
               and tries < repair_attempts
               and (deadline is None or time.time() < deadline)):
            tries += 1
            repairs += 1
            log("  %-26s probe verification failed, retrying (attempt %d)"
                % (rec["layer"], tries))
            rec = handlers[item["kind"]](item)
        rec["repair_attempts_used"] = tries
        records.append(rec)
        log("  %-26s %-9s sess=%-6d exact=%-5s frame=%-5s loc=%s/%s probe=%s"
            % (rec["layer"], rec["kind"], rec["sessions_used"],
               rec.get("exact_up_to_row_perm"), rec.get("frame_check_exact"),
               rec.get("probe_located_needed", "-"),
               rec.get("probe_needed_coordinates", "-"),
               rec.get("probe_verified", "-")))

    ext = None
    if not stopped:
        try:
            blocks = []
            for blk in net.blocks:
                nm = blk["name"]
                blocks.append({
                    "name": nm, "W1": recovered[nm + ".conv1"]["W"],
                    "b1": recovered[nm + ".conv1"]["b"], "eta1": blk["eta1"],
                    "W2": recovered[nm + ".conv2"]["W"],
                    "b2": recovered[nm + ".conv2"]["b"],
                    "S": recovered[nm + ".shortcut"]["W"],
                    "bs": np.zeros_like(recovered[nm + ".shortcut"]["b"]),
                    "eta": blk["eta"], "shortcut": blk["shortcut"],
                    "C_in": blk["C_in"], "C_out": blk["C_out"]})
            ext = IntResNet20({"W": recovered["conv1"]["W"],
                               "b": recovered["conv1"]["b"],
                               "eta": net.stem["eta"]}, blocks,
                              {"W": recovered["fc"]["W"],
                               "b": recovered["fc"]["b"]}, A)
        except Exception as exc:
            stopped = "assembly failed: %s: %s" % (type(exc).__name__, exc)
    summary = {
        "mode": "R3", "interface": "feature map",
        "permutation": "all output coordinates, fresh per group per session",
        "T": int(T), "noise_law": noise_law, "rounds": len(rounds),
        "layers_attacked": len(records), "layers_total": len(meta),
        "layers_exact": int(sum(1 for r in records
                                if r.get("exact_up_to_row_perm"))),
        "sessions_total": int(oracle.n_sessions),
        "round_evaluations_total": int(oracle.n_round_evaluations),
        "replies_read": int(oracle.n_replies_read),
        "memoised_round_evaluations": int(oracle.cache_hits),
        "probe_rounds": int(attack.probe_rounds),
        "probe_multiset_failures": int(attack.probe_multiset_failures),
        "frame_unavailable_events": int(attack.frame_unavailable),
        "missing_coordinate_events": int(attack.missing_coordinate_events),
        "missing_coordinates": int(attack.missing_coordinates),
        "oracle_rounding_exact": bool(oracle.rounding_exact),
        "oracle_rounding_failures": int(oracle.n_rounding_failures),
        "inadmissible_query_entries": int(oracle.inadmissible_entries),
        "isolation_failures": int(sum(r.get("isolation_failures", 0)
                                      for r in records)),
        "repair_attempts_allowed": int(repair_attempts),
        "repairs_triggered": int(repairs),
        "stopped_reason": stopped,
        "wall_time_sec": round(time.time() - t0, 2),
    }
    return records, ext, summary

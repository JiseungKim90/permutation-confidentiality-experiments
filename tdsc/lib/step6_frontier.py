"""Step 6: the precision frontier of whole-network extraction.

New module.  Nothing here changes the behaviour of any existing lib entry point.
It reuses lib/fmap.py (``conv_int``, ``out_size``, ``block_stride``), and
lib/fmap_partial.py (the locate-then-control implementation of step 5:
``placements``, ``reach``, ``required_placements``, ``pullback_pixels``,
``sparse_map``, ``build_r3_rounds``, ``PartialOracle``,
``extract_chain_partial``).

Why this module exists.  ``REPORT_step5.md`` Section 6 extracts the complete
ResNet-20 under a fresh uniform permutation of all output coordinates at 8 bits
and fails at 4, and its Section 10 localises the failure to one quantity: the
attained span ``R`` of a sparse probe against the number ``n`` of informative
values the locate step must separate inside it.  ``R`` is set by the weight and
activation precision, not chosen independently, so the question "at what
precision does whole-network extraction become possible" is the question this
module is built to measure.

Four things are added.

1. ``forward_fmap_instrumented`` -- the honest feature-map forward pass of
   lib/fmap.py with two additions: it reports the largest accumulator magnitude
   it saw over the images it was given, and it can CLIP every accumulator into
   ``[-2^(B-1), 2^(B-1) - 1]`` before the requantisation, which is the
   server-side accumulator cap of E6-B.  With ``cap_bits=None`` it is bit
   identical to ``lib.fmap.forward_fmap`` (unit tested).

2. ``capped_oracle(B)`` -- a context manager that makes
   ``fmap_partial.extract_chain_partial`` build an oracle which clips every
   round's output into the same interval before permuting it.  The existing
   module is not edited: a subclass of ``fmap_partial.PartialOracle`` is
   substituted for the name for the duration of the call and restored
   afterwards, and the subclass records how often the cap bound.

3. ``locate_diagnostic`` -- the locate step of ``REPORT_step5.md`` Section 3.1
   measured in isolation on one convolution, with the probe-value search of
   ``install_conv_probe`` reproduced (random draws plus one-pixel hill climbing,
   no oracle query).  It reports the attained span ``R``, the informative count
   ``n``, the collision count, the birthday quantity ``n^2 / (2R)`` and how many
   of the coordinates the chain needs are located.  It takes a probe variant, so
   that E6-D's single-input-channel probe and E6-D's one-placement-at-a-time
   probe are measured by the same code as the baseline.

4. ``conv_inventory`` -- the 21 convolutions of ResNet-20 with the geometry the
   chain uses, the live input and output channels, and the placements their
   output group must make locatable (``group_needs``, which reproduces the
   ``group_need`` table ``extract_chain_partial`` builds).

Recorded limitation of ``locate_diagnostic`` on a ``conv2`` round: the chain's
real background there is ``b2 + bs + S(x_probe)``, which varies over space,
while the diagnostic uses the plain convolution background ``b2 + bs`` unless an
``extra`` map is supplied.  The chain's own per-layer records
(``probe_needed_coordinates``, ``probe_located_needed``) are the measurement of
the real thing; the diagnostic exists to supply the span ``R`` and the birthday
quantity, which the chain does not record.
"""
import contextlib

import numpy as np

from .fmap import conv_int, out_size, block_stride
from . import fmap_partial as fp
from .checkpoint import load_verified_checkpoint


# ------------------------------------------------------- honest forward pass
def forward_fmap_instrumented(net, X, cap_bits=None, batch=200):
    """``lib.fmap.forward_fmap`` with an accumulator cap and instrumentation.

    Returns ``(logits, accumulator_absmax, clipped_entries)``.  The cap is
    applied to every quantity the server holds in its accumulator: the stem
    output, each block's first convolution, each block's sum of the second
    convolution and the shortcut, and the final linear output.  With
    ``cap_bits=None`` nothing is clipped and the logits are bit identical to
    ``forward_fmap``.
    """
    A = net.A
    lo = hi = None
    if cap_bits is not None:
        lo = -(1 << (int(cap_bits) - 1))
        hi = (1 << (int(cap_bits) - 1)) - 1
    state = {"absmax": 0, "nclip": 0}

    def track(u):
        state["absmax"] = max(state["absmax"], int(np.abs(u).max()))
        if cap_bits is None:
            return u
        n = int(np.sum((u < lo) | (u > hi)))
        if n:
            state["nclip"] += n
            return np.clip(u, lo, hi)
        return u

    Xi = np.clip(np.rint(np.asarray(X, dtype=np.float64) * A), 0, A).astype(np.int64)
    outs = []
    for s in range(0, len(Xi), batch):
        chunk = Xi[s:s + batch]
        acts = []
        for x in chunk:
            u = track(conv_int(x, net.stem["W"], net.stem["b"], 1, 3))
            a = np.clip(np.rint(np.maximum(u, 0) / net.stem["eta"]), 0,
                        A).astype(np.int64)
            for blk in net.blocks:
                st = blk.get("stride", block_stride(blk["name"]))
                u1 = track(conv_int(a, blk["W1"], blk["b1"], st, 3))
                a1 = np.clip(np.rint(np.maximum(u1, 0) / blk["eta1"]), 0,
                             A).astype(np.int64)
                v = track(conv_int(a1, blk["W2"], blk["b2"], 1, 3)
                          + conv_int(a, blk["S"], blk["bs"], st, 1))
                a = np.clip(np.rint(np.maximum(v, 0) / blk["eta"]), 0,
                            A).astype(np.int64)
            acts.append(a.mean(axis=(1, 2)))
        pooled = np.rint(np.stack(acts))
        outs.append(track(pooled @ net.fc["W"].T + net.fc["b"]))
    return (np.concatenate(outs).astype(np.int64), int(state["absmax"]),
            int(state["nclip"]))


# --------------------------------------------------------- the capped oracle
@contextlib.contextmanager
def capped_oracle(cap_bits):
    """Make ``extract_chain_partial`` use an accumulator-capped server.

    Substitutes a subclass of ``fmap_partial.PartialOracle`` for that name, so
    that every round's output vector is clipped into
    ``[-2^(B-1), 2^(B-1) - 1]`` before the session permutation is applied.  The
    substitution is undone on exit; the existing module is not edited.  The
    yielded dict collects the oracles that were built, so the caller can read
    ``clipped_rounds`` and ``clipped_entries`` afterwards -- which is what tells
    the reader whether the cap bound on the attack at all.
    """
    stats = {"cap_bits": (None if cap_bits is None else int(cap_bits)),
             "oracles": []}
    if cap_bits is None:
        yield stats
        return
    original = fp.PartialOracle
    lo = -(1 << (int(cap_bits) - 1))
    hi = (1 << (int(cap_bits) - 1)) - 1

    class _CappedOracle(original):
        def __init__(self, *a, **kw):
            super().__init__(*a, **kw)
            self.cap_bits = int(cap_bits)
            self.cap_lo = lo
            self.cap_hi = hi
            self.clipped_rounds = 0
            self.clipped_entries = 0
            stats["oracles"].append(self)

        def eval_round(self, r, true_maps):
            v = original.eval_round(self, r, true_maps)
            n = int(np.sum((v < lo) | (v > hi)))
            if n:
                self.clipped_rounds += 1
                self.clipped_entries += n
                return np.clip(v, lo, hi)
            return v

    fp.PartialOracle = _CappedOracle
    try:
        yield stats
    finally:
        fp.PartialOracle = original


def cap_stats(stats):
    return {"cap_bits": stats["cap_bits"],
            "clipped_rounds": int(sum(getattr(o, "clipped_rounds", 0)
                                      for o in stats["oracles"])),
            "clipped_entries": int(sum(getattr(o, "clipped_entries", 0)
                                       for o in stats["oracles"])),
            "oracles_built": len(stats["oracles"])}


# -------------------------------------------------------------- the network
def group_needs(rounds):
    """The output placements each group must make locatable.

    Reproduces the ``group_need`` table ``extract_chain_partial`` builds: the
    union, over the consumers of a group, of ``required_placements`` for that
    consumer's geometry.
    """
    need = {}
    for rr in rounds:
        for sp in rr.inputs:
            if sp["kind"] != "new" or sp["frame"] < 0:
                continue
            _C, Hi, Wi = sp["in_shape"]
            cur = need.setdefault(sp["frame"], [])
            for p in fp.required_placements(Hi, Wi, sp["stride"], sp["k"],
                                            bool(sp.get("pool"))):
                if p not in cur:
                    cur.append(p)
    return need


def live_channels(net, H0=32):
    """group id -> the live output channels of that group.

    A channel whose matrix row is identically zero is constant in every reply
    and cannot be located; the chain certifies it from its own recovered matrix
    and restricts every query to the live channels (``REPORT_step5.md``
    Section 3.4).  For the stem and a block's first convolution the row is the
    convolution's row; for a block's output the channel is live when either the
    second convolution's row or the shortcut's row is non-zero.
    """
    rounds, meta = fp.build_r3_rounds(net, H0)
    out = {-1: None}
    g = rounds[0].group
    Ws = net.stem["W"]
    out[g] = np.array([c for c in range(Ws.shape[0]) if Ws[c].any()],
                      dtype=np.int64)
    i = 1
    for blk in net.blocks:
        g1 = rounds[i].group
        W1 = blk["W1"]
        out[g1] = np.array([c for c in range(W1.shape[0]) if W1[c].any()],
                           dtype=np.int64)
        g2 = rounds[i + 1].group
        W2, S = blk["W2"], blk["S"]
        out[g2] = np.array([c for c in range(W2.shape[0])
                            if W2[c].any() or S[c].any()], dtype=np.int64)
        i += 2
    return out, rounds, meta


def conv_inventory(net, H0=32):
    """The 21 convolutions of ResNet-20 as the chain sees them.

    One entry per convolution: the matrix and bias in TRUE row order, the input
    geometry, the stride and kernel, the live input and output channels, the
    placements the output group must make locatable (``need``) and the placements
    the input group makes available (``avail_in``), which is what
    ``install_conv_probe`` receives.  The two projection shortcuts are included,
    as in ``REPORT_step5.md`` Section 4, so the table has 21 rows; their output
    group is the one they share with the block's second convolution.
    """
    live, rounds, _meta = live_channels(net, H0)
    need = group_needs(rounds)
    items = []

    def entry(name, kind, W, b, spec, out_group, in_group):
        C_in, Hi, Wi = spec["in_shape"]
        st, k = spec["stride"], spec["k"]
        Ho, Wo = out_size(Hi, st), out_size(Wi, st)
        av = (fp.required_placements(Hi, Wi, st, k) if in_group < 0
              else need.get(in_group, fp.placements(Hi, Wi)))
        return {"layer": name, "kind": kind, "W": W, "b": b,
                "C_out": int(W.shape[0]), "C_in": int(C_in),
                "H_in": int(Hi), "W_in": int(Wi), "H_out": int(Ho),
                "W_out": int(Wo), "stride": int(st), "kernel": int(k),
                "coordinates_N": int(W.shape[0] * Ho * Wo),
                "out_group": int(out_group), "in_group": int(in_group),
                "need": [tuple(p) for p in need.get(out_group,
                                                    fp.placements(Ho, Wo))],
                "avail_in": [tuple(p) for p in av],
                "live_out": live.get(out_group),
                "live_in": live.get(in_group)}

    r0 = rounds[0]
    items.append(entry("conv1", "conv", net.stem["W"], r0.b, r0.inputs[0],
                       r0.group, -1))
    i = 1
    for blk in net.blocks:
        ra, rb = rounds[i], rounds[i + 1]
        g_in = ra.inputs[0]["frame"]
        items.append(entry(blk["name"] + ".conv1", "conv", blk["W1"], ra.b,
                           ra.inputs[0], ra.group, g_in))
        if blk["shortcut"] == "conv":
            items.append(entry(blk["name"] + ".shortcut", "shortcut",
                               blk["S"], rb.b, rb.inputs[1], rb.group, g_in))
        items.append(entry(blk["name"] + ".conv2", "conv2", blk["W2"], rb.b,
                           rb.inputs[0], rb.group, ra.group))
        i += 2
    return items


def min_reach_pixels(need, H, Win, stride, k, Ho, Wo, pls):
    """A cover of `need` by pixels of `pls` chosen to minimise the number of
    output positions the probe makes informative.

    `fmap_partial.pullback_pixels` takes the first pixel of `pls` that covers
    each needed position, which for the placement lattice is the interior pixel
    (1, 1) and therefore all nine output positions of a stride-1 3x3 kernel.  A
    corner pixel covers the same needed position through four output positions
    only, so it leaves `n` smaller, which is the quantity the locate step's
    birthday condition is in.  This is the adversarially better choice and E6-D
    uses it.
    """
    cov = {p: set((y, x) for (y, x, _) in fp.reach(p, stride, k, Ho, Wo))
           for p in pls}
    out, done = [], set()
    want = [tuple(q) for q in need]
    while True:
        rest = [q for q in want if q not in done]
        if not rest:
            break
        best, key = None, None
        for p in pls:
            gain = len(cov[p] & set(rest))
            if gain <= 0:
                continue
            kk = (-gain, len(cov[p]))
            if key is None or kk < key:
                best, key = p, kk
        if best is None:
            break
        out.append(best)
        done |= cov[best] & set(want)
    return out


# ------------------------------------------------------------ locate step
def locate_diagnostic(item, A, rng, need=None, variant="full_channel",
                      cap_bits=None, probe_tries=80, hill_mult=20,
                      extra=None, pixel_policy="chain"):
    """Measure the locate step on one convolution.

    ``item`` is one entry of ``conv_inventory``.  ``need`` overrides the output
    placements that must be located (E6-D's one-placement-at-a-time variant
    passes a single placement).  ``variant`` is ``"full_channel"`` -- the chain's
    probe, every live input channel non-zero at each probe pixel -- or
    ``"single_channel"`` -- exactly one live input channel non-zero, which is the
    escape route ``REPORT_step5.md`` Section 10 names.  ``cap_bits`` clips the
    predicted reply, which is the clip-aware client of E6-B.

    The probe-value search is the one ``install_conv_probe`` performs: up to
    ``probe_tries`` random draws, then hill climbing on one pixel at a time,
    scored by how many of the needed coordinates are uniquely valued in the whole
    predicted reply.  No oracle query is involved.
    """
    W = np.asarray(item["W"], dtype=np.int64)
    b = np.asarray(item["b"], dtype=np.int64)
    C_out, C_in = int(item["C_out"]), int(item["C_in"])
    Hi, Wi, st, k = item["H_in"], item["W_in"], item["stride"], item["kernel"]
    Ho, Wo = item["H_out"], item["W_out"]
    need = [tuple(p) for p in (item["need"] if need is None else need)]
    chans = (list(range(C_out)) if item["live_out"] is None
             else [int(c) for c in item["live_out"]])
    li = (np.arange(C_in, dtype=np.int64) if item["live_in"] is None
          else np.asarray(item["live_in"], dtype=np.int64))
    need_idx = np.array([c * (Ho * Wo) + y * Wo + x
                         for c in chans for (y, x) in need], dtype=np.int64)
    if pixel_policy == "min_reach":
        ppls = min_reach_pixels(need, Hi, Wi, st, k, Ho, Wo, item["avail_in"])
    else:
        ppls = fp.pullback_pixels(need, Hi, Wi, st, k, Ho, Wo, item["avail_in"])
    if not ppls:
        ppls = fp.placements(Hi, Wi)
    lo = hi = None
    if cap_bits is not None:
        lo = -(1 << (int(cap_bits) - 1))
        hi = (1 << (int(cap_bits) - 1)) - 1
    bg = np.repeat(b, Ho * Wo)
    if extra is not None:
        bg = bg + np.asarray(extra, dtype=np.int64).reshape(-1)

    def _vec(sel):
        v = np.zeros(C_in, dtype=np.int64)
        v[sel] = rng.integers(0, A + 1, size=sel.size).astype(np.int64)
        return v

    def _sel():
        if variant == "single_channel":
            return np.array([int(li[int(rng.integers(0, li.size))])],
                            dtype=np.int64)
        return li

    def _score(vals):
        cmap = fp.sparse_map(C_in, Hi, Wi, dict(zip(ppls, vals)))
        raw = conv_int(cmap, W, b, st, k).reshape(-1)
        if extra is not None:
            raw = raw + np.asarray(extra, dtype=np.int64).reshape(-1)
        seen = raw if cap_bits is None else np.clip(raw, lo, hi)
        nclip = (0 if cap_bits is None
                 else int(np.sum((raw < lo) | (raw > hi))))
        u, inv, c = np.unique(seen, return_inverse=True, return_counts=True)
        mask = c[inv] == 1
        return int(mask[need_idx].sum()), raw, seen, mask, nclip, cmap

    best = None
    for _ in range(probe_tries):
        sel = _sel()
        vals = [_vec(sel) for _ in ppls]
        sc, raw, seen, mask, nclip, cmap = _score(vals)
        if best is None or sc > best[0]:
            best = (sc, [v.copy() for v in vals], raw, seen, mask, nclip, cmap,
                    sel)
        if sc == need_idx.size:
            break
    steps = 0
    while best[0] < need_idx.size and steps < hill_mult * len(ppls):
        steps += 1
        sel = best[7]
        if variant == "single_channel" and rng.random() < 0.34:
            sel = _sel()
            vals = [_vec(sel) for _ in ppls]
        else:
            vals = [v.copy() for v in best[1]]
            j = int(rng.integers(0, len(ppls)))
            vals[j] = _vec(sel)
        sc, raw, seen, mask, nclip, cmap = _score(vals)
        if sc > best[0]:
            best = (sc, vals, raw, seen, mask, nclip, cmap, sel)
    score, vals, raw, seen, mask, nclip, cmap, sel = best

    inf = raw != bg
    n_inf = int(inf.sum())
    vals_inf = seen[inf]
    span = int(vals_inf.max() - vals_inf.min() + 1) if n_inf else 1
    _u2, c2 = np.unique(vals_inf, return_counts=True)
    pairs = int(np.sum(c2 * (c2 - 1) // 2))
    positions = sorted(set((int(j) % (Ho * Wo)) for j in np.nonzero(inf)[0]))
    return {
        "layer": item["layer"], "kind": item["kind"],
        "C_out": C_out, "C_in": C_in, "H_in": Hi, "H_out": Ho,
        "stride": st, "kernel": k, "coordinates_N": item["coordinates_N"],
        "variant": variant, "pixel_policy": pixel_policy,
        "cap_bits": (None if cap_bits is None else int(cap_bits)),
        "input_channels_live": int(li.size),
        "output_channels_live": len(chans),
        "probe_channels_used": int(sel.size),
        "probe_pixels": len(ppls),
        "probe_pixel_positions": [list(p) for p in ppls],
        "needed_placements": [list(p) for p in need],
        "needed_coordinates": int(need_idx.size),
        "located_needed": int(score),
        "all_needed_located": bool(score == need_idx.size),
        "located_needed_fraction": round(float(score) / max(1, need_idx.size), 6),
        "informative_entries_n": n_inf,
        "informative_positions": len(positions),
        "distinct_informative_values": int(_u2.size),
        "collisions_n_minus_distinct": int(n_inf - _u2.size),
        "colliding_pairs_measured": pairs,
        "informative_value_span_R": span,
        "birthday_pairs_predicted": round(n_inf * n_inf / (2.0 * span), 4),
        "locatable_coordinates": int(mask.sum()),
        "locatable_fraction_of_N": round(float(mask.sum())
                                         / max(1, mask.size), 6),
        "predicted_entries_clipped": int(nclip),
        "hill_climb_steps": int(steps),
        "distinct_bias_values": int(np.unique(b).size),
        "bias_collisions": int(C_out - np.unique(b).size),
    }


def aggregate_locate(rows):
    """Aggregate a list of ``locate_diagnostic`` rows over the layers."""
    if not rows:
        return {}
    n = sum(r["informative_entries_n"] for r in rows)
    need = sum(r["needed_coordinates"] for r in rows)
    got = sum(r["located_needed"] for r in rows)
    bd = [r["birthday_pairs_predicted"] for r in rows]
    return {
        "layers": len(rows),
        "layers_all_needed_located": int(sum(1 for r in rows
                                             if r["all_needed_located"])),
        "needed_coordinates_total": int(need),
        "located_needed_total": int(got),
        "located_needed_fraction": round(got / max(1, need), 6),
        "informative_entries_total": int(n),
        "colliding_pairs_total": int(sum(r["colliding_pairs_measured"]
                                         for r in rows)),
        "span_R_min": int(min(r["informative_value_span_R"] for r in rows)),
        "span_R_max": int(max(r["informative_value_span_R"] for r in rows)),
        "birthday_median": float(np.median(bd)),
        "birthday_min": float(min(bd)),
        "birthday_max": float(max(bd)),
        "predicted_entries_clipped_total": int(
            sum(r["predicted_entries_clipped"] for r in rows)),
    }


# ------------------------------------------------------------- bookkeeping
def single_patch_formula(net, T, H0=32):
    """``sum_r (1 + T d_r)`` for the single-patch interface of step 4.

    Copied from ``scripts/step5_c.py`` so that the reference number in this
    step's tables is produced by the same expression.
    """
    tot = 1 + T * net.stem["W"].shape[1]
    H = H0
    for blk in net.blocks:
        st = block_stride(blk["name"])
        tot += 1 + T * blk["W1"].shape[1]
        tot += 1 + T * blk["W2"].shape[1]
        tot += 1 + T * blk["S"].shape[1]
        H = out_size(H, st)
    tot += 1 + T * net.fc["W"].shape[1]
    return int(tot)


def implicated_mechanism(rec):
    """Which of ``REPORT_step5.md`` Section 3.4's three mechanisms a non-exact
    layer record implicates.  Reports every signal present, not a guess at a
    single cause."""
    flags = []
    if rec.get("input_channels_dead", 0) or rec.get("output_channels_dead", 0):
        flags.append("dead channels")
    asm = rec.get("assembly") or {}
    if (rec.get("rows_still_ambiguous", 0)
            or asm.get("matching_failures", 0)
            or rec.get("membership_anchor_collisions", 0)
            or asm.get("rows_with_no_candidate", 0)):
        flags.append("tap labelling")
    if rec.get("bias_groups_oversized", 0) or rec.get("linkage_failures", 0):
        flags.append("channel linkage")
    if rec.get("isolation_failures", 0):
        flags.append("isolation by multiset subtraction")
    if rec.get("probe_verified") is False:
        flags.append("probe verification failed")
    if rec.get("all_needed_located") is False or (
            rec.get("probe_needed_all_located") is False):
        flags.append("locate step (needed coordinate unlocated)")
    return flags or ["none of the three; LTA or the frame algebra"]


def first_failure(records):
    """The first layer record that is not exact, with the signals it carries."""
    for rec in records:
        if not rec.get("exact_up_to_row_perm"):
            return {"layer": rec.get("layer"), "kind": rec.get("kind"),
                    "max_abs_error_up_to_row_perm":
                        rec.get("max_abs_error_up_to_row_perm"),
                    "frame_check_exact": rec.get("frame_check_exact"),
                    "probe_verified": rec.get("probe_verified"),
                    "probe_located_needed": rec.get("probe_located_needed"),
                    "probe_needed_coordinates":
                        rec.get("probe_needed_coordinates"),
                    "isolation_failures": rec.get("isolation_failures"),
                    "rows_still_ambiguous": rec.get("rows_still_ambiguous"),
                    "bias_groups_oversized": rec.get("bias_groups_oversized"),
                    "linkage_failures": rec.get("linkage_failures"),
                    "input_channels_dead": rec.get("input_channels_dead"),
                    "output_channels_dead": rec.get("output_channels_dead"),
                    "mechanisms_implicated": implicated_mechanism(rec)}
    return None


def chain_probe_summary(records):
    """The locate-step outcome the chain itself measured, aggregated."""
    rows = [r for r in records if r.get("probe_needed_coordinates") is not None]
    if not rows:
        return {}
    need = sum(r["probe_needed_coordinates"] for r in rows)
    got = sum(r["probe_located_needed"] for r in rows)
    return {"probe_installations": len(rows),
            "probe_needed_coordinates_total": int(need),
            "probe_located_needed_total": int(got),
            "probe_located_needed_fraction": round(got / max(1, need), 6),
            "probes_with_all_needed_located": int(
                sum(1 for r in rows if r.get("probe_needed_all_located"))),
            "probes_verified": int(sum(1 for r in rows
                                       if r.get("probe_verified"))),
            "informative_coordinates_total": int(
                sum(r.get("probe_informative_coordinates", 0) for r in rows))}


# ------------------------------------------------- QAT weights (E6-A, QAT arm)
# `REPORT_qat.md` Section 5 measures that post-hoc per-layer quantisation with
# step max|W|/7 leaves a median zero fraction of 0.35 on CIFAR ResNet-20 and
# occupies 13 of 15 lattice levels, against 0.18 and 15 of 15 for a
# QAT-trained model, and Section 7 measures that the width the attack attains
# rises under QAT because sum_j |W_int[k,j]| grows.  The locate step of step 5
# depends on the informative values being distinct inside the attained span, so
# a wider span makes the locate step MORE likely to succeed at low precision and
# the precision frontier measured on post-hoc weights may sit too high.  The two
# helpers below let E6-A be re-run on the QAT weight lattice.
#
# Nothing under `scripts/qat_*` or `lib_qat/` is imported, modified or executed:
# the checkpoint is read directly and the layer naming is remapped onto
# `lib.models._ResNetCIFAR`, which `lib.resnet20.resnet20_float_params` already
# knows how to BatchNorm-fold.  The QAT layers carry an all-zero `conv_bias`
# buffer (checked), so that fold agrees with `QLayer.folded()` of
# `scripts/qat_common.py`.

def load_qat_resnet20(path, expected_sha256):
    """Rebuild `lib.models._ResNetCIFAR(n=3)` from a QAT ResNet-20 checkpoint.

    Returns `(model, info)` where `info["weight_steps"]` are the LEARNED
    per-layer weight step sizes, keyed exactly as
    `lib.fmap.quantise_resnet20_fmap` keys its own `weight_scales`, and
    `info["mean"]`, `info["std"]` are the input normalisation the QAT run used
    (which is not the same as `lib.cifar10.MEAN/STD`).
    """
    from .models import _ResNetCIFAR
    ck = load_verified_checkpoint(path, expected_sha256)
    sd = ck["model"] if isinstance(ck, dict) and "model" in ck else ck
    out, steps = {}, {}
    out["conv1.weight"] = sd["conv1.weight"]
    out["bn1.weight"] = sd["conv1.gamma"]
    out["bn1.bias"] = sd["conv1.beta"]
    out["bn1.running_mean"] = sd["conv1.run_mean"]
    out["bn1.running_var"] = sd["conv1.run_var"]
    steps["conv1"] = abs(float(sd["conv1.wq.step"]))
    conv_bias_absmax = 0.0
    for k, v in sd.items():
        if k.endswith("conv_bias"):
            conv_bias_absmax = max(conv_bias_absmax, float(v.abs().max()))
    for st in (1, 2, 3):
        for i in range(3):
            pre = "layer%d.%d" % (st, i)
            for src, bn in (("conv1", "bn1"), ("conv2", "bn2")):
                out["%s.%s.weight" % (pre, src)] = sd["%s.%s.weight" % (pre, src)]
                out["%s.%s.weight" % (pre, bn)] = sd["%s.%s.gamma" % (pre, src)]
                out["%s.%s.bias" % (pre, bn)] = sd["%s.%s.beta" % (pre, src)]
                out["%s.%s.running_mean" % (pre, bn)] = sd["%s.%s.run_mean" % (pre, src)]
                out["%s.%s.running_var" % (pre, bn)] = sd["%s.%s.run_var" % (pre, src)]
                steps["%s.%s" % (pre, src)] = abs(float(
                    sd["%s.%s.wq.step" % (pre, src)]))
            sk = "%s.short" % pre
            if sk + ".weight" in sd:
                out["%s.shortcut.0.weight" % pre] = sd[sk + ".weight"]
                out["%s.shortcut.1.weight" % pre] = sd[sk + ".gamma"]
                out["%s.shortcut.1.bias" % pre] = sd[sk + ".beta"]
                out["%s.shortcut.1.running_mean" % pre] = sd[sk + ".run_mean"]
                out["%s.shortcut.1.running_var" % pre] = sd[sk + ".run_var"]
                steps["%s.shortcut" % pre] = abs(float(sd[sk + ".wq.step"]))
    out["fc.weight"] = sd["fc.weight"]
    out["fc.bias"] = sd["fc.beta"]
    steps["fc"] = abs(float(sd["fc.wq.step"]))
    # The learned ACTIVATION step of each consumer, keyed by the PRODUCER whose
    # requantisation scalar has to realise it.  `quantise_resnet20_fmap` sets
    # eta so that the produced activation scale is `s_w s_a_in eta`; asking that
    # this equal the consumer's learned input step transplants the QAT
    # activation quantiser as well as its weight lattice.
    order = ["layer%d.%d" % (st, i) for st in (1, 2, 3) for i in range(3)]
    acts = {}

    def _astep(key):
        k = key + ".aq.step"
        return abs(float(sd[k])) if k in sd else None

    acts["conv1"] = _astep(order[0] + ".conv1")
    for j, pre in enumerate(order):
        acts[pre + ".conv1"] = _astep(pre + ".conv2")
        nxt = (order[j + 1] + ".conv1") if j + 1 < len(order) else "fc"
        acts[pre + ".block"] = _astep(nxt)
    model = _ResNetCIFAR(n=3)
    missing, unexpected = model.load_state_dict(out, strict=False)
    model.eval()
    info = {"weight_steps": steps, "activation_steps": acts,
            "mean": np.asarray(sd["mean"], dtype=np.float64).reshape(-1),
            "std": np.asarray(sd["std"], dtype=np.float64).reshape(-1),
            "conv_bias_absmax": conv_bias_absmax,
            "missing_keys": list(missing), "unexpected_keys": list(unexpected),
            "training_args": (ck.get("args", {}) if isinstance(ck, dict) else {}),
            "trained_w_bits": int((ck.get("args", {}) or {}).get("w_bits", 4)
                                  if isinstance(ck, dict) else 4)}
    return model, info


def quantise_resnet20_fmap_scales(model, w_bits, a_bits, calib_images,
                                  normalise=None, scales=None,
                                  act_scales=None):
    """`lib.fmap.quantise_resnet20_fmap` with the per-layer weight step
    overridable.

    Identical to that function in every other respect -- the BatchNorm fold, the
    input-normalisation fold, the bias convention, the requantisation scalars
    calibrated on real feature maps and the accumulator bookkeeping -- so that a
    QAT arm and a post-hoc arm differ in exactly one variable, the weight
    lattice.  A supplied step is applied as
    `W_int = clip(rint(W / s), -qmax, qmax)`; the clip matters here because a
    learned step, unlike `max|W| / qmax`, does not bound the quotient.
    """
    from .resnet20 import (resnet20_float_params, fold_input_normalisation,
                           IntResNet20)
    scales = dict(scales or {})
    act_scales = dict(act_scales or {})
    A = 2 ** a_bits - 1
    qmax = 2 ** (w_bits - 1) - 1
    fpar = resnet20_float_params(model)
    stem_fold_gain = 1.0
    if normalise is not None:
        fpar = fold_input_normalisation(fpar, normalise[0], normalise[1])
        # Folding the input normalisation multiplies the stem's weights by
        # 1/std per input channel.  A weight step learned by QAT was learned on
        # the UNFOLDED weight, so it must be scaled by the same gain or the
        # folded stem clips against the lattice and the stem is destroyed.  The
        # three per-channel gains agree to about 1.5 %, so the scalar mean is
        # used and the residual is reported as `clipped_weights["conv1"]`.
        stem_fold_gain = float(np.mean(1.0 / np.asarray(normalise[1],
                                                        dtype=np.float64)))
        if "conv1" in scales:
            scales["conv1"] = scales["conv1"] * stem_fold_gain
    s_in = 1.0 / A
    X = np.clip(np.rint(np.asarray(calib_images, dtype=np.float64) * A), 0,
                A).astype(np.int64)
    info = {"w_bits": w_bits, "a_bits": a_bits, "A": A, "weight_scales": {},
            "weight_scale_source": {}, "requant_source": {}, "W_int_absmax": {},
            "shortcut_int_absmax": {}, "requant_scales": {},
            "accumulator_absmax_calibration": 0,
            "calibration_images": int(len(X)), "clipped_weights": {},
            "stem_input_fold_gain": 1.0}
    acc = 0

    def q(W, key):
        s = scales.get(key)
        src = "supplied"
        if s is None or not np.isfinite(s) or s <= 0:
            mx = float(np.abs(W).max())
            s = mx / qmax if mx > 0 else 1.0
            src = "max_over_qmax"
        raw = np.rint(W / s)
        nclip = int(np.sum(np.abs(raw) > qmax))
        info["weight_scale_source"][key] = src
        info["clipped_weights"][key] = nclip
        return np.clip(raw, -qmax, qmax).astype(np.int64), s

    def eta_for(key, default, denom):
        """The requantisation scalar.  `default` is the max-calibrated value of
        `lib.fmap.quantise_resnet20_fmap`; a supplied activation step overrides
        it so that the produced activation scale `denom * eta` equals the
        consumer's learned input step."""
        tgt = act_scales.get(key)
        if tgt is None or not np.isfinite(tgt) or tgt <= 0 or denom <= 0:
            info["requant_source"][key] = "max_calibrated"
            return default
        info["requant_source"][key] = "supplied"
        return float(tgt) / float(denom)

    Ws, ss = q(fpar["stem"]["W"], "conv1")
    b0 = np.rint(fpar["stem"]["b"] / (ss * s_in)).astype(np.int64)
    maps = [conv_int(x, Ws, b0, 1, 3) for x in X]
    acc = max(acc, max(int(np.abs(u).max()) for u in maps))
    mx = max(int(np.maximum(u, 0).max()) for u in maps)
    eta = eta_for("conv1", mx / A if mx > 0 else 1.0, ss * s_in)
    acts = [np.clip(np.rint(np.maximum(u, 0) / eta), 0, A).astype(np.int64)
            for u in maps]
    s_a = ss * s_in * eta
    stem = {"W": Ws, "b": b0, "eta": eta}
    info["weight_scales"]["conv1"] = ss
    info["W_int_absmax"]["conv1"] = int(np.abs(Ws).max())
    info["requant_scales"]["conv1"] = eta
    blocks = []
    for blk in fpar["blocks"]:
        nm = blk["name"]
        st = block_stride(nm)
        W1, s1 = q(blk["W1"], nm + ".conv1")
        b1 = np.rint(blk["b1"] / (s1 * s_a)).astype(np.int64)
        u1 = [conv_int(a, W1, b1, st, 3) for a in acts]
        acc = max(acc, max(int(np.abs(u).max()) for u in u1))
        m1 = max(int(np.maximum(u, 0).max()) for u in u1)
        eta1 = eta_for(nm + ".conv1", m1 / A if m1 > 0 else 1.0, s1 * s_a)
        a1 = [np.clip(np.rint(np.maximum(u, 0) / eta1), 0, A).astype(np.int64)
              for u in u1]
        s_a1 = s1 * s_a * eta1
        W2, s2 = q(blk["W2"], nm + ".conv2")
        b2 = np.rint(blk["b2"] / (s2 * s_a1)).astype(np.int64)
        s_z = s2 * s_a1
        S = np.rint(blk["Ws"] * s_a / s_z).astype(np.int64)
        bsc = np.rint(blk["bs"] / s_z).astype(np.int64)
        u = [conv_int(z, W2, b2, 1, 3) + conv_int(a, S, bsc, st, 1)
             for z, a in zip(a1, acts)]
        acc = max(acc, max(int(np.abs(v).max()) for v in u))
        mx = max(int(np.maximum(v, 0).max()) for v in u)
        etab = eta_for(nm + ".block", mx / A if mx > 0 else 1.0, s_z)
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
    Wf, sf = q(fpar["fc"]["W"], "fc")
    bf = np.rint(fpar["fc"]["b"] / (sf * s_a)).astype(np.int64)
    outl = np.rint(pooled) @ Wf.T + bf
    acc = max(acc, int(np.abs(outl).max()))
    info["weight_scales"]["fc"] = sf
    info["W_int_absmax"]["fc"] = int(np.abs(Wf).max())
    info["accumulator_absmax_calibration"] = int(acc)
    info["accumulator_bits_calibration"] = int(acc).bit_length() + 1
    info["stem_input_fold_gain"] = stem_fold_gain
    net = IntResNet20(stem, blocks, {"W": Wf, "b": bf}, A)
    return net, info


def lattice_occupancy(net):
    """Zero fraction and occupied-level count per weight tensor, the quantity
    `REPORT_qat.md` Section 5 tabulates, so that the QAT arm of this step can be
    checked against it."""
    rows = []
    tensors = [("conv1", net.stem["W"])]
    for blk in net.blocks:
        tensors.append((blk["name"] + ".conv1", blk["W1"]))
        tensors.append((blk["name"] + ".conv2", blk["W2"]))
    tensors.append(("fc", net.fc["W"]))
    for name, W in tensors:
        W = np.asarray(W, dtype=np.int64)
        rows.append({"layer": name,
                     "zero_fraction": round(float(np.mean(W == 0)), 4),
                     "levels_occupied": int(np.unique(W).size),
                     "absmax": int(np.abs(W).max()),
                     "l1_row_max": int(np.abs(W).sum(axis=1).max())})
    z = [r["zero_fraction"] for r in rows]
    lv = [r["levels_occupied"] for r in rows]
    return {"per_tensor": rows, "tensors": len(rows),
            "zero_fraction_median": float(np.median(z)),
            "zero_fraction_min": float(min(z)),
            "zero_fraction_max": float(max(z)),
            "levels_occupied_median": float(np.median(lv)),
            "levels_occupied_min": int(min(lv)),
            "row_l1_max": int(max(r["l1_row_max"] for r in rows))}

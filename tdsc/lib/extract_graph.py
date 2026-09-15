"""Sequential extraction of a residual network over a round graph (step 4, T4-A).

New module; it reuses lib.lta_run.run_lta unchanged through the GraphChannel
adapter of lib.graph_oracle.
"""
import time
import numpy as np

from .lta import noise_fns_bounded
from .lta_run import run_lta
from .graph_oracle import GraphOracle, GraphAttack, GraphChannel, labelled_readout
from .session import find_separating_anchor
from .resnet20 import IntResNet20, build_rounds, _embed_centre
from .verify import (canonicalise, row_exactness, row_permutation, frame_check,
                     canonical_column_matrix)
from .extract import _order_final_layer


def _canon_cols(W_true, colmap, slots):
    return canonical_column_matrix(np.asarray(W_true, dtype=np.int64), colmap, slots)


def _true_from_canon(xc, colmap, slots):
    """The true-frame vector that a canonical-coordinate vector is scattered to."""
    xc = np.asarray(xc, dtype=np.int64)
    if colmap is None:
        return xc
    idx = (np.asarray(colmap, dtype=np.int64)[:, None] * slots
           + np.arange(slots, dtype=np.int64)).ravel() if slots > 1 else \
        np.asarray(colmap, dtype=np.int64)
    out = np.empty_like(xc)
    out[idx] = xc
    return out



def _best_constant(B, rowsum, A):
    """Pick the frame-invariant activation level c in {0..A} that makes the
    matching key B + c * rowsum(W2) as injective as possible (client-side, free)."""
    best_c, best_bad = 0, None
    for c in range(A + 1):
        k = B + c * rowsum
        bad = k.size - int(np.unique(k).size)
        if best_bad is None or bad < best_bad:
            best_c, best_bad = c, bad
        if bad == 0:
            break
    return int(best_c), int(best_bad)


def _perfect_matching(compat):
    """Hopcroft-Karp-free augmenting-path perfect matching on a boolean matrix."""
    n = compat.shape[0]
    matchR = [-1] * n

    def try_k(i, seen):
        for j in range(n):
            if compat[i, j] and not seen[j]:
                seen[j] = True
                if matchR[j] == -1 or try_k(matchR[j], seen):
                    matchR[j] = i
                    return True
        return False

    for i in range(n):
        if not try_k(i, [False] * n):
            return None
    perm = np.empty(n, dtype=np.int64)
    for j, i in enumerate(matchR):
        perm[i] = j
    return perm



def _duplicate_groups(W, b):
    """Canonical row indices grouped by identical [W|b] (client-side knowledge)."""
    M = np.concatenate([np.asarray(W), np.asarray(b)[:, None]], axis=1)
    seen = {}
    for k in range(M.shape[0]):
        seen.setdefault(M[k].tobytes(), []).append(k)
    return [v for v in seen.values() if len(v) > 1]


def _symmetrise(x, groups, slots):
    """Make a canonical-coordinate vector constant across each group of provably
    identical channels.  Such channels carry equal activations in honest
    inference, so no query can tell them apart; keeping the query symmetric on
    them makes the server-side true-frame vector well defined however the frame
    probe happened to resolve the tie."""
    if not groups:
        return x
    x = np.array(x, dtype=np.int64, copy=True)
    for g in groups:
        for sl in range(slots):
            idx = [c * slots + sl for c in g]
            x[idx] = x[idx[0]]
    return x


def extract_resnet20(net, mode, T, seed, noise_law="Gaussian", const_value=None,
                     global_deadline=None, diagnostics=True, log=print,
                     repair_attempts=0, channel_factory=None):
    """Extract a quantised ResNet-20 through the round graph of `mode`."""
    t0 = time.time()
    A = net.A
    rounds, meta = build_rounds(net, mode)
    rng_oracle = np.random.default_rng([seed, 1])
    rng_attack = np.random.default_rng([seed, 2])
    noise_fn = noise_fns_bounded(rng_oracle)[noise_law]
    oracle = GraphOracle(rounds, A, noise_fn, rng_oracle)
    attack = GraphAttack(oracle, A, rng_attack)
    if const_value is None:
        const_value = max(1, A // 2)

    colmap = {-1: None}          # group -> canonical->true row map
    dupgroups = {-1: []}         # group -> provably identical canonical channels
    recovered = {}               # layer name -> dict
    records = []
    stopped = ""
    primary_store = {}           # round -> (W_rec, b_rec, anchor, B)

    attempts = {}
    repairs = 0
    mi = 0
    while mi < len(meta):
        item = meta[mi]
        if global_deadline is not None and time.time() > global_deadline:
            stopped = "time box reached before %s" % item["layer"]
            break
        name = item["layer"]
        r = item["round"]
        obs = item["observe"]
        spec = rounds[r]
        s_before = oracle.n_sessions
        rec = {"layer": name, "round": r, "observe": obs, "kind": item["kind"],
               "group": item["group"], "mode": mode}

        if item["kind"] == "labelled":
            inp = spec.inputs[item["vary"]]
            gcol = inp["frame"]
            Sc = _canon_cols(inp["W"], colmap.get(gcol), inp["slots"])
            sched = {r: [("zero",)] * spec.n_new}
            W_rec, b_rec, ns, unl = labelled_readout(
                attack, r, item["vary"], obs, sched, inp["d"], spec.m)
            rec["sessions_used"] = int(oracle.n_sessions - s_before)
            rec["labelled_readout"] = True
            rec["unlabelled_sessions"] = int(unl)
            rec["sessions_formula"] = "d + 1 = %d" % (inp["d"] + 1)
            if W_rec is None:
                rec["status"] = "labelled_readout_failed"
                rec["exact_up_to_row_perm"] = False
                records.append(rec)
                log("  %-28s labelled readout FAILED (%d unlabelled)" % (name, unl))
                mi += 1
                continue
            Pg = colmap[item["group"]]
            true_W = Sc[Pg]
            true_b = np.asarray(spec.b, dtype=np.int64)[Pg]
            err = int(np.abs(W_rec - true_W).max()) if W_rec.size else 0
            errb = int(np.abs(b_rec - true_b).max())
            rec["status"] = "ok"
            rec["max_abs_error"] = max(err, errb)
            rec["exact_up_to_row_perm"] = bool(max(err, errb) == 0)
            rec["frame_check_exact"] = rec["exact_up_to_row_perm"]
            rec["rows_recovered_exactly"] = int(np.sum(
                np.all(W_rec == true_W, axis=1) & (b_rec == true_b)))
            rec["m"] = int(spec.m)
            rec["d"] = int(inp["d"])
            recovered[name] = {"W": W_rec, "b": b_rec}
            records.append(rec)
            log("  %-28s labelled  sessions=%-6d exact=%-5s" %
                (name, rec["sessions_used"], rec["exact_up_to_row_perm"]))
            mi += 1
            continue

        # ---------------- LTA-based rounds
        embed = None
        if item["kind"] == "sum_secondary" and item.get("centre_only"):
            inp = spec.inputs[item["vary"]]
            C_in = int(item["d_eff"])
            gcol = inp["frame"]
            Sfull = rounds[item["primary_round"]].inputs[1]["W"]
            W_for_lta = Sfull[:, np.arange(C_in) * 9 + 4]
            Sc = _canon_cols(W_for_lta, colmap.get(gcol), 1)
            idx = np.arange(C_in) * 9 + 4

            def embed(xq, _idx=idx, _d=inp["d"]):
                v = np.zeros(_d, dtype=np.int64)
                v[_idx] = xq
                return v
            d_lta = C_in
        else:
            inp = spec.inputs[item["vary"]]
            gcol = inp["frame"]
            Sc = _canon_cols(inp["W"], colmap.get(gcol), inp["slots"])
            d_lta = int(Sc.shape[1])

        # other inputs of the observed/varied rounds, and the effective true bias
        sched = {}
        b_eff = np.asarray(rounds[obs].b, dtype=np.int64).copy()
        oplans = []
        k = 0
        for ii, s in enumerate(rounds[obs].inputs):
            if s["kind"] != "new":
                continue
            if obs == r and ii == item["vary"]:
                oplans.append(("zero",))      # placeholder, replaced per query
            elif item["kind"] == "sum_secondary":
                ps = primary_store[item["primary_round"]]
                cv = ps[6]
                if obs == r:
                    # R4: the two inputs are independent in one round, so the
                    # primary input can carry its separating anchor and the
                    # secondary LTA inherits a tie-free, predictable intercept set.
                    anch = ps[2]
                    oplans.append(("canon", anch))
                    gfr = s["frame"]
                    b_eff = b_eff + s["W"] @ _true_from_canon(
                        anch, colmap.get(gfr), s["slots"])
                else:
                    oplans.append(("const", cv))
                    b_eff = b_eff + s["W"] @ np.full(s["d"], cv, dtype=np.int64)
            else:
                oplans.append(("zero",))
            k += 1
        sched[obs] = oplans
        if obs != r:
            sched[r] = [("zero",)] * rounds[r].n_new
        # the retained term of an R3 sum round, when we are not varying it
        for s in rounds[obs].inputs:
            if s["kind"] == "ret" and not (item["kind"] == "sum_secondary"):
                src_r = s["src"][0]
                pv = attack.probes[src_r]["vectors"][s["src"][1]]
                sinp = rounds[src_r].inputs[s["src"][1]]
                xt = _true_from_canon(pv, colmap.get(sinp["frame"]), sinp["slots"])
                b_eff = b_eff + s["W"] @ xt

        rng = rng_attack
        if channel_factory is not None:
            ch = channel_factory(Sc, b_eff, A, attack, r, item["vary"], obs, sched,
                                 embed, item, T)
        else:
            ch = GraphChannel(Sc, b_eff, A, attack, r, item["vary"], obs, sched,
                              embed=embed)
        dl = global_deadline
        out = run_lta(Sc, b_eff, A, T, rng, None, channel=ch, deadline=dl,
                      diagnostics=diagnostics)
        W_rec = out.pop("_W_rec")
        b_rec = out.pop("_b_rec")
        x0 = out.pop("_x0")
        anchor_resp = None
        for xa, ya in ch.anchor_responses:
            if np.array_equal(xa, x0):
                anchor_resp = ya
                break
        rec.update({k2: v for k2, v in out.items() if not k2.startswith("_")})
        rec["sessions_used"] = int(oracle.n_sessions - s_before)
        rec["sessions_formula"] = "1 + T d = %d" % (1 + T * d_lta)
        rec["lta_probes"] = int(getattr(ch, "n_queries", 0))
        if hasattr(ch, "packed_sessions"):
            rec["packed_sessions"] = int(ch.packed_sessions)
            rec["probes_per_session"] = int(ch.n_probes_per_session)
            rec["zero_checks"] = int(ch.zero_checks)
            rec["zero_check_mismatches"] = int(ch.zero_check_mismatches)
            rec["sessions_formula_packed"] = "1 + ceil(T d / (P-1)) = %d" % (
                1 + int(np.ceil(T * d_lta / max(1, ch.n_probes_per_session))))

        if item["kind"] == "sum_secondary":
            (W_pri, b_pri, anch_pri, Bc, W2sum, delta_fn, cv,
             cbad, xfix) = primary_store[item["primary_round"]]
            W_sec, b_sec = canonicalise(W_rec, b_rec)
            if obs == r:
                # R4: the primary input carried its separating anchor, so both
                # sides are the same predictable, tie-free intercept set.
                key1 = W_pri @ anch_pri + Bc
                key2 = b_sec        # run_lta already removed S x0 from the intercept
            else:
                # R3: the primary input can only be frame-invariant, so the link
                # runs through rowsum(W2) and the retained probe vector.
                key1 = Bc + cv * W2sum
                key2 = b_sec + delta_fn(W_sec)
            uniq = bool(np.unique(key1).size == key1.size)
            rec["secondary_constant"] = int(cv)
            rec["secondary_match_keys_distinct"] = uniq
            rec["secondary_second_lta"] = False
            if uniq:
                o1 = np.argsort(key1, kind="stable")
                o2 = np.argsort(key2, kind="stable")
                rec["secondary_match_exact"] = bool(
                    np.array_equal(key1[o1], key2[o2]))
                perm = np.empty(key1.size, dtype=np.int64)
                perm[o1] = o2
                rec["secondary_match_candidates_max"] = 1
            else:
                # the single-constant key is not injective: spend a second
                # secondary LTA at a different frame-invariant level, which yields
                # rowsum(W2) per secondary row and makes the match injective.
                cv2 = (cv + 1) if cv + 1 <= A else (cv - 1)
                sched2 = {k4: list(v) for k4, v in sched.items()}
                p4 = list(sched2[obs])
                for ii2, s2 in enumerate(
                        [q for q in rounds[obs].inputs if q["kind"] == "new"]):
                    if not (obs == r and ii2 == item["vary"]):
                        p4[ii2] = ("const", cv2)
                sched2[obs] = p4
                b_eff2 = np.asarray(rounds[obs].b, dtype=np.int64).copy()
                kk2 = 0
                for s2 in rounds[obs].inputs:
                    if s2["kind"] != "new":
                        continue
                    if not (obs == r and kk2 == item["vary"]):
                        b_eff2 = b_eff2 + s2["W"] @ np.full(
                            s2["d"], cv2, dtype=np.int64)
                    kk2 += 1
                ch2 = GraphChannel(Sc, b_eff2, A, attack, r, item["vary"], obs,
                                   sched2, embed=embed)
                out2 = run_lta(Sc, b_eff2, A, T, rng, None, channel=ch2,
                               deadline=dl, diagnostics=diagnostics)
                W_rec2 = out2.pop("_W_rec")
                b_rec2 = out2.pop("_b_rec")
                out2.pop("_x0")
                W_sec2, b_sec2 = canonicalise(W_rec2, b_rec2)
                rec["secondary_second_lta"] = True
                rec["secondary_second_constant"] = int(cv2)
                # align run 2 to run 1 by their (identical) S rows
                a1s = np.lexsort(W_sec.T[::-1])
                a2s = np.lexsort(W_sec2.T[::-1])
                b_sec2_al = np.empty_like(b_sec2)
                b_sec2_al[a1s] = b_sec2[a2s]
                rs_sec = (b_sec - b_sec2_al) // (cv - cv2)
                rec["secondary_rowsum_recovered"] = True
                compat = ((rs_sec[None, :] == W2sum[:, None])
                          & (key2[None, :] == key1[:, None]))
                rec["secondary_match_candidates_max"] = int(compat.sum(axis=1).max())
                # Tie-break sessions.  A frame-invariant primary input can never
                # separate two output coordinates that share rowsum(W2) and the
                # block constant, so the tie-break session puts the primary input
                # at its separating anchor (which needs the preceding round's
                # frame, hence its probe) and changes the secondary input.  In R3
                # the secondary input IS the preceding round's input, so changing
                # it requires installing a second probe there; in R4 the two
                # inputs are independent and no probe swap is needed.
                nt = 0
                d_sec = int(Sc.shape[1])
                xz = np.zeros(d_sec, dtype=np.int64) if xfix is None else xfix
                base_i = key1
                from .lta import _isin_sorted
                while int(compat.sum(axis=1).max()) > 1 and nt < 3:
                    saved = None
                    if obs != r:
                        # R3: swap round r's probe for a fresh separating anchor
                        nm_r = rounds[r].name
                        if nm_r not in recovered:
                            break
                        Wr = recovered[nm_r]["W"]
                        br = recovered[nm_r]["b"]
                        xfp, bad6, st6, fl6 = find_separating_anchor(Wr, br, A, rng)
                        predr = Wr @ xfp + br
                        saved = attack.probes.get(r)
                        attack.install_probe(r, [xfp], predr)
                        C_in6 = d_sec
                        xf = xfp[np.arange(C_in6) * 9 + 4]
                        sched3 = {k5: list(v) for k5, v in sched.items()}
                        sched3.pop(r, None)
                        p5 = list(sched3[obs])
                        kk5 = 0
                        for s5 in rounds[obs].inputs:
                            if s5["kind"] != "new":
                                continue
                            p5[kk5] = ("canon", anch_pri)
                            kk5 += 1
                        sched3[obs] = p5
                    else:
                        xf = rng.integers(0, A + 1, size=d_sec).astype(np.int64)
                        sched3 = {k5: list(v) for k5, v in sched.items()}
                        p5 = list(sched3[obs])
                        kk5 = 0
                        for s5 in rounds[obs].inputs:
                            if s5["kind"] != "new":
                                continue
                            p5[kk5] = (("canon", xf) if kk5 == item["vary"]
                                       else ("canon", anch_pri))
                            kk5 += 1
                        sched3[obs] = p5
                    y3, pi3 = attack.run_session(sched3, obs, return_pi=True)
                    nt += 1
                    if saved is not None:
                        attack.probes[r] = saved
                        attack.group_probe[rounds[r].group] = r
                    need = rounds[obs].inputs[0]["frame"]
                    if need >= 0 and need not in pi3:
                        continue
                    vals = base_i[:, None] + (W_sec @ (xf - xz))[None, :]
                    compat = compat & _isin_sorted(vals, np.unique(np.sort(y3)))
                rec["secondary_tiebreak_sessions"] = nt
                rec["secondary_match_candidates_max"] = int(compat.sum(axis=1).max())
                perm = _perfect_matching(compat)
                if perm is None:
                    o1 = np.argsort(key1, kind="stable")
                    o2 = np.argsort(key2, kind="stable")
                    perm = np.empty(key1.size, dtype=np.int64)
                    perm[o1] = o2
                    rec["secondary_match_exact"] = False
                else:
                    rec["secondary_match_exact"] = True
                rec["sessions_used"] = int(oracle.n_sessions - s_before)
            W_can = W_sec[perm]
            b_can = b_sec[perm]
            rec["block_bias_sum_recovered"] = True
            # the block's total constant is b2 + b_s; in R3 the primary bias also
            # carries S x_probe, which the recovered S now removes.
            b_block = Bc - delta_fn(W_can)
            conv2_name = name[: -len(".shortcut")] + ".conv2"
            recovered[conv2_name]["b"] = b_block
            rec["block_constant_absmax"] = int(np.abs(b_block).max())
            b_can = np.zeros_like(b_can)
        elif item.get("unshuffled"):
            W_can, b_can, fin = _order_final_layer(W_rec, b_rec, x0, anchor_resp)
            if not fin.startswith("true order"):
                xc2, bad2, st2, fl2 = find_separating_anchor(W_rec, b_rec, A, rng)
                sch2 = {k3: list(v) for k3, v in sched.items()}
                p2 = list(sch2[r])
                p2[item["vary"]] = ("canon", xc2 if embed is None else embed(xc2))
                sch2[r] = p2
                y2 = attack.run_session(sch2, obs)
                W2c, b2c, fin2 = _order_final_layer(W_rec, b_rec, xc2, y2)
                if fin2.startswith("true order"):
                    W_can, b_can, fin = W2c, b2c, fin2 + " (one extra session)"
                else:
                    fin = fin2 + " (extra separating-anchor session did not resolve it)"
            rec["final_layer_row_order"] = fin
            rec["sessions_used"] = int(oracle.n_sessions - s_before)
        else:
            W_can, b_can = canonicalise(W_rec, b_rec)

        if item["kind"] == "sum_secondary":
            max_err = int(np.abs(W_can - Sc[np.asarray(
                colmap[item["group"]], dtype=np.int64)]).max()) if W_can.size else 0
            n_exact = int(np.sum(np.all(
                W_can == Sc[np.asarray(colmap[item["group"]], dtype=np.int64)],
                axis=1)))
            P = np.asarray(colmap[item["group"]], dtype=np.int64)
            n_matched = n_exact
            fok, ferr = bool(max_err == 0), max_err
        else:
            max_err, n_exact = row_exactness(W_can, b_can, Sc, b_eff)
            P, n_matched = row_permutation(W_can, b_can, Sc, b_eff)
            fok, ferr = frame_check(W_can, b_can, Sc, b_eff, P)
        rec["max_abs_error_up_to_row_perm"] = max_err
        rec["exact_up_to_row_perm"] = bool(max_err == 0)
        rec["rows_recovered_exactly"] = n_exact
        rec["frame_check_exact"] = fok
        rec["frame_check_max_abs_error"] = ferr
        rec["rows_matched_to_true_rows"] = n_matched
        rec["W_int_absmax"] = int(np.abs(Sc).max())
        rec["status"] = out.get("status", "ok")
        recovered[name] = {"W": W_can, "b": b_can}

        if item["kind"] != "sum_secondary":
            colmap[item["group"]] = P
            pred = W_can @ x0 + b_can
            pvecs = []
            kk = 0
            for ii, s in enumerate(rounds[r].inputs):
                if s["kind"] != "new":
                    continue
                if ii == item["vary"]:
                    pvecs.append(x0 if embed is None else embed(x0))
                else:
                    pvecs.append(np.zeros(s["d"], dtype=np.int64))
                kk += 1
            # The LTA anchor reproduces its own reply by construction, so it is
            # not a test of the recovered matrix.  Always install a FRESH
            # separating anchor and spend one session verifying it: that reply is
            # a genuine prediction, it is what makes the probe tie-free (which the
            # labelled readout needs), and it is the detect-and-retry signal.
            xc2, bad2, st2, fl2 = find_separating_anchor(W_can, b_can, A, rng)
            _vin = rounds[r].inputs[item["vary"]]
            xc2 = _symmetrise(xc2, dupgroups.get(_vin.get("frame", -1), []),
                              _vin.get("slots", 1) if embed is None else 1)
            pred2 = W_can @ xc2 + b_can
            pv2 = list(pvecs)
            pv2[item["vary"]] = xc2 if embed is None else embed(xc2)
            sch2 = {k3: list(v) for k3, v in sched.items()}
            p2 = list(sch2[r])
            p2[item["vary"]] = ("canon", pv2[item["vary"]])
            sch2[r] = p2
            y2 = attack.run_session(sch2, obs)
            rec["probe_extra_session"] = True
            rec["probe_verified"] = bool(np.array_equal(np.sort(pred2), np.sort(y2)))
            rec["probe_colliding_rows"] = int(bad2)
            rec["probe_search_steps"] = int(st2)
            pvecs, pred = pv2, pred2
            tf = attack.install_probe(r, pvecs, pred)
            rec["probe_tie_free"] = bool(tf)
            dupgroups[item["group"]] = _duplicate_groups(W_can, b_can)
            rec["identical_output_channel_groups"] = [
                [int(c) for c in g] for g in dupgroups[item["group"]]]
            rec["sessions_used"] = int(oracle.n_sessions - s_before)
            if item["kind"] == "sum_primary":
                W2sum = W_can.sum(axis=1)
                retv = None
                for s in rounds[obs].inputs:
                    if s["kind"] == "ret":
                        src_r = s["src"][0]
                        sinp = rounds[src_r].inputs[s["src"][1]]
                        pvv = attack.probes[src_r]["vectors"][s["src"][1]]
                        C_in = sinp["d"] // sinp["slots"]
                        retv = pvv[np.arange(C_in) * 9 + 4]
                if retv is None:
                    d_sec = None
                    for s5 in rounds[obs].inputs:
                        if s5["kind"] == "new" and s5 is not rounds[obs].inputs[
                                item["vary"]]:
                            pass
                    retv0 = None

                    def delta_fn(Ws):
                        return np.zeros(Ws.shape[0], dtype=np.int64)
                else:
                    retv0 = retv

                    def delta_fn(Ws, _rv=retv):
                        return Ws @ _rv
                cbest, cbad = _best_constant(b_can, W2sum, A)
                rec["secondary_constant"] = cbest
                rec["secondary_key_collisions"] = cbad
                # use the PROBE anchor (separating by construction) as the
                # primary input of the secondary phase, so that R4's matching key
                # is tie-free
                anch_store = xc2
                primary_store[r] = (W_can, b_can, anch_store, b_can.copy(), W2sum,
                                    delta_fn, cbest, cbad, retv0)
        if (repair_attempts and rec.get("probe_verified") is False
                and attempts.get(mi, 0) < repair_attempts):
            attempts[mi] = attempts.get(mi, 0) + 1
            repairs += 1
            log("  %-28s probe verification failed, retrying (attempt %d)"
                % (name, attempts[mi]))
            continue
        rec["repair_attempts_used"] = int(attempts.get(mi, 0))
        mi += 1
        records.append(rec)
        log("  %-28s %-14s sessions=%-6d exact=%-5s frame=%-5s unres=%-3s"
            % (name, item["kind"], rec["sessions_used"],
               rec["exact_up_to_row_perm"], rec.get("frame_check_exact"),
               rec.get("unresolved_rows")))

    ext = None
    if not stopped and all(r.get("exact_up_to_row_perm") is not None
                           for r in records):
        try:
            ext = _assemble(net, recovered)
        except Exception as exc:
            ext = None
            stopped = "assembly failed: %s: %s" % (type(exc).__name__, exc)
    summary = {
        "mode": mode, "T": int(T), "noise_law": noise_law,
        "rounds": len(rounds), "layers_attacked": len(records),
        "layers_exact": int(sum(1 for r in records if r.get("exact_up_to_row_perm"))),
        "layers_total": len(meta),
        "sessions_total": int(oracle.n_sessions),
        "round_evaluations_total": int(oracle.n_round_evaluations),
        "round_evaluations_expected": int(oracle.n_sessions * len(rounds)),
        "probe_rounds": int(attack.probe_rounds),
        "probe_failures": int(attack.probe_failures),
        "labelled_reads": int(attack.labelled_reads),
        "oracle_rounding_exact": bool(oracle.rounding_exact),
        "inadmissible_query_entries": int(oracle.inadmissible_entries),
        "memoised_round_evaluations": int(oracle.cache_hits),
        "frame_unavailable_events": int(attack.frame_unavailable),
        "packed_sessions": int(sum(
            r2.get("packed_sessions", 0) for r2 in records)),
        "zero_patch_checks": int(sum(r2.get("zero_checks", 0) for r2 in records)),
        "zero_patch_mismatches": int(sum(
            r2.get("zero_check_mismatches", 0) for r2 in records)),
        "repair_attempts_allowed": int(repair_attempts),
        "repairs_triggered": int(repairs),
        "stopped_reason": stopped,
        "wall_time_sec": round(time.time() - t0, 2),
    }
    return records, ext, summary


def _assemble(net, recovered):
    stem = {"W": recovered["conv1"]["W"], "b": recovered["conv1"]["b"],
            "eta": net.stem["eta"]}
    blocks = []
    for blk in net.blocks:
        nm = blk["name"]
        blocks.append({"name": nm,
                       "W1": recovered[nm + ".conv1"]["W"],
                       "b1": recovered[nm + ".conv1"]["b"],
                       "eta1": blk["eta1"],
                       "W2": recovered[nm + ".conv2"]["W"],
                       "b2": recovered[nm + ".conv2"]["b"],
                       "S": recovered[nm + ".shortcut"]["W"],
                       "bs": recovered[nm + ".shortcut"]["b"],
                       "eta": blk["eta"],
                       "shortcut": blk["shortcut"],
                       "C_in": blk["C_in"], "C_out": blk["C_out"]})
    fc = {"W": recovered["fc"]["W"], "b": recovered["fc"]["b"]}
    return IntResNet20(stem, blocks, fc, net.A)

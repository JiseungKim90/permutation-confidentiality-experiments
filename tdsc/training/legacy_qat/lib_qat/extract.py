"""Sequential frame-tracked extraction of a whole chain (plan Section 2)."""
import time
import numpy as np

from .lta import noise_fns_bounded
from .lta_run import run_lta
from .quant import QuantChain, accumulator_absmax_for_queries
from .session import (SessionOracle, SessionChannel, FrameTrackedAttack,
                      find_separating_anchor)
from .verify import (canonicalise, row_exactness, row_permutation, frame_check,
                     canonical_column_matrix)


def _order_final_layer(W_rec, b_rec, x0, anchor_response):
    """The final layer is unshuffled (sigma_R = id), so the ordered anchor
    response fixes the true row order directly."""
    pred = W_rec @ np.asarray(x0, dtype=np.int64) + b_rec
    if np.unique(pred).size != pred.size or anchor_response is None:
        W, b = canonicalise(W_rec, b_rec)
        return W, b, "canonical (anchor values not distinct)"
    order = np.argsort(pred, kind="stable")
    ps = pred[order]
    idx = np.searchsorted(ps, anchor_response)
    np.clip(idx, 0, ps.size - 1, out=idx)
    if not bool(np.all(ps[idx] == anchor_response)):
        W, b = canonicalise(W_rec, b_rec)
        return W, b, "canonical (anchor response did not match the prediction)"
    rowfor = order[idx]
    if np.unique(rowfor).size != rowfor.size:
        W, b = canonicalise(W_rec, b_rec)
        return W, b, "canonical (anchor match was not a bijection)"
    return W_rec[rowfor], b_rec[rowfor], "true order from the unshuffled response"


def extract_chain(true_chain, T, seed, noise_law="Gaussian", layer_budget=None,
                  global_deadline=None, backends=None, diagnostics=True,
                  log=print, repair_attempts=0):
    """Run the sequential attack.  Returns (records, extracted_chain_or_None, summary).

    repair_attempts (step 4, default 0 = the step-3 behaviour): when a layer's
    probe verification fails, discard that layer's result and re-run LTA for it
    with a fresh anchor, at most this many extra times.  The detection signal is
    the client's own probe verification, so the repair uses no ground truth.
    """
    t0 = time.time()
    A = true_chain.A
    slots = true_chain.slots
    R = true_chain.R
    rng_oracle = np.random.default_rng([seed, 1])
    rng_attack = np.random.default_rng([seed, 2])
    noise_fn = noise_fns_bounded(rng_oracle)[noise_law]
    oracle = SessionOracle(true_chain, noise_fn, rng_oracle, backends=backends)
    attack = FrameTrackedAttack(oracle, A, slots, rng_attack)

    colmap = None
    repairs = 0
    records = []
    ext_W, ext_b = [], []
    K = 0
    stopped = ""
    for r in range(R):
        if global_deadline is not None and time.time() > global_deadline:
            stopped = ("global time box reached before layer %d of %d" % (r, R))
            break
        Wc = canonical_column_matrix(true_chain.W[r], colmap, slots)
        bc = np.asarray(true_chain.b[r], dtype=np.int64)
        s_before = oracle.n_sessions
        e_before = oracle.n_layer_evaluations
        attempt = 0
        # --- detect-and-retry loop; with repair_attempts = 0 it runs exactly once
        while True:
            ch = SessionChannel(Wc, bc, A, rng_attack, attack)
            dl = None
            if layer_budget is not None:
                dl = time.time() + layer_budget
            if global_deadline is not None:
                dl = global_deadline if dl is None else min(dl, global_deadline)
            rec = run_lta(Wc, bc, A, T, rng_attack, None, channel=ch, deadline=dl,
                          diagnostics=diagnostics)
            W_rec = rec.pop("_W_rec")
            b_rec = rec.pop("_b_rec")
            x0 = rec.pop("_x0")
            anchor_resp = None
            for xa, ya in ch.anchor_responses:
                if np.array_equal(xa, x0):
                    anchor_resp = ya
                    break
            if r == R - 1:
                break
            W_try, b_try = canonicalise(W_rec, b_rec)
            pinfo = attack.install_probe(W_try, b_try, lta_anchor=x0,
                                         lta_anchor_response=anchor_resp)
            if (repair_attempts > 0 and pinfo.get("probe_verified") is False
                    and attempt < repair_attempts):
                attack.uninstall_last_probe()
                attempt += 1
                repairs += 1
                continue
            break
        rec["repair_attempts_used"] = attempt
        if r == R - 1:
            W_can, b_can, fin = _order_final_layer(W_rec, b_rec, x0, anchor_resp)
            extra = 0
            if not fin.startswith("true order"):
                # the LTA anchor did not separate the rows, so the unshuffled
                # response cannot be read off; spend one session on a separating
                # anchor found offline (same mechanism as a frame probe).
                xc2, bad2, steps2, floor2 = find_separating_anchor(
                    W_rec, b_rec, A, rng_attack)
                y2 = attack.run_attack_session(xc2)
                extra = 1
                W2, b2, fin2 = _order_final_layer(W_rec, b_rec, xc2, y2)
                if fin2.startswith("true order"):
                    W_can, b_can = W2, b2
                    fin = fin2 + " (one extra session with a separating anchor)"
                else:
                    fin = (fin2 + " (an extra session with a separating anchor did "
                                  "not resolve it; %d colliding rows, %d unseparable)"
                           % (bad2, floor2))
            rec["final_layer_row_order"] = fin
            rec["final_layer_extra_sessions"] = extra
            rec["sessions_used"] = int(oracle.n_sessions - s_before)
        else:
            W_can, b_can = W_try, b_try
        max_err, n_exact = row_exactness(W_can, b_can, Wc, bc)
        P, n_matched = row_permutation(W_can, b_can, Wc, bc)
        fok, ferr = frame_check(W_can, b_can, Wc, bc, P)
        rec["layer_index"] = r
        rec["sessions_used"] = int(oracle.n_sessions - s_before)
        rec["layer_evaluations"] = int(oracle.n_layer_evaluations - e_before)
        rec["rows_matched_to_true_rows"] = n_matched
        rec["frame_check_exact"] = fok
        rec["frame_check_max_abs_error"] = ferr
        rec["max_abs_error_up_to_row_perm"] = max_err
        rec["exact_up_to_row_perm"] = bool(max_err == 0)
        rec["rows_recovered_exactly"] = n_exact
        rec["accumulator_absmax_worstcase_query"] = \
            accumulator_absmax_for_queries(Wc, bc, A)
        rec["accumulator_bits_worstcase_query"] = \
            rec["accumulator_absmax_worstcase_query"].bit_length() + 1
        rec["W_int_absmax"] = int(np.abs(Wc).max())
        rec["zero_weight_fraction"] = float(np.mean(Wc == 0))
        if r < R - 1:
            rec["sessions_used"] = int(oracle.n_sessions - s_before)
            rec.update(pinfo)
        rec["probe_rounds_cumulative"] = int(attack.probe_rounds)
        rec["probe_failures_cumulative"] = int(attack.probe_failures)
        ext_W.append(W_can)
        ext_b.append(b_can)
        colmap = P
        K = r + 1
        records.append(rec)
        log("  layer %3d/%d %5dx%-5d sessions=%-7d exact=%-5s frame=%-5s unres=%-4d "
            "q/d=%-6.3f probe_fail=%-4d %.1fs"
            % (r + 1, R, rec["m"], rec["d"], rec["sessions_used"],
               rec["exact_up_to_row_perm"], rec["frame_check_exact"],
               rec.get("unresolved_rows", -1), rec["queries_over_d"],
               attack.probe_failures, rec["wall_time_sec"]))
    ext_chain = None
    if K == R:
        ext_chain = QuantChain(ext_W, ext_b, true_chain.eta, A, slots)
    summary = {
        "layers_total": int(R),
        "layers_completed": int(K),
        "stopped_reason": stopped,
        "T": int(T),
        "noise_law": noise_law,
        "sessions_total": int(oracle.n_sessions),
        "layer_evaluations_total": int(oracle.n_layer_evaluations),
        "layer_evaluations_expected_sessions_times_R": int(oracle.n_sessions * R),
        "probe_rounds": int(attack.probe_rounds),
        "probe_failures": int(attack.probe_failures),
        "probe_distinctness_violations": int(attack.probe_distinctness_violations),
        "probe_ambiguous_matches": int(attack.probe_ambiguous_matches),
        "extra_probe_sessions": int(attack.extra_probe_sessions),
        "zero_rounds": int(attack.zero_rounds),
        "oracle_rounding_exact": bool(oracle.rounding_exact),
        "oracle_rounding_failures": int(oracle.n_rounding_failures),
        "inadmissible_query_entries": int(oracle.inadmissible_entries),
        "memoised_layer_evaluations": int(oracle.cache_hits),
        "wall_time_sec": round(time.time() - t0, 2),
        "repair_attempts_allowed": int(repair_attempts),
        "repairs_triggered": int(repairs),
    }
    if backends is not None:
        summary["tfhe_evaluations"] = int(oracle.backend_calls)
        summary["tfhe_decryption_mismatches"] = int(oracle.backend_mismatches)
        summary["tfhe_decryption_exact"] = bool(oracle.backend_mismatches == 0)
        summary["tfhe_eval_time_sec"] = round(oracle.backend_time, 2)
    sess_formula = sum(1 + T * int(w.shape[1]) for w in true_chain.W[:K])
    summary["sessions_formula_sum_r_1_plus_T_d_r"] = int(sess_formula)
    summary["sessions_above_formula"] = int(oracle.n_sessions - sess_formula)
    return records, ext_chain, summary

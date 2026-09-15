"""Assembly, second pass and verification for LTA.

Re-materialised from the feasibility step-2b implementation.  The only change
for step 3 is the optional `channel` argument of run_lta, which lets a caller
supply a pre-built channel (the session oracle adapter of lib/session.py);
when it is omitted the behaviour is identical to step 2b.
"""
import time
from collections import Counter
import numpy as np

from .lta import Channel, lta_pass, noise_fns_bounded


_LTA_ZERO_FIELDS = (
    "columns_failed",
    "columns_backtracked",
    "unresolved_rows",
    "channel_rounding_failures",
    "inadmissible_query_entries",
)


def _is_explicit_zero(record, field):
    """Accept a required counter only when it is present and integer zero."""
    value = record.get(field)
    return bool(
        field in record
        and isinstance(value, (int, np.integer))
        and not isinstance(value, (bool, np.bool_))
        and int(value) == 0
    )


def _nested_lta_schema(record):
    """Return valid nested runs and any record-layout schema errors."""
    if not isinstance(record, dict):
        return [], [{"layout": "record", "run_index": None,
                     "reason": "record is not an object"}]

    runs, errors = [], []
    has_single = "lta" in record
    has_many = "lta_runs" in record
    if has_single:
        single = record["lta"]
        if isinstance(single, dict):
            runs.append(("lta", 0, single))
        else:
            errors.append({"layout": "lta", "run_index": 0,
                           "reason": "member is not an object"})
    if has_many:
        many = record["lta_runs"]
        if not isinstance(many, (list, tuple)):
            errors.append({"layout": "lta_runs", "run_index": None,
                           "reason": "container is not a list"})
        elif not many:
            errors.append({"layout": "lta_runs", "run_index": None,
                           "reason": "container has no LTA invocation"})
        else:
            for run_index, run in enumerate(many):
                if isinstance(run, dict):
                    runs.append(("lta_runs", run_index, run))
                else:
                    errors.append({"layout": "lta_runs",
                                   "run_index": run_index,
                                   "reason": "member is not an object"})

    if not has_single and not has_many:
        explicit_readout = bool(
            record.get("kind") == "fc"
            and isinstance(record.get("readout"), str)
            and record["readout"]
        )
        if not explicit_readout:
            errors.append({"layout": "record", "run_index": None,
                           "reason": "missing LTA layout or explicit non-LTA readout"})
    return runs, errors


def lta_run_is_certified(record):
    """Return whether one LTA invocation is safe to consume.

    Diagnostics against a private reference model are deliberately excluded:
    a real attacker does not possess that model.  Certification instead uses
    only transcript-side conditions that must hold before recovered arrays are
    consumed.
    """
    return bool(
        isinstance(record, dict)
        and record.get("status") == "ok"
        and all(_is_explicit_zero(record, field)
                for field in _LTA_ZERO_FIELDS)
    )


def require_lta_run_certified(record, context="LTA recovery"):
    """Reject a partial/ambiguous LTA result before arrays are materialised."""
    if lta_run_is_certified(record):
        return record
    view = record if isinstance(record, dict) else {}
    raise RuntimeError(
        "%s is uncertified: status=%r, columns_failed=%r, "
        "columns_backtracked=%r, unresolved_rows=%r"
        % (context, view.get("status"), view.get("columns_failed"),
           view.get("columns_backtracked"), view.get("unresolved_rows"))
    )


def iter_lta_runs(records):
    """Yield object-valued LTA invocations from both supported layouts."""
    for record_index, record in enumerate(records):
        runs, _ = _nested_lta_schema(record)
        for layout, run_index, run in runs:
            yield record_index, layout, run_index, run


def recovery_record_is_certified(record):
    """Fail-closed predicate for one layer-level recovery record."""
    nested, schema_errors = _nested_lta_schema(record)
    if schema_errors or not isinstance(record, dict):
        return False
    isolation_ok = (
        "isolation_failures" not in record
        or _is_explicit_zero(record, "isolation_failures")
    )
    probe_value = record.get("probe_verified", True)
    probe_ok = bool(
        isinstance(probe_value, (bool, np.bool_)) and probe_value
    )
    return bool(
        all(lta_run_is_certified(run) for _, _, run in nested)
        and isolation_ok
        and probe_ok
    )


def summarize_recovery_records(records):
    """Common final gate and accounting for layer and LTA records."""
    records = list(records)
    runs = list(iter_lta_runs(records))
    malformed = []
    for record_index, record in enumerate(records):
        _, errors = _nested_lta_schema(record)
        for error in errors:
            malformed.append({"record_index": record_index, **error})
    failed = [
        {"record_index": ri, "layout": layout, "run_index": rj,
         "status": run.get("status")}
        for ri, layout, rj, run in runs
        if not lta_run_is_certified(run)
    ]
    certified_records = sum(
        recovery_record_is_certified(record) for record in records
    )
    return {
        "records_total": int(len(records)),
        "records_certified": int(certified_records),
        "lta_invocations_total": int(len(runs)),
        "lta_invocations_certified": int(len(runs) - len(failed)),
        "lta_passes_total": int(sum(
            int(run.get("n_passes", 0)) for _, _, _, run in runs
        )),
        "failed_lta_runs": failed,
        "malformed_lta_records": malformed,
        "success": bool(certified_records == len(records)
                        and not failed and not malformed),
    }


def _singles(p, x0, d):
    """Rows of a pass whose intercept is unique: (alpha, W, b)."""
    a_vals, mu, off, slopes = p["a_vals"], p["mu"], p["off"], p["slopes"]
    sing = np.nonzero(mu == 1)[0]
    W = np.zeros((sing.size, d), dtype=np.int64)
    for j in range(d):
        if slopes[j] is None:
            continue
        W[:, j] = slopes[j][off[sing]]
    b = a_vals[sing] - W @ np.asarray(x0, dtype=np.int64)
    return sing, a_vals[sing], W, b


def _group_matrix(p, i, d):
    """(mu_i x d) matrix of the slope multisets of tied group i, columns sorted."""
    mu, off, slopes = p["mu"], p["off"], p["slopes"]
    n = int(mu[i])
    G = np.zeros((n, d), dtype=np.int64)
    for j in range(d):
        if slopes[j] is not None:
            G[:, j] = slopes[j][off[i]:off[i] + n]
    return G


def _pick_jstar(p, d):
    """Column maximising the number of tied groups whose slope multiset there
    has all-distinct values."""
    mu, off, slopes = p["mu"], p["off"], p["slopes"]
    groups = np.nonzero(mu > 1)[0]
    if groups.size == 0:
        return None, 0
    slots = np.concatenate([np.arange(off[i], off[i] + mu[i]) for i in groups])
    grp_of = np.concatenate([np.full(int(mu[i]), gi) for gi, i in enumerate(groups)])
    adj_same = grp_of[1:] == grp_of[:-1]
    best_j, best_c = None, -1
    for j in range(d):
        if slopes[j] is None:
            continue
        col = slopes[j][slots]
        bad = (col[1:] == col[:-1]) & adj_same
        nbad = np.unique(grp_of[1:][bad]).size
        c = groups.size - nbad
        if c > best_c:
            best_c, best_j = c, j
        if best_c == groups.size:
            break
    return best_j, best_c


def _verify(W_rec, b_rec, W_int, b_int):
    M_rec = np.concatenate([W_rec, b_rec[:, None]], axis=1)
    M_true = np.concatenate([W_int, b_int[:, None]], axis=1)
    o1 = np.lexsort(M_rec.T[::-1])
    o2 = np.lexsort(M_true.T[::-1])
    diff = np.abs(M_rec[o1] - M_true[o2])
    max_err = int(diff.max()) if diff.size else 0
    c1 = Counter(M_rec[i].tobytes() for i in range(M_rec.shape[0]))
    c2 = Counter(M_true[i].tobytes() for i in range(M_true.shape[0]))
    inter = sum(min(v, c2.get(k, 0)) for k, v in c1.items())
    return max_err, int(inter)


def run_lta(W_int, b_int, A, T, rng, noise_fn, backend=None, deadline=None,
            node_budget=4000, max_anchor_tries=5, diagnostics=True,
            allow_pass2=True, x0_fixed=None, channel=None):
    """Full LTA on one quantised layer.  Returns a JSON-ready record.

    If `channel` is given it is used instead of a fresh lib.lta.Channel; it must
    expose the same interface (query / prepare_anchor / set_anchor / query_step
    and the query counters).
    """
    t_start = time.time()
    m, d = W_int.shape
    rec = {
        "m": int(m), "d": int(d), "T": int(T), "A": int(A),
        "status": "ok", "error": "",
    }
    hi = A - 2 * T
    if hi < 0:
        rec["status"] = "anchor_range_empty"
        rec["error"] = "A - 2T = %d < 0" % hi
        rec["wall_time_sec"] = round(time.time() - t_start, 3)
        return rec
    key_mul = 2 * int(np.abs(W_int).max()) + 3 if W_int.size else 3
    ch = channel if channel is not None else Channel(
        W_int, b_int, A, noise_fn, rng, backend=backend)
    tW = W_int if diagnostics else None

    # ---------------- anchor selection
    attempts = []
    best = None
    tries = 1 if x0_fixed is not None else max_anchor_tries
    for _ in range(tries):
        x0c = (np.asarray(x0_fixed, dtype=np.int64) if x0_fixed is not None
               else rng.integers(0, hi + 1, size=d).astype(np.int64))
        S0c = ch.set_anchor(x0c)
        ties = int(m - np.unique(S0c).size)
        attempts.append(ties)
        if best is None or ties < best[2]:
            best = (x0c, S0c, ties)
        if ties == 0:
            break
    x0, S0, _ = best
    rec["anchor_queries"] = len(attempts)
    rec["anchor_tie_attempts"] = attempts
    rec["anchor_range"] = [0, int(hi)]

    # ---------------- pass 1
    p1 = lta_pass(ch, x0, T, S0=S0, deadline=deadline, node_budget=node_budget,
                  true_W=tW, key_mul=key_mul)
    rec["pass1"] = {k: v for k, v in p1.items()
                    if k not in ("slopes", "a_vals", "mu", "off", "S0")}
    rec["n_ties_anchor1"] = p1["n_ties"]
    if p1["status"] != "ok":
        rec["status"] = p1["status"]
        rec["error"] = p1["error"]

    mu1 = p1["mu"]
    groups1 = np.nonzero(mu1 > 1)[0]
    sing_idx, sing_alpha, W1, b1 = _singles(p1, x0, d)

    rows_W = [W1]
    rows_b = [b1]
    unresolved = 0
    rec["pass2_used"] = False
    rec["pass2_anchor_prediction_exact"] = None
    rec["pass2_overlap_checked"] = 0
    rec["pass2_overlap_mismatches"] = 0
    rec["rows_from_pass1"] = int(W1.shape[0])
    rec["rows_from_pass2"] = 0
    rec["jstar"] = None

    p2 = None
    dict2 = {}
    if groups1.size and allow_pass2 and p1["status"] == "ok" and hi >= 0:
        jstar, ndist = _pick_jstar(p1, d)
        rec["jstar"] = None if jstar is None else int(jstar)
        rec["jstar_groups_all_distinct"] = int(ndist)
        if jstar is not None:
            x0p = x0.copy()
            x0p[jstar] += T
            pred = [sing_alpha + T * W1[:, jstar]]
            for i in groups1:
                G = _group_matrix(p1, i, d)
                pred.append(p1["a_vals"][i] + T * G[:, jstar])
            pred = np.sort(np.concatenate(pred))
            S0p = ch.set_anchor(x0p)
            rec["pass2_anchor_prediction_exact"] = bool(
                np.array_equal(pred, np.sort(S0p)))
            p2 = lta_pass(ch, x0p, T, S0=S0p, deadline=deadline,
                          node_budget=node_budget, true_W=tW, key_mul=key_mul)
            rec["pass2"] = {k: v for k, v in p2.items()
                            if k not in ("slopes", "a_vals", "mu", "off", "S0")}
            rec["pass2_used"] = True
            rec["n_ties_anchor2"] = p2["n_ties"]
            if p2["status"] != "ok" and rec["status"] == "ok":
                rec["status"] = p2["status"]
                rec["error"] = p2["error"]
            s2_idx, s2_alpha, W2, b2 = _singles(p2, x0p, d)
            for r in range(W2.shape[0]):
                dict2[int(s2_alpha[r])] = (W2[r], int(b2[r]))
            for r in range(W1.shape[0]):
                ap = int(sing_alpha[r] + T * W1[r, jstar])
                if ap in dict2:
                    rec["pass2_overlap_checked"] += 1
                    if not np.array_equal(dict2[ap][0], W1[r]):
                        rec["pass2_overlap_mismatches"] += 1

    # ---------------- fill tied groups
    inconsistent_matches = 0
    unresolved_group_ids = []
    rec["rows_forced_as_group_residual"] = 0
    rec["rows_in_provably_identical_residuals"] = 0
    for i in groups1:
        n = int(mu1[i])
        G = _group_matrix(p1, i, d)
        alive = np.ones((n, d), dtype=bool)
        taken = []
        if p2 is not None and rec["jstar"] is not None:
            js = rec["jstar"]
            used = set()
            for g in G[:, js]:
                ap = int(p1["a_vals"][i] + T * g)
                if ap in dict2 and ap not in used:
                    w2, bb2 = dict2[ap]
                    if int(w2[js]) != int(g):
                        continue
                    eq = (G == w2[None, :]) & alive
                    has = eq.any(axis=0)
                    if not bool(has.all()):
                        inconsistent_matches += 1
                        continue
                    alive[eq.argmax(axis=0), np.arange(d)] = False
                    used.add(ap)
                    taken.append((w2, bb2))
        for w2, bb2 in taken:
            rows_W.append(w2[None, :])
            rows_b.append(np.array([bb2], dtype=np.int64))
        rec["rows_from_pass2"] += len(taken)
        left = n - len(taken)
        if left > 0:
            Wl = np.zeros((left, d), dtype=np.int64)
            for j in range(d):
                Wl[:, j] = G[alive[:, j], j]
            bl = p1["a_vals"][i] - Wl @ x0
            rows_W.append(Wl)
            rows_b.append(bl)
            if left == 1:
                rec["rows_forced_as_group_residual"] += 1
            elif bool(np.all(Wl == Wl[0])):
                rec["rows_in_provably_identical_residuals"] += int(left)
            else:
                unresolved += left
                unresolved_group_ids.append((i, int(left), int(n)))

    rec["inconsistent_pass2_matches"] = inconsistent_matches
    W_rec = np.concatenate(rows_W, axis=0) if rows_W else np.zeros((0, d), np.int64)
    b_rec = np.concatenate(rows_b, axis=0) if rows_b else np.zeros((0,), np.int64)
    if W_rec.shape[0] != m:
        rec["status"] = "assembly_row_count_mismatch"
        rec["error"] = "assembled %d rows, expected %d" % (W_rec.shape[0], m)
        pad = m - W_rec.shape[0]
        if pad > 0:
            W_rec = np.concatenate([W_rec, np.zeros((pad, d), np.int64)])
            b_rec = np.concatenate([b_rec, np.zeros(pad, np.int64)])
        else:
            W_rec, b_rec = W_rec[:m], b_rec[:m]

    max_err, n_exact_rows = _verify(W_rec, b_rec, W_int, b_int)
    rec["unresolved_rows"] = int(unresolved)
    rec["unresolved_groups"] = int(len(unresolved_group_ids))
    ident_copies = 0
    grp_all_ident = 0
    rows_all_ident = 0
    if unresolved_group_ids:
        base1 = W_int @ np.asarray(x0, dtype=np.int64) + b_int
        for i, left, nrows in unresolved_group_ids:
            ks = np.nonzero(base1 == p1["a_vals"][i])[0]
            cnt = Counter(W_int[k].tobytes() + b_int[k].tobytes() for k in ks)
            ident_copies += sum(v - 1 for v in cnt.values() if v > 1)
            if len(cnt) == 1:
                grp_all_ident += 1
                rows_all_ident += left
    rec["redundant_true_row_copies_in_unresolved_groups"] = int(ident_copies)
    rec["unresolved_groups_all_rows_identical"] = int(grp_all_ident)
    rec["unresolved_rows_in_all_identical_groups"] = int(rows_all_ident)
    rec["unresolved_rows_in_mixed_groups"] = int(unresolved - rows_all_ident)
    rec["max_abs_error_up_to_row_perm"] = max_err
    rec["exact_up_to_row_perm"] = bool(max_err == 0)
    rec["rows_recovered_exactly"] = n_exact_rows
    rec["n_queries"] = int(ch.n_queries)
    rec["queries_over_d"] = round(ch.n_queries / d, 3)
    npass = 2 if rec["pass2_used"] else 1
    n_anchor = rec["anchor_queries"] + (1 if rec["pass2_used"] else 0)
    rec["n_passes"] = npass
    rec["n_queries_formula"] = (
        "%d anchor queries + %d pass(es) x T d = %d + %d*%d*%d = %d"
        % (n_anchor, npass, n_anchor, npass, T, d, n_anchor + npass * T * d))
    rec["channel_rounding_exact"] = bool(ch.rounding_exact)
    rec["channel_rounding_failures"] = int(ch.n_rounding_failures)
    rec["inadmissible_query_entries"] = int(ch.inadmissible_query_entries)
    cf = p1["columns_forced"] + (p2["columns_forced"] if p2 else 0)
    ct = p1["columns_total"] + (p2["columns_total"] if p2 else 0)
    rec["columns_forced_fraction"] = round(cf / ct, 4) if ct else None
    rec["columns_failed"] = int(p1["columns_failed"] + (p2["columns_failed"] if p2 else 0))
    rec["false_line_events"] = int(p1["false_line_events"]
                                   + (p2["false_line_events"] if p2 else 0))
    rec["columns_backtracked"] = int(p1["columns_backtracked"]
                                     + (p2["columns_backtracked"] if p2 else 0))
    if rec["status"] == "ok" and rec["columns_failed"]:
        rec["status"] = "uncertified_columns"
        rec["error"] = "%d LTA column(s) are uncertified" % rec["columns_failed"]
    if rec["status"] == "ok" and rec["columns_backtracked"]:
        rec["status"] = "uncertified_backtracking"
        rec["error"] = "%d LTA column(s) used an uncertified branch" % rec["columns_backtracked"]
    if rec["status"] == "ok" and rec["unresolved_rows"]:
        rec["status"] = "unresolved_rows"
        rec["error"] = "%d recovered row(s) remain unresolved" % rec["unresolved_rows"]
    rec["certified_complete"] = lta_run_is_certified(rec)
    if backend is not None:
        rec["n_tfhe_evaluations"] = int(ch.backend_calls)
        rec["tfhe_eval_time_sec"] = round(ch.backend_time, 3)
        rec["tfhe_decryption_mismatches"] = int(ch.backend_mismatches)
        rec["tfhe_decryption_exact"] = bool(ch.backend_mismatches == 0)
    rec["wall_time_sec"] = round(time.time() - t_start, 3)
    rec["_W_rec"] = W_rec
    rec["_b_rec"] = b_rec
    rec["_x0"] = x0
    return rec

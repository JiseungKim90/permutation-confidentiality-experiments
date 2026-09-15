"""LTA: line tracking with a dense anchor.

Re-materialised verbatim from the feasibility step-2b implementation.

Channel model:
    y = P_pi (W_int x + b_int) + eta,  pi fresh per query,
    |eta| <= 1/2 - 1e-9 in integer units, six bounded laws,
    noise added in np.longdouble, observation = rint(.) read as an unordered multiset.
"""
import time
import numpy as np

NOISE_DELTA = 0.5


def noise_fns_bounded(rng, Delta=NOISE_DELTA):
    """Six bounded laws, ported from lib/attack.py::noise_fns_bounded of the prior
    artifact with the strict bound set to Delta - 1e-9."""
    B = Delta - 1e-9
    return {
        "Zero": lambda n: np.zeros(n, dtype=np.longdouble),
        "Gaussian": lambda n: np.asarray(
            np.clip(rng.normal(0.0, Delta / 3.0, n), -B, B), dtype=np.longdouble),
        "Uniform": lambda n: np.asarray(rng.uniform(-B, B, n), dtype=np.longdouble),
        "Triangular": lambda n: np.asarray(
            rng.triangular(-B, 0.0, B, n), dtype=np.longdouble),
        "Laplace": lambda n: np.asarray(
            np.clip(rng.laplace(0.0, Delta / 2.0, n), -B, B), dtype=np.longdouble),
        "Worst-case": lambda n: (np.full(n, B, dtype=np.longdouble)
                                 * np.where(np.arange(n) % 2 == 0, 1.0, -1.0)),
    }


class Channel:
    """Permuting, bounded-noise oracle over a quantised linear layer.

    backend, if given, is backend(x_int, perm) -> decrypted permuted integer
    vector (TFHE transcript); otherwise the permuted exact vector is used.
    """

    def __init__(self, W_int, b_int, A, noise_fn, rng, backend=None):
        self.W = np.ascontiguousarray(W_int, dtype=np.int64)
        self.b = np.ascontiguousarray(b_int, dtype=np.int64)
        self.m, self.d = self.W.shape
        self.A = int(A)
        self.noise_fn = noise_fn
        self.rng = rng
        self.backend = backend
        self.n_queries = 0
        self.n_rounding_failures = 0
        self.inadmissible_query_entries = 0
        self.backend_mismatches = 0
        self.backend_calls = 0
        self.backend_time = 0.0
        self._base = None
        self._x0 = None

    def exact(self, x):
        return self.W @ np.asarray(x, dtype=np.int64) + self.b

    def _check_admissible(self, x):
        bad = int(np.sum((x < 0) | (x > self.A)))
        self.inadmissible_query_entries += bad
        return bad

    def _observe(self, true_vals, x):
        perm = self.rng.permutation(self.m)
        permuted = true_vals[perm]
        if self.backend is not None:
            t0 = time.time()
            dec = np.asarray(self.backend(x, perm), dtype=np.int64)
            self.backend_time += time.time() - t0
            self.backend_calls += 1
            if not np.array_equal(dec, permuted):
                self.backend_mismatches += 1
            permuted = dec
        obs = permuted.astype(np.longdouble) + self.noise_fn(self.m)
        rounded = np.rint(obs).astype(np.int64)
        if not np.array_equal(rounded, permuted):
            self.n_rounding_failures += 1
        self.n_queries += 1
        return rounded

    def query(self, x):
        x = np.asarray(x, dtype=np.int64)
        self._check_admissible(x)
        return self._observe(self.exact(x), x)

    def prepare_anchor(self, x0):
        """Set the anchor context without issuing a query."""
        x0 = np.asarray(x0, dtype=np.int64)
        self._x0 = x0
        self._base = self.exact(x0)

    def set_anchor(self, x0):
        x0 = np.asarray(x0, dtype=np.int64)
        self._check_admissible(x0)
        self.prepare_anchor(x0)
        return self._observe(self._base, x0)

    def query_step(self, j, t):
        """Query x0 + t e_j.  True value base + t W[:, j] is exact integer
        arithmetic, identical to W (x0 + t e_j) + b."""
        x = self._x0.copy()
        x[j] += t
        self._check_admissible(x)
        return self._observe(self._base + t * self.W[:, j], x)

    @property
    def rounding_exact(self):
        return self.n_rounding_failures == 0


def _isin_sorted(vals, sorted_unique):
    if sorted_unique.size == 0:
        return np.zeros(vals.shape, dtype=bool)
    idx = np.searchsorted(sorted_unique, vals)
    np.clip(idx, 0, sorted_unique.size - 1, out=idx)
    return sorted_unique[idx] == vals


class CoverBudget(Exception):
    pass


def solve_cover(a_vals, mu, cand, uniq, cnt, T, off, m, node_budget=4000):
    """Exact multiset cover with a conservative uniqueness certificate.

    A cover is returned only when multiplicity propagation completes the cover
    at the root search node.  A feasible cover first reached after branching is
    deliberately reported as unresolved: the first feasible branch need not be
    the unique cover.  This fail-closed rule is stricter than testing uniqueness
    by exhaustive enumeration, but keeps every accepted column sound within a
    fixed node budget.
    """
    K = a_vals.size
    res_mu = np.array(mu, dtype=np.int64)
    cc = [c.astype(np.int64).copy() for c in cnt]
    out_flat = np.zeros(m, dtype=np.int64)
    sizes = np.array([c.size for c in cand], dtype=np.int64)
    if np.any(sizes == 0):
        i = int(np.nonzero(sizes == 0)[0][0])
        return {"ok": False, "slopes": None, "forced": False, "nodes": 0,
                "reason": "no surviving candidate slope for intercept %d" % int(a_vals[i])}

    one = np.nonzero(sizes == 1)[0]
    if one.size:
        g_one = np.array([cand[i][0] for i in one], dtype=np.int64)
        for t in range(1, T + 1):
            k = np.searchsorted(uniq[t - 1], a_vals[one] + t * g_one)
            np.subtract.at(cc[t - 1], k, mu[one])
            if np.any(cc[t - 1] < 0):
                return {"ok": False, "slopes": None, "forced": False, "nodes": 0,
                        "reason": "multiplicity underflow at t=%d in the forced pass" % t}
        slot_row = np.repeat(np.arange(K), mu)
        sel = np.isin(slot_row, one)
        out_flat[sel] = np.repeat(g_one, mu[one])
        res_mu[one] = 0

    gen_rows = np.nonzero(sizes > 1)[0]
    if gen_rows.size == 0:
        ok = all(bool(np.all(c == 0)) for c in cc)
        return {"ok": ok, "slopes": out_flat if ok else None, "forced": True,
                "nodes": 0, "reason": "" if ok else "residual multiplicity after forced pass"}

    pair_row = np.repeat(gen_rows, sizes[gen_rows])
    pair_g = np.concatenate([cand[i] for i in gen_rows]).astype(np.int64)
    P = pair_row.size
    pv = np.zeros((T, P), dtype=np.int64)
    for t in range(1, T + 1):
        pv[t - 1] = np.searchsorted(uniq[t - 1], a_vals[pair_row] + t * pair_g)
    nodes = [0]

    def propagate(res_mu, cc, active, comm):
        while True:
            if not np.any(active):
                return True
            ub = res_mu[pair_row].copy()
            for t in range(T):
                ub = np.minimum(ub, cc[t][pv[t]])
            ub[~active] = 0
            np.clip(ub, 0, None, out=ub)
            row_sum = np.bincount(pair_row, weights=ub, minlength=K).astype(np.int64)
            if np.any(row_sum < res_mu):
                return False
            exact = np.zeros(P, dtype=bool)
            eq_rows = (res_mu > 0) & (row_sum == res_mu)
            if np.any(eq_rows):
                exact |= eq_rows[pair_row] & active
            need = np.where(active, res_mu[pair_row] - (row_sum[pair_row] - ub), 0)
            for t in range(T):
                vs = np.bincount(pv[t], weights=ub, minlength=cc[t].size).astype(np.int64)
                if np.any(vs < cc[t]):
                    return False
                eq_v = (cc[t] > 0) & (vs == cc[t])
                if np.any(eq_v):
                    exact |= eq_v[pv[t]] & active
                need = np.maximum(need, np.where(
                    active, cc[t][pv[t]] - (vs[pv[t]] - ub), 0))
            np.clip(need, 0, None, out=need)
            delta = np.where(exact, ub, need)
            delta[~active] = 0
            progress = bool(np.any(delta)) or bool(np.any(exact))
            if not progress:
                return True
            comm += delta
            res_mu -= np.bincount(pair_row, weights=delta, minlength=K).astype(np.int64)
            if np.any(res_mu < 0):
                return False
            for t in range(T):
                cc[t] -= np.bincount(pv[t], weights=delta,
                                     minlength=cc[t].size).astype(np.int64)
                if np.any(cc[t] < 0):
                    return False
            active &= ~exact
            active &= (res_mu[pair_row] > 0)

    def complete(res_mu, cc):
        return bool(np.all(res_mu == 0)) and all(bool(np.all(c == 0)) for c in cc)

    def search(res_mu, cc, active, comm):
        nodes[0] += 1
        if nodes[0] > node_budget:
            raise CoverBudget("node budget %d exhausted" % node_budget)
        if not propagate(res_mu, cc, active, comm):
            return None
        if complete(res_mu, cc):
            return comm
        live = np.nonzero(active)[0]
        if live.size == 0:
            return None
        counts = np.bincount(pair_row[live], minlength=K)
        rows_hot = np.nonzero((counts > 0) & (res_mu > 0))[0]
        if rows_hot.size == 0:
            return None
        i = int(rows_hot[np.argmin(counts[rows_hot])])
        p = int(live[pair_row[live] == i][0])
        for branch in (1, 0):
            rm = res_mu.copy()
            c2 = [c.copy() for c in cc]
            act = active.copy()
            cm = comm.copy()
            if branch == 1:
                cm[p] += 1
                rm[pair_row[p]] -= 1
                bad = rm[pair_row[p]] < 0
                for t in range(T):
                    c2[t][pv[t][p]] -= 1
                    if c2[t][pv[t][p]] < 0:
                        bad = True
                if bad:
                    continue
            else:
                act[p] = False
            got = search(rm, c2, act, cm)
            if got is not None:
                return got
        return None

    try:
        sol = search(res_mu, cc, np.ones(P, dtype=bool), np.zeros(P, dtype=np.int64))
    except CoverBudget as exc:
        return {"ok": False, "slopes": None, "forced": False, "nodes": nodes[0],
                "reason": str(exc)}
    if sol is None:
        return {"ok": False, "slopes": None, "forced": False, "nodes": nodes[0],
                "reason": "no feasible exact cover"}
    if nodes[0] > 1:
        return {"ok": False, "slopes": None, "forced": False, "nodes": nodes[0],
                "reason": "uniqueness not certified: exact-cover branching was required"}
    for i in gen_rows:
        sel = (pair_row == i)
        out_flat[off[i]:off[i] + mu[i]] = np.sort(np.repeat(pair_g[sel], sol[sel]))
    return {"ok": True, "slopes": out_flat, "forced": True,
            "nodes": nodes[0], "reason": ""}


def track_column(a_vals, mu, off, m, S_list, T, node_budget=4000,
                 true_keys=None, key_mul=None, chunk=2_000_000):
    """Candidate generation + exact cover for one column."""
    K = a_vals.size
    uniq, cnt = [], []
    for t in range(1, T + 1):
        u, c = np.unique(S_list[t - 1], return_counts=True)
        uniq.append(u.astype(np.int64))
        cnt.append(c.astype(np.int64))
    V1 = uniq[0]
    n_before = int(K) * int(V1.size)
    cand = [None] * K
    n_after = 0
    n_false = 0
    rows_per_chunk = max(1, chunk // max(1, V1.size))
    for s in range(0, K, rows_per_chunk):
        e = min(K, s + rows_per_chunk)
        blk = a_vals[s:e][:, None]
        gam = V1[None, :] - blk
        mask = np.ones(gam.shape, dtype=bool)
        for t in range(2, T + 1):
            mask &= _isin_sorted(blk + t * gam, uniq[t - 1])
        rr, cci = np.nonzero(mask)
        vals = gam[rr, cci]
        n_after += vals.size
        if true_keys is not None and vals.size:
            keys = a_vals[s + rr] * key_mul + vals
            n_false += int(np.sum(~_isin_sorted(keys, true_keys)))
        bounds = np.searchsorted(rr, np.arange(1, e - s))
        parts = np.split(vals, bounds)
        for r in range(e - s):
            cand[s + r] = parts[r]
    out = solve_cover(a_vals, mu, cand, uniq, cnt, T, off, m, node_budget=node_budget)
    out["n_candidates_before"] = n_before
    out["n_candidates_after"] = int(n_after)
    out["n_false_lines"] = int(n_false)
    return out


def lta_pass(channel, x0, T, S0=None, deadline=None, node_budget=4000,
             true_W=None, key_mul=None):
    """One LTA pass: dense anchor x0 plus T d line-tracking queries.

    If S0 is given, the anchor query was already issued by the caller."""
    m, d = channel.m, channel.d
    if S0 is None:
        S0 = channel.set_anchor(x0)
    else:
        channel.prepare_anchor(x0)
    a_vals, mu = np.unique(S0, return_counts=True)
    a_vals = a_vals.astype(np.int64)
    mu = mu.astype(np.int64)
    K = a_vals.size
    off = np.concatenate([[0], np.cumsum(mu)]).astype(np.int64)
    prof = {str(int(v)): int(c) for v, c in zip(*np.unique(mu, return_counts=True))}
    res = {
        "n_distinct_intercepts": int(K),
        "n_ties": int(m - K),
        "n_tied_groups": int(np.sum(mu > 1)),
        "n_tied_rows": int(np.sum(mu[mu > 1])),
        "max_multiplicity": int(mu.max()),
        "multiplicity_profile": prof,
        "columns_total": int(d),
        "columns_forced": 0,
        "columns_backtracked": 0,
        "columns_failed": 0,
        "column_failures": [],
        "false_line_events": 0,
        "cand_before_total": 0,
        "cand_after_total": 0,
        "backtrack_nodes_max": 0,
        "status": "ok",
        "error": "",
    }
    slopes = [None] * d
    base = channel._base
    for j in range(d):
        if deadline is not None and time.time() > deadline:
            res["status"] = "time_budget_exceeded"
            res["error"] = "per-layer time budget exceeded (column %d/%d)" % (j, d)
            break
        S_list = [channel.query_step(j, t) for t in range(1, T + 1)]
        tk = None
        if true_W is not None:
            tk = np.sort(base * key_mul + true_W[:, j])
        out = track_column(a_vals, mu, off, m, S_list, T, node_budget=node_budget,
                           true_keys=tk, key_mul=key_mul)
        res["cand_before_total"] += out["n_candidates_before"]
        res["cand_after_total"] += out["n_candidates_after"]
        res["false_line_events"] += out["n_false_lines"]
        res["backtrack_nodes_max"] = max(res["backtrack_nodes_max"], out["nodes"])
        if not out["ok"]:
            res["columns_failed"] += 1
            if len(res["column_failures"]) < 5:
                res["column_failures"].append({"column": int(j), "reason": out["reason"]})
            slopes[j] = None
        else:
            if out["forced"]:
                res["columns_forced"] += 1
            else:
                res["columns_backtracked"] += 1
            slopes[j] = out["slopes"]
    if res["status"] == "ok" and res["columns_failed"]:
        res["status"] = "uncertified_columns"
        res["error"] = (
            "%d/%d column(s) lack a unique exact-cover certificate"
            % (res["columns_failed"], res["columns_total"])
        )
    if res["status"] == "ok" and res["columns_backtracked"]:
        res["status"] = "uncertified_backtracking"
        res["error"] = (
            "%d/%d column(s) were accepted without root propagation"
            % (res["columns_backtracked"], res["columns_total"])
        )
    res["slopes"] = slopes
    res["a_vals"] = a_vals
    res["mu"] = mu
    res["off"] = off
    res["S0"] = S0
    return res

"""Session oracle, frame tracking and the sequential frame-tracked LTA attack
(plan Sections 1 and 2).

Frame conventions.  For a permutation array p, apply_perm(p, v)[i] = v[p[i]] and
unapply_perm is its inverse.  In session s the server draws sigma_r for
r = 1..R-1 and sets sigma_R = id.  Round r receives the client-frame vector x_r,
forms the true-frame input unapply_perm(rho_r, x_r) with rho_1 = id and
rho_r = expand(sigma_{r-1}, slots), evaluates W_r (.) + b_r, applies sigma_r and
returns the ordered noisy integer vector.

The client's frame for round r is therefore "position i holds true channel
sigma_{r-1}[i]".  If canonical row c of layer r-1 is true row tau_{r-1}(c), then a
canonical-coordinate vector xc is submitted as apply_perm(expand(pi_{r-1}), xc)
with pi_{r-1}[i] = tau_{r-1}^{-1}(sigma_{r-1}[i]); this composite is exactly what a
frame probe on layer r-1 measures, and it never requires tau or sigma separately.
"""
import time
import numpy as np

from .lta import Channel


def expand(perm, slots):
    """Lift a channel permutation to (channel, slot) coordinates, flat index
    channel * slots + slot (the PyTorch conv flattening)."""
    perm = np.asarray(perm, dtype=np.int64)
    if slots == 1:
        return perm
    return (perm[:, None] * slots + np.arange(slots, dtype=np.int64)).ravel()


def apply_perm(perm, v):
    return np.asarray(v)[perm]


def unapply_perm(perm, v):
    v = np.asarray(v)
    out = np.empty_like(v)
    out[perm] = v
    return out



def n_distinct_rows(W, b):
    M = np.concatenate([np.asarray(W), np.asarray(b)[:, None]], axis=1)
    return len(set(M[k].tobytes() for k in range(M.shape[0])))


def find_separating_anchor(W, b, A, rng, n_random=200, max_steps=5000):
    """Search (client side, no oracle query) for x in {0..A}^d whose predicted
    values W x + b separate as many rows as possible.

    Random draws are the plan's rule; at 4-bit weight scales the intercepts of a
    wide layer are far too concentrated for a random draw to separate m rows
    (measured in REPORT_step3.md), so the best random draw is refined by greedy
    repair: pick a colliding pair, pick a coordinate where the two rows differ and
    retune that coordinate.  Rows that are identical can never be separated; the
    search stops once only such collisions remain.
    """
    W = np.asarray(W, dtype=np.int64)
    b = np.asarray(b, dtype=np.int64)
    m, d = W.shape
    floor_bad = m - n_distinct_rows(W, b)
    best_x, best_bad = None, None
    for _ in range(max(1, n_random)):
        x = rng.integers(0, A + 1, size=d).astype(np.int64)
        v = W @ x + b
        bad = m - int(np.unique(v).size)
        if best_bad is None or bad < best_bad:
            best_x, best_bad = x, bad
        if bad <= floor_bad:
            return x, bad, 0, floor_bad
    x = best_x.copy()
    v = W @ x + b
    bad = best_bad
    steps = 0
    for steps in range(1, max_steps + 1):
        if bad <= floor_bad:
            break
        u, inv, cnt = np.unique(v, return_inverse=True, return_counts=True)
        dup = np.nonzero(cnt > 1)[0]
        if dup.size == 0:
            break
        g = int(dup[rng.integers(0, dup.size)])
        rows = np.nonzero(inv == g)[0]
        diff = None
        for p in range(rows.size - 1):
            cand = np.nonzero(W[rows[p]] != W[rows[p + 1]])[0]
            if cand.size:
                diff = cand
                break
        if diff is None:
            continue
        j = int(diff[rng.integers(0, diff.size)])
        col = W[:, j]
        cur = int(x[j])
        bv, bb, bval = None, None, None
        for val in range(0, A + 1):
            vv = v + (val - cur) * col
            cb = m - int(np.unique(vv).size)
            if bb is None or cb < bb:
                bb, bval, bv = cb, val, vv
        if bb <= bad:
            x[j] = bval
            v = bv
            bad = bb
    return x, bad, steps, floor_bad


class SessionOracle:
    """Safhire-style session oracle over an integer chain.

    One query per layer per session, ordered responses, fresh permutations per
    layer per session.  `backends[r]`, when given, is a callable
    backend(x_true, perm) -> decrypted permuted integer vector, used for the TFHE
    transcript; every round of every session then runs under encryption.
    """

    def __init__(self, chain, noise_fn, rng, backends=None):
        self.chain = chain
        self.W = chain.W
        self.b = chain.b
        self.A = chain.A
        self.slots = chain.slots
        self.R = chain.R
        self.noise_fn = noise_fn
        self.rng = rng
        self.backends = backends
        self.n_sessions = 0
        self.n_layer_evaluations = 0
        self.n_rounding_failures = 0
        self.inadmissible_entries = 0
        self.backend_calls = 0
        self.backend_mismatches = 0
        self.backend_time = 0.0
        self._cache = [dict() for _ in range(self.R)]
        self._seen = [dict() for _ in range(self.R)]
        self.cache_hits = 0

    @property
    def rounding_exact(self):
        return self.n_rounding_failures == 0

    def eval_layer(self, r, x_true):
        """Deterministic map x_true -> W_r x_true + b_r, memoised.

        The memo is a cache of a deterministic integer function; the session
        randomness (sigma_r and eta) is drawn outside it, so caching cannot
        change the distribution of any response.
        """
        key = x_true.tobytes()
        hit = self._cache[r].get(key)
        if hit is not None:
            self.cache_hits += 1
            return hit
        v = self.W[r] @ x_true + self.b[r]
        s = self._seen[r]
        if s.pop(key, 0):
            self._cache[r][key] = v
        else:
            if len(s) > 4096:
                s.clear()
            s[key] = 1
        return v

    def session(self):
        self.n_sessions += 1
        return Session(self)


class Session:
    def __init__(self, oracle):
        self.o = oracle
        R = oracle.R
        self.sigmas = [oracle.rng.permutation(oracle.W[r].shape[0])
                       for r in range(R - 1)]
        self.sigmas.append(np.arange(oracle.W[R - 1].shape[0], dtype=np.int64))
        self.r = 0

    def round(self, x_client):
        o = self.o
        r = self.r
        if r >= o.R:
            raise RuntimeError("session already used all %d rounds" % o.R)
        x_client = np.asarray(x_client, dtype=np.int64)
        o.inadmissible_entries += int(np.sum((x_client < 0) | (x_client > o.A)))
        if r == 0:
            x_true = x_client
        else:
            x_true = unapply_perm(expand(self.sigmas[r - 1], o.slots), x_client)
        v = o.eval_layer(r, x_true)
        o.n_layer_evaluations += 1
        perm = self.sigmas[r]
        permuted = v[perm]
        if o.backends is not None and o.backends[r] is not None:
            t0 = time.time()
            dec = np.asarray(o.backends[r](x_true, perm), dtype=np.int64)
            o.backend_time += time.time() - t0
            o.backend_calls += 1
            if not np.array_equal(dec, permuted):
                o.backend_mismatches += 1
            permuted = dec
        obs = permuted.astype(np.longdouble) + o.noise_fn(permuted.size)
        y = np.rint(obs).astype(np.int64)
        if not np.array_equal(y, permuted):
            o.n_rounding_failures += 1
        self.r += 1
        return y


class SessionChannel(Channel):
    """Adapter presenting the step-2b Channel interface over the session oracle.

    Wc / bc are the canonical-frame true matrix and bias of the layer under
    attack; they are used only for the verification baseline and the false-line
    diagnostic of lta_pass (step 2b established that the recovery path is
    independent of them).  Every query is one full session.
    """

    def __init__(self, Wc, bc, A, rng, attack):
        super().__init__(Wc, bc, A, None, rng)
        self.attack = attack
        self.anchor_responses = []

    def query(self, x):
        x = np.asarray(x, dtype=np.int64)
        self._check_admissible(x)
        self.n_queries += 1
        return self.attack.run_attack_session(x)

    def set_anchor(self, x0):
        x0 = np.asarray(x0, dtype=np.int64)
        self.prepare_anchor(x0)
        y = self.query(x0)
        self.anchor_responses.append((x0.copy(), y.copy()))
        return y

    def query_step(self, j, t):
        x = self._x0.copy()
        x[j] += t
        return self.query(x)

    @property
    def rounding_exact(self):
        return self.attack.oracle.rounding_exact


class FrameTrackedAttack:
    """Sequential extraction of the whole chain (plan Section 2)."""

    def __init__(self, oracle, A, slots, rng):
        self.oracle = oracle
        self.A = int(A)
        self.slots = int(slots)
        self.rng = rng
        self.R = oracle.R
        self.r = 0                     # index of the layer currently under attack
        self.state = []                 # per recovered layer
        self.probe_rounds = 0
        self.probe_failures = 0
        self.probe_distinctness_violations = 0
        self.probe_ambiguous_matches = 0
        self.extra_probe_sessions = 0
        self.zero_rounds = 0

    # ---------------- frame probe ----------------
    def _probe_match(self, st, y):
        """Multiset match of the ordered response against the predicted values.

        Sorting both sides gives a bijection whenever the multisets agree, which
        also handles predicted values that repeat (the assignment inside such a
        group is then arbitrary; st["ambiguous"] records that the frame is only
        determined up to those channels).
        """
        pred_sorted = st["pred_sorted"]
        order = st["pred_order"]
        ys = np.argsort(y, kind="stable")
        good = bool(np.array_equal(y[ys], pred_sorted))
        pi = np.empty(y.size, dtype=np.int64)
        pi[ys] = order
        if st["ambiguous"]:
            self.probe_ambiguous_matches += 1
        return pi, good

    def run_attack_session(self, xc):
        """One session: frame probes for layers 0..r-1, the attack query for
        layer r, zeros afterwards.  Returns the ordered response of round r."""
        xc = np.asarray(xc, dtype=np.int64)
        sess = self.oracle.session()
        pi_prev = None
        for q in range(self.r):
            st = self.state[q]
            xq = (st["probe_xc"] if q == 0
                  else apply_perm(expand(pi_prev, self.slots), st["probe_xc"]))
            y = sess.round(xq)
            self.probe_rounds += 1
            pi_prev, good = self._probe_match(st, y)
            if not good:
                self.probe_failures += 1
        xr = (xc if self.r == 0
              else apply_perm(expand(pi_prev, self.slots), xc))
        yr = sess.round(xr)
        for q in range(self.r + 1, self.R):
            sess.round(np.zeros(self.oracle.W[q].shape[1], dtype=np.int64))
            self.zero_rounds += 1
        return yr

    # ---------------- probe selection ----------------
    def install_probe(self, W_rec, b_rec, lta_anchor=None, lta_anchor_response=None):
        """Choose a separating probe anchor for the freshly recovered layer self.r.

        The LTA anchor is reused when its predicted values are already distinct
        (no extra session).  Otherwise a separating anchor is searched for offline
        with find_separating_anchor (no oracle query) and exactly one extra
        session is spent verifying its predicted multiset against the oracle.
        """
        d = int(W_rec.shape[1])
        info = {"probe_from_lta_anchor": False, "probe_search_steps": 0,
                "probe_colliding_rows": None, "probe_unseparable_rows": None,
                "probe_tie_free": False, "probe_verified": None,
                "probe_extra_session": False}
        xc = None
        if lta_anchor is not None:
            pred = W_rec @ np.asarray(lta_anchor, dtype=np.int64) + b_rec
            if np.unique(pred).size == pred.size:
                xc = np.asarray(lta_anchor, dtype=np.int64)
                info["probe_from_lta_anchor"] = True
                info["probe_tie_free"] = True
                info["probe_colliding_rows"] = 0
                info["probe_unseparable_rows"] = 0
                if lta_anchor_response is not None:
                    info["probe_verified"] = bool(np.array_equal(
                        np.sort(pred), np.sort(lta_anchor_response)))
        if xc is None:
            xc, bad, steps, floor_bad = find_separating_anchor(
                W_rec, b_rec, self.A, self.rng)
            info["probe_search_steps"] = int(steps)
            info["probe_colliding_rows"] = int(bad)
            info["probe_unseparable_rows"] = int(floor_bad)
            info["probe_tie_free"] = bool(bad == 0)
            if bad > floor_bad:
                self.probe_distinctness_violations += 1
            y = self.run_attack_session(xc)
            self.extra_probe_sessions += 1
            info["probe_extra_session"] = True
            info["probe_verified"] = bool(np.array_equal(
                np.sort(W_rec @ xc + b_rec), np.sort(y)))
        pred = W_rec @ xc + b_rec
        order = np.argsort(pred, kind="stable")
        self.state.append({"probe_xc": xc, "pred": pred,
                           "pred_sorted": pred[order], "pred_order": order,
                           "ambiguous": bool(np.unique(pred).size != pred.size),
                           "W": W_rec, "b": b_rec})
        self.r += 1
        return info

    def uninstall_last_probe(self):
        """Discard the most recently installed probe (step-4 detect-and-retry).

        Additive helper: it only undoes install_probe's two state changes, so code
        that never calls it behaves exactly as before.
        """
        if not self.state:
            return False
        self.state.pop()
        self.r -= 1
        return True

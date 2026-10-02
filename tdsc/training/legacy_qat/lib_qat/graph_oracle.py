"""Round-graph session oracle for networks with residual additions (step 4).

New module; nothing here changes the behaviour of lib/session.py or lib/extract.py.

A network is a list of RoundSpec.  One round is one server round, i.e. one client
query and one ordered reply.  A round may read several inputs: vectors the client
supplies in this round ("new") and vectors the client supplied at an earlier round
that the server retained ("ret").  Rounds that belong to the same permutation
group share one sigma inside a session, which is how the branchwise residual mode
(R2) makes client-side addition well defined.

Frames.  A "new" input declares the group whose output frame it lives in (-1 for
the network input, which is in the true frame) and how many spatial slots each
channel occupies (9 for a 3x3 patch, 1 for a 1x1 branch or a linear layer).  The
server un-permutes it with expand(sigma_group, slots), exactly as in lib/session.
An input may instead be declared frame-invariant ("const"), i.e. a vector whose
entries are all equal; un-permuting such a vector is the identity, so the client
can use it without knowing the frame.
"""
import time
import numpy as np

from .session import expand, apply_perm, unapply_perm


class RoundSpec:
    """One server round.  inputs[i] is a dict with
        kind:  "new" or "ret"
        W:     (m, d) integer matrix applied to that input
        d:     input dimension
        frame: group id whose output frame a "new" input lives in (-1 = true frame)
        slots: channels-to-coordinates expansion of a "new" input
        src:   (round index, input index) of the retained vector, for kind "ret"
    """

    def __init__(self, name, group, m, b, inputs, unshuffled=False, role=""):
        self.name = name
        self.group = int(group)
        self.m = int(m)
        self.b = np.asarray(b, dtype=np.int64)
        self.inputs = inputs
        self.unshuffled = bool(unshuffled)
        self.role = role
        self.n_new = sum(1 for s in inputs if s["kind"] == "new")


class GraphOracle:
    def __init__(self, rounds, A, noise_fn, rng):
        self.rounds = rounds
        self.A = int(A)
        self.noise_fn = noise_fn
        self.rng = rng
        self.R = len(rounds)
        self.groups = sorted(set(r.group for r in rounds))
        self.group_m = {}
        self.group_unshuffled = {}
        for r in rounds:
            self.group_m[r.group] = r.m
            self.group_unshuffled[r.group] = (
                self.group_unshuffled.get(r.group, True) and r.unshuffled)
        self.n_sessions = 0
        self.n_round_evaluations = 0
        self.n_rounding_failures = 0
        self.inadmissible_entries = 0
        self._cache = [dict() for _ in range(self.R)]
        self._seen = [dict() for _ in range(self.R)]
        self.cache_hits = 0

    @property
    def rounding_exact(self):
        return self.n_rounding_failures == 0

    def eval_round(self, r, true_vecs):
        """Deterministic map (true-frame inputs) -> sum_i W_i x_i + b, memoised.
        The session randomness is drawn outside this cache."""
        key = b"|".join(v.tobytes() for v in true_vecs)
        hit = self._cache[r].get(key)
        if hit is not None:
            self.cache_hits += 1
            return hit
        spec = self.rounds[r]
        v = spec.b.copy()
        for s, x in zip(spec.inputs, true_vecs):
            v = v + s["W"] @ x
        seen = self._seen[r]
        if seen.pop(key, 0):
            self._cache[r][key] = v
        else:
            if len(seen) > 2048:
                seen.clear()
            seen[key] = 1
        return v

    def session(self):
        self.n_sessions += 1
        return GraphSession(self)


class GraphSession:
    def __init__(self, oracle):
        self.o = oracle
        self.sigmas = {}
        for g in oracle.groups:
            m = oracle.group_m[g]
            self.sigmas[g] = (np.arange(m, dtype=np.int64)
                              if oracle.group_unshuffled[g]
                              else oracle.rng.permutation(m))
        self.r = 0
        self.stored = {}

    def round(self, vectors):
        """vectors: one client-frame vector per "new" input of the current round."""
        o = self.o
        r = self.r
        if r >= o.R:
            raise RuntimeError("session already used all %d rounds" % o.R)
        spec = o.rounds[r]
        vectors = list(vectors)
        if len(vectors) != spec.n_new:
            raise ValueError("round %d expects %d new inputs, got %d"
                             % (r, spec.n_new, len(vectors)))
        true_vecs = []
        k = 0
        for i, s in enumerate(spec.inputs):
            if s["kind"] == "new":
                x = np.asarray(vectors[k], dtype=np.int64)
                k += 1
                o.inadmissible_entries += int(np.sum((x < 0) | (x > o.A)))
                g = s["frame"]
                if g < 0:
                    xt = x
                else:
                    xt = unapply_perm(expand(self.sigmas[g], s["slots"]), x)
                self.stored[(r, i)] = xt
                true_vecs.append(xt)
            else:
                true_vecs.append(self.stored[s["src"]])
        v = o.eval_round(r, true_vecs)
        o.n_round_evaluations += 1
        perm = self.sigmas[spec.group]
        permuted = v[perm]
        obs = permuted.astype(np.longdouble) + o.noise_fn(permuted.size)
        y = np.rint(obs).astype(np.int64)
        if not np.array_equal(y, permuted):
            o.n_rounding_failures += 1
        self.r += 1
        return y

    def round_multi(self, vector_sets):
        """One round carrying several independent probes (the packed feature-map
        query of step 4, sub-problem B): every set is evaluated under this
        session's permutation and one ordered reply per set is returned.  The
        round is consumed once, because it is one client query."""
        o = self.o
        r = self.r
        if r >= o.R:
            raise RuntimeError("session already used all %d rounds" % o.R)
        spec = o.rounds[r]
        outs = []
        first = True
        perm = self.sigmas[spec.group]
        for vectors in vector_sets:
            vectors = list(vectors)
            true_vecs = []
            k = 0
            for i, s in enumerate(spec.inputs):
                if s["kind"] == "new":
                    x = np.asarray(vectors[k], dtype=np.int64)
                    k += 1
                    o.inadmissible_entries += int(np.sum((x < 0) | (x > o.A)))
                    g2 = s["frame"]
                    xt = x if g2 < 0 else unapply_perm(
                        expand(self.sigmas[g2], s["slots"]), x)
                    if first:
                        self.stored[(r, i)] = xt
                    true_vecs.append(xt)
                else:
                    true_vecs.append(self.stored[s["src"]])
            v = o.eval_round(r, true_vecs)
            o.n_round_evaluations += 1
            permuted = v[perm]
            obs = permuted.astype(np.longdouble) + o.noise_fn(permuted.size)
            y = np.rint(obs).astype(np.int64)
            if not np.array_equal(y, permuted):
                o.n_rounding_failures += 1
            outs.append(y)
            first = False
        self.r += 1
        return outs



class GraphAttack:
    """Frame-tracked attack over a round graph (step 4, sub-problem A)."""

    def __init__(self, oracle, A, rng):
        self.o = oracle
        self.A = int(A)
        self.rng = rng
        self.probes = {}          # round index -> probe record
        self.group_probe = {}     # group id -> round index
        self.probe_rounds = 0
        self.probe_failures = 0
        self.extra_sessions = 0
        self.labelled_reads = 0
        self.frame_unavailable = 0

    # ---------------- sessions ----------------
    def _build_vectors(self, q, plans, pi):
        spec = self.o.rounds[q]
        vecs = []
        k = 0
        for s in spec.inputs:
            if s["kind"] != "new":
                continue
            p = plans[k]
            k += 1
            if p[0] == "zero":
                vecs.append(np.zeros(s["d"], dtype=np.int64))
            elif p[0] == "const":
                vecs.append(np.full(s["d"], int(p[1]), dtype=np.int64))
            elif p[0] == "canon":
                xc = np.asarray(p[1], dtype=np.int64)
                g = s["frame"]
                if g < 0:
                    vecs.append(xc)
                else:
                    if g not in pi:
                        # an earlier probe did not resolve its frame (its predicted
                        # multiset did not match): degrade to zeros and record it
                        # rather than aborting, so the damage is measured.
                        self.frame_unavailable += 1
                        vecs.append(np.zeros(s["d"], dtype=np.int64))
                    else:
                        vecs.append(apply_perm(expand(pi[g], s["slots"]), xc))
            else:
                raise ValueError("unknown input plan %r" % (p,))
        return vecs

    def run_session(self, schedule, observe, return_pi=False):
        """One session.  schedule maps a round index to a list of input plans; any
        round not in the schedule sends its installed probe (if it has one and
        comes before `observe`) and otherwise zeros.  Returns round `observe`'s
        ordered response, or (response, pi) when return_pi is set."""
        sess = self.o.session()
        pi = {}
        out = None
        for q in range(self.o.R):
            spec = self.o.rounds[q]
            plans = schedule.get(q)
            use_probe = False
            if plans is None:
                if q in self.probes and q < observe:
                    plans = [("canon", v) for v in self.probes[q]["vectors"]]
                    use_probe = True
                else:
                    plans = [("zero",)] * spec.n_new
            y = sess.round(self._build_vectors(q, plans, pi))
            if q == observe:
                out = y
            if use_probe:
                self.probe_rounds += 1
                pr = self.probes[q]
                ys = np.argsort(y, kind="stable")
                if not bool(np.array_equal(y[ys], pr["pred_sorted"])):
                    self.probe_failures += 1
                else:
                    p = np.empty(y.size, dtype=np.int64)
                    p[ys] = pr["pred_order"]
                    pi[spec.group] = p
        return (out, pi) if return_pi else out


    def run_session_multi(self, schedule, observe, vary_round, vary_input, vecs):
        """One session in which round `observe` carries len(vecs) packed probes on
        input `vary_input` (the packed round must be the observed round)."""
        if vary_round != observe:
            raise ValueError("packed probes require vary_round == observe")
        sess = self.o.session()
        pi = {}
        outs = None
        for q in range(self.o.R):
            spec = self.o.rounds[q]
            plans = schedule.get(q)
            use_probe = False
            if plans is None:
                if q in self.probes and q < observe:
                    plans = [("canon", v) for v in self.probes[q]["vectors"]]
                    use_probe = True
                else:
                    plans = [("zero",)] * spec.n_new
            if q == observe:
                sets = []
                for xv in vecs:
                    pl = list(plans)
                    pl[vary_input] = ("canon", xv)
                    sets.append(self._build_vectors(q, pl, pi))
                outs = sess.round_multi(sets)
                continue
            y = sess.round(self._build_vectors(q, plans, pi))
            if use_probe:
                self.probe_rounds += 1
                pr = self.probes[q]
                ys = np.argsort(y, kind="stable")
                if not bool(np.array_equal(y[ys], pr["pred_sorted"])):
                    self.probe_failures += 1
                else:
                    p = np.empty(y.size, dtype=np.int64)
                    p[ys] = pr["pred_order"]
                    pi[spec.group] = p
        return outs

    # ---------------- probes ----------------
    def install_probe(self, q, vectors, pred):
        """Install round q as the frame probe of its group."""
        pred = np.asarray(pred, dtype=np.int64)
        order = np.argsort(pred, kind="stable")
        self.probes[q] = {"vectors": [np.asarray(v, dtype=np.int64) for v in vectors],
                          "pred": pred, "pred_sorted": pred[order],
                          "pred_order": order,
                          "tie_free": bool(np.unique(pred).size == pred.size)}
        self.group_probe[self.o.rounds[q].group] = q
        return self.probes[q]["tie_free"]


class GraphChannel:
    """lib.lta.Channel-compatible adapter: one LTA query is one session in which
    input `vary_input` of round `vary_round` carries the query vector (in canonical
    coordinates) and round `observe` is read."""

    def __init__(self, W_eff, b_eff, A, attack, vary_round, vary_input, observe,
                 schedule, embed=None):
        self.W = np.ascontiguousarray(W_eff, dtype=np.int64)
        self.b = np.ascontiguousarray(b_eff, dtype=np.int64)
        self.m, self.d = self.W.shape
        self.A = int(A)
        self.attack = attack
        self.vary_round = int(vary_round)
        self.vary_input = int(vary_input)
        self.observe = int(observe)
        self.schedule = {k: list(v) for k, v in schedule.items()}
        self.embed = embed      # optional map query -> the round's full input vector
        self.n_queries = 0
        self.inadmissible_query_entries = 0
        self.backend = None
        self.anchor_responses = []
        self._base = None
        self._x0 = None

    def exact(self, x):
        return self.W @ np.asarray(x, dtype=np.int64) + self.b

    def _check_admissible(self, x):
        bad = int(np.sum((x < 0) | (x > self.A)))
        self.inadmissible_query_entries += bad
        return bad

    def query(self, x):
        x = np.asarray(x, dtype=np.int64)
        self._check_admissible(x)
        self.n_queries += 1
        sched = {k: list(v) for k, v in self.schedule.items()}
        plans = list(sched[self.vary_round])
        plans[self.vary_input] = ("canon", x if self.embed is None else self.embed(x))
        sched[self.vary_round] = plans
        return self.attack.run_session(sched, self.observe)

    def prepare_anchor(self, x0):
        x0 = np.asarray(x0, dtype=np.int64)
        self._x0 = x0
        self._base = self.exact(x0)

    def set_anchor(self, x0):
        x0 = np.asarray(x0, dtype=np.int64)
        self._check_admissible(x0)
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
        return self.attack.o.rounding_exact

    @property
    def n_rounding_failures(self):
        return self.attack.o.n_rounding_failures


def labelled_readout(attack, vary_round, input_index, observe, schedule, d, m):
    """Recover one matrix when the observed round's group already has a probe at an
    EARLIER round of the same group: that probe fixes the frame inside the session,
    so every reply entry carries a canonical row label and the matrix can be read
    column by column with d + 1 sessions and no anchor, no ties and no cover.

    Returns (W (m x d) in canonical row order, b (m,), n_sessions, n_unlabelled).
    """
    W = np.zeros((m, d), dtype=np.int64)
    b = np.zeros(m, dtype=np.int64)
    n_sessions = 0
    unlabelled = 0
    g = attack.o.rounds[observe].group

    def one(xc):
        nonlocal n_sessions, unlabelled
        sched = {k: list(v) for k, v in schedule.items()}
        plans = list(sched[vary_round])
        plans[input_index] = ("canon", xc)
        sched[vary_round] = plans
        y, pi = attack.run_session(sched, observe, return_pi=True)
        n_sessions += 1
        if g not in pi:
            unlabelled += 1
            return None
        out = np.empty(y.size, dtype=np.int64)
        out[pi[g]] = y
        attack.labelled_reads += 1
        return out

    z = one(np.zeros(d, dtype=np.int64))
    if z is None:
        return None, None, n_sessions, unlabelled
    b[:] = z
    for j in range(d):
        e = np.zeros(d, dtype=np.int64)
        e[j] = 1
        v = one(e)
        if v is None:
            return None, None, n_sessions, unlabelled
        W[:, j] = v - b
    return W, b, n_sessions, unlabelled

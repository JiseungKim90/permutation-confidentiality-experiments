"""Verification for step 3 (plan Section 3)."""
from collections import Counter
import numpy as np


def canonicalise(W_rec, b_rec):
    """Canonical row order = lexicographic order of the recovered rows [W|b]."""
    M = np.concatenate([W_rec, b_rec[:, None]], axis=1)
    order = np.lexsort(M.T[::-1])
    return W_rec[order], b_rec[order]


def row_exactness(W_rec, b_rec, W_true, b_true):
    """Exactness up to a row permutation, as in step 2b."""
    M_rec = np.concatenate([W_rec, b_rec[:, None]], axis=1)
    M_true = np.concatenate([W_true, b_true[:, None]], axis=1)
    o1 = np.lexsort(M_rec.T[::-1])
    o2 = np.lexsort(M_true.T[::-1])
    diff = np.abs(M_rec[o1] - M_true[o2])
    max_err = int(diff.max()) if diff.size else 0
    c1 = Counter(M_rec[i].tobytes() for i in range(M_rec.shape[0]))
    c2 = Counter(M_true[i].tobytes() for i in range(M_true.shape[0]))
    inter = sum(min(v, c2.get(k, 0)) for k, v in c1.items())
    return max_err, int(inter)


def row_permutation(W_rec, b_rec, W_true, b_true):
    """P with recovered row c equal to true row P[c], by exact row matching.

    Returns (P, n_matched).  Unmatched recovered rows are assigned the remaining
    true indices in order, so P is always a permutation; n_matched < m signals
    that the extracted layer is not a row permutation of the true one.
    """
    m = W_rec.shape[0]
    buckets = {}
    for k in range(m):
        key = W_true[k].tobytes() + b_true[k].tobytes()
        buckets.setdefault(key, []).append(k)
    P = np.full(m, -1, dtype=np.int64)
    for c in range(m):
        key = W_rec[c].tobytes() + b_rec[c].tobytes()
        lst = buckets.get(key)
        if lst:
            P[c] = lst.pop()
    n_matched = int(np.sum(P >= 0))
    free = [k for k in range(m) if k not in set(P[P >= 0].tolist())]
    for c in range(m):
        if P[c] < 0:
            P[c] = free.pop()
    return P, n_matched


def frame_check(W_rec, b_rec, W_true_canon_cols, b_true, P):
    """Explicit composed-permutation check W~_r = P_r W_r P_{r-1}^T.

    W_true_canon_cols is W_r with its columns already written in the canonical
    frame of layer r-1, so the residual claim is W~_r = P_r (that matrix).
    """
    ok_W = bool(np.array_equal(W_rec, W_true_canon_cols[P]))
    ok_b = bool(np.array_equal(b_rec, b_true[P]))
    err = int(np.abs(W_rec - W_true_canon_cols[P]).max()) if W_rec.size else 0
    errb = int(np.abs(b_rec - b_true[P]).max()) if b_rec.size else 0
    return bool(ok_W and ok_b), max(err, errb)


def canonical_column_matrix(W_true, colmap, slots):
    """W_r with columns rewritten in the canonical frame of layer r-1:
    column (c, slot) of the result is column (colmap[c], slot) of W_r."""
    if colmap is None:
        return np.array(W_true, dtype=np.int64, copy=True)
    colmap = np.asarray(colmap, dtype=np.int64)
    if slots == 1:
        idx = colmap
    else:
        idx = (colmap[:, None] * slots + np.arange(slots, dtype=np.int64)).ravel()
    return np.ascontiguousarray(W_true[:, idx])


def end_to_end(true_chain, ext_chain, X):
    """Compare integer logits of the true and extracted chains."""
    Yt, acc_t = true_chain.forward_int(X, return_acc_max=True)
    Ye, acc_e = ext_chain.forward_int(X, return_acc_max=True)
    same_rows = np.all(Yt == Ye, axis=1)
    return {
        "n_inputs": int(X.shape[0]),
        "identical_logit_vectors": int(np.sum(same_rows)),
        "identical_logit_fraction": float(np.mean(same_rows)),
        "identical_argmax": int(np.sum(np.argmax(Yt, axis=1) == np.argmax(Ye, axis=1))),
        "identical_argmax_fraction": float(
            np.mean(np.argmax(Yt, axis=1) == np.argmax(Ye, axis=1))),
        "max_abs_logit_difference": int(np.abs(Yt - Ye).max()) if Yt.size else 0,
        "accumulator_absmax_true": int(acc_t),
        "accumulator_absmax_extracted": int(acc_e),
    }

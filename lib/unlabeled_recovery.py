"""Exact row recovery from independently shuffled affine-layer transcripts.

The oracle returns a fresh permutation of ``W @ x + b`` for every query.  All
values in this module are integer lattice indices, i.e., the result after the
protocol's fixed-point output has been rounded to the nearest lattice point.
This makes the separation assumptions and collision failures explicit.

Two recovery modes are provided:

* ``recover_bias_anchored`` uses distinct biases as persistent row anchors.
* ``recover_bias_free`` uses one weight column with distinct entries as the
  anchor and aligns every other column with mixed-coordinate queries.

Both algorithms reject non-unique matchings instead of silently selecting one.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, Mapping, Sequence, Tuple

import numpy as np


class RecoveryError(RuntimeError):
    """Base class for transcript-recovery failures."""


class AmbiguousRecoveryError(RecoveryError):
    """Raised when the observations admit more than one aligned model."""


class InconsistentTranscriptError(RecoveryError):
    """Raised when no model satisfies all observed multisets."""


@dataclass(frozen=True)
class RecoveryResult:
    """Recovered affine layer in a canonical row order."""

    weight: np.ndarray
    bias: np.ndarray
    queries: int
    anchor_coordinate: int | None
    coefficients: Tuple[int, ...]


class FreshPermutationOracle:
    """Simulation of a rounded affine transcript with fresh output shuffling.

    ``weight`` and ``bias`` must be integer lattice indices.  Optional noise is
    measured in lattice units and must have magnitude strictly below 1/2; the
    receiver rounds before attempting recovery.
    """

    def __init__(
        self,
        weight: np.ndarray,
        bias: np.ndarray | None = None,
        *,
        seed: int = 0,
        noise_bound: float = 0.0,
    ) -> None:
        w = np.asarray(weight, dtype=np.int64)
        if w.ndim != 2:
            raise ValueError("weight must be a rank-two matrix")
        if bias is None:
            b = np.zeros(w.shape[0], dtype=np.int64)
        else:
            b = np.asarray(bias, dtype=np.int64)
        if b.shape != (w.shape[0],):
            raise ValueError("bias has incompatible shape")
        if not 0.0 <= noise_bound < 0.5:
            raise ValueError("noise_bound must satisfy 0 <= bound < 1/2")
        self.weight = w
        self.bias = b
        self.noise_bound = float(noise_bound)
        self.rng = np.random.default_rng(seed)
        self.query_count = 0

    def query(self, x: Sequence[int]) -> np.ndarray:
        vec = np.asarray(x, dtype=np.int64)
        if vec.shape != (self.weight.shape[1],):
            raise ValueError("query has incompatible shape")
        exact = self.weight @ vec + self.bias
        if self.noise_bound:
            noise = self.rng.uniform(
                -self.noise_bound, self.noise_bound, size=exact.shape
            )
            observed = np.rint(exact.astype(np.float64) + noise).astype(np.int64)
        else:
            observed = exact.copy()
        self.query_count += 1
        return observed[self.rng.permutation(observed.size)]


QueryFunction = Callable[[Sequence[int]], np.ndarray]


def _as_counter(values: Iterable[int]) -> Counter[int]:
    return Counter(int(v) for v in values)


def _unique_multiset_matching(
    anchors: Sequence[int],
    observations: Mapping[int, Sequence[int]],
    *,
    max_nodes: int = 2_000_000,
) -> np.ndarray:
    """Find the unique vector ``c`` with multisets ``anchor + lambda*c``.

    ``observations[lambda]`` is the shuffled response multiset.  The search
    consumes multiplicities exactly and stops as soon as two distinct aligned
    vectors are found.
    """

    coeffs = tuple(sorted(int(lam) for lam in observations))
    if not coeffs or coeffs[0] <= 0:
        raise ValueError("positive coefficients are required")
    anchors_arr = np.asarray(anchors, dtype=np.int64)
    m = anchors_arr.size
    counters: Dict[int, Counter[int]] = {
        lam: _as_counter(observations[lam]) for lam in coeffs
    }
    if any(sum(counter.values()) != m for counter in counters.values()):
        raise ValueError("every observation must contain one value per row")

    # Candidate c values can be derived from any response.  Using the smallest
    # coefficient also avoids division and keeps the search exactly integral.
    seed_lam = coeffs[0]
    candidate_values: Dict[int, Tuple[int, ...]] = {}
    for idx, anchor in enumerate(anchors_arr):
        vals = []
        for observed in counters[seed_lam]:
            delta = int(observed) - int(anchor)
            if delta % seed_lam:
                continue
            c = delta // seed_lam
            if all(counters[lam][int(anchor) + lam * c] > 0 for lam in coeffs):
                vals.append(c)
        candidate_values[idx] = tuple(sorted(set(vals)))

    assigned: Dict[int, int] = {}
    solutions: set[Tuple[int, ...]] = set()
    nodes = 0

    def feasible(index: int) -> Tuple[int, ...]:
        anchor = int(anchors_arr[index])
        return tuple(
            c
            for c in candidate_values[index]
            if all(counters[lam][anchor + lam * c] > 0 for lam in coeffs)
        )

    def dfs() -> None:
        nonlocal nodes
        if len(solutions) >= 2:
            return
        nodes += 1
        if nodes > max_nodes:
            raise AmbiguousRecoveryError(
                f"matching search exceeded {max_nodes:,} nodes"
            )
        if len(assigned) == m:
            if all(not +(counter) for counter in counters.values()):
                solutions.add(tuple(assigned[i] for i in range(m)))
            return

        unassigned = [i for i in range(m) if i not in assigned]
        choices = [(feasible(i), i) for i in unassigned]
        choices.sort(key=lambda item: (len(item[0]), item[1]))
        values, index = choices[0]
        if not values:
            return
        anchor = int(anchors_arr[index])
        for c in values:
            keys = {lam: anchor + lam * c for lam in coeffs}
            for lam, key in keys.items():
                counters[lam][key] -= 1
                if counters[lam][key] == 0:
                    del counters[lam][key]
            assigned[index] = c
            dfs()
            del assigned[index]
            for lam, key in keys.items():
                counters[lam][key] += 1
            if len(solutions) >= 2:
                return

    dfs()
    if not solutions:
        raise InconsistentTranscriptError("no aligned model fits the transcript")
    if len(solutions) > 1:
        raise AmbiguousRecoveryError("transcript admits multiple aligned models")
    return np.asarray(next(iter(solutions)), dtype=np.int64)


def _checked_coefficients(coefficients: Sequence[int]) -> Tuple[int, ...]:
    coeffs = tuple(sorted(set(int(v) for v in coefficients)))
    if not coeffs or coeffs[0] <= 0:
        raise ValueError("coefficients must be distinct positive integers")
    return coeffs


def recover_bias_anchored(
    query: QueryFunction,
    *,
    output_dim: int,
    input_dim: int,
    coefficients: Sequence[int] = (1, 2, 3),
) -> RecoveryResult:
    """Recover ``W,b`` when the rounded biases are pairwise distinct.

    Query count is ``1 + input_dim * len(coefficients)``.  The output rows are
    ordered by increasing bias, which fixes the unavoidable global row-label
    symmetry of an independently shuffled transcript.
    """

    coeffs = _checked_coefficients(coefficients)
    zero = np.zeros(input_dim, dtype=np.int64)
    anchors = np.sort(np.asarray(query(zero), dtype=np.int64))
    if anchors.shape != (output_dim,):
        raise ValueError("oracle returned an incompatible output")
    if np.unique(anchors).size != output_dim:
        raise AmbiguousRecoveryError("bias anchors are not pairwise distinct")

    recovered = np.empty((output_dim, input_dim), dtype=np.int64)
    for column in range(input_dim):
        observations: Dict[int, np.ndarray] = {}
        for lam in coeffs:
            x = np.zeros(input_dim, dtype=np.int64)
            x[column] = lam
            observations[lam] = np.asarray(query(x), dtype=np.int64)
        recovered[:, column] = _unique_multiset_matching(anchors, observations)

    return RecoveryResult(
        weight=recovered,
        bias=anchors,
        queries=1 + input_dim * len(coeffs),
        anchor_coordinate=None,
        coefficients=coeffs,
    )


def recover_bias_free(
    query: QueryFunction,
    *,
    output_dim: int,
    input_dim: int,
    anchor_coordinate: int = 0,
    coefficients: Sequence[int] = (1, 2, 3),
) -> RecoveryResult:
    """Recover a bias-free ``W`` using a distinct reference column.

    For every non-anchor column j, the algorithm observes the column multiset
    with query ``e_j`` and the mixed multisets from ``e_r + lambda e_j``.
    Query count is ``1 + (input_dim-1) * (1 + len(coefficients))``.
    """

    coeffs = _checked_coefficients(coefficients)
    if not 0 <= anchor_coordinate < input_dim:
        raise ValueError("anchor_coordinate is out of range")
    e_anchor = np.zeros(input_dim, dtype=np.int64)
    e_anchor[anchor_coordinate] = 1
    anchors = np.sort(np.asarray(query(e_anchor), dtype=np.int64))
    if anchors.shape != (output_dim,):
        raise ValueError("oracle returned an incompatible output")
    if np.unique(anchors).size != output_dim:
        raise AmbiguousRecoveryError("reference column is not pairwise distinct")

    recovered = np.empty((output_dim, input_dim), dtype=np.int64)
    recovered[:, anchor_coordinate] = anchors
    for column in range(input_dim):
        if column == anchor_coordinate:
            continue
        e_column = np.zeros(input_dim, dtype=np.int64)
        e_column[column] = 1
        column_values = np.asarray(query(e_column), dtype=np.int64)
        observations: Dict[int, np.ndarray] = {}
        for lam in coeffs:
            x = e_anchor.copy()
            x[column] = lam
            observations[lam] = np.asarray(query(x), dtype=np.int64)

        # Matching consumes mixed responses.  Restrict candidate weights to the
        # separately observed column multiset by adding a zero-anchor response.
        matched = _unique_multiset_matching(anchors, observations)
        if Counter(int(v) for v in matched) != Counter(int(v) for v in column_values):
            raise InconsistentTranscriptError(
                f"mixed-query solution for column {column} disagrees with e_j"
            )
        recovered[:, column] = matched

    return RecoveryResult(
        weight=recovered,
        bias=np.zeros(output_dim, dtype=np.int64),
        queries=1 + (input_dim - 1) * (1 + len(coeffs)),
        anchor_coordinate=anchor_coordinate,
        coefficients=coeffs,
    )


def row_canonicalize(weight: np.ndarray, anchor: np.ndarray) -> np.ndarray:
    """Return rows sorted by an anchor vector (used only for verification)."""

    w = np.asarray(weight)
    a = np.asarray(anchor)
    if w.shape[0] != a.size:
        raise ValueError("anchor has incompatible length")
    if np.unique(a).size != a.size:
        raise ValueError("anchor must be pairwise distinct")
    return w[np.argsort(a, kind="stable")]

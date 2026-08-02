"""Deterministic row recovery by exact polynomial interpolation.

For distinct anchors a_i and an unknown column c_i, a shuffled query at
coefficient k reveals the multiset {a_i + k c_i}.  Define

    F(t, k) = product_i (t - a_i - k c_i).

For fixed t this is a degree-at-most-m polynomial in k.  The observations at
k=0,...,m determine its derivative at zero.  Evaluating at t=a_i gives

    dF/dk(a_i, 0) = -c_i product_{r != i}(a_i-a_r),

so every c_i is recovered exactly.  This removes the empirical
unique-matching assumption at the cost of m probes per recovered column.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from math import comb, prod
from typing import Callable, Mapping, Sequence, Tuple

import numpy as np


class InterpolationRecoveryError(RuntimeError):
    """Raised when an exact interpolation transcript is malformed."""


@dataclass(frozen=True)
class InterpolationRecoveryResult:
    """Recovered affine layer in increasing-anchor row order."""

    weight: np.ndarray
    bias: np.ndarray
    queries: int
    anchor_coordinate: int | None
    coefficients: Tuple[int, ...]


QueryFunction = Callable[[Sequence[int]], np.ndarray]


def derivative_weights_at_zero(degree: int) -> Tuple[Fraction, ...]:
    """Lagrange weights for f'(0) from f(0),...,f(degree).

    The identity is exact for every polynomial of degree at most degree.
    """

    if degree < 1:
        raise ValueError("degree must be positive")
    harmonic = sum((Fraction(1, k) for k in range(1, degree + 1)), Fraction())
    weights = [-harmonic]
    for k in range(1, degree + 1):
        weights.append(Fraction(((-1) ** (k + 1)) * comb(degree, k), k))
    return tuple(weights)


def _checked_multiset(values: Sequence[int], output_dim: int) -> Tuple[int, ...]:
    arr = np.asarray(values)
    if arr.shape != (output_dim,):
        raise InterpolationRecoveryError("oracle returned an incompatible output")
    rounded = np.rint(arr).astype(np.int64)
    if not np.array_equal(arr, rounded):
        raise InterpolationRecoveryError("transcript must contain rounded lattice indices")
    return tuple(int(v) for v in rounded)


def recover_vector_by_interpolation(
    anchors: Sequence[int],
    observations: Mapping[int, Sequence[int]],
) -> np.ndarray:
    """Recover c from multisets observations[k] = ms(a + k*c).

    Observations must contain all integer coefficients 0,...,m, where m is
    the number of anchors.  Multiplicities are retained through the product
    polynomial; no matching or generic-position assumption is used.
    """

    a = tuple(int(v) for v in anchors)
    m = len(a)
    if len(set(a)) != m:
        raise InterpolationRecoveryError("anchors must be pairwise distinct")
    expected = set(range(m + 1))
    if set(int(k) for k in observations) != expected:
        raise InterpolationRecoveryError(
            f"observations must contain coefficients 0,...,{m}"
        )
    rows = {
        k: _checked_multiset(observations[k], m)
        for k in range(m + 1)
    }
    if sorted(rows[0]) != sorted(a):
        raise InterpolationRecoveryError("coefficient-zero response disagrees with anchors")

    weights = derivative_weights_at_zero(m)
    recovered = []
    for i, anchor in enumerate(a):
        denominator = prod(anchor - other for j, other in enumerate(a) if j != i)
        if denominator == 0:
            raise InterpolationRecoveryError("anchor derivative denominator is zero")
        derivative = Fraction()
        for k in range(m + 1):
            value = prod(anchor - observed for observed in rows[k])
            derivative += weights[k] * value
        coordinate = -derivative / denominator
        if coordinate.denominator != 1:
            raise InterpolationRecoveryError(
                f"nonintegral recovered coordinate for anchor index {i}: {coordinate}"
            )
        recovered.append(int(coordinate))
    return np.asarray(recovered, dtype=np.int64)


def recover_bias_anchored_interpolation(
    query: QueryFunction,
    *,
    output_dim: int,
    input_dim: int,
) -> InterpolationRecoveryResult:
    """Recover an affine layer using distinct bias anchors and m probes/column."""

    zero = np.zeros(input_dim, dtype=np.int64)
    anchors = np.sort(np.asarray(query(zero), dtype=np.int64))
    if anchors.shape != (output_dim,) or np.unique(anchors).size != output_dim:
        raise InterpolationRecoveryError("bias anchors are not pairwise distinct")
    coefficient_zero = tuple(int(v) for v in anchors)

    recovered = np.empty((output_dim, input_dim), dtype=np.int64)
    for column in range(input_dim):
        observations = {0: coefficient_zero}
        for coefficient in range(1, output_dim + 1):
            x = np.zeros(input_dim, dtype=np.int64)
            x[column] = coefficient
            observations[coefficient] = query(x)
        recovered[:, column] = recover_vector_by_interpolation(
            anchors, observations
        )

    return InterpolationRecoveryResult(
        weight=recovered,
        bias=anchors,
        queries=1 + output_dim * input_dim,
        anchor_coordinate=None,
        coefficients=tuple(range(1, output_dim + 1)),
    )


def recover_bias_free_interpolation(
    query: QueryFunction,
    *,
    output_dim: int,
    input_dim: int,
    anchor_coordinate: int = 0,
) -> InterpolationRecoveryResult:
    """Recover a bias-free layer from one distinct reference column."""

    if not 0 <= anchor_coordinate < input_dim:
        raise ValueError("anchor_coordinate is out of range")
    e_anchor = np.zeros(input_dim, dtype=np.int64)
    e_anchor[anchor_coordinate] = 1
    anchors = np.sort(np.asarray(query(e_anchor), dtype=np.int64))
    if anchors.shape != (output_dim,) or np.unique(anchors).size != output_dim:
        raise InterpolationRecoveryError("reference column is not pairwise distinct")
    coefficient_zero = tuple(int(v) for v in anchors)

    recovered = np.empty((output_dim, input_dim), dtype=np.int64)
    recovered[:, anchor_coordinate] = anchors
    for column in range(input_dim):
        if column == anchor_coordinate:
            continue
        observations = {0: coefficient_zero}
        for coefficient in range(1, output_dim + 1):
            x = e_anchor.copy()
            x[column] = coefficient
            observations[coefficient] = query(x)
        recovered[:, column] = recover_vector_by_interpolation(
            anchors, observations
        )

    return InterpolationRecoveryResult(
        weight=recovered,
        bias=np.zeros(output_dim, dtype=np.int64),
        queries=1 + output_dim * (input_dim - 1),
        anchor_coordinate=anchor_coordinate,
        coefficients=tuple(range(1, output_dim + 1)),
    )

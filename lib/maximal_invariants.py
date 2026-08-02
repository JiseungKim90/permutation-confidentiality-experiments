"""Maximal invariants and attacks for invertible linear obfuscation."""

from __future__ import annotations

import numpy as np


def rowspace_basis(
    observed: np.ndarray,
    *,
    relative_tolerance: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return an orthonormal row-space basis and retained singular values."""
    matrix = np.asarray(observed, dtype=np.float64)
    if matrix.ndim != 2 or min(matrix.shape) == 0:
        raise ValueError("observed must be a non-empty matrix")
    _, singular, vt = np.linalg.svd(matrix, full_matrices=False)
    if relative_tolerance is None:
        relative_tolerance = max(matrix.shape) * np.finfo(matrix.dtype).eps
    threshold = float(singular[0]) * float(relative_tolerance)
    rank = int(np.sum(singular > threshold))
    if rank == 0:
        return np.empty((0, matrix.shape[1]), dtype=np.float64), singular[:0]
    return vt[:rank], singular[:rank]


def rowspace_projector(basis: np.ndarray) -> np.ndarray:
    """Return the feature-side orthogonal projector of an orthonormal basis."""
    rows = np.asarray(basis, dtype=np.float64)
    if rows.ndim != 2:
        raise ValueError("basis must be two-dimensional")
    return rows.T @ rows


def row_residuals(candidates: np.ndarray, basis: np.ndarray) -> np.ndarray:
    """Return relative distance of every candidate row from the row space."""
    rows = np.asarray(candidates, dtype=np.float64)
    space = np.asarray(basis, dtype=np.float64)
    if rows.ndim < 2 or space.ndim != 2 or rows.shape[-1] != space.shape[-1]:
        raise ValueError("candidate and basis feature dimensions must agree")
    norms_sq = np.sum(rows * rows, axis=-1)
    coordinates = np.einsum("...d,rd->...r", rows, space, optimize=True)
    projected_sq = np.sum(coordinates * coordinates, axis=-1)
    residual_sq = np.maximum(norms_sq - projected_sq, 0.0)
    denominator = np.maximum(norms_sq, np.finfo(np.float64).tiny)
    return np.sqrt(residual_sq / denominator)


def candidate_presence_scores(
    candidates: np.ndarray,
    basis: np.ndarray,
    *,
    sampled_rows: int,
) -> np.ndarray:
    """Score candidates by their sampled_rows smallest subspace residuals."""
    bank = np.asarray(candidates, dtype=np.float64)
    if bank.ndim != 3:
        raise ValueError("candidates must have shape (candidate, row, feature)")
    if sampled_rows <= 0 or sampled_rows > bank.shape[1]:
        raise ValueError("sampled_rows must be within the candidate row count")
    residual = row_residuals(bank, basis)
    smallest = np.partition(residual, sampled_rows - 1, axis=1)[:, :sampled_rows]
    return np.mean(smallest, axis=1)


def scale_permutation_signature(
    rows: np.ndarray,
    *,
    decimals: int | None = None,
) -> np.ndarray:
    """Complete signature for nonzero scaling and coordinate permutation."""
    values = np.asarray(rows, dtype=np.float64)
    if values.ndim < 2:
        raise ValueError("rows must have a feature dimension")
    magnitudes = np.abs(values)
    scale = np.max(magnitudes, axis=-1, keepdims=True)
    if np.any(scale == 0):
        raise ValueError("zero rows have no nonzero-scale signature")
    signature = np.sort(magnitudes / scale, axis=-1)
    if decimals is not None:
        signature = np.round(signature, decimals=decimals)
    return signature

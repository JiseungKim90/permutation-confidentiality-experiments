"""Utilities for replaying and attacking STIP's final cloud boundary.

The NDSS 2026 paper specifies the value released by the embedding TEE as
``alpha @ Emb(A) @ P``.  Here ``alpha`` is diagonal over token rows and ``P``
permutes hidden features.  The helpers below deliberately model only this
released boundary; they do not claim that the public STIP repository implements
the final TEE extension.
"""

from __future__ import annotations

import torch


def canonical_orbit(rows: torch.Tensor) -> torch.Tensor:
    """Return a unit-norm complete representative of a scale/permutation orbit.

    For every nonzero row ``v``, this computes the lexicographically smaller of
    ``sort(v / ||v||_inf)`` and ``sort(-v / ||v||_inf)``.  Consequently the
    output is unchanged by any nonzero row scale and feature permutation.
    """

    if rows.ndim < 2:
        raise ValueError("rows must have a feature dimension")
    shape = rows.shape
    flat = rows.detach().to(dtype=torch.float32, device="cpu").reshape(-1, shape[-1])
    scales = flat.abs().amax(dim=1, keepdim=True)
    if bool((scales == 0).any()):
        raise ValueError("zero rows have no nonzero scale/permutation orbit")

    ascending = torch.sort(flat / scales, dim=1).values
    negative = -torch.flip(ascending, dims=(1,))
    unresolved = torch.ones(ascending.shape[0], dtype=torch.bool)
    choose_negative = torch.zeros(ascending.shape[0], dtype=torch.bool)
    for column in range(ascending.shape[1]):
        smaller = negative[:, column] < ascending[:, column]
        larger = negative[:, column] > ascending[:, column]
        choose_negative[unresolved & smaller] = True
        unresolved &= ~(smaller | larger)
        if not bool(unresolved.any()):
            break

    representative = torch.where(choose_negative[:, None], negative, ascending)
    representative = torch.nn.functional.normalize(representative, dim=1)
    return representative.reshape(shape)


def apply_final_boundary(
    representations: torch.Tensor,
    row_scales: torch.Tensor,
    feature_permutation: torch.Tensor,
) -> torch.Tensor:
    """Apply the paper-specified ``alpha @ Emb(A) @ P`` transformation."""

    if representations.ndim != 3:
        raise ValueError("representations must have shape [batch, tokens, hidden]")
    if row_scales.shape != representations.shape[:2]:
        raise ValueError("row_scales must have shape [batch, tokens]")
    if feature_permutation.ndim != 1 or feature_permutation.numel() != representations.shape[-1]:
        raise ValueError("feature_permutation has the wrong dimension")
    if bool((row_scales == 0).any()):
        raise ValueError("STIP row scales must be nonzero")
    return row_scales[..., None] * representations[..., feature_permutation]


def invert_final_boundary(
    transcript: torch.Tensor,
    row_scales: torch.Tensor,
    feature_permutation: torch.Tensor,
) -> torch.Tensor:
    """Trusted inverse used only as a replay correctness control."""

    inverse = torch.argsort(feature_permutation)
    return (transcript / row_scales[..., None])[..., inverse]


def cosine_topk(
    queries: torch.Tensor,
    dictionary: torch.Tensor,
    k: int = 5,
    query_chunk: int = 128,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Exact cosine top-k search without an optional ANN dependency."""

    if queries.ndim != 2 or dictionary.ndim != 2:
        raise ValueError("queries and dictionary must be matrices")
    if queries.shape[1] != dictionary.shape[1]:
        raise ValueError("query and dictionary dimensions differ")
    if not 0 < k <= dictionary.shape[0]:
        raise ValueError("invalid k")

    q = torch.nn.functional.normalize(queries.float().cpu(), dim=1)
    d = torch.nn.functional.normalize(dictionary.float().cpu(), dim=1)
    all_scores: list[torch.Tensor] = []
    all_indices: list[torch.Tensor] = []
    for start in range(0, q.shape[0], query_chunk):
        scores = q[start : start + query_chunk] @ d.T
        values, indices = torch.topk(scores, k=k, dim=1, largest=True, sorted=True)
        all_scores.append(values)
        all_indices.append(indices)
    return torch.cat(all_scores), torch.cat(all_indices)

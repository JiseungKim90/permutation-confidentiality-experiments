from __future__ import annotations

import torch

from lib.stip_orbit import (
    apply_final_boundary,
    canonical_orbit,
    cosine_topk,
    invert_final_boundary,
)


def test_complete_invariance_and_recovery() -> None:
    generator = torch.Generator().manual_seed(20260802)
    vocabulary = torch.randn(64, 24, generator=generator)
    token_ids = torch.randint(0, vocabulary.shape[0], (5, 9), generator=generator)
    clean = vocabulary[token_ids]
    permutation = torch.randperm(clean.shape[-1], generator=generator)
    scales = torch.exp(torch.empty(5, 9).uniform_(-4, 4, generator=generator))
    scales[0, ::2] *= -1

    transcript = apply_final_boundary(clean, scales, permutation)
    recovered_clean = invert_final_boundary(transcript, scales, permutation)
    assert torch.allclose(recovered_clean, clean, atol=1e-6, rtol=1e-6)
    assert torch.allclose(
        canonical_orbit(transcript), canonical_orbit(clean), atol=2e-6, rtol=0
    )

    _, recovered = cosine_topk(
        canonical_orbit(transcript).reshape(-1, clean.shape[-1]),
        canonical_orbit(vocabulary),
        k=1,
    )
    assert torch.equal(recovered[:, 0].reshape_as(token_ids), token_ids)


def test_zero_row_is_rejected() -> None:
    try:
        canonical_orbit(torch.zeros(1, 8))
    except ValueError as error:
        assert "zero" in str(error)
    else:
        raise AssertionError("zero row was not rejected")

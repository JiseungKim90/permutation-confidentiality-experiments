import numpy as np

from lib.maximal_invariants import (
    candidate_presence_scores,
    row_residuals,
    rowspace_basis,
    rowspace_projector,
    scale_permutation_signature,
)


def random_invertible(size: int, rng: np.random.Generator) -> np.ndarray:
    q1, _ = np.linalg.qr(rng.standard_normal((size, size)))
    q2, _ = np.linalg.qr(rng.standard_normal((size, size)))
    singular = np.geomspace(0.2, 7.0, size)
    return q1 @ np.diag(singular) @ q2.T


def test_rowspace_is_invariant_under_arbitrary_left_mixing():
    rng = np.random.default_rng(7)
    hidden = rng.standard_normal((12, 31))
    mixed = random_invertible(12, rng) @ hidden
    basis_h, _ = rowspace_basis(hidden)
    basis_u, _ = rowspace_basis(mixed)
    assert np.allclose(
        rowspace_projector(basis_h),
        rowspace_projector(basis_u),
        atol=1e-10,
    )


def test_candidate_presence_survives_high_energy_shields():
    rng = np.random.default_rng(8)
    bank = rng.standard_normal((40, 9, 64))
    source_ids = np.array([3, 17, 31])
    sampled_rows = 5
    selected = np.concatenate([bank[i, :sampled_rows] for i in source_ids])
    shields = 25.0 * rng.standard_normal((7, 64))
    full = np.concatenate([selected, shields])
    mixed = random_invertible(full.shape[0], rng) @ full
    basis, _ = rowspace_basis(mixed)
    scores = candidate_presence_scores(
        bank,
        basis,
        sampled_rows=sampled_rows,
    )
    predicted = np.argsort(scores)[: len(source_ids)]
    assert set(predicted.tolist()) == set(source_ids.tolist())
    assert np.max(scores[source_ids]) < np.min(np.delete(scores, source_ids))


def test_full_column_rank_is_the_trivial_rowspace_phase():
    rng = np.random.default_rng(9)
    hidden = rng.standard_normal((24, 11))
    basis, _ = rowspace_basis(hidden)
    projector = rowspace_projector(basis)
    assert basis.shape[0] == 11
    assert np.allclose(projector, np.eye(11), atol=1e-10)
    candidates = rng.standard_normal((5, 4, 11))
    assert np.max(row_residuals(candidates, basis)) < 1e-7


def test_scale_permutation_signature_is_complete_on_examples():
    rng = np.random.default_rng(10)
    rows = rng.standard_normal((5, 23))
    permutations = np.stack([rng.permutation(23) for _ in range(5)])
    scales = np.array([-3.0, -2.0, -1.0, 2.0, 3.0])
    transformed = np.stack(
        [scales[i] * rows[i, permutations[i]] for i in range(5)]
    )
    assert np.allclose(
        scale_permutation_signature(rows),
        scale_permutation_signature(transformed),
    )


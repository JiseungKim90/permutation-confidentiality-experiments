from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


ARTIFACT = Path(__file__).resolve().parents[1]


def load_script(name: str):
    path = ARTIFACT / "scripts" / name
    spec = importlib.util.spec_from_file_location(name.replace(".py", ""), path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_trial_pools_are_disjoint_and_scale_paired():
    prepare = load_script("69_gelo_msmarco_prepare.py")
    trials = prepare.build_trials(120_000, 7, 20, 4)
    seen = set()
    for record in trials["records"]["closed"]:
        assert len(record["known"]) == 4
        assert all(0 <= value < 10_000 for value in record["known"])
        seen.update(record["known"])
    for record in trials["records"]["calibration"]:
        assert not record["known"]
        assert all(100_000 <= value < 110_000 for value in record["unknown"])
    for record in trials["records"]["open"]:
        assert not record["known"]
        assert all(110_000 <= value < 120_000 for value in record["unknown"])
    assert seen


def test_chunked_float64_scores_match_reference(tmp_path):
    evaluate = load_script("71_gelo_msmarco_evaluate.py")
    rng = np.random.default_rng(11)
    candidates = rng.standard_normal((9, 6, 13)).astype(np.float16)
    observed = np.asarray(candidates[2, :3], dtype=np.float64)
    basis, _ = evaluate.rowspace_basis(observed)
    raw = tmp_path / "bank.f16"
    mapping = np.memmap(str(raw), mode="w+", dtype=np.float16, shape=candidates.shape)
    mapping[:] = candidates
    mapping.flush()
    actual, _, _ = evaluate.score_bank(mapping, 9, basis, 3, 2, "float64")

    rows = candidates.astype(np.float64)
    norms_sq = np.sum(rows * rows, axis=-1)
    coordinates = np.einsum("...d,rd->...r", rows, basis, optimize=True)
    residual = np.sqrt(
        np.maximum(norms_sq - np.sum(coordinates * coordinates, axis=-1), 0.0)
        / np.maximum(norms_sq, np.finfo(np.float64).tiny)
    )
    expected = np.mean(np.partition(residual, 2, axis=1)[:, :3], axis=1)
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)


def test_open_set_threshold_metrics_count_false_positives():
    evaluate = load_script("71_gelo_msmarco_evaluate.py")
    scores = np.asarray([0.01, 0.03, 0.20, 0.40], dtype=np.float64)
    metric = evaluate.metrics_at_size(scores, 4, [], [], 2, 0.05)
    assert metric["false_accepted"] == 2
    assert metric["candidate_false_positive_rate"] == 0.5
    assert metric["trial_false_positive"] is True

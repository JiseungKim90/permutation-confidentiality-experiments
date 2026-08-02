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


def test_one_percent_calibration_uses_first_of_one_hundred():
    evaluate = load_script("71_gelo_msmarco_evaluate.py")
    assert evaluate.calibration_order_index(20, 0.05) == 0
    assert evaluate.calibration_order_index(100, 0.01) == 0
    assert evaluate.calibration_order_index(199, 0.01) == 1


def test_identity_representation_has_zero_drift():
    drift = load_script("77_gelo_private_drift_sweep.py")
    rng = np.random.default_rng(17)
    public = rng.standard_normal((3, 5, 11)).astype(np.float32)
    metrics = drift.representation_metrics(public, public.copy())
    assert metrics["relative_hidden_l2_drift"] == 0.0
    assert abs(metrics["mean_row_cosine"] - 1.0) < 1e-12
    assert metrics["p95_public_to_private_rowspace_residual"] < 1e-7
    assert metrics["p95_private_to_public_rowspace_residual"] < 1e-7


def test_zero_step_schedule_is_empty_and_well_shaped():
    drift = load_script("77_gelo_private_drift_sweep.py")
    schedule = drift.deterministic_schedule(10, 20, 4, 0, 7)
    assert schedule.shape == (0, 4)


def test_drift_finalizer_requires_all_550_records():
    finalize = load_script("79_finalize_gelo_private_drift.py")
    rows = []
    offsets = {"calibration": 0, "closed": 1000, "partial": 2000, "open": 3000}
    for split, count in finalize.EXPECTED.items():
        for trial in range(count):
            rows.append(
                {
                    "source_model": "private",
                    "layer": 8,
                    "condition": "ideal",
                    "split": split,
                    "trial": offsets[split] + trial,
                }
            )
    assert finalize.validate(rows)["complete"] is True
    assert finalize.validate(rows[:-1])["complete"] is False


def test_zero_of_three_hundred_wilson_upper_bound_is_about_one_percent():
    finalize = load_script("79_finalize_gelo_private_drift.py")
    lower, upper = finalize.wilson(0, 300)
    assert lower == 0.0
    assert 0.012 < upper < 0.013

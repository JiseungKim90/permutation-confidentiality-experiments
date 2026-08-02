from __future__ import annotations

import importlib.util
from pathlib import Path


ARTIFACT = Path(__file__).resolve().parents[1]


def load_finalize():
    path = ARTIFACT / "scripts" / "74_finalize_gelo_msmarco_campaign.py"
    spec = importlib.util.spec_from_file_location("finalize_gelo", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def complete_core_rows(finalize):
    rows = []
    trial_offsets = {"calibration": 0, "closed": 100, "partial": 200, "open": 300}
    for source_model in finalize.SOURCE_MODELS:
        for layer in finalize.CAMPAIGNS["core"]["layers"]:
            for condition in finalize.CAMPAIGNS["core"]["conditions"]:
                for split, count in finalize.EXPECTED_SPLITS.items():
                    for trial in range(count):
                        rows.append(
                            {
                                "source_model": source_model,
                                "layer": layer,
                                "condition": condition,
                                "split": split,
                                "trial": trial + trial_offsets[split],
                            }
                        )
    return rows


def test_completeness_detects_missing_and_duplicate_records():
    finalize = load_finalize()
    rows = complete_core_rows(finalize)
    audit = finalize.validate_campaign("core", rows)
    assert audit["complete"] is True
    assert audit["actual_records"] == 3060

    broken = rows[:-1] + [rows[0]]
    audit = finalize.validate_campaign("core", broken)
    assert audit["complete"] is False
    assert audit["duplicate_keys"]
    assert audit["missing_or_extra_groups"]


def test_rank_pr_is_exact_at_positive_rank_events():
    finalize = load_finalize()
    row = {
        "source_model": "public",
        "layer": 8,
        "condition": "ideal",
        "split": "closed",
        "trial": 0,
        "metrics": [
            {
                "bank_size": 100000,
                "known_ranks": {"10": 1, "20": 4},
            }
        ],
    }
    points = finalize.aggregate_rank_pr("core", [row])
    assert [point["recall"] for point in points] == [0.5, 1.0]
    assert [point["precision"]["mean"] for point in points] == [1.0, 0.5]


def test_percentile_uses_linear_interpolation():
    finalize = load_finalize()
    assert finalize.percentile([0.0, 10.0], 0.5) == 5.0

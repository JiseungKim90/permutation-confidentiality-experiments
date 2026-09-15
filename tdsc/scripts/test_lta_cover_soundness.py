#!/usr/bin/env python3
"""Regression tests for fail-closed LTA exact-cover handling."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np


REPRO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPRO_ROOT))

from lib.lta import track_column  # noqa: E402
from lib.lta_run import (  # noqa: E402
    lta_run_is_certified,
    recovery_record_is_certified,
    require_lta_run_certified,
    run_lta,
    summarize_recovery_records,
)


def _track(anchors: np.ndarray, slopes: np.ndarray) -> dict:
    anchors = np.asarray(anchors, dtype=np.int64)
    slopes = np.asarray(slopes, dtype=np.int64)
    multiplicities = np.ones(anchors.size, dtype=np.int64)
    offsets = np.arange(anchors.size + 1, dtype=np.int64)
    replies = [np.sort(anchors + t * slopes) for t in (1, 2, 3)]
    return track_column(
        anchors,
        multiplicities,
        offsets,
        anchors.size,
        replies,
        3,
    )


def main() -> None:
    # These two distinct row matchings generate identical multisets at every
    # queried time.  Returning the first branch would be an unsound recovery.
    anchors = np.array([-9, -8, -7, -6, -3, -2, -1, 0], dtype=np.int64)
    slopes_a = np.array([5, 3, 3, 3, 2, 2, 2, 0], dtype=np.int64)
    slopes_b = np.array([4, 4, 4, 2, 3, 1, 1, 1], dtype=np.int64)
    assert all(
        np.array_equal(
            np.sort(anchors + t * slopes_a),
            np.sort(anchors + t * slopes_b),
        )
        for t in (0, 1, 2, 3)
    )
    ambiguous = _track(anchors, slopes_a)
    assert not ambiguous["ok"]
    assert not ambiguous["forced"]
    assert "uniqueness not certified" in ambiguous["reason"]

    # A root-propagated singleton cover remains accepted and exact.
    unique_anchors = np.array([0, 100], dtype=np.int64)
    unique_slopes = np.array([1, 7], dtype=np.int64)
    unique = _track(unique_anchors, unique_slopes)
    assert unique["ok"]
    assert unique["forced"]
    assert np.array_equal(unique["slopes"], unique_slopes)

    # The same ambiguous transcript must remain rejected through the complete
    # wrapper; zero-filled rows must never become a successful recovery.
    ambiguous_run = run_lta(
        slopes_a[:, None], anchors, 16, 3,
        np.random.default_rng(1), lambda n: np.zeros(n),
        diagnostics=False, allow_pass2=False,
        x0_fixed=np.zeros(1, dtype=np.int64),
    )
    assert ambiguous_run["status"] == "uncertified_columns"
    assert ambiguous_run["columns_failed"] == 1
    assert not ambiguous_run["certified_complete"]
    assert not lta_run_is_certified(ambiguous_run)
    try:
        require_lta_run_certified(ambiguous_run, "ambiguous fixture")
    except RuntimeError as exc:
        assert "uncertified" in str(exc)
    else:
        raise AssertionError("ambiguous wrapper result was consumed")

    positive_run = run_lta(
        unique_slopes[:, None], unique_anchors, 16, 3,
        np.random.default_rng(2), lambda n: np.zeros(n),
        diagnostics=False, allow_pass2=False,
        x0_fixed=np.zeros(1, dtype=np.int64),
    )
    assert positive_run["status"] == "ok"
    assert positive_run["certified_complete"]
    assert lta_run_is_certified(positive_run)

    # Certification is schema-strict: success cannot be manufactured by
    # omitted counters or silently ignored malformed nested members.
    assert not lta_run_is_certified({"status": "ok"})
    missing_counter = dict(positive_run)
    del missing_counter["unresolved_rows"]
    assert not lta_run_is_certified(missing_counter)
    malformed_nested = {
        "lta_runs": [dict(positive_run), None],
        "isolation_failures": 0,
    }
    assert not recovery_record_is_certified(malformed_nested)
    malformed_summary = summarize_recovery_records([malformed_nested])
    assert not malformed_summary["success"]
    assert len(malformed_summary["malformed_lta_records"]) == 1
    assert not recovery_record_is_certified({"lta": None})

    # An explicitly present but empty invocation list must not pass by
    # vacuous truth, including when the layer advertises itself as a conv.
    empty_lta_runs = {
        "kind": "conv",
        "lta_runs": [],
        "isolation_failures": 0,
    }
    assert not recovery_record_is_certified(empty_lta_runs)
    empty_summary = summarize_recovery_records([empty_lta_runs])
    assert not empty_summary["success"]
    assert len(empty_summary["malformed_lta_records"]) == 1
    assert (
        empty_summary["malformed_lta_records"][0]["reason"]
        == "container has no LTA invocation"
    )

    # The final unshuffled classifier is an explicit non-LTA recovery variant.
    non_lta_readout = {
        "kind": "fc",
        "readout": "unshuffled final round, labelled read",
    }
    assert recovery_record_is_certified(non_lta_readout)

    # The submission gate must traverse both historical record layouts.
    bad_single = {"lta": dict(ambiguous_run), "isolation_failures": 0}
    bad_list = {"lta_runs": [dict(positive_run), dict(ambiguous_run)],
                "isolation_failures": 0}
    good_single = {"lta": dict(positive_run), "isolation_failures": 0}
    assert not recovery_record_is_certified(bad_single)
    assert not recovery_record_is_certified(bad_list)
    assert recovery_record_is_certified(good_single)
    summary = summarize_recovery_records([good_single, bad_list])
    assert summary["lta_invocations_total"] == 3
    assert summary["lta_invocations_certified"] == 2
    assert not summary["success"]

    # The retained current v7 record set must continue to pass the stricter
    # schema: 29 layer/readout records and 34 LTA invocations.
    current_path = (
        REPRO_ROOT / "reference" / "extraction"
        / "attacker.json"
    )
    with current_path.open("r", encoding="utf-8") as fh:
        current_records = json.load(fh)["records"]
    current_summary = summarize_recovery_records(current_records)
    assert current_summary["success"]
    assert current_summary["records_certified"] == 29
    assert current_summary["lta_invocations_certified"] == 34

    print(json.dumps({
        "ambiguous_case": {
            "ok": bool(ambiguous["ok"]),
            "nodes": int(ambiguous["nodes"]),
            "reason": ambiguous["reason"],
        },
        "root_certified_case": {
            "ok": bool(unique["ok"]),
            "nodes": int(unique["nodes"]),
            "slopes": unique["slopes"].tolist(),
        },
        "wrapper_gate": {
            "ambiguous_status": ambiguous_run["status"],
            "ambiguous_certified_complete": ambiguous_run["certified_complete"],
            "positive_status": positive_run["status"],
            "positive_certified_complete": positive_run["certified_complete"],
            "mixed_layout_summary": summary,
            "malformed_layout_summary": malformed_summary,
            "empty_layout_summary": empty_summary,
            "current_v7_summary": current_summary,
        },
    }, sort_keys=True))


if __name__ == "__main__":
    main()

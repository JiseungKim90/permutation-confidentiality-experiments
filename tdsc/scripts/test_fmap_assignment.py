#!/usr/bin/env python3
"""Regression tests for certified global feature-map row assignment."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


REPRO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPRO_ROOT))

from lib.fmap_partial import (assignment_query_counter,
                              assemble_from_candidates,
                              certify_with_separating_queries,
                              retained_shortcut_positions,
                              separating_multiplacement_query)  # noqa: E402


def main() -> None:
    # The first row is globally forced to channel 1 because the second row can
    # occupy only channel 0.  The former per-channel greedy choice failed here.
    W, _, info = assemble_from_candidates(
        [[(0, 0), (1, 0)], [(0, 0)]],
        [np.array([11]), np.array([22])], [0, 0], 2, 1, 1,
    )
    assert info["assignment_certified"]
    assert info["matching_failures"] == 0
    assert np.array_equal(W[:, 0], np.array([22, 11]))

    # Two different slopes that can exchange slots induce different tensors and
    # must remain uncertified.
    _, _, info = assemble_from_candidates(
        [[(0, 0), (1, 0)], [(0, 0), (1, 0)]],
        [np.array([11]), np.array([22])], [0, 0], 2, 1, 1,
    )
    assert not info["assignment_certified"]
    assert info["assignment_ambiguous_components"] == 1

    # Exchanging identical slopes changes no tensor entry and is safe.
    W, _, info = assemble_from_candidates(
        [[(0, 0), (1, 0)], [(0, 0), (1, 0)]],
        [np.array([7]), np.array([7])], [0, 0], 2, 1, 1,
    )
    assert info["assignment_certified"]
    assert info["assignment_harmless_components"] == 1
    assert np.array_equal(W[:, 0], np.array([7, 7]))

    # Exact reply multisets distinguish the two otherwise exchangeable rows.
    # The observed values are 11+0 and 22+100, which only the identity matching
    # reproduces; set-membership alone cannot certify that global assignment.
    W, _, info = assemble_from_candidates(
        [[(0, 0), (1, 0)], [(0, 0), (1, 0)]],
        [np.array([11]), np.array([22])], [0, 0], 2, 1, 1,
        evidence=[{"posof": {0: (0, 0)}, "u": np.array([1]),
                   "observed": {11: 1, 122: 1}}],
        bgmap=np.array([[[0]], [[100]]], dtype=np.int64),
    )
    assert info["assignment_certified"]
    assert info["evidence_certified"]
    assert info["evidence_distinct_tensors"] == 1
    assert np.array_equal(W[:, 0], np.array([11, 22]))

    # With equal backgrounds the same multiset is compatible with both tensors,
    # so the certifier must retain the ambiguity.
    _, _, info = assemble_from_candidates(
        [[(0, 0), (1, 0)], [(0, 0), (1, 0)]],
        [np.array([11]), np.array([22])], [0, 0], 2, 1, 1,
        evidence=[{"posof": {0: (0, 0)}, "u": np.array([1]),
                   "observed": {11: 1, 22: 1}}],
        bgmap=np.zeros((2, 1, 1), dtype=np.int64),
    )
    assert not info["assignment_certified"]
    assert info["evidence_distinct_tensors"] == 2

    # When every independent single-pixel multiset is compatible with two
    # assignments, overlapping pixels expose which recovered taps add at a
    # common output coordinate.  The client can find such a query without any
    # private model access.
    slopes = [np.array([i + 1]) for i in range(9)]
    assign_a = list(range(9))
    assign_b = list(range(9))
    assign_b[0], assign_b[1] = assign_b[1], assign_b[0]
    sep = separating_multiplacement_query(
        assign_a, assign_b, slopes,
        np.zeros((1, 3, 3), dtype=np.int64),
        [(0, 0), (0, 1), (1, 0), (1, 1)],
        1, 3, 3, 3, 1, 255, np.random.default_rng(7), tries=10,
    )
    assert sep is not None
    assert sep[2] != sep[3]

    # A whole-channel swap is invisible while both channels have the same
    # retained-input background.  An alternative retained input can separate
    # those backgrounds; evidence must be checked against the background that
    # was active for that query, not the original fixed one.
    channel_cands = [[(0, 0), (1, 0)], [(0, 0), (1, 0)]]
    channel_slopes = [np.array([11]), np.array([22])]
    fixed_bg = np.zeros((2, 1, 2), dtype=np.int64)
    initial = assemble_from_candidates(
        channel_cands, channel_slopes, [0, 0], 2, 1, 1,
        evidence=[{"terms": [{"posof": {0: (0, 0)},
                                "u": np.array([1])}],
                   "observed": {11: 1, 22: 1}}],
        bgmap=fixed_bg, include_intercepts=False,
    )
    assert not initial[2]["assignment_certified"]
    alternative_bg = np.array([[[0, 0]], [[100, 100]]], dtype=np.int64)
    alt_terms = [{"posof": {0: (0, 0)}, "u": np.array([1])}]

    def alternative_query(a, b):
        pa = assignment_query_counter(
            a, channel_slopes, alternative_bg, 1, alt_terms)
        pb = assignment_query_counter(
            b, channel_slopes, alternative_bg, 1, alt_terms)
        assert pa != pb
        identity = [0, 1]
        return {"terms": alt_terms,
                "observed": assignment_query_counter(
                    identity, channel_slopes, alternative_bg, 1, alt_terms),
                "bgmap": alternative_bg, "pred_a": pa, "pred_b": pb,
                "source": "retained-input"}

    _, _, context_info, context_queries = certify_with_separating_queries(
        channel_cands, channel_slopes, [0, 0], 2, 1, fixed_bg,
        [(0, 0), (0, 1)], 1, 1, 1, 2, 1, 255,
        [{"terms": [{"posof": {0: (0, 0)}, "u": np.array([1])}],
          "observed": {11: 1, 22: 1}}],
        initial, lambda _pixels, _terms: np.array([11, 22]),
        np.random.default_rng(8), include_intercepts=False,
        search_tries=2, alternative_query=alternative_query,
    )
    assert context_queries == 1
    assert context_info["assignment_certified"]
    assert context_info["separating_query_sources"] == ["retained-input"]

    # A stride-2 conv1 probe may use only odd pixels, which the 1x1 shortcut
    # never samples.  The retained-context search must add the controllable
    # even coordinate instead of merely redrawing the old probe values.
    old_probe_positions = [(1, 1), (0, 15), (15, 0), (15, 15)]
    shortcut_positions = retained_shortcut_positions(
        [(0, 0), (0, 1), (1, 0), (1, 1)], 2, 16, 16,
        [(0, 0), (0, 1), (1, 0), (1, 1)])
    assert shortcut_positions == [(0, 0)]
    assert not set(shortcut_positions).intersection(old_probe_positions)

    # A dead 2x2 kernel can collapse every recovered row onto the same apparent
    # tap.  Since all four slopes are zero, assigning them to the four taps is
    # tensor-identical and must not collapse the following network frame.
    W, _, info = assemble_from_candidates(
        [[(0, 0)]] * 4,
        [np.array([0])] * 4, [3] * 4, 1, 1, 2,
    )
    assert info["assignment_certified"]
    assert info["homogeneous_channels_relaxed"] == 1
    assert np.array_equal(W, np.zeros((1, 4), dtype=np.int64))


if __name__ == "__main__":
    main()

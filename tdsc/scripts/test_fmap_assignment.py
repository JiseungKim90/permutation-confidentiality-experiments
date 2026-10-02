#!/usr/bin/env python3
"""Regression tests for certified global feature-map row assignment."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


REPRO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPRO_ROOT))

from lib.fmap_partial import assemble_from_candidates  # noqa: E402


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


if __name__ == "__main__":
    main()

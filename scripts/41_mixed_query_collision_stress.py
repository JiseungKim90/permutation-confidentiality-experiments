"""Stress-test the separation condition for mixed-query row alignment."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from lib.unlabeled_recovery import (
    AmbiguousRecoveryError,
    FreshPermutationOracle,
    InconsistentTranscriptError,
    recover_bias_anchored,
    row_canonicalize,
)


def parse_args():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260801)
    parser.add_argument(
        "--output", type=Path,
        default=root / "outputs" / "mixed_query_collision_stress.json"
    )
    return parser.parse_args()


def run_setting(trials, seed, coefficients):
    statuses = []
    for trial in range(trials):
        rng = np.random.default_rng(seed + trial)
        m, d = 8, 8
        weight = rng.integers(-2, 3, size=(m, d), dtype=np.int64)
        bias = np.arange(m, dtype=np.int64)
        oracle = FreshPermutationOracle(weight, bias, seed=seed + 10_000 + trial)
        try:
            recovered = recover_bias_anchored(
                oracle.query,
                output_dim=m,
                input_dim=d,
                coefficients=coefficients,
            )
            expected = row_canonicalize(weight, bias)
            status = "exact" if np.array_equal(recovered.weight, expected) else "wrong"
        except AmbiguousRecoveryError:
            status = "ambiguous"
        except InconsistentTranscriptError:
            status = "inconsistent"
        statuses.append(status)
    return {
        "coefficients": list(coefficients),
        "queries": 1 + 8 * len(coefficients),
        "exact": statuses.count("exact"),
        "ambiguous": statuses.count("ambiguous"),
        "inconsistent": statuses.count("inconsistent"),
        "wrong": statuses.count("wrong"),
    }


def main():
    args = parse_args()
    result = {
        "experiment": "mixed-query separation stress test",
        "matrix_shape": [8, 8],
        "weight_lattice_range": [-2, 2],
        "biases": "consecutive integers 0,...,7",
        "trials": args.trials,
        "seed": args.seed,
        "settings": [
            run_setting(args.trials, args.seed, (1,)),
            run_setting(args.trials, args.seed, (1, 2)),
            run_setting(args.trials, args.seed, (1, 2, 3)),
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

"""Finite exhaustive checks for activation-compatible linear masks.

The paper's classification results are analytic.  This script is a bounded
sanity check: it enumerates small integer 2x2 matrices and verifies the claimed
families on a finite grid.  It is not used as a proof.
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np


def relu(x):
    return np.maximum(x, 0.0)


def gelu(x):
    # Exact GELU definition, not the tanh approximation.
    return 0.5 * x * (1.0 + np.vectorize(__import__("math").erf)(x / np.sqrt(2.0)))


def commutes(matrix, activation, grid, atol=1e-11):
    for x in grid:
        if not np.allclose(
            activation(matrix @ x), matrix @ activation(x),
            rtol=0.0, atol=atol
        ):
            return False
    return True


def is_positive_monomial(matrix):
    positive = matrix > 0
    return (
        np.all(matrix >= 0)
        and np.all(positive.sum(axis=0) == 1)
        and np.all(positive.sum(axis=1) == 1)
    )


def is_permutation(matrix):
    return is_positive_monomial(matrix) and np.all(matrix[ matrix > 0] == 1)


def is_signed_permutation(matrix):
    nonzero = matrix != 0
    return (
        np.all(nonzero.sum(axis=0) == 1)
        and np.all(nonzero.sum(axis=1) == 1)
        and np.all(np.abs(matrix[nonzero]) == 1)
    )


def enumerate_case(entries, activation, classifier, grid):
    invertible = 0
    compatible = []
    classified = []
    for values in itertools.product(entries, repeat=4):
        matrix = np.asarray(values, dtype=np.float64).reshape(2, 2)
        if abs(np.linalg.det(matrix)) < 1e-12:
            continue
        invertible += 1
        if commutes(matrix, activation, grid):
            compatible.append(matrix.astype(int).tolist())
        if classifier(matrix):
            classified.append(matrix.astype(int).tolist())
    return {
        "invertible_matrices": invertible,
        "grid_compatible": len(compatible),
        "classified_family": len(classified),
        "sets_equal": compatible == classified,
        "compatible_matrices": compatible,
    }


def parse_args():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output", type=Path,
        default=root / "outputs" / "activation_mask_classification.json"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    grid = [
        np.asarray(x, dtype=np.float64)
        for x in itertools.product((-2, -1, 0, 1, 2), repeat=2)
    ]
    result = {
        "experiment": "finite activation-mask classification sanity check",
        "dimension": 2,
        "grid": [-2, -1, 0, 1, 2],
        "proof_status": "sanity check only; analytic proofs are in the manuscript",
        "relu": enumerate_case(
            (-1, 0, 1, 2), relu, is_positive_monomial, grid
        ),
        "gelu": enumerate_case(
            (-1, 0, 1), gelu, is_permutation, grid
        ),
        "tanh": enumerate_case(
            (-1, 0, 1), np.tanh, is_signed_permutation, grid
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
    print(json.dumps({
        name: {
            "grid_compatible": data["grid_compatible"],
            "classified_family": data["classified_family"],
            "sets_equal": data["sets_equal"],
        }
        for name, data in result.items()
        if name in ("relu", "gelu", "tanh")
    }, indent=2))


if __name__ == "__main__":
    main()

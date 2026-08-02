"""Recovery consequences of permutation lifetime in a hybrid FHE session.

The experiment distinguishes two protocol states:

1. A fixed input/output permutation is reused for repeated queries to a layer.
   The client-coordinate affine map is then labeled and zero/basis queries
   recover it exactly.
2. The hidden input permutation is resampled independently for every request.
   Column relabelings are distributionally indistinguishable, so an ordered
   effective matrix is not identifiable from that oracle.

The first condition is an explicit capability assumption.  The Safhire paper
derives permutations from a session identifier and round number, but its public
description does not establish whether a client may replay a round within the
same session.
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from lib.attack import get_conv_layers
from lib.models import ResNet20


def parse_args():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint", type=Path,
        default=root / "models" / "resnet20_seed0.pt"
    )
    parser.add_argument("--precision", type=int, default=256)
    parser.add_argument("--seed", type=int, default=20260801)
    parser.add_argument(
        "--output", type=Path,
        default=root / "outputs" / "session_lifecycle_recovery.json"
    )
    return parser.parse_args()


def load_model(path):
    model = ResNet20()
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    state = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state)
    return model.eval(), checkpoint


class FixedSessionOracle:
    """Rounded affine map in the fixed client-visible session coordinates."""

    def __init__(self, weight, bias, seed, noise_bound=0.499):
        self.weight = np.asarray(weight, dtype=np.int64)
        self.bias = np.asarray(bias, dtype=np.int64)
        self.rng = np.random.default_rng(seed)
        output_dim, input_dim = self.weight.shape
        self.input_permutation = self.rng.permutation(input_dim)
        self.output_permutation = self.rng.permutation(output_dim)
        self.effective_weight = self.weight[
            self.output_permutation
        ][:, self.input_permutation]
        self.effective_bias = self.bias[self.output_permutation]
        self.noise_bound = noise_bound
        self.query_count = 0

    def query(self, x):
        x = np.asarray(x, dtype=np.int64)
        exact = self.effective_weight @ x + self.effective_bias
        noise = self.rng.uniform(
            -self.noise_bound, self.noise_bound, size=exact.shape
        )
        self.query_count += 1
        return np.rint(exact.astype(np.float64) + noise).astype(np.int64)


def recover_labeled_affine(oracle):
    output_dim, input_dim = oracle.effective_weight.shape
    zero = np.zeros(input_dim, dtype=np.int64)
    recovered_bias = oracle.query(zero)
    recovered_weight = np.empty((output_dim, input_dim), dtype=np.int64)
    for column in range(input_dim):
        x = np.zeros(input_dim, dtype=np.int64)
        x[column] = 1
        recovered_weight[:, column] = oracle.query(x) - recovered_bias
    return recovered_weight, recovered_bias


def all_layer_fixed_session(model, precision, seed):
    rows = []
    total_queries = 0
    global_max_error = 0
    for index, (name, weight, bias, _has_bias, output_dim, input_dim) in enumerate(
        get_conv_layers(model)
    ):
        weight_int = np.rint(weight * precision).astype(np.int64)
        bias_int = np.rint(bias * precision).astype(np.int64)
        oracle = FixedSessionOracle(
            weight_int, bias_int, seed + index * 10_000
        )
        recovered_weight, recovered_bias = recover_labeled_affine(oracle)
        weight_error = int(np.max(np.abs(
            recovered_weight - oracle.effective_weight
        )))
        bias_error = int(np.max(np.abs(
            recovered_bias - oracle.effective_bias
        )))
        error = max(weight_error, bias_error)
        global_max_error = max(global_max_error, error)
        total_queries += oracle.query_count
        rows.append({
            "layer": name,
            "shape": [int(output_dim), int(input_dim)],
            "queries": oracle.query_count,
            "max_lattice_error": error,
            "exact": error == 0,
        })
    return {
        "layers": rows,
        "layer_count": len(rows),
        "total_queries": total_queries,
        "global_max_lattice_error": global_max_error,
        "all_exact": all(row["exact"] for row in rows),
    }


def transcript_distribution(weight, query):
    """Enumerate the exact fresh-input/fresh-output multiset distribution."""

    weight = np.asarray(weight, dtype=np.int64)
    query = np.asarray(query, dtype=np.int64)
    distribution = Counter()
    for permutation in itertools.permutations(range(weight.shape[1])):
        response = weight[:, permutation] @ query
        distribution[tuple(sorted(int(v) for v in response))] += 1
    return distribution


def fresh_input_nonidentifiability():
    weight = np.asarray([
        [2, -1, 4],
        [0, 3, -2],
        [5, 1, 2],
    ], dtype=np.int64)
    column_relabeling = [2, 0, 1]
    relabeled = weight[:, column_relabeling]
    queries = [
        [1, 0, 0],
        [1, 2, 0],
        [1, -1, 3],
        [2, 4, -3],
    ]
    comparisons = []
    for query in queries:
        first = transcript_distribution(weight, query)
        second = transcript_distribution(relabeled, query)
        comparisons.append({
            "query": query,
            "support_size": len(first),
            "distributions_equal": first == second,
        })
    return {
        "shape": [3, 3],
        "column_relabeling": column_relabeling,
        "queries_checked": comparisons,
        "all_equal": all(row["distributions_equal"] for row in comparisons),
        "interpretation": (
            "With an independent uniform hidden input permutation per request, "
            "W and every column relabeling WQ induce the same transcript law."
        ),
    }


def main():
    args = parse_args()
    started = time.time()
    model, checkpoint = load_model(args.checkpoint)
    fixed = all_layer_fixed_session(
        model, args.precision, args.seed
    )
    fresh = fresh_input_nonidentifiability()
    output = {
        "experiment": "permutation session-lifecycle audit",
        "checkpoint": str(args.checkpoint),
        "checkpoint_test_accuracy": (
            checkpoint.get("test_accuracy")
            if isinstance(checkpoint, dict) else None
        ),
        "precision": args.precision,
        "seed": args.seed,
        "fixed_session_repeated_layer_queries": fixed,
        "fresh_input_per_request": fresh,
        "scope": {
            "fixed_session_result_requires": (
                "the service accepts repeated chosen inputs to the same round "
                "while retaining its round permutations"
            ),
            "implementation_status": (
                "not verified: no public Safhire source/state-machine artifact "
                "was located as of 2026-08-01"
            ),
        },
        "wall_seconds": time.time() - started,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(output, handle, indent=2)
    print(json.dumps(output, indent=2), flush=True)


if __name__ == "__main__":
    main()

"""Recover an affine layer up to column permutation across fresh sessions.

The oracle independently permutes input and output coordinates on every query.
The attack uses a permutation-invariant constant input as a row anchor and a
one-coordinate perturbation.  If the anchor intervals do not overlap, sorting
aligns every response to the same rows.  Each query then reveals one uniformly
sampled original column.  Coupon collection recovers all distinct columns
without same-session replay.
"""

from __future__ import annotations

import argparse
import json
import math
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
        "--checkpoint",
        type=Path,
        default=root / "models" / "resnet20_seed0.pt",
    )
    parser.add_argument("--precision", type=int, default=256)
    parser.add_argument("--input-min", type=int, default=0)
    parser.add_argument("--input-max", type=int, default=255)
    parser.add_argument("--beta", type=int, default=1)
    parser.add_argument("--failure-probability", type=float, default=1e-6)
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260801)
    parser.add_argument(
        "--output",
        type=Path,
        default=root / "outputs" / "fresh_session_orbit_recovery.json",
    )
    return parser.parse_args()


def load_model(path: Path):
    model = ResNet20()
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    state = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state)
    return model.eval(), checkpoint


def _column_counter(matrix: np.ndarray) -> Counter:
    return Counter(tuple(int(v) for v in matrix[:, j]) for j in range(matrix.shape[1]))


def _order_is_preserved(anchor: np.ndarray, weight: np.ndarray, beta: int) -> bool:
    order = np.argsort(anchor, kind="stable")
    return all(
        np.array_equal(
            np.argsort(anchor + beta * weight[:, column], kind="stable"),
            order,
        )
        for column in range(weight.shape[1])
    )


def select_constant_anchor(
    weight: np.ndarray,
    bias: np.ndarray,
    input_min: int,
    input_max: int,
    beta: int,
):
    """Evaluate a fixed, model-independent constant anchor choice."""

    if beta == 0:
        raise ValueError("beta must be nonzero")
    lower = input_min
    upper = input_max
    if beta > 0:
        upper -= beta
    else:
        lower -= beta
    if lower > upper:
        raise ValueError("input range cannot contain the requested perturbation")

    # The attacker always chooses the largest admissible constant.  Selection
    # therefore uses only the public input range, not hidden model parameters.
    alpha = upper
    row_sums = weight.sum(axis=1, dtype=np.int64)
    anchor = alpha * row_sums + bias
    sorted_anchor = np.sort(anchor)
    distinct = np.unique(anchor).size == anchor.size
    min_gap = (
        int(np.min(np.diff(sorted_anchor))) if anchor.size > 1 and distinct else 0
    )
    perturbation_bound = int(abs(beta) * np.max(np.abs(weight)))
    certified = distinct and min_gap > 2 * perturbation_bound
    exact_order = distinct and _order_is_preserved(anchor, weight, beta)
    return {
        "alpha": int(alpha),
        "anchor": anchor,
        "distinct": bool(distinct),
        "minimum_anchor_gap": min_gap,
        "perturbation_bound": perturbation_bound,
        "certified_interval_separation": bool(certified),
        "all_columns_preserve_order": bool(exact_order),
    }


class FreshSessionOracle:
    """Fresh hidden input/output permutation and bounded decoding noise."""

    def __init__(self, weight, bias, seed, noise_bound=0.499):
        self.weight = np.asarray(weight, dtype=np.int64)
        self.bias = np.asarray(bias, dtype=np.int64)
        self.rng = np.random.default_rng(seed)
        self.noise_bound = noise_bound
        self.query_count = 0
        self.sampled_columns = []

    def query(self, x):
        x = np.asarray(x, dtype=np.int64)
        output_dim, input_dim = self.weight.shape
        if x.shape != (input_dim,):
            raise ValueError("oracle input has the wrong dimension")
        input_permutation = self.rng.permutation(input_dim)
        output_permutation = self.rng.permutation(output_dim)
        exact = self.weight[:, input_permutation] @ x + self.bias
        noise = self.rng.uniform(
            -self.noise_bound, self.noise_bound, size=output_dim
        )
        self.query_count += 1
        values, counts = np.unique(x, return_counts=True)
        if values.size == 2 and int(np.max(counts)) == input_dim - 1:
            baseline = values[int(np.argmax(counts))]
            perturbed = int(np.flatnonzero(x != baseline)[0])
            self.sampled_columns.append(int(input_permutation[perturbed]))
        return np.rint(exact[output_permutation] + noise).astype(np.int64)


def coupon_query_bound(input_dim: int, failure_probability: float) -> int:
    if input_dim < 1:
        raise ValueError("input dimension must be positive")
    if not 0 < failure_probability < 1:
        raise ValueError("failure probability must be in (0,1)")
    return int(math.ceil(input_dim * (math.log(input_dim) - math.log(failure_probability))))


def recover_orbit(
    weight: np.ndarray,
    bias: np.ndarray,
    anchor_choice,
    beta: int,
    failure_probability: float,
    seed: int,
):
    output_dim, input_dim = weight.shape
    alpha = anchor_choice["alpha"]
    oracle = FreshSessionOracle(weight, bias, seed)
    constant = np.full(input_dim, alpha, dtype=np.int64)
    anchor = np.sort(oracle.query(constant))
    query_bound = coupon_query_bound(input_dim, failure_probability)
    recovered_columns = []
    nonintegral = 0
    for _ in range(query_bound):
        query = constant.copy()
        query[0] += beta
        delta = np.sort(oracle.query(query)) - anchor
        if np.any(delta % beta):
            nonintegral += 1
            continue
        recovered_columns.append(tuple(int(v) for v in delta // beta))

    recovered_counter = Counter(recovered_columns)
    true_order = np.argsort(anchor_choice["anchor"], kind="stable")
    canonical_weight = weight[true_order]
    true_counter = _column_counter(canonical_weight)
    distinct_columns = len(true_counter)
    all_true_columns_seen = all(column in recovered_counter for column in true_counter)
    no_false_columns = all(column in true_counter for column in recovered_counter)

    recovered_bias_exact = False
    if len(recovered_counter) == input_dim:
        recovered_matrix = np.asarray(
            list(recovered_counter.keys()), dtype=np.int64
        ).T
        recovered_bias = anchor - alpha * recovered_matrix.sum(axis=1)
        recovered_bias_exact = bool(np.array_equal(recovered_bias, bias[true_order]))

    sampled_indices = set(oracle.sampled_columns)
    return {
        "queries": int(oracle.query_count),
        "coupon_queries": query_bound,
        "failure_probability_bound": failure_probability,
        "sampled_latent_column_count": len(sampled_indices),
        "true_distinct_column_count": distinct_columns,
        "recovered_distinct_column_count": len(recovered_counter),
        "pairwise_distinct_columns": distinct_columns == input_dim,
        "all_latent_columns_sampled": len(sampled_indices) == input_dim,
        "all_true_columns_seen": bool(all_true_columns_seen),
        "no_false_columns": bool(no_false_columns),
        "nonintegral_recoveries": nonintegral,
        "bias_recovered_exactly": recovered_bias_exact,
        "status": (
            "exact_orbit"
            if all_true_columns_seen
            and no_false_columns
            and distinct_columns == input_dim
            and recovered_bias_exact
            else "not_exact"
        ),
    }


def evaluate_layers(model, args):
    rows = []
    for layer_index, (name, weight, bias, _, output_dim, input_dim) in enumerate(
        get_conv_layers(model)
    ):
        weight_int = np.rint(weight * args.precision).astype(np.int64)
        bias_int = np.rint(bias * args.precision).astype(np.int64)
        anchor = select_constant_anchor(
            weight_int,
            bias_int,
            args.input_min,
            args.input_max,
            args.beta,
        )
        trials = []
        for trial in range(args.trials):
            trials.append(
                recover_orbit(
                    weight_int,
                    bias_int,
                    anchor,
                    args.beta,
                    args.failure_probability,
                    args.seed + layer_index * 100_000 + trial,
                )
            )
        rows.append({
            "layer": name,
            "shape": [int(output_dim), int(input_dim)],
            "alpha": anchor["alpha"],
            "beta": args.beta,
            "minimum_anchor_gap": anchor["minimum_anchor_gap"],
            "perturbation_bound": anchor["perturbation_bound"],
            "certified_interval_separation": anchor[
                "certified_interval_separation"
            ],
            "all_columns_preserve_order": anchor["all_columns_preserve_order"],
            "trials": trials,
        })
    return rows


def evaluate_final_classifier(model, args):
    weight = np.rint(
        model.fc.weight.detach().cpu().numpy() * args.precision
    ).astype(np.int64)
    bias = np.rint(
        model.fc.bias.detach().cpu().numpy() * args.precision
    ).astype(np.int64)
    anchor = select_constant_anchor(
        weight, bias, args.input_min, args.input_max, args.beta
    )
    trials = [
        recover_orbit(
            weight,
            bias,
            anchor,
            args.beta,
            args.failure_probability,
            args.seed + 9_000_000 + trial,
        )
        for trial in range(args.trials)
    ]
    return {
        "layer": "fc",
        "shape": [int(weight.shape[0]), int(weight.shape[1])],
        "alpha": anchor["alpha"],
        "beta": args.beta,
        "minimum_anchor_gap": anchor["minimum_anchor_gap"],
        "perturbation_bound": anchor["perturbation_bound"],
        "certified_interval_separation": anchor[
            "certified_interval_separation"
        ],
        "all_columns_preserve_order": anchor["all_columns_preserve_order"],
        "trials": trials,
    }


def main():
    args = parse_args()
    started = time.time()
    model, checkpoint = load_model(args.checkpoint)
    layers = evaluate_layers(model, args)
    classifier = evaluate_final_classifier(model, args)
    output = {
        "experiment": "fresh-session orbit recovery",
        "checkpoint": str(args.checkpoint),
        "checkpoint_test_accuracy": (
            checkpoint.get("test_accuracy") if isinstance(checkpoint, dict) else None
        ),
        "precision": args.precision,
        "input_range": [args.input_min, args.input_max],
        "beta": args.beta,
        "seed": args.seed,
        "trials_per_layer": args.trials,
        "failure_probability_per_trial": args.failure_probability,
        "theorem_conditions": {
            "fresh_hidden_input_permutation_each_query": True,
            "fresh_hidden_output_permutation_each_query": True,
            "constant_anchor_is_permutation_invariant": True,
            "sufficient_row_condition": "minimum anchor gap > 2*|beta|*max|W_ij|",
            "completion_condition": "pairwise-distinct columns",
            "query_bound": "1 + ceil(d*(ln d + ln(1/delta)))",
            "recovery_scope": "W up to column permutation and b in anchor row order",
        },
        "layers": layers,
        "final_classifier": classifier,
        "summary": {
            "layer_count": len(layers),
            "certified_layer_count": sum(
                row["certified_interval_separation"] for row in layers
            ),
            "order_preserving_layer_count": sum(
                row["all_columns_preserve_order"] for row in layers
            ),
            "all_trials_exact_layer_count": sum(
                bool(row["trials"])
                and all(trial["status"] == "exact_orbit" for trial in row["trials"])
                for row in layers
            ),
            "total_exact_trials": sum(
                trial["status"] == "exact_orbit"
                for row in layers
                for trial in row["trials"]
            ),
            "total_trials": sum(len(row["trials"]) for row in layers),
            "classifier_exact_trials": sum(
                trial["status"] == "exact_orbit"
                for trial in classifier["trials"]
            ),
            "classifier_total_trials": len(classifier["trials"]),
        },
        "wall_seconds": time.time() - started,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(output, indent=2), flush=True)


if __name__ == "__main__":
    main()

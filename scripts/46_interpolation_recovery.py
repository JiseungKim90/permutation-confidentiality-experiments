"""Deterministic interpolation recovery under fresh output permutations.

This experiment validates the m-probe interpolation theorem on deliberately
collision-heavy integer layers and on the folded first Conv+BN block of a
trained ResNet-20.  It also compares the universal method with the practical
three-coefficient matching solver.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from lib.interpolation_recovery import (
    derivative_weights_at_zero,
    recover_bias_anchored_interpolation,
    recover_bias_free_interpolation,
)
from lib.models import ResNet20
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
    parser.add_argument(
        "--checkpoint", type=Path,
        default=root / "models" / "resnet20_seed0.pt"
    )
    parser.add_argument("--precision", type=int, default=256)
    parser.add_argument("--trials", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260801)
    parser.add_argument(
        "--output", type=Path,
        default=root / "outputs" / "interpolation_recovery.json"
    )
    return parser.parse_args()


def check_derivative_identity(max_degree=16):
    records = []
    for degree in range(1, max_degree + 1):
        weights = derivative_weights_at_zero(degree)
        exact = True
        for power in range(degree + 1):
            estimate = sum(
                weights[k] * (k ** power) for k in range(degree + 1)
            )
            target = 1 if power == 1 else 0
            exact = exact and estimate == target
        records.append({"degree": degree, "exact": bool(exact)})
    return records


def verify(result, weight, anchor):
    expected = row_canonicalize(weight, anchor)
    return bool(np.array_equal(result.weight, expected))


def collision_trials(trials, seed):
    settings = []
    for output_dim in (4, 8):
        input_dim = output_dim
        counts = {
            "interpolation_exact": 0,
            "interpolation_wrong": 0,
            "practical_exact": 0,
            "practical_ambiguous": 0,
            "practical_wrong": 0,
        }
        interpolation_queries = None
        for trial in range(trials):
            rng = np.random.default_rng(
                seed + output_dim * 100_000 + trial
            )
            weight = rng.integers(
                -2, 3, size=(output_dim, input_dim), dtype=np.int64
            )
            bias = np.arange(output_dim, dtype=np.int64)

            oracle = FreshPermutationOracle(
                weight, bias,
                seed=seed + output_dim * 200_000 + trial,
                noise_bound=0.499,
            )
            recovered = recover_bias_anchored_interpolation(
                oracle.query, output_dim=output_dim, input_dim=input_dim
            )
            interpolation_queries = oracle.query_count
            if verify(recovered, weight, bias):
                counts["interpolation_exact"] += 1
            else:
                counts["interpolation_wrong"] += 1

            oracle_practical = FreshPermutationOracle(
                weight, bias,
                seed=seed + output_dim * 300_000 + trial,
                noise_bound=0.499,
            )
            try:
                practical = recover_bias_anchored(
                    oracle_practical.query,
                    output_dim=output_dim,
                    input_dim=input_dim,
                    coefficients=(1, 2, 3),
                )
                if verify(practical, weight, bias):
                    counts["practical_exact"] += 1
                else:
                    counts["practical_wrong"] += 1
            except (AmbiguousRecoveryError, InconsistentTranscriptError):
                counts["practical_ambiguous"] += 1

        settings.append({
            "shape": [output_dim, input_dim],
            "weight_range": [-2, 2],
            "biases": list(range(output_dim)),
            "trials": trials,
            "interpolation_queries": interpolation_queries,
            "practical_queries_if_completed": 1 + 3 * input_dim,
            **counts,
        })
    return settings


def load_model(path):
    model = ResNet20()
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    state = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state)
    return model.eval()


def fold_first_conv_bn(model, precision):
    conv = model.conv1
    bn = model.bn1
    raw = conv.weight.detach().cpu().numpy().reshape(conv.out_channels, -1)
    conv_bias = np.zeros(conv.out_channels, dtype=np.float64)
    if conv.bias is not None:
        conv_bias = conv.bias.detach().cpu().numpy().astype(np.float64)
    scale = (
        bn.weight.detach().cpu().numpy()
        / np.sqrt(bn.running_var.detach().cpu().numpy() + bn.eps)
    )
    weight = np.rint(raw * scale[:, None] * precision).astype(np.int64)
    bias = np.rint((
        bn.bias.detach().cpu().numpy()
        + scale * (conv_bias - bn.running_mean.detach().cpu().numpy())
    ) * precision).astype(np.int64)
    return weight, bias


def trained_layer(checkpoint, precision, seed):
    model = load_model(checkpoint)
    weight, bias = fold_first_conv_bn(model, precision)
    output_dim, input_dim = weight.shape

    biased_oracle = FreshPermutationOracle(
        weight, bias, seed=seed + 1_000_000, noise_bound=0.499
    )
    biased = recover_bias_anchored_interpolation(
        biased_oracle.query,
        output_dim=output_dim,
        input_dim=input_dim,
    )

    anchor = next(
        column for column in range(input_dim)
        if np.unique(weight[:, column]).size == output_dim
    )
    bias_free_oracle = FreshPermutationOracle(
        weight, seed=seed + 2_000_000, noise_bound=0.499
    )
    bias_free = recover_bias_free_interpolation(
        bias_free_oracle.query,
        output_dim=output_dim,
        input_dim=input_dim,
        anchor_coordinate=anchor,
    )

    return {
        "shape": [int(output_dim), int(input_dim)],
        "precision": precision,
        "distinct_biases": int(np.unique(bias).size),
        "bias_anchored": {
            "status": "exact" if verify(biased, weight, bias) else "wrong",
            "queries": biased_oracle.query_count,
            "theoretical_queries": 1 + output_dim * input_dim,
        },
        "bias_free": {
            "status": (
                "exact" if verify(bias_free, weight, weight[:, anchor])
                else "wrong"
            ),
            "anchor_coordinate": anchor,
            "queries": bias_free_oracle.query_count,
            "theoretical_queries": 1 + output_dim * (input_dim - 1),
        },
    }


def main():
    args = parse_args()
    started = time.time()
    derivative_checks = check_derivative_identity()
    collisions = collision_trials(args.trials, args.seed)
    trained = trained_layer(
        args.checkpoint, args.precision, args.seed
    )
    output = {
        "experiment": "deterministic interpolation row recovery",
        "seed": args.seed,
        "noise_bound_lattice_units": 0.499,
        "theorem": {
            "coefficients": "0,...,m",
            "bias_anchored_queries": "1+m*d",
            "bias_free_queries": "1+m*(d-1)",
            "requires_unique_matching": False,
            "requires_pairwise_distinct_anchors": True,
        },
        "derivative_identity": derivative_checks,
        "collision_stress": collisions,
        "trained_resnet20_first_conv_bn": trained,
        "wall_seconds": time.time() - started,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(output, handle, indent=2)
    print(json.dumps(output, indent=2), flush=True)


if __name__ == "__main__":
    main()

"""Evaluate full row recovery under independent output permutations.

This journal-extension experiment covers random integer layers, collision-heavy
stress tests, and the first Conv+BN block of the trained CIFAR-10 ResNet-20.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from lib.models import ResNet20
from lib.unlabeled_recovery import (
    AmbiguousRecoveryError,
    FreshPermutationOracle,
    recover_bias_anchored,
    recover_bias_free,
    row_canonicalize,
)


def parse_args():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path,
                        default=root / "models" / "resnet20_seed0.pt")
    parser.add_argument("--output", type=Path,
                        default=root / "outputs" / "unlabeled_row_recovery.json")
    parser.add_argument("--precision", type=int, default=256)
    parser.add_argument("--trials", type=int, default=30)
    parser.add_argument("--seed", type=int, default=20260801)
    return parser.parse_args()


def exact_equal(recovered, true_weight, anchor):
    canonical = row_canonicalize(true_weight, anchor)
    return bool(np.array_equal(recovered.weight, canonical))


def random_trials(n_trials, seed, noise_bound):
    records = []
    for trial in range(n_trials):
        rng = np.random.default_rng(seed + trial)
        m, d = 16, 27
        weight = rng.integers(-4096, 4097, size=(m, d), dtype=np.int64)

        # Pairwise-distinct bias anchors.
        bias = rng.choice(np.arange(-8192, 8193), size=m, replace=False)
        oracle = FreshPermutationOracle(
            weight, bias, seed=seed + 10_000 + trial, noise_bound=noise_bound
        )
        recovered = recover_bias_anchored(
            oracle.query, output_dim=m, input_dim=d
        )
        biased_ok = exact_equal(recovered, weight, bias)

        # Regenerate the reference column until it is pairwise distinct.
        while np.unique(weight[:, 0]).size != m:
            weight[:, 0] = rng.integers(-4096, 4097, size=m)
        oracle0 = FreshPermutationOracle(
            weight, seed=seed + 20_000 + trial, noise_bound=noise_bound
        )
        recovered0 = recover_bias_free(
            oracle0.query, output_dim=m, input_dim=d, anchor_coordinate=0
        )
        bias_free_ok = exact_equal(recovered0, weight, weight[:, 0])
        records.append({
            "trial": trial,
            "biased_exact": biased_ok,
            "bias_free_exact": bias_free_ok,
            "biased_queries": oracle.query_count,
            "bias_free_queries": oracle0.query_count,
        })
    return records


def collision_stress(n_trials, seed):
    """Measure how often exact recovery correctly rejects colliding instances."""
    records = []
    for trial in range(n_trials):
        rng = np.random.default_rng(seed + 30_000 + trial)
        m, d = 8, 8
        weight = rng.integers(-2, 3, size=(m, d), dtype=np.int64)
        # Biases remain distinct: failures are caused by mixed-sum collisions.
        bias = rng.choice(np.arange(-16, 17), size=m, replace=False)
        oracle = FreshPermutationOracle(weight, bias, seed=seed + 40_000 + trial)
        status = "exact"
        try:
            recovered = recover_bias_anchored(
                oracle.query, output_dim=m, input_dim=d
            )
            if not exact_equal(recovered, weight, bias):
                status = "wrong"
        except AmbiguousRecoveryError:
            status = "ambiguous"
        records.append({"trial": trial, "status": status})
    return records


def load_resnet(path):
    model = ResNet20()
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    state = checkpoint.get("model_state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    model.load_state_dict(state)
    model.eval()
    return model


def fold_first_conv_bn(model, precision):
    conv = model.conv1
    bn = model.bn1
    weight = conv.weight.detach().cpu().numpy().reshape(conv.out_channels, -1)
    if conv.bias is None:
        bias = np.zeros(conv.out_channels, dtype=np.float64)
    else:
        bias = conv.bias.detach().cpu().numpy().astype(np.float64)
    scale = (
        bn.weight.detach().cpu().numpy()
        / np.sqrt(bn.running_var.detach().cpu().numpy() + bn.eps)
    )
    folded_weight = weight * scale[:, None]
    folded_bias = (
        bn.bias.detach().cpu().numpy()
        + scale * (bias - bn.running_mean.detach().cpu().numpy())
    )
    return (
        np.rint(folded_weight * precision).astype(np.int64),
        np.rint(folded_bias * precision).astype(np.int64),
    )


def trained_first_layer(checkpoint, precision, seed):
    model = load_resnet(checkpoint)
    weight, bias = fold_first_conv_bn(model, precision)
    m, d = weight.shape
    result = {
        "shape": [int(m), int(d)],
        "precision": precision,
        "distinct_folded_biases": int(np.unique(bias).size),
        "bias_anchored": None,
        "bias_free": None,
    }

    oracle = FreshPermutationOracle(weight, bias, seed=seed + 50_000, noise_bound=0.499)
    try:
        recovered = recover_bias_anchored(
            oracle.query, output_dim=m, input_dim=d
        )
        result["bias_anchored"] = {
            "status": "exact" if exact_equal(recovered, weight, bias) else "wrong",
            "queries": oracle.query_count,
            "theoretical_queries": recovered.queries,
        }
    except AmbiguousRecoveryError as exc:
        result["bias_anchored"] = {"status": "ambiguous", "reason": str(exc)}

    # Also test the bias-free variant on raw folded weights.  Select the first
    # quantized column that gives distinct row anchors.
    anchor = next(
        (j for j in range(d) if np.unique(weight[:, j]).size == m), None
    )
    result["bias_free_anchor_coordinate"] = anchor
    if anchor is not None:
        oracle0 = FreshPermutationOracle(weight, seed=seed + 60_000, noise_bound=0.499)
        try:
            recovered0 = recover_bias_free(
                oracle0.query, output_dim=m, input_dim=d,
                anchor_coordinate=anchor
            )
            result["bias_free"] = {
                "status": "exact" if exact_equal(
                    recovered0, weight, weight[:, anchor]
                ) else "wrong",
                "queries": oracle0.query_count,
                "theoretical_queries": recovered0.queries,
            }
        except AmbiguousRecoveryError as exc:
            result["bias_free"] = {"status": "ambiguous", "reason": str(exc)}
    else:
        result["bias_free"] = {
            "status": "not_applicable",
            "reason": "no pairwise-distinct quantized weight column",
        }
    return result


def main():
    args = parse_args()
    started = time.time()
    exact = random_trials(args.trials, args.seed, noise_bound=0.499)
    collisions = collision_stress(args.trials, args.seed)
    trained = trained_first_layer(args.checkpoint, args.precision, args.seed)

    summary = {
        "experiment": "fresh-output-permutation full row recovery",
        "scope": (
            "affine layers whose input coordinates are attacker-controlled; "
            "does not cover an independently hidden input permutation"
        ),
        "seed": args.seed,
        "coefficients": [1, 2, 3],
        "noise_bound_lattice_units": 0.499,
        "random_trials": exact,
        "random_summary": {
            "trials": len(exact),
            "biased_exact": sum(row["biased_exact"] for row in exact),
            "bias_free_exact": sum(row["bias_free_exact"] for row in exact),
        },
        "collision_stress": collisions,
        "collision_summary": {
            key: sum(row["status"] == key for row in collisions)
            for key in ("exact", "ambiguous", "wrong")
        },
        "trained_resnet20_first_conv_bn": trained,
        "wall_seconds": time.time() - started,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print(json.dumps(summary["random_summary"], indent=2))
    print(json.dumps(summary["collision_summary"], indent=2))
    print(json.dumps(trained, indent=2))
    print(f"[done] wrote {args.output} in {summary['wall_seconds']:.2f}s")


if __name__ == "__main__":
    main()

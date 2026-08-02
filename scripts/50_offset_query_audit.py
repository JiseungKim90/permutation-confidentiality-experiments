"""Dense offset interpolation and simple query-audit defenses.

Zero and basis probes are easy to flag as sparse.  The interpolation theorem
can instead be centered at any dense admissible input x0:

    F_j(k) = ms(W x0 + b + k W[:,j]).

If the anchor response W x0+b is distinct, k=0,...,m recovers every column and
then b.  This experiment checks the attack on a trained ResNet-20 first block
while keeping all coordinates in an 8-bit range and measures whether range and
sparsity audits reject the transcript.
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

from lib.interpolation_recovery import recover_vector_by_interpolation
from lib.models import ResNet20
from lib.unlabeled_recovery import FreshPermutationOracle


def parse_args():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint", type=Path,
        default=root / "models" / "resnet20_seed0.pt"
    )
    parser.add_argument("--precision", type=int, default=256)
    parser.add_argument("--seed", type=int, default=20260801)
    parser.add_argument("--audit-repetitions", type=int, default=10000)
    parser.add_argument(
        "--output", type=Path,
        default=root / "outputs" / "offset_query_audit.json"
    )
    return parser.parse_args()


def load_folded_first_layer(path, precision):
    model = ResNet20()
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    state = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state)
    model.eval()
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


def choose_dense_anchor(weight, bias, rng, input_min, input_max, margin):
    for attempt in range(10000):
        base = rng.integers(
            input_min + margin,
            input_max - margin + 1,
            size=weight.shape[1],
            dtype=np.int64,
        )
        anchor = weight @ base + bias
        if np.unique(anchor).size == weight.shape[0]:
            return base, anchor, attempt + 1
    raise RuntimeError("failed to find a distinct dense anchor")


def audit(query, input_min=0, input_max=255, sparse_threshold=0.25):
    query = np.asarray(query, dtype=np.int64)
    in_range = bool(np.all(
        (query >= input_min) & (query <= input_max)
    ))
    nonzero_fraction = float(np.count_nonzero(query) / query.size)
    return {
        "range_accept": in_range,
        "sparsity_accept": nonzero_fraction > sparse_threshold,
        "nonzero_fraction": nonzero_fraction,
        "minimum": int(query.min()),
        "maximum": int(query.max()),
    }


def run_attack(weight, bias, seed):
    rng = np.random.default_rng(seed)
    output_dim, input_dim = weight.shape
    base, true_anchor, attempts = choose_dense_anchor(
        weight, bias, rng, 0, 255, output_dim
    )
    canonical_order = np.argsort(true_anchor, kind="stable")
    oracle = FreshPermutationOracle(
        weight, bias, seed=seed + 1, noise_bound=0.499
    )

    anchor_observation = oracle.query(base)
    anchors = np.sort(anchor_observation)
    transcripts = [base.copy()]
    recovered_weight = np.empty_like(weight)
    for column in range(input_dim):
        observations = {0: anchor_observation}
        for coefficient in range(1, output_dim + 1):
            query = base.copy()
            query[column] += coefficient
            observations[coefficient] = oracle.query(query)
            transcripts.append(query)
        recovered_weight[:, column] = recover_vector_by_interpolation(
            anchors, observations
        )

    recovered_bias = anchors - recovered_weight @ base
    expected_weight = weight[canonical_order]
    expected_bias = bias[canonical_order]
    audits = [audit(query) for query in transcripts]

    baseline_queries = [
        np.zeros(input_dim, dtype=np.int64),
        *[
            np.eye(1, input_dim, column, dtype=np.int64).reshape(-1)
            for column in range(input_dim)
        ],
    ]
    baseline_audits = [audit(query) for query in baseline_queries]
    return {
        "shape": [int(output_dim), int(input_dim)],
        "anchor_search_attempts": attempts,
        "queries": oracle.query_count,
        "theoretical_queries": 1 + output_dim * input_dim,
        "weight_status": (
            "exact" if np.array_equal(recovered_weight, expected_weight)
            else "wrong"
        ),
        "bias_status": (
            "exact" if np.array_equal(recovered_bias, expected_bias)
            else "wrong"
        ),
        "max_weight_lattice_error": int(np.max(np.abs(
            recovered_weight - expected_weight
        ))),
        "max_bias_lattice_error": int(np.max(np.abs(
            recovered_bias - expected_bias
        ))),
        "input_range": [0, 255],
        "attack_query_minimum": min(row["minimum"] for row in audits),
        "attack_query_maximum": max(row["maximum"] for row in audits),
        "range_audit_accept_percent": (
            100.0 * sum(row["range_accept"] for row in audits) / len(audits)
        ),
        "sparsity_audit_accept_percent": (
            100.0 * sum(row["sparsity_accept"] for row in audits) / len(audits)
        ),
        "minimum_nonzero_fraction": min(
            row["nonzero_fraction"] for row in audits
        ),
        "baseline_sparse_probe_accept_percent": (
            100.0 * sum(
                row["sparsity_accept"] for row in baseline_audits
            ) / len(baseline_audits)
        ),
        "transcripts": transcripts,
    }


def benchmark_audit(transcripts, repetitions):
    started = time.perf_counter()
    accepted = 0
    count = 0
    for repetition in range(repetitions):
        query = transcripts[repetition % len(transcripts)]
        row = audit(query)
        accepted += int(row["range_accept"] and row["sparsity_accept"])
        count += 1
    elapsed = time.perf_counter() - started
    return {
        "checks": count,
        "accepted": accepted,
        "total_seconds": elapsed,
        "microseconds_per_query": 1e6 * elapsed / count,
    }


def main():
    args = parse_args()
    started = time.time()
    weight, bias = load_folded_first_layer(
        args.checkpoint, args.precision
    )
    result = run_attack(weight, bias, args.seed)
    transcripts = result.pop("transcripts")
    audit_cost = benchmark_audit(
        transcripts, args.audit_repetitions
    )
    output = {
        "experiment": "dense offset-query audit bypass",
        "checkpoint": str(args.checkpoint),
        "precision": args.precision,
        "seed": args.seed,
        "fresh_output_permutation_per_query": True,
        "attack": result,
        "audit_cost": audit_cost,
        "conclusion": {
            "range_check": "does not stop the dense offset attack",
            "sparsity_check": "does not stop the dense offset attack",
            "required_break": (
                "cryptographic provenance/well-formedness of each layer input, "
                "or hiding the first activation inside the protected domain"
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

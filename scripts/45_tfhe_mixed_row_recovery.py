"""Concrete-TFHE replay of mixed-query row recovery on ResNet-20 conv1.

Each query is evaluated by a separately compiled affine circuit with an
independently sampled server-side output permutation.  The experiment covers
the raw bias-free convolution and the deployment-style folded Conv+BN layer.
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

from lib.models import ResNet20
from lib.unlabeled_recovery import (
    recover_bias_anchored,
    recover_bias_free,
    row_canonicalize,
)

try:
    from concrete import fhe
except ImportError as exc:  # pragma: no cover - exercised on the Linux runner
    raise SystemExit("concrete-python is required for this experiment") from exc


def parse_args():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path,
                        default=root / "models" / "resnet20_seed0.pt")
    parser.add_argument("--precision", type=int, default=256)
    parser.add_argument("--seed", type=int, default=20260801)
    parser.add_argument(
        "--output", type=Path,
        default=root / "outputs" / "logs" / "45_tfhe_mixed_row_recovery.json"
    )
    return parser.parse_args()


def load_model(path):
    model = ResNet20()
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    state = checkpoint.get("model_state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    model.load_state_dict(state)
    return model.eval()


def raw_and_folded(model, precision):
    conv = model.conv1
    bn = model.bn1
    raw = conv.weight.detach().cpu().numpy().reshape(conv.out_channels, -1)
    raw_int = np.rint(raw * precision).astype(np.int64)
    conv_bias = np.zeros(conv.out_channels, dtype=np.float64)
    if conv.bias is not None:
        conv_bias = conv.bias.detach().cpu().numpy().astype(np.float64)
    scale = (
        bn.weight.detach().cpu().numpy()
        / np.sqrt(bn.running_var.detach().cpu().numpy() + bn.eps)
    )
    folded_weight = np.rint(raw * scale[:, None] * precision).astype(np.int64)
    folded_bias = np.rint((
        bn.bias.detach().cpu().numpy()
        + scale * (conv_bias - bn.running_mean.detach().cpu().numpy())
    ) * precision).astype(np.int64)
    return raw_int, folded_weight, folded_bias


class FreshTFHEOracle:
    def __init__(self, weight, bias, seed, cache_dir):
        self.weight = np.asarray(weight, dtype=np.int64)
        self.bias = np.asarray(bias, dtype=np.int64)
        self.rng = np.random.default_rng(seed)
        self.cache_dir = cache_dir
        self.query_count = 0
        self.compile_seconds = 0.0
        self.evaluate_seconds = 0.0

    def _compile(self, permutation, requested_x):
        weight = self.weight
        bias = self.bias
        p_matrix = np.eye(weight.shape[0], dtype=np.int64)[permutation]

        @fhe.compiler({"x": "encrypted"})
        def circuit_function(x):
            return p_matrix @ (weight @ x + bias)

        # Cover every coordinate amplitude used by Lambda={1,2,3}.  Including
        # the requested point makes the compiler range explicit for this run.
        inputset = [
            np.zeros(weight.shape[1], dtype=np.int64),
            np.full(weight.shape[1], 3, dtype=np.int64),
            np.asarray(requested_x, dtype=np.int64),
            *[
                self.rng.integers(0, 4, size=weight.shape[1], dtype=np.int64)
                for _ in range(16)
            ],
        ]
        circuit = circuit_function.compile(
            inputset,
            configuration=fhe.Configuration(
                enable_unsafe_features=True,
                use_insecure_key_cache=True,
                insecure_key_cache_location=str(self.cache_dir),
            ),
        )
        circuit.keygen()
        return circuit

    def query(self, x):
        x = np.asarray(x, dtype=np.int64)
        permutation = self.rng.permutation(self.weight.shape[0]).astype(np.int64)
        started = time.time()
        circuit = self._compile(permutation, x)
        self.compile_seconds += time.time() - started
        started = time.time()
        observed = circuit.encrypt_run_decrypt(x)
        self.evaluate_seconds += time.time() - started
        self.query_count += 1
        return np.asarray(observed, dtype=np.int64)


def summarize(result, oracle, expected):
    return {
        "status": "exact" if np.array_equal(result.weight, expected) else "wrong",
        "queries": oracle.query_count,
        "theoretical_queries": result.queries,
        "compile_and_keygen_seconds": oracle.compile_seconds,
        "encrypted_evaluation_seconds": oracle.evaluate_seconds,
    }


def main():
    args = parse_args()
    started = time.time()
    model = load_model(args.checkpoint)
    raw_weight, folded_weight, folded_bias = raw_and_folded(model, args.precision)
    m, d = raw_weight.shape
    cache_root = Path("/tmp/p050_tfhe_mixed_keys")

    raw_anchor = next(
        j for j in range(d) if np.unique(raw_weight[:, j]).size == m
    )
    raw_oracle = FreshTFHEOracle(
        raw_weight, np.zeros(m, dtype=np.int64), args.seed, cache_root / "raw"
    )
    raw_result = recover_bias_free(
        raw_oracle.query, output_dim=m, input_dim=d,
        anchor_coordinate=raw_anchor
    )
    raw_expected = row_canonicalize(raw_weight, raw_weight[:, raw_anchor])

    folded_oracle = FreshTFHEOracle(
        folded_weight, folded_bias, args.seed + 1, cache_root / "folded"
    )
    folded_result = recover_bias_anchored(
        folded_oracle.query, output_dim=m, input_dim=d
    )
    folded_expected = row_canonicalize(folded_weight, folded_bias)

    output = {
        "experiment": "Concrete-TFHE fresh-shuffle mixed-query row recovery",
        "concrete_version": getattr(fhe, "__version__", "unknown"),
        "checkpoint": str(args.checkpoint),
        "precision": args.precision,
        "coefficients": [1, 2, 3],
        "shape": [m, d],
        "fresh_output_permutation_per_query": True,
        "raw_bias_free": {
            **summarize(raw_result, raw_oracle, raw_expected),
            "anchor_coordinate": raw_anchor,
        },
        "folded_bias_anchored": summarize(
            folded_result, folded_oracle, folded_expected
        ),
        "wall_seconds": time.time() - started,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(output, handle, indent=2)
    print(json.dumps(output, indent=2), flush=True)


if __name__ == "__main__":
    main()

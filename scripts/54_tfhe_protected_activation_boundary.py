"""Measure the Concrete-TFHE boundary for protecting the first ReLU.

For a requested unsigned input range, this script compiles the folded
ResNet-20 first affine operator and then attempts to compile the same operator
followed by ReLU.  Unsupported accumulator widths are recorded as a result,
not hidden as a crashed run.
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

try:
    from concrete import fhe
except ImportError as exc:
    raise SystemExit("concrete-python is required for this experiment") from exc


def parse_args():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint", type=Path, default=root / "models" / "resnet20_seed0.pt"
    )
    parser.add_argument("--precision", type=int, default=256)
    parser.add_argument("--input-max", type=int, default=255)
    parser.add_argument("--seed", type=int, default=20260801)
    parser.add_argument("--evaluations", type=int, default=10)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def folded_first_layer(path: Path, precision: int):
    model = ResNet20()
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint.get("model_state_dict", checkpoint))
    model.eval()
    conv, bn = model.conv1, model.bn1
    raw = conv.weight.detach().cpu().numpy().reshape(conv.out_channels, -1)
    conv_bias = np.zeros(conv.out_channels, dtype=np.float64)
    if conv.bias is not None:
        conv_bias = conv.bias.detach().cpu().numpy().astype(np.float64)
    scale = bn.weight.detach().cpu().numpy() / np.sqrt(
        bn.running_var.detach().cpu().numpy() + bn.eps
    )
    weight = np.rint(raw * scale[:, None] * precision).astype(np.int64)
    bias = np.rint(
        (bn.bias.detach().cpu().numpy()
         + scale * (conv_bias - bn.running_mean.detach().cpu().numpy()))
        * precision
    ).astype(np.int64)
    return weight, bias


def config(cache: str):
    return fhe.Configuration(
        enable_unsafe_features=True,
        use_insecure_key_cache=True,
        insecure_key_cache_location=cache,
    )


def compile_affine(weight, bias, inputset, cache):
    @fhe.compiler({"x": "encrypted"})
    def function(x):
        return weight @ x + bias

    started = time.time()
    circuit = function.compile(inputset, configuration=config(cache))
    circuit.keygen()
    return circuit, time.time() - started


def compile_protected(weight, bias, inputset, cache):
    @fhe.compiler({"x": "encrypted"})
    def function(x):
        y = weight @ x + bias
        return np.maximum(y, 0)

    started = time.time()
    circuit = function.compile(inputset, configuration=config(cache))
    circuit.keygen()
    return circuit, time.time() - started


def evaluate(circuit, inputs):
    started = time.time()
    outputs = [
        np.asarray(circuit.encrypt_run_decrypt(x), dtype=np.int64) for x in inputs
    ]
    return outputs, time.time() - started


def main():
    args = parse_args()
    wall_started = time.time()
    if args.input_max < 1:
        raise ValueError("input-max must be positive")
    root = Path(__file__).resolve().parents[1]
    output_path = args.output or (
        root / "outputs" / "logs" /
        f"54_tfhe_protected_activation_input{args.input_max}.json"
    )
    weight, bias = folded_first_layer(args.checkpoint, args.precision)
    rng = np.random.default_rng(args.seed + args.input_max)
    inputset = [
        np.zeros(weight.shape[1], dtype=np.int64),
        np.full(weight.shape[1], args.input_max, dtype=np.int64),
        *[
            rng.integers(
                0, args.input_max + 1, size=weight.shape[1], dtype=np.int64
            )
            for _ in range(96)
        ],
    ]
    eval_inputs = [
        rng.integers(0, args.input_max + 1, size=weight.shape[1], dtype=np.int64)
        for _ in range(args.evaluations)
    ]

    affine, affine_compile = compile_affine(
        weight, bias, inputset, f"/tmp/p050_tfhe_affine_{args.input_max}"
    )
    affine_outputs, affine_eval = evaluate(affine, eval_inputs)
    clear_affine = [weight @ x + bias for x in eval_inputs]
    affine_exact = all(
        np.array_equal(a, b) for a, b in zip(affine_outputs, clear_affine)
    )

    base = {
        "experiment": "Concrete-TFHE protected first activation boundary",
        "concrete_version": getattr(fhe, "__version__", "unknown"),
        "checkpoint": str(args.checkpoint),
        "layer": "folded conv1+bn followed by relu",
        "shape": [int(weight.shape[0]), int(weight.shape[1])],
        "precision": args.precision,
        "input_range": [0, args.input_max],
        "evaluations": args.evaluations,
        "baseline_affine": {
            "compile_and_keygen_seconds": affine_compile,
            "total_encrypt_run_decrypt_seconds": affine_eval,
            "seconds_per_evaluation": affine_eval / args.evaluations,
            "exact": affine_exact,
        },
        "scope": (
            "Single folded first-layer vector microbenchmark; excludes network, "
            "ciphertext traffic, and end-to-end model latency."
        ),
    }

    try:
        protected, protected_compile = compile_protected(
            weight, bias, inputset, f"/tmp/p050_tfhe_relu_{args.input_max}"
        )
    except RuntimeError as exc:
        output = {
            **base,
            "protected_affine_relu": {
                "status": "compile_unsupported",
                "error": str(exc),
            },
            "wall_seconds": time.time() - wall_started,
        }
    else:
        protected_outputs, protected_eval = evaluate(protected, eval_inputs)
        clear_relu = [np.maximum(y, 0) for y in clear_affine]
        protected_exact = all(
            np.array_equal(a, b) for a, b in zip(protected_outputs, clear_relu)
        )
        output = {
            **base,
            "protected_affine_relu": {
                "status": "exact" if protected_exact else "wrong",
                "compile_and_keygen_seconds": protected_compile,
                "total_encrypt_run_decrypt_seconds": protected_eval,
                "seconds_per_evaluation": protected_eval / args.evaluations,
                "exact": protected_exact,
            },
            "online_latency_ratio": (
                protected_eval / affine_eval if affine_eval > 0 else None
            ),
            "wall_seconds": time.time() - wall_started,
        }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(output, indent=2), flush=True)
    if not affine_exact:
        raise SystemExit("baseline affine output mismatch")


if __name__ == "__main__":
    main()

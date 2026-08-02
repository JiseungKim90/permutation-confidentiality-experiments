"""Concrete-TFHE recovery of a later layer with fixed hidden permutations.

The selected ResNet-20 shortcut convolution has both a hidden input-coordinate
permutation and a hidden output-coordinate permutation.  When those
permutations remain fixed across repeated layer queries, their composition with
the trained layer is simply a labeled affine map in client coordinates.
Zero/basis queries recover that effective map exactly.
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

from lib.attack import get_conv_layers
from lib.models import ResNet20

try:
    from concrete import fhe
except ImportError as exc:
    raise SystemExit("concrete-python is required for this experiment") from exc


def parse_args():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint", type=Path,
        default=root / "models" / "resnet20_seed0.pt"
    )
    parser.add_argument(
        "--layer", default="layer2.0.shortcut.0"
    )
    parser.add_argument("--precision", type=int, default=256)
    parser.add_argument("--seed", type=int, default=20260801)
    parser.add_argument(
        "--output", type=Path,
        default=root / "outputs" / "logs" /
        "49_tfhe_fixed_session_later_layer.json"
    )
    return parser.parse_args()


def load_layer(checkpoint_path, layer_name, precision):
    model = ResNet20()
    checkpoint = torch.load(
        checkpoint_path, map_location="cpu", weights_only=False
    )
    state = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state)
    layers = {
        name: (weight, bias)
        for name, weight, bias, _has_bias, _output_dim, _input_dim
        in get_conv_layers(model.eval())
    }
    if layer_name not in layers:
        raise KeyError(f"unknown convolutional layer: {layer_name}")
    weight, bias = layers[layer_name]
    return (
        np.rint(weight * precision).astype(np.int64),
        np.rint(bias * precision).astype(np.int64),
    )


def compile_circuit(effective_weight, effective_bias, seed):
    rng = np.random.default_rng(seed)
    weight = np.asarray(effective_weight, dtype=np.int64)
    bias = np.asarray(effective_bias, dtype=np.int64)

    @fhe.compiler({"x": "encrypted"})
    def fixed_session_layer(x):
        return weight @ x + bias

    inputset = [
        np.zeros(weight.shape[1], dtype=np.int64),
        np.ones(weight.shape[1], dtype=np.int64),
        *[
            rng.integers(0, 2, size=weight.shape[1], dtype=np.int64)
            for _ in range(64)
        ],
    ]
    circuit = fixed_session_layer.compile(
        inputset,
        configuration=fhe.Configuration(
            enable_unsafe_features=True,
            use_insecure_key_cache=True,
            insecure_key_cache_location="/tmp/p050_tfhe_session_keys",
        ),
    )
    circuit.keygen()
    return circuit


def main():
    args = parse_args()
    started = time.time()
    weight, bias = load_layer(
        args.checkpoint, args.layer, args.precision
    )
    output_dim, input_dim = weight.shape
    rng = np.random.default_rng(args.seed)
    input_permutation = rng.permutation(input_dim)
    output_permutation = rng.permutation(output_dim)
    effective_weight = weight[
        output_permutation
    ][:, input_permutation]
    effective_bias = bias[output_permutation]

    compile_started = time.time()
    circuit = compile_circuit(
        effective_weight, effective_bias, args.seed + 1
    )
    compile_seconds = time.time() - compile_started

    evaluation_started = time.time()
    zero = np.zeros(input_dim, dtype=np.int64)
    recovered_bias = np.asarray(
        circuit.encrypt_run_decrypt(zero), dtype=np.int64
    )
    recovered_weight = np.empty(
        (output_dim, input_dim), dtype=np.int64
    )
    for column in range(input_dim):
        x = np.zeros(input_dim, dtype=np.int64)
        x[column] = 1
        response = np.asarray(
            circuit.encrypt_run_decrypt(x), dtype=np.int64
        )
        recovered_weight[:, column] = response - recovered_bias
    evaluation_seconds = time.time() - evaluation_started

    weight_error = int(np.max(np.abs(
        recovered_weight - effective_weight
    )))
    bias_error = int(np.max(np.abs(
        recovered_bias - effective_bias
    )))
    output = {
        "experiment": "Concrete-TFHE fixed-session later-layer recovery",
        "concrete_version": getattr(fhe, "__version__", "unknown"),
        "checkpoint": str(args.checkpoint),
        "layer": args.layer,
        "shape": [int(output_dim), int(input_dim)],
        "precision": args.precision,
        "fixed_hidden_input_permutation": True,
        "fixed_hidden_output_permutation": True,
        "queries": 1 + input_dim,
        "max_weight_lattice_error": weight_error,
        "max_bias_lattice_error": bias_error,
        "status": "exact" if max(weight_error, bias_error) == 0 else "wrong",
        "compile_and_keygen_seconds": compile_seconds,
        "encrypted_evaluation_seconds": evaluation_seconds,
        "scope": (
            "Requires repeated access to the same layer while its session "
            "permutations are retained."
        ),
        "wall_seconds": time.time() - started,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(output, handle, indent=2)
    print(json.dumps(output, indent=2), flush=True)


if __name__ == "__main__":
    main()

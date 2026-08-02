"""Concrete-TFHE validation of state-machine-compliant orbit recovery.

The affine computation is executed by Concrete-TFHE.  At the protocol wrapper
boundary, every query uses a unique session identifier and a monotonic round
counter; stale and duplicate requests are rejected.  Input and output
permutations are derived independently for every session/round pair.  A
constant anchor plus one-coordinate perturbations recovers a later ResNet-20
layer up to column permutation while using the target round only once in each
fresh session.
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

try:
    from concrete import fhe
except ImportError as exc:
    raise SystemExit("concrete-python is required for this experiment") from exc


def parse_args():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=root / "models" / "resnet20_seed0.pt",
    )
    parser.add_argument("--layer", default="layer1.1.conv2")
    parser.add_argument("--precision", type=int, default=256)
    parser.add_argument("--input-min", type=int, default=0)
    parser.add_argument("--input-max", type=int, default=15)
    parser.add_argument("--beta", type=int, default=1)
    parser.add_argument("--target-round", type=int, default=5)
    parser.add_argument("--failure-probability", type=float, default=1e-6)
    parser.add_argument("--seed", type=int, default=20260801)
    parser.add_argument(
        "--output",
        type=Path,
        default=root / "outputs" / "logs" /
        "58_tfhe_fresh_session_orbit_recovery.json",
    )
    return parser.parse_args()


def load_layer(checkpoint_path: Path, layer_name: str, precision: int):
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


def compile_circuit(weight, bias, input_min, input_max, seed):
    rng = np.random.default_rng(seed)

    @fhe.compiler({"x": "encrypted"})
    def affine(x):
        return weight @ x + bias

    inputset = [
        np.full(weight.shape[1], input_min, dtype=np.int64),
        np.full(weight.shape[1], input_max, dtype=np.int64),
        *[
            rng.integers(
                input_min,
                input_max + 1,
                size=weight.shape[1],
                dtype=np.int64,
            )
            for _ in range(96)
        ],
    ]
    circuit = affine.compile(
        inputset,
        configuration=fhe.Configuration(
            enable_unsafe_features=True,
            use_insecure_key_cache=True,
            insecure_key_cache_location="/tmp/p050_tfhe_fresh_orbit_keys",
        ),
    )
    circuit.keygen()
    return circuit


def column_counter(matrix):
    return Counter(
        tuple(int(v) for v in matrix[:, column])
        for column in range(matrix.shape[1])
    )


class StrictSessionStateMachine:
    """Executable unique-session and monotonic-round validation wrapper."""

    def __init__(self, target_round):
        if target_round < 0:
            raise ValueError("target round must be nonnegative")
        self.target_round = target_round
        self.next_session_id = 0
        self.next_round = {}
        self.accepted_target_pairs = set()
        self.rejected_stale_or_duplicate = 0

    def open_session(self):
        session_id = self.next_session_id
        self.next_session_id += 1
        self.next_round[session_id] = 0
        return session_id

    def _accept_round(self, session_id, round_index):
        expected = self.next_round.get(session_id)
        if expected is None or round_index != expected:
            self.rejected_stale_or_duplicate += 1
            raise ValueError("unknown, stale, skipped, or duplicate round")
        self.next_round[session_id] = expected + 1

    def advance_control_round(self, session_id, round_index):
        if round_index >= self.target_round:
            raise ValueError("control advance reached the protected target round")
        self._accept_round(session_id, round_index)

    def authorize_target(self, session_id, round_index):
        if round_index != self.target_round:
            self.rejected_stale_or_duplicate += 1
            raise ValueError("request does not address the configured target round")
        self._accept_round(session_id, round_index)
        pair = (session_id, round_index)
        if pair in self.accepted_target_pairs:
            raise AssertionError("duplicate target pair passed monotonic validation")
        self.accepted_target_pairs.add(pair)

    def statistics(self):
        return {
            "unique_sessions_opened": self.next_session_id,
            "accepted_target_requests": len(self.accepted_target_pairs),
            "unique_session_round_pairs": len(self.accepted_target_pairs),
            "rejected_stale_or_duplicate_requests": (
                self.rejected_stale_or_duplicate
            ),
            "all_target_rounds_used_once": (
                len(self.accepted_target_pairs) == self.next_session_id
            ),
        }


def main():
    args = parse_args()
    started = time.time()
    if args.beta <= 0:
        raise ValueError("this experiment requires positive beta")
    alpha = args.input_max - args.beta
    if alpha < args.input_min:
        raise ValueError("input range cannot contain the perturbation")

    weight, bias = load_layer(args.checkpoint, args.layer, args.precision)
    output_dim, input_dim = weight.shape
    true_anchor = alpha * weight.sum(axis=1, dtype=np.int64) + bias
    order = np.argsort(true_anchor, kind="stable")
    sorted_anchor = true_anchor[order]
    distinct_anchor = np.unique(sorted_anchor).size == output_dim
    minimum_gap = (
        int(np.min(np.diff(sorted_anchor))) if distinct_anchor else 0
    )
    perturbation_bound = int(args.beta * np.max(np.abs(weight)))
    certified = distinct_anchor and minimum_gap > 2 * perturbation_bound
    order_preserving = distinct_anchor and all(
        np.array_equal(
            np.argsort(
                true_anchor + args.beta * weight[:, column],
                kind="stable",
            ),
            order,
        )
        for column in range(input_dim)
    )

    compile_started = time.time()
    circuit = compile_circuit(
        weight, bias, args.input_min, args.input_max, args.seed + 1
    )
    compile_seconds = time.time() - compile_started

    state_machine = StrictSessionStateMachine(args.target_round)
    encrypted_evaluation_seconds = 0.0
    sampled_columns = []

    def query(client_input, record_column=False):
        nonlocal encrypted_evaluation_seconds
        session_id = state_machine.open_session()
        for round_index in range(args.target_round):
            state_machine.advance_control_round(session_id, round_index)
        state_machine.authorize_target(session_id, args.target_round)
        round_rng = np.random.default_rng(
            np.random.SeedSequence([args.seed, session_id, args.target_round])
        )
        input_permutation = round_rng.permutation(input_dim)
        output_permutation = round_rng.permutation(output_dim)
        semantic_server_input = np.ascontiguousarray(
            client_input[input_permutation], dtype=np.int64
        )
        eval_started = time.time()
        response = np.asarray(
            circuit.encrypt_run_decrypt(semantic_server_input), dtype=np.int64
        )
        encrypted_evaluation_seconds += time.time() - eval_started
        if record_column:
            sampled_columns.append(
                int(np.flatnonzero(input_permutation == 0)[0])
            )
        try:
            state_machine.authorize_target(session_id, args.target_round)
        except ValueError:
            pass
        else:
            raise AssertionError("duplicate target request was not rejected")
        return response[output_permutation]

    constant = np.full(input_dim, alpha, dtype=np.int64)
    observed_anchor = np.sort(query(constant))
    coupon_queries = int(math.ceil(
        input_dim * (
            math.log(input_dim) - math.log(args.failure_probability)
        )
    ))
    recovered_columns = []
    for _ in range(coupon_queries):
        client_input = constant.copy()
        client_input[0] += args.beta
        delta = np.sort(query(client_input, record_column=True)) - observed_anchor
        recovered_columns.append(tuple(int(v) for v in delta // args.beta))

    recovered_counter = Counter(recovered_columns)
    true_counter = column_counter(weight[order])
    no_false_columns = all(column in true_counter for column in recovered_counter)
    all_true_columns_seen = all(column in recovered_counter for column in true_counter)
    attack_declared_complete = len(recovered_counter) == input_dim
    recovered_bias_exact = False
    if attack_declared_complete:
        recovered_matrix = np.asarray(
            list(recovered_counter.keys()), dtype=np.int64
        ).T
        recovered_bias = observed_anchor - alpha * recovered_matrix.sum(axis=1)
        recovered_bias_exact = bool(np.array_equal(recovered_bias, bias[order]))

    exact = (
        order_preserving
        and attack_declared_complete
        and all_true_columns_seen
        and no_false_columns
        and recovered_bias_exact
    )
    output = {
        "experiment": "Concrete-TFHE state-machine-compliant orbit recovery",
        "concrete_version": getattr(fhe, "__version__", "unknown"),
        "checkpoint": str(args.checkpoint),
        "layer": args.layer,
        "shape": [int(output_dim), int(input_dim)],
        "precision": args.precision,
        "input_range": [args.input_min, args.input_max],
        "alpha": alpha,
        "beta": args.beta,
        "target_round": args.target_round,
        "unique_session_id_each_query": True,
        "monotonic_round_counter_enforced": True,
        "stale_and_duplicate_requests_rejected": True,
        "fresh_hidden_input_permutation_each_query": True,
        "fresh_hidden_output_permutation_each_query": True,
        "same_round_replay": False,
        "minimum_anchor_gap": minimum_gap,
        "perturbation_bound": perturbation_bound,
        "certified_interval_separation": bool(certified),
        "all_columns_preserve_order": bool(order_preserving),
        "failure_probability_bound": args.failure_probability,
        "queries": 1 + coupon_queries,
        "sampled_latent_column_count": len(set(sampled_columns)),
        "recovered_distinct_column_count": len(recovered_counter),
        "attack_declared_complete": attack_declared_complete,
        "all_true_columns_seen": bool(all_true_columns_seen),
        "no_false_columns": bool(no_false_columns),
        "bias_recovered_exactly": recovered_bias_exact,
        "status": "exact_orbit" if exact else "not_exact",
        "state_machine": state_machine.statistics(),
        "compile_and_keygen_seconds": compile_seconds,
        "encrypted_evaluation_seconds": encrypted_evaluation_seconds,
        "seconds_per_encrypted_query": (
            encrypted_evaluation_seconds / (1 + coupon_queries)
        ),
        "scope": (
            "The target affine circuit is Concrete-TFHE.  An executable "
            "wrapper enforces unique sessions and monotonic rounds and derives "
            "fresh secret permutations per session/round.  Earlier rounds are "
            "validated control transitions, not a complete encrypted network."
        ),
        "wall_seconds": time.time() - started,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(output, indent=2), flush=True)


if __name__ == "__main__":
    main()

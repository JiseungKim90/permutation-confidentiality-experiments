#!/usr/bin/env python3
"""Exhaustive checks for finite-domain transcript fibres and frame control.

The script uses only the Python standard library.  A model is represented by a
multiset of affine rows, so distinct enumerated models are already quotiented by
the unavoidable common row permutation.  It is intended to run on ubuntu02;
local execution is only a syntax/smoke check.
"""

import argparse
import hashlib
import itertools
import json
import os
import platform
import socket
import subprocess
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path


def git_head(path: Path):
    try:
        return subprocess.check_output(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def row_space(dimension, coefficients):
    return tuple(itertools.product(coefficients, repeat=dimension + 1))


def input_grid(dimension, values):
    return tuple(itertools.product(values, repeat=dimension))


def affine_value(row, point):
    return sum(w * x for w, x in zip(row[:-1], point)) + row[-1]


def clipped(value, bounds):
    if bounds is None:
        return value
    low, high = bounds
    return min(max(value, low), high)


def model_signature(model, points, clip_bounds=None):
    return tuple(
        tuple(sorted(clipped(affine_value(row, point), clip_bounds) for row in model))
        for point in points
    )


def serialise_model(model):
    return [list(row) for row in model]


def analyse_fibres(name, dimension, width, coefficients, grid_values, clip_bounds=None):
    rows = row_space(dimension, coefficients)
    points = input_grid(dimension, grid_values)
    models = itertools.combinations_with_replacement(rows, width)
    fibres = defaultdict(list)
    orbit_count = 0
    for model in models:
        orbit_count += 1
        fibres[model_signature(model, points, clip_bounds)].append(model)

    sizes = [len(fibre) for fibre in fibres.values()]
    ambiguous = [fibre for fibre in fibres.values() if len(fibre) > 1]
    witness = None
    if ambiguous:
        first = sorted(ambiguous, key=lambda f: (-len(f), f))[0]
        witness = {
            "fibre_size": len(first),
            "models": [serialise_model(model) for model in first[:3]],
            "signature": [list(reply) for reply in model_signature(first[0], points, clip_bounds)],
        }

    return {
        "name": name,
        "dimension": dimension,
        "width": width,
        "coefficient_alphabet": list(coefficients),
        "input_grid_values": list(grid_values),
        "input_count": len(points),
        "clip_bounds": list(clip_bounds) if clip_bounds is not None else None,
        "row_permutation_orbits": orbit_count,
        "transcript_classes": len(fibres),
        "singleton_classes": sum(size == 1 for size in sizes),
        "nontrivial_classes": len(ambiguous),
        "orbits_in_nontrivial_classes": sum(len(fibre) for fibre in ambiguous),
        "largest_fibre": max(sizes),
        "witness": witness,
    }


def permute(vector, permutation):
    return tuple(vector[index] for index in permutation)


def inverse_permutation(permutation):
    inverse = [0] * len(permutation)
    for output_index, input_index in enumerate(permutation):
        inverse[input_index] = output_index
    return tuple(inverse)


def partition_condition(response, target):
    return all(
        response[i] != response[j] or target[i] == target[j]
        for i in range(len(response))
        for j in range(i + 1, len(response))
    )


def stabilizer_condition(response, target, permutations):
    stabilizer = [p for p in permutations if permute(response, p) == response]
    return all(permute(target, p) == target for p in stabilizer)


def controller(response, ordered_response, target, permutations):
    representative = next(p for p in permutations if permute(response, p) == ordered_response)
    return permute(target, representative)


def analyse_control(max_width):
    records = []
    total_mismatches = 0
    total_controller_failures = 0
    for width in range(1, max_width + 1):
        vectors = tuple(itertools.product((0, 1), repeat=width))
        permutations = tuple(itertools.permutations(range(width)))
        mismatches = 0
        controller_failures = 0
        controllable = 0
        for response in vectors:
            for target in vectors:
                direct = partition_condition(response, target)
                brute = stabilizer_condition(response, target, permutations)
                if direct != brute:
                    mismatches += 1
                if not direct:
                    continue
                controllable += 1
                for actual in permutations:
                    ordered = permute(response, actual)
                    message = controller(response, ordered, target, permutations)
                    decoded = permute(message, inverse_permutation(actual))
                    if decoded != target:
                        controller_failures += 1
        total_mismatches += mismatches
        total_controller_failures += controller_failures
        records.append(
            {
                "width": width,
                "response_target_pairs": len(vectors) ** 2,
                "controllable_pairs": controllable,
                "condition_mismatches": mismatches,
                "controller_failures": controller_failures,
                "permutations_per_pair": len(permutations),
            }
        )
    return {
        "alphabet": [0, 1],
        "max_width": max_width,
        "records": records,
        "total_condition_mismatches": total_mismatches,
        "total_controller_failures": total_controller_failures,
    }


def explicit_finite_domain_witness():
    points = ((0,), (1,))
    model_a = ((-1, 1), (1, 0))
    model_b = ((0, 0), (0, 1))
    signature_a = model_signature(model_a, points)
    signature_b = model_signature(model_b, points)
    at_two_a = model_signature(model_a, ((2,),))
    at_two_b = model_signature(model_b, ((2,),))
    return {
        "inputs": [0, 1],
        "model_a_rows": serialise_model(model_a),
        "model_b_rows": serialise_model(model_b),
        "signature_a": [list(reply) for reply in signature_a],
        "signature_b": [list(reply) for reply in signature_b],
        "same_on_admissible_domain": signature_a == signature_b,
        "distinguished_at_input_2": at_two_a != at_two_b,
        "response_at_2_a": list(at_two_a[0]),
        "response_at_2_b": list(at_two_b[0]),
    }


def second_layer_value(rows, point, clip_bounds=None):
    return tuple(
        clipped(affine_value(row, point), clip_bounds)
        for row in rows
    )


def chain_kernel_signature(first_rows, second_rows, first_inputs, second_inputs,
                           clip_first=None, clip_second=None):
    """Complete one-session kernel for a two-round width-two protocol.

    The first reply is shuffled by one hidden permutation.  After observing it,
    the client may choose any listed second-round message.  The final reply is
    labelled and unshuffled.  Recording every conditional output distribution
    therefore determines the transcript law against every adaptive client; fresh
    multi-session laws are products of this kernel.
    """
    permutations = tuple(itertools.permutations(range(2)))
    signature = []
    for scalar_input in first_inputs:
        point = (scalar_input,)
        canonical = tuple(
            clipped(affine_value(row, point), clip_first)
            for row in first_rows
        )
        posterior = defaultdict(list)
        for permutation in permutations:
            posterior[permute(canonical, permutation)].append(permutation)
        branches = []
        for ordered_reply in sorted(posterior):
            compatible = posterior[ordered_reply]
            continuations = []
            for message in second_inputs:
                outputs = []
                for permutation in compatible:
                    true_input = permute(message, inverse_permutation(permutation))
                    outputs.append(second_layer_value(
                        second_rows, true_input, clip_second
                    ))
                continuations.append((message, tuple(sorted(outputs))))
            branches.append((
                ordered_reply,
                len(compatible),
                tuple(continuations),
            ))
        signature.append((scalar_input, tuple(branches)))
    return tuple(signature)


def transform_two_round_gauge(first_rows, second_rows, permutation):
    transformed_first = permute(first_rows, permutation)
    transformed_second = tuple(
        permute(row[:-1], permutation) + (row[-1],)
        for row in second_rows
    )
    return transformed_first, transformed_second


def canonical_two_round_gauge(first_rows, second_rows):
    permutations = tuple(itertools.permutations(range(2)))
    return min(
        transform_two_round_gauge(first_rows, second_rows, permutation)
        for permutation in permutations
    )


def serialise_chain(chain):
    first_rows, second_rows = chain
    return {
        "first_layer_rows": serialise_model(first_rows),
        "second_layer_rows": serialise_model(second_rows),
    }


def analyse_two_round_kernels(name, first_inputs, second_inputs,
                              coefficients=(-1, 0, 1),
                              clip_first=None, clip_second=None):
    """Enumerate gauge orbits and complete adaptive kernels for two layers.

    Dimensions are 1 -> 2 -> 2.  The only hidden gauge swaps the two internal
    coordinates; the final two output coordinates remain labelled.
    """
    first_row_space = row_space(1, coefficients)
    second_row_space = row_space(2, coefficients)
    gauge_orbits = {}
    gauge_invariance_failures = 0
    for first_rows in itertools.product(first_row_space, repeat=2):
        for second_rows in itertools.product(second_row_space, repeat=2):
            canonical = canonical_two_round_gauge(first_rows, second_rows)
            if canonical in gauge_orbits:
                continue
            first_can, second_can = canonical
            signature = chain_kernel_signature(
                first_can,
                second_can,
                first_inputs,
                second_inputs,
                clip_first,
                clip_second,
            )
            swapped = transform_two_round_gauge(first_can, second_can, (1, 0))
            if chain_kernel_signature(
                swapped[0],
                swapped[1],
                first_inputs,
                second_inputs,
                clip_first,
                clip_second,
            ) != signature:
                gauge_invariance_failures += 1
            separating = any(
                len(set(
                    clipped(affine_value(row, (x,)), clip_first)
                    for row in first_can
                )) == 2
                for x in first_inputs
            )
            gauge_orbits[canonical] = (signature, separating)

    fibres = defaultdict(list)
    separating_fibres = defaultdict(list)
    for chain, (signature, separating) in gauge_orbits.items():
        fibres[signature].append(chain)
        if separating:
            separating_fibres[signature].append(chain)

    sizes = [len(fibre) for fibre in fibres.values()]
    separating_sizes = [len(fibre) for fibre in separating_fibres.values()]
    ambiguous = [fibre for fibre in fibres.values() if len(fibre) > 1]
    separating_ambiguous = [
        fibre for fibre in separating_fibres.values() if len(fibre) > 1
    ]
    witness = None
    if ambiguous:
        first = sorted(ambiguous, key=lambda fibre: (-len(fibre), fibre))[0]
        witness = {
            "fibre_size": len(first),
            "chains": [serialise_chain(chain) for chain in first[:3]],
        }
    return {
        "name": name,
        "dimensions": [1, 2, 2],
        "coefficient_alphabet": list(coefficients),
        "first_inputs": list(first_inputs),
        "second_inputs": [list(point) for point in second_inputs],
        "clip_first": list(clip_first) if clip_first is not None else None,
        "clip_second": list(clip_second) if clip_second is not None else None,
        "gauge_orbits": len(gauge_orbits),
        "transcript_classes": len(fibres),
        "nontrivial_classes": len(ambiguous),
        "orbits_in_nontrivial_classes": sum(len(fibre) for fibre in ambiguous),
        "largest_fibre": max(sizes),
        "separating_gauge_orbits": sum(
            separating for _, separating in gauge_orbits.values()
        ),
        "separating_transcript_classes": len(separating_fibres),
        "separating_nontrivial_classes": len(separating_ambiguous),
        "separating_orbits_in_nontrivial_classes": sum(
            len(fibre) for fibre in separating_ambiguous
        ),
        "separating_largest_fibre": max(separating_sizes, default=0),
        "gauge_invariance_failures": gauge_invariance_failures,
        "witness": witness,
    }


def build_report(max_control_width):
    source = Path(__file__).resolve()
    source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    cases = [
        analyse_fibres("m2_d1_signed_two_inputs", 1, 2, (-1, 0, 1), (0, 1)),
        analyse_fibres("m2_d1_signed_grid_m_plus_1", 1, 2, (-1, 0, 1), (-1, 0, 1)),
        analyse_fibres("m3_d1_signed_two_inputs", 1, 3, (-1, 0, 1), (-1, 0)),
        analyse_fibres("m3_d1_signed_grid_m_plus_1", 1, 3, (-1, 0, 1), (-1, 0, 1, 2)),
        analyse_fibres("m2_d2_signed_boolean_grid", 2, 2, (-1, 0, 1), (0, 1)),
        analyse_fibres("m2_d2_signed_grid_m_plus_1", 2, 2, (-1, 0, 1), (-1, 0, 1)),
        analyse_fibres(
            "m2_d2_signed_grid_with_clipping",
            2,
            2,
            (-1, 0, 1),
            (-1, 0, 1),
            (-1, 1),
        ),
        analyse_fibres(
            "m2_d1_wide_grid_m_plus_1",
            1,
            2,
            (-2, -1, 0, 1, 2),
            (-1, 0, 1),
        ),
        analyse_fibres(
            "m2_d1_wide_grid_with_clipping",
            1,
            2,
            (-2, -1, 0, 1, 2),
            (-1, 0, 1),
            (-1, 1),
        ),
    ]
    witness = explicit_finite_domain_witness()
    control = analyse_control(max_control_width)
    second_grid = tuple(itertools.product((-1, 0, 1), repeat=2))
    two_round = [
        analyse_two_round_kernels(
            "two_round_rich_exact",
            (-1, 0, 1),
            second_grid,
        ),
        analyse_two_round_kernels(
            "two_round_restricted_first_domain",
            (0, 1),
            second_grid,
        ),
        analyse_two_round_kernels(
            "two_round_rich_clipped",
            (-1, 0, 1),
            second_grid,
            clip_first=(-1, 1),
            clip_second=(-1, 1),
        ),
    ]
    rich_two_round = next(
        case for case in two_round if case["name"] == "two_round_rich_exact"
    )
    success = (
        witness["same_on_admissible_domain"]
        and witness["distinguished_at_input_2"]
        and control["total_condition_mismatches"] == 0
        and control["total_controller_failures"] == 0
        and all(
            case["nontrivial_classes"] == 0
            for case in cases
            if case["name"].endswith("grid_m_plus_1")
        )
        and next(
            case for case in cases if case["name"] == "m2_d1_wide_grid_with_clipping"
        )["nontrivial_classes"] > 0
        and rich_two_round["gauge_invariance_failures"] == 0
        and rich_two_round["separating_nontrivial_classes"] == 0
    )
    return {
        "schema": "p050-tdsc-identifiability-exhaustive-v2",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "success": success,
        "source": {
            "path": str(source),
            "sha256": source_hash,
            "git_head": git_head(source.parent),
        },
        "environment": {
            "hostname": socket.gethostname(),
            "python": sys.version,
            "platform": platform.platform(),
            "pid": os.getpid(),
        },
        "finite_domain_witness": witness,
        "fibre_cases": cases,
        "stabilizer_control": control,
        "two_round_kernels": two_round,
        "success_criteria": {
            "explicit_witness_verified": True,
            "all_stabilizer_checks_zero_mismatch": True,
            "all_constructive_controller_checks_zero_failure": True,
            "unclipped_m_plus_1_grids_have_only_singleton_fibres": True,
            "wider_coefficient_clipped_case_has_nontrivial_fibre": True,
            "two_round_gauge_transform_preserves_kernel": True,
            "rich_separating_two_round_kernels_equal_gauge_orbits": True,
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-control-width", type=int, default=6)
    args = parser.parse_args()
    if args.max_control_width < 1 or args.max_control_width > 8:
        parser.error("--max-control-width must lie in [1,8]")
    report = build_report(args.max_control_width)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "output": str(args.output),
        "success": report["success"],
        "source_sha256": report["source"]["sha256"],
        "finite_cases": len(report["fibre_cases"]),
        "control_max_width": report["stabilizer_control"]["max_width"],
        "condition_mismatches": report["stabilizer_control"]["total_condition_mismatches"],
        "controller_failures": report["stabilizer_control"]["total_controller_failures"],
        "two_round_cases": len(report["two_round_kernels"]),
    }, sort_keys=True))
    return 0 if report["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Process-isolated completion of tied-class columns in the P050 R3 attack.

The canonical extractor recovers all affine rows that can be driven through
uniquely located reply coordinates.  Its current implementation writes zero to
an unresolved reply class.  That is unnecessarily restrictive: if several
canonical coordinates have the same predicted reply value, a client can still
write one common value to the entire class.

This experiment starts from the process-isolated 29-record artifact.  The
attacker discovers constant-zero stem channels from that artifact alone.  It
then builds class-constant first-block inputs, varies one tied class over a
public set of activation values, and reads the slopes of uniquely anchored
output coordinates.  Nine boundary masks invert a 3x3 kernel (their integer
matrix is unimodular, with determinant +1 or -1); a single anchored coordinate
recovers a 1x1
shortcut coefficient.  A third process, launched only after the oracle and
attacker exit, checks every round map in one coherent channel gauge and compares
10,000 logits.

The private checkpoint and CIFAR data are never opened by the attacker process.
The oracle transport is inherited from run_extraction and uses
raw byte messages only.
"""

from __future__ import print_function

import json
import multiprocessing as mp
import os
import time
import traceback

import numpy as np

from _bootstrap import DATA, RESULTS
from lib import fmap_partial as fp
from lib.fmap import conv_int
from lib.verify import (canonical_column_matrix, frame_check, row_permutation)
from run_extraction import (
    RpcOracleView,
    audit_attacker_view,
    canonical_json,
    install_data_guard,
    jsonable,
    load_network,
    model_digest,
    oracle_worker,
    remove_private_environment_for_spawn,
    restore_private_environment,
    save_network,
    secret_nonzero_count,
    sha256_bytes,
    sha256_file,
    zero_network_from_manifest,
)


RUN_NAME = os.environ.get(
    "TDSC_CLASS_RUN", "tdsc-completion")
OUT_DIR = os.path.join(RESULTS, "provenance", RUN_NAME)
SOURCE_RUN = os.environ.get(
    "TDSC_CLASS_SOURCE", "tdsc-extraction")
SOURCE_DIR = os.path.join(RESULTS, "provenance", SOURCE_RUN)
W_BITS = int(os.environ.get("TDSC_CLASS_WBITS", "8"))
N_TEST = int(os.environ.get("TDSC_CLASS_NTEST", "10000"))
SEARCH_LIMIT = int(os.environ.get("TDSC_CLASS_SEARCH", "256"))
MIN_MARKER_COVER = int(os.environ.get("TDSC_CLASS_COVER", "2"))
SHORTCUT_CARRIER_REPEATS = int(os.environ.get("TDSC_CLASS_SREPEATS", "2"))
TRACE_ARITHMETIC = bool(int(os.environ.get("TDSC_CLASS_TRACE", "0")))
SKIP_EVALUATOR = bool(int(os.environ.get("TDSC_CLASS_SKIP_EVAL", "0")))
CONTROL_VALUES = tuple(int(value) for value in os.environ.get(
    "TDSC_CLASS_VALUES", "1,2,4,8,16,32,64,128,255").split(","))


def read_json(path):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def integer_error(left, right):
    left = np.asarray(left, dtype=np.int64)
    right = np.asarray(right, dtype=np.int64)
    if left.shape != right.shape:
        return None
    return int(np.abs(left - right).max()) if left.size else 0


def exact_integer_determinant(matrix):
    """Fraction-free Bareiss determinant for the small integer systems."""
    values = np.asarray(matrix, dtype=np.int64)
    if values.ndim != 2 or values.shape[0] != values.shape[1]:
        raise ValueError("determinant requires a square matrix")
    n = int(values.shape[0])
    if n == 0:
        return 1
    work = [[int(value) for value in row] for row in values.tolist()]
    sign, previous = 1, 1
    for k in range(n - 1):
        pivot_row = next((r for r in range(k, n) if work[r][k]), None)
        if pivot_row is None:
            return 0
        if pivot_row != k:
            work[k], work[pivot_row] = work[pivot_row], work[k]
            sign = -sign
        pivot = work[k][k]
        for i in range(k + 1, n):
            for j in range(k + 1, n):
                numerator = work[i][j] * pivot - work[i][k] * work[k][j]
                if numerator % previous:
                    raise ArithmeticError("non-exact Bareiss division")
                work[i][j] = numerator // previous
            work[i][k] = 0
        previous = pivot
    return sign * int(work[n - 1][n - 1])


def class_index(sorted_values, observations):
    """Map observed reply values to their public predicted value classes."""
    values = np.asarray(sorted_values, dtype=np.int64)
    reply = np.asarray(observations, dtype=np.int64)
    indices = np.searchsorted(values, reply)
    valid = indices < values.size
    clipped = np.clip(indices, 0, max(0, values.size - 1))
    if not np.all(valid & (values[clipped] == reply)):
        raise AssertionError("oracle reply contains an unpredicted class value")
    return clipped


def unique_mask(values):
    flat = np.asarray(values, dtype=np.int64).reshape(-1)
    _, inverse, counts = np.unique(
        flat, return_inverse=True, return_counts=True)
    return (counts[inverse] == 1).reshape(np.asarray(values).shape)


def boundary_type(y, x, height, width):
    row = 0 if y == 0 else (2 if y == height - 1 else 1)
    col = 0 if x == 0 else (2 if x == width - 1 else 1)
    return 3 * row + col


def boundary_mask(y, x, height, width):
    return np.asarray([
        int(0 <= y - 1 + kh < height and 0 <= x - 1 + kw < width)
        for kh in range(3) for kw in range(3)
    ], dtype=np.int64)


def candidate_markers(baseline1, baseline2):
    """Return globally unique markers for every recoverable boundary type."""
    first_unique = unique_mask(baseline1)
    second_unique = unique_mask(baseline2)
    channels, height, width = baseline1.shape
    w_markers = {}
    s_markers = {}
    for channel in range(channels):
        for y in range(height):
            for x in range(width):
                if first_unique[channel, y, x]:
                    key = (int(channel), boundary_type(
                        y, x, height, width))
                    w_markers.setdefault(key, (int(y), int(x)))
                if second_unique[channel, y, x]:
                    s_markers.setdefault(int(channel), (int(y), int(x)))
    return w_markers, s_markers


def universe(n_channels):
    # At this stage the clone may contain two shortcut rows that become equal
    # precisely because their tied-class coefficients are missing.  Such rows
    # cannot have a unique zero-primary marker.  Shortcut markers are therefore
    # created later, after W1 recovery permits a controlled W2 carrier.
    return {("w", channel, kind)
            for channel in range(n_channels) for kind in range(9)}


def anchor_coverage(anchor):
    return {("w", channel, kind)
            for channel, kind in anchor["w_markers"]}


def choose_anchors(candidates, required, multiplicity):
    """Greedy multicover; all choices are made from clone predictions only."""
    remaining = {key: int(multiplicity) for key in required}
    available = list(range(len(candidates)))
    selected = []
    while any(value > 0 for value in remaining.values()):
        best = None
        best_score = -1
        for index in available:
            score = sum(
                1 for key in anchor_coverage(candidates[index])
                if remaining.get(key, 0) > 0)
            if score > best_score:
                best, best_score = index, score
        if best is None or best_score <= 0:
            missing = [key for key, value in remaining.items() if value > 0]
            raise RuntimeError("marker multicover failed for %d elements" % len(missing))
        selected.append(candidates[best])
        available.remove(best)
        for key in anchor_coverage(candidates[best]):
            if key in remaining and remaining[key] > 0:
                remaining[key] -= 1
    return selected


def find_public_image_and_anchors(net, rng, search_limit, min_cover):
    """Search using only the recovered network and the public alphabet."""
    height = width = 32
    block = net.blocks[0]
    stem_w = np.asarray(net.stem["W"], dtype=np.int64)
    stem_b = np.asarray(net.stem["b"], dtype=np.int64)
    zero_rows = np.all(stem_w == 0, axis=1)
    constants = np.clip(np.rint(
        np.maximum(stem_b, 0) / float(net.stem["eta"])),
        0, net.A).astype(np.int64)
    dead = np.nonzero(zero_rows & (constants == 0))[0].astype(np.int64)
    if dead.size == 0:
        raise RuntimeError("recovered clone exposes no constant-zero stem row")
    dead_values = [int(stem_b[channel]) for channel in dead]
    if len(set(dead_values)) != len(dead_values):
        raise RuntimeError("constant-zero stem rows do not have distinct affine values")

    image = None
    pred0 = None
    class_values = None
    inverse = None
    for _ in range(64):
        trial = rng.integers(
            0, net.A + 1, size=(3, height, width), dtype=np.int64)
        predicted = conv_int(trial, stem_w, stem_b, 1, 3)
        values, inv, counts = np.unique(
            predicted.reshape(-1), return_inverse=True, return_counts=True)
        count_map = dict(zip(values.tolist(), counts.tolist()))
        if all(count_map.get(value, 0) == height * width
               for value in dead_values):
            image, pred0 = trial, predicted
            class_values, inverse = values, inv
            break
    if image is None:
        raise RuntimeError("could not isolate every tied stem class by value")

    dead_class_indices = {
        int(channel): int(np.searchsorted(class_values, stem_b[channel]))
        for channel in dead
    }
    required = universe(block["C_out"])
    counts = {key: 0 for key in required}
    candidates = []
    for attempt in range(1, search_limit + 1):
        class_targets = rng.integers(
            0, net.A + 1, size=class_values.size, dtype=np.int64)
        for index in dead_class_indices.values():
            class_targets[index] = 0
        retained = class_targets[inverse].reshape(stem_w.shape[0], height, width)
        baseline1 = conv_int(
            retained, block["W1"], block["b1"], 1, 3)
        primary = np.zeros(
            (block["C_out"], height, width), dtype=np.int64)
        main = conv_int(
            primary, block["W2"], block["b2"] + block["bs"], 1, 3)
        shortcut = conv_int(
            retained, block["S"],
            np.zeros(block["C_out"], dtype=np.int64), 1, 1)
        baseline2 = main + shortcut
        w_markers, s_markers = candidate_markers(baseline1, baseline2)
        candidate = {
            "attempt": int(attempt),
            "class_targets": class_targets,
            "retained": retained,
            "baseline1": baseline1,
            "baseline2": baseline2,
            "w_markers": w_markers,
            "s_markers": s_markers,
        }
        candidates.append(candidate)
        for key in anchor_coverage(candidate):
            if key in counts:
                counts[key] += 1
        if all(value >= min_cover for value in counts.values()):
            break
    if not all(value >= min_cover for value in counts.values()):
        missing = [key for key, value in counts.items() if value < min_cover]
        raise RuntimeError(
            "anchor search exhausted with under-covered markers %r" % missing)
    selected = choose_anchors(candidates, required, min_cover)
    return {
        "image": image,
        "pred0": pred0,
        "class_values": class_values,
        "dead_channels": dead,
        "dead_class_indices": dead_class_indices,
        "candidates_tested": len(candidates),
        "anchors": selected,
    }


def line_candidates(intercept, observations, control_values):
    """All integer slopes whose full public control line occurs in the replies."""
    intercept = int(intercept)
    first = int(control_values[0])
    if first != 1:
        raise ValueError("the first control value must be one")
    candidates = np.unique(np.asarray(observations[first], dtype=np.int64)) - intercept
    for value in control_values[1:]:
        observed = np.unique(np.asarray(observations[int(value)], dtype=np.int64))
        wanted = intercept + int(value) * candidates
        indices = np.searchsorted(observed, wanted)
        valid = indices < observed.size
        clipped = np.clip(indices, 0, max(0, observed.size - 1))
        candidates = candidates[valid & (observed[clipped] == wanted)]
        if candidates.size == 0:
            break
    return np.unique(candidates).astype(np.int64)


def recover_w1_columns(net, plan, observations, control_values):
    """Recover all first-convolution columns fed by tied stem classes."""
    block = net.blocks[0]
    recovered_w1 = np.asarray(block["W1"], dtype=np.int64).copy()
    channel_reports = []
    candidate_histogram = {}

    for dead_channel in plan["dead_channels"].tolist():
        conv_equations = {row: [] for row in range(block["C_out"])}
        for anchor_index, anchor in enumerate(plan["anchors"]):
            round1 = observations[anchor_index][dead_channel]
            for (row, kind), (y, x) in anchor["w_markers"].items():
                intercept = int(anchor["baseline1"][row, y, x])
                candidates = line_candidates(intercept, round1, control_values)
                candidate_histogram[str(int(candidates.size))] = (
                    candidate_histogram.get(str(int(candidates.size)), 0) + 1)
                if candidates.size == 1:
                    conv_equations[row].append({
                        "mask": boundary_mask(y, x, 32, 32),
                        "slope": int(candidates[0]),
                        "anchor": int(anchor_index),
                        "boundary_type": int(kind),
                        "coordinate": [int(y), int(x)],
                    })
        row_reports = []
        for row in range(block["C_out"]):
            chosen = []
            rank = 0
            for equation in conv_equations[row]:
                trial = chosen + [equation]
                matrix = np.stack([item["mask"] for item in trial], axis=0)
                new_rank = int(np.linalg.matrix_rank(matrix))
                if new_rank > rank:
                    chosen, rank = trial, new_rank
                if rank == 9:
                    break
            if rank != 9:
                raise RuntimeError(
                    "rank-%d W1 system for input %d output %d" %
                    (rank, dead_channel, row))
            matrix = np.stack([item["mask"] for item in chosen], axis=0)
            determinant = exact_integer_determinant(matrix)
            if abs(determinant) != 1:
                raise AssertionError(
                    "selected 3x3 mask system is not unimodular")
            slopes = np.asarray(
                [item["slope"] for item in chosen], dtype=np.int64)
            estimate_float = np.linalg.solve(
                matrix.astype(np.float64), slopes.astype(np.float64))
            estimate = np.rint(estimate_float).astype(np.int64)
            if not np.array_equal(matrix @ estimate, slopes):
                raise AssertionError("3x3 integer inversion was not exact")
            all_matrix = np.stack(
                [item["mask"] for item in conv_equations[row]], axis=0)
            all_slopes = np.asarray(
                [item["slope"] for item in conv_equations[row]], dtype=np.int64)
            if not np.array_equal(all_matrix @ estimate, all_slopes):
                raise AssertionError("redundant W1 marker equations disagree")
            start = dead_channel * 9
            recovered_w1[row, start:start + 9] = estimate
            row_reports.append({
                "output_row": int(row),
                "w1_kernel": estimate.tolist(),
                "w1_nonzero": int(np.count_nonzero(estimate)),
                "w1_equations": len(conv_equations[row]),
                "w1_selected_determinant": int(determinant),
                "w1_selected_unimodular": True,
                "w1_solver": (
                    "floating solve rounded to integers, then accepted only "
                    "after exact integer equation checks"),
            })
        channel_reports.append({
            "input_channel": int(dead_channel),
            "rows": row_reports,
        })

    original_w1 = np.asarray(block["W1"], dtype=np.int64)
    block["W1"] = recovered_w1
    return {
        "channels": channel_reports,
        "line_candidate_count_histogram": candidate_histogram,
        "w1_changed_coefficients": int(np.count_nonzero(recovered_w1 - original_w1)),
        "w1_recovered_nonzero_in_tied_columns": int(sum(
            np.count_nonzero(recovered_w1[:, channel * 9:(channel + 1) * 9])
            for channel in plan["dead_channels"])),
    }


def run_anchor(proxy, plan, anchor, dead_channel, value):
    session = proxy.session()
    y0 = session.round([plan["image"].reshape(-1)], read=True)
    if not np.array_equal(np.sort(y0), np.sort(plan["pred0"].reshape(-1))):
        raise AssertionError("stem prediction multiset mismatch")
    targets = anchor["class_targets"].copy()
    for index in plan["dead_class_indices"].values():
        targets[index] = 0
    if dead_channel is not None:
        targets[plan["dead_class_indices"][int(dead_channel)]] = int(value)
    z1 = targets[class_index(plan["class_values"], y0)]
    y1 = session.round([z1], read=True)
    return y1


def varying_baseline_line_candidates(known, observations, control_values):
    """Slopes for a coordinate with a public, control-dependent W2 carrier."""
    first = int(control_values[0])
    if first != 1:
        raise ValueError("the first control value must be one")
    candidates = (
        np.unique(np.asarray(observations[first], dtype=np.int64))
        - int(known[first]))
    for value in control_values[1:]:
        value = int(value)
        observed = np.unique(np.asarray(observations[value], dtype=np.int64))
        wanted = int(known[value]) + value * candidates
        indices = np.searchsorted(observed, wanted)
        valid = indices < observed.size
        clipped = np.clip(indices, 0, max(0, observed.size - 1))
        candidates = candidates[valid & (observed[clipped] == wanted)]
        if candidates.size == 0:
            break
    return np.unique(candidates).astype(np.int64)


def prepare_shortcut_carrier(net, plan, anchor, dead_channel, value, rng):
    """Make a controllable W2 input and its known shortcut-free baseline."""
    block = net.blocks[0]
    retained = np.asarray(anchor["retained"], dtype=np.int64).copy()
    for channel in plan["dead_channels"]:
        retained[int(channel), :, :] = 0
    retained[int(dead_channel), :, :] = int(value)
    predicted1 = conv_int(retained, block["W1"], block["b1"], 1, 3)
    class_values, inverse = np.unique(
        predicted1.reshape(-1), return_inverse=True)
    class_targets = rng.integers(
        0, net.A + 1, size=class_values.size, dtype=np.int64)
    primary = class_targets[inverse].reshape(predicted1.shape)
    main = conv_int(
        primary, block["W2"], block["b2"] + block["bs"], 1, 3)
    known_shortcut = conv_int(
        retained, block["S"],
        np.zeros(block["C_out"], dtype=np.int64), 1, 1)
    return {
        "retained": retained,
        "predicted1": predicted1,
        "class_values": class_values,
        "class_targets": class_targets,
        "known_baseline2": main + known_shortcut,
    }


def run_shortcut_carrier(proxy, plan, anchor, carrier):
    """Run one three-round session using clone-predicted class controllers."""
    session = proxy.session()
    y0 = session.round([plan["image"].reshape(-1)], read=True)
    if not np.array_equal(np.sort(y0), np.sort(plan["pred0"].reshape(-1))):
        raise AssertionError("stem prediction multiset mismatch")
    # The retained map is class-constant by construction.  Convert its canonical
    # class values to the fresh reply order without learning the permutation.
    stem_targets = anchor["class_targets"].copy()
    for channel, index in plan["dead_class_indices"].items():
        stem_targets[index] = int(carrier["retained"][int(channel), 0, 0])
    z1 = stem_targets[class_index(plan["class_values"], y0)]
    y1 = session.round([z1], read=True)
    if not np.array_equal(
            np.sort(y1), np.sort(carrier["predicted1"].reshape(-1))):
        raise AssertionError("completed W1 prediction multiset mismatch")
    z2 = carrier["class_targets"][class_index(
        carrier["class_values"], y1)]
    return session.round([z2], read=True)


def recover_shortcut_columns(net, plan, proxy, rng, control_values, repeats):
    """Recover tied shortcut columns after W1 enables a public W2 carrier."""
    block = net.blocks[0]
    original_s = np.asarray(block["S"], dtype=np.int64)
    recovered_s = original_s.copy()
    candidate_histogram = {}
    channel_reports = []
    anchor = plan["anchors"][0]

    for dead_channel in plan["dead_channels"].tolist():
        singleton_slopes = {row: [] for row in range(block["C_out"])}
        repeat_reports = []
        for repeat in range(int(repeats)):
            carriers = {}
            observed = {}
            for value in control_values:
                value = int(value)
                carriers[value] = prepare_shortcut_carrier(
                    net, plan, anchor, dead_channel, value, rng)
                observed[value] = run_shortcut_carrier(
                    proxy, plan, anchor, carriers[value])

            unique_at_one = unique_mask(carriers[1]["known_baseline2"])
            markers_used = 0
            for row in range(block["C_out"]):
                coords = np.argwhere(unique_at_one[row])
                # A few independent spatial carriers per row are enough, while
                # redundant singleton slopes provide an internal consistency check.
                for y, x in coords[:32]:
                    known = {
                        int(value): int(
                            carriers[int(value)]["known_baseline2"][row, y, x])
                        for value in control_values
                    }
                    candidates = varying_baseline_line_candidates(
                        known, observed, control_values)
                    key = str(int(candidates.size))
                    candidate_histogram[key] = candidate_histogram.get(key, 0) + 1
                    markers_used += 1
                    if candidates.size == 1:
                        singleton_slopes[row].append(int(candidates[0]))
            repeat_reports.append({
                "repeat": int(repeat),
                "markers_tested": int(markers_used),
            })

        row_reports = []
        for row in range(block["C_out"]):
            slopes = singleton_slopes[row]
            if not slopes or len(set(slopes)) != 1:
                raise RuntimeError(
                    "shortcut carrier slopes are missing or inconsistent for "
                    "input %d output %d: %r" %
                    (dead_channel, row, sorted(set(slopes))))
            recovered_s[row, dead_channel] = int(slopes[0])
            row_reports.append({
                "output_row": int(row),
                "shortcut": int(slopes[0]),
                "singleton_carrier_markers": len(slopes),
            })
        channel_reports.append({
            "input_channel": int(dead_channel),
            "repeats": repeat_reports,
            "rows": row_reports,
        })

    block["S"] = recovered_s
    return {
        "shortcut_channels": channel_reports,
        "shortcut_line_candidate_count_histogram": candidate_histogram,
        "shortcut_changed_coefficients": int(
            np.count_nonzero(recovered_s - original_s)),
        "shortcut_recovered_nonzero_in_tied_columns": int(sum(
            np.count_nonzero(recovered_s[:, channel])
            for channel in plan["dead_channels"])),
        "shortcut_carrier_repeats": int(repeats),
    }


def completed_clone_closure_audit(proxy, net, dead_channels):
    """Replay the four earlier distinguisher probes against the completed clone."""
    height = width = 32
    zero_image = np.zeros((3, height, width), dtype=np.int64)
    predicted0 = conv_int(
        zero_image, net.stem["W"], net.stem["b"], 1, 3)
    class_values = np.unique(predicted0.reshape(-1))
    block = net.blocks[0]
    records = []
    for dead_channel in np.asarray(dead_channels, dtype=np.int64).tolist():
        for value in (1, 255):
            session = proxy.session()
            y0 = session.round([zero_image.reshape(-1)], read=True)
            if not np.array_equal(np.sort(y0), np.sort(predicted0.reshape(-1))):
                raise AssertionError("closure-audit stem multiset mismatch")
            targets = np.zeros(class_values.size, dtype=np.int64)
            index = int(np.searchsorted(
                class_values, int(net.stem["b"][dead_channel])))
            targets[index] = int(value)
            z1 = targets[class_index(class_values, y0)]
            y1 = session.round([z1], read=True)

            retained = np.zeros(
                (block["C_in"], height, width), dtype=np.int64)
            retained[dead_channel, :, :] = int(value)
            predicted1 = conv_int(
                retained, block["W1"], block["b1"], 1, 3).reshape(-1)
            z2 = np.zeros(predicted1.size, dtype=np.int64)
            y2 = session.round([z2], read=True)
            primary = np.zeros(
                (block["C_out"], height, width), dtype=np.int64)
            main = conv_int(
                primary, block["W2"], block["b2"] + block["bs"], 1, 3)
            shortcut = conv_int(
                retained, block["S"],
                np.zeros(block["C_out"], dtype=np.int64), 1, 1)
            predicted2 = (main + shortcut).reshape(-1)
            records.append({
                "channel": int(dead_channel),
                "control_value": int(value),
                "round1_multiset_equal": bool(np.array_equal(
                    np.sort(y1), np.sort(predicted1))),
                "round2_multiset_equal": bool(np.array_equal(
                    np.sort(y2), np.sort(predicted2))),
            })
    return {
        "probes": len(records),
        "round1_equal": int(sum(
            record["round1_multiset_equal"] for record in records)),
        "round2_equal": int(sum(
            record["round2_multiset_equal"] for record in records)),
        "success": bool(records and all(
            record["round1_multiset_equal"]
            and record["round2_multiset_equal"] for record in records)),
        "records": records,
    }


def attacker_worker(connection, manifest, config):
    started = time.time()
    report = {
        "schema": "p050-class-aware-attacker-v3",
        "status": "initialising",
        "pid": os.getpid(),
        "start_method": "spawn",
    }
    proxy = None
    try:
        view_audit = audit_attacker_view(config, globals())
        guard = install_data_guard(config["forbidden_data_root"])
        public_net = zero_network_from_manifest(manifest)
        if secret_nonzero_count(public_net) != 0:
            raise AssertionError("public network contains a nonzero secret entry")
        if sha256_bytes(canonical_json(manifest)) != config[
                "public_manifest_sha256"]:
            raise AssertionError("public manifest hash mismatch")
        recovered = load_network(
            config["source_arrays"], config["source_meta"])
        source_digest = model_digest(recovered)
        rng = np.random.default_rng([config["attack_seed"], 41])
        plan = find_public_image_and_anchors(
            recovered, rng, config["search_limit"], config["min_cover"])

        public_rounds, _ = fp.build_r3_rounds(public_net, 32)
        proxy = RpcOracleView(connection, public_rounds, public_net.A)
        observations = {}
        baseline_checks = []
        for anchor_index, anchor in enumerate(plan["anchors"]):
            base1 = run_anchor(proxy, plan, anchor, None, 0)
            check1 = bool(np.array_equal(
                np.sort(base1), np.sort(anchor["baseline1"].reshape(-1))))
            baseline_checks.append({
                "anchor": int(anchor_index),
                "round1_multiset_equal": check1,
            })
            if not check1:
                raise AssertionError("zero-tied-class baseline does not match clone")
            observations[anchor_index] = {}
            for dead_channel in plan["dead_channels"].tolist():
                observations[anchor_index][dead_channel] = {}
                for value in config["control_values"]:
                    y1 = run_anchor(
                        proxy, plan, anchor, dead_channel, value)
                    observations[anchor_index][dead_channel][int(value)] = y1

        w1_recovery = recover_w1_columns(
            recovered, plan, observations, config["control_values"])
        shortcut_recovery = recover_shortcut_columns(
            recovered, plan, proxy, rng, config["control_values"],
            config["shortcut_carrier_repeats"])
        recovery = dict(w1_recovery)
        recovery.update(shortcut_recovery)
        closure_audit = completed_clone_closure_audit(
            proxy, recovered, plan["dead_channels"])
        save_network(
            recovered, config["completed_arrays"], config["completed_meta"])
        oracle_stats = proxy._stats()
        server_close = proxy.close()
        success = bool(
            all(item["round1_multiset_equal"] for item in baseline_checks)
            and recovery["w1_changed_coefficients"] > 0
            and recovery["shortcut_changed_coefficients"] > 0
            and closure_audit["success"]
            and guard["forbidden_open_attempts"] == 0
            and oracle_stats.get("inadmissible_entries") == 0
            and oracle_stats.get("n_rounding_failures") == 0
            and server_close.get("transcript_hash_match")
        )
        report.update({
            "status": "complete",
            "success": success,
            "public_manifest_sha256": config["public_manifest_sha256"],
            "public_secret_nonzero_entries": secret_nonzero_count(public_net),
            "forbidden_open_attempts": int(guard["forbidden_open_attempts"]),
            "forbidden_open_sample": guard["sample"],
            "attacker_view_audit": view_audit,
            "source_model_sha256": source_digest,
            "completed_model_sha256": model_digest(recovered),
            "source_arrays_sha256": sha256_file(config["source_arrays"]),
            "source_meta_sha256": sha256_file(config["source_meta"]),
            "completed_arrays_sha256": sha256_file(config["completed_arrays"]),
            "completed_meta_sha256": sha256_file(config["completed_meta"]),
            "dead_channels_discovered_from_clone": plan[
                "dead_channels"].tolist(),
            "stem_reply_class_sizes": {
                str(int(channel)): int(np.sum(
                    plan["pred0"] == recovered.stem["b"][channel]))
                for channel in plan["dead_channels"]
            },
            "anchor_candidates_tested": int(plan["candidates_tested"]),
            "anchors_selected": len(plan["anchors"]),
            "selected_anchor_attempts": [
                int(anchor["attempt"]) for anchor in plan["anchors"]],
            "minimum_marker_cover": int(config["min_cover"]),
            "shortcut_carrier_repeats": int(
                config["shortcut_carrier_repeats"]),
            "control_values": list(config["control_values"]),
            "baseline_checks": baseline_checks,
            "recovery": recovery,
            "completed_clone_closure_audit": closure_audit,
            "oracle_stats_before_close": oracle_stats,
            "server_close": server_close,
        })
    except Exception as exc:
        report.update({
            "status": "error",
            "success": False,
            "error": "%s: %s" % (type(exc).__name__, exc),
            "traceback": traceback.format_exc(),
        })
        if proxy is not None:
            try:
                report["server_close"] = proxy.close()
            except Exception:
                pass
    finally:
        report["wall_time_sec"] = round(time.time() - started, 3)
        with open(config["attacker_report"], "w", encoding="utf-8") as handle:
            json.dump(jsonable(report), handle, indent=2, sort_keys=True)
            handle.write("\n")
        try:
            connection.close()
        except Exception:
            pass
    raise SystemExit(0 if report.get("success") else 1)


def layer_match(recovered_w, recovered_b, expected_w, expected_b):
    permutation, matched = row_permutation(
        recovered_w, recovered_b, expected_w, expected_b)
    exact, error = frame_check(
        recovered_w, recovered_b, expected_w, expected_b, permutation)
    return permutation, int(matched), bool(exact), int(error)


def strict_structural_certificate(private, recovered):
    """Check every observable affine round map in one coherent channel gauge."""
    records = []
    eta_exact = bool(
        private.A == recovered.A
        and private.stem["eta"] == recovered.stem["eta"]
        and len(private.blocks) == len(recovered.blocks)
        and all(
            truth["name"] == clone["name"]
            and truth["shortcut"] == clone["shortcut"]
            and truth["eta1"] == clone["eta1"]
            and truth["eta"] == clone["eta"]
            for truth, clone in zip(private.blocks, recovered.blocks)))

    previous, matched, exact, error = layer_match(
        recovered.stem["W"], recovered.stem["b"],
        private.stem["W"], private.stem["b"])
    records.append({
        "name": "conv1", "kind": "stem", "rows_matched": matched,
        "row_count": int(recovered.stem["W"].shape[0]),
        "frame_exact": exact, "max_abs_error": error, "success": exact})

    for truth, clone in zip(private.blocks, recovered.blocks):
        name = str(truth["name"])
        w1_expected = canonical_column_matrix(truth["W1"], previous, 9)
        p1, matched1, exact1, error1 = layer_match(
            clone["W1"], clone["b1"], w1_expected, truth["b1"])
        records.append({
            "name": name + ".conv1", "kind": "conv1",
            "rows_matched": matched1,
            "row_count": int(clone["W1"].shape[0]),
            "frame_exact": exact1, "max_abs_error": error1,
            "success": exact1})

        shortcut_expected = canonical_column_matrix(truth["S"], previous, 1)
        w2_expected = canonical_column_matrix(truth["W2"], p1, 9)
        combined_expected = np.concatenate(
            [shortcut_expected, w2_expected], axis=1)
        combined_recovered = np.concatenate(
            [clone["S"], clone["W2"]], axis=1)
        expected_bias = np.asarray(truth["b2"] + truth["bs"], dtype=np.int64)
        recovered_bias = np.asarray(clone["b2"] + clone["bs"], dtype=np.int64)
        p2, matched2, exact2, error2 = layer_match(
            combined_recovered, recovered_bias,
            combined_expected, expected_bias)
        shortcut_error = integer_error(
            clone["S"], shortcut_expected[p2])
        conv2_error = integer_error(clone["W2"], w2_expected[p2])
        bias_error = integer_error(recovered_bias, expected_bias[p2])
        records.append({
            "name": name + ".shortcut", "kind": "shortcut",
            "max_abs_error": shortcut_error,
            "success": bool(shortcut_error == 0)})
        records.append({
            "name": name + ".conv2", "kind": "conv2",
            "combined_rows_matched": matched2,
            "combined_row_count": int(combined_recovered.shape[0]),
            "combined_frame_exact": exact2,
            "combined_max_abs_error": error2,
            "weight_max_abs_error": conv2_error,
            "combined_bias_max_abs_error": bias_error,
            "bias_scope": "b2+bs is observable; the separate split is not",
            "success": bool(
                exact2 and conv2_error == 0 and bias_error == 0)})
        previous = p2

    fc_expected = canonical_column_matrix(private.fc["W"], previous, 1)
    fc_w_error = integer_error(recovered.fc["W"], fc_expected)
    fc_b_error = integer_error(recovered.fc["b"], private.fc["b"])
    records.append({
        "name": "fc", "kind": "labelled_fc",
        "weight_max_abs_error": fc_w_error,
        "bias_max_abs_error": fc_b_error,
        "success": bool(fc_w_error == 0 and fc_b_error == 0)})
    success = bool(
        eta_exact and len(records) == 29
        and all(item["success"] for item in records))
    return {
        "success": success,
        "eta_metadata_exact": eta_exact,
        "parameter_records_checked": len(records),
        "parameter_records_successful": int(sum(
            item["success"] for item in records)),
        "equivalence_scope": (
            "all observable affine round maps in one coherent channel gauge; "
            "residual bias is identified as b2+bs"),
        "records": records,
    }


def evaluator_worker(config):
    started = time.time()
    report = {
        "schema": "p050-class-aware-independent-evaluator-v2",
        "status": "initialising",
        "pid": os.getpid(),
        "start_method": "spawn",
    }
    try:
        from lib import cifar10
        from lib.fmap import forward_fmap, quantise_resnet20_fmap
        from lib.models import env_info
        from lib.resnet20 import load_resnet20_cifar10

        model, _ = load_resnet20_cifar10(config["checkpoint"])
        data = cifar10.load(
            config["cifar"], extract_dir=os.path.join(DATA, "cifar10_extract"))
        images, labels = data["test_batch"]
        private, qinfo = quantise_resnet20_fmap(
            model, config["w_bits"], config["w_bits"], images[:64],
            normalise=(cifar10.MEAN, cifar10.STD))
        recovered = load_network(
            config["completed_arrays"], config["completed_meta"])
        certificate = strict_structural_certificate(private, recovered)
        truth = forward_fmap(private, images[:config["n_test"]])
        candidate = forward_fmap(recovered, images[:config["n_test"]])
        equal = np.all(truth == candidate, axis=1)
        calibration_count = min(64, int(config["n_test"]))
        post_truth = truth[calibration_count:]
        post_candidate = candidate[calibration_count:]
        post_labels = labels[calibration_count:config["n_test"]]
        post_equal = np.all(post_truth == post_candidate, axis=1)
        report.update({
            "status": "complete",
            "environment": env_info(),
            "n_images": int(config["n_test"]),
            "identical_logit_vectors": int(np.sum(equal)),
            "identical_logit_fraction": float(np.mean(equal)),
            "identical_argmax_fraction": float(np.mean(
                truth.argmax(1) == candidate.argmax(1))),
            "max_abs_logit_difference": int(np.abs(truth - candidate).max()),
            "accuracy_private": 100.0 * float(np.mean(
                truth.argmax(1) == labels[:config["n_test"]])),
            "accuracy_recovered": 100.0 * float(np.mean(
                candidate.argmax(1) == labels[:config["n_test"]])),
            "accuracy_scope": (
                "descriptive score on all n_images; the first 64 images were "
                "also used only to calibrate the public integer range"),
            "calibration_images": int(calibration_count),
            "accuracy_images_overlapping_calibration": int(calibration_count),
            "post_calibration_images": int(post_truth.shape[0]),
            "post_calibration_identical_logit_vectors": int(
                np.sum(post_equal)),
            "post_calibration_identical_logit_fraction": float(
                np.mean(post_equal)) if post_equal.size else None,
            "post_calibration_accuracy_private": (
                100.0 * float(np.mean(
                    post_truth.argmax(1) == post_labels))
                if post_truth.shape[0] else None),
            "post_calibration_accuracy_recovered": (
                100.0 * float(np.mean(
                    post_candidate.argmax(1) == post_labels))
                if post_candidate.shape[0] else None),
            "private_model_sha256": model_digest(private),
            "completed_model_sha256": model_digest(recovered),
            "completed_arrays_sha256": sha256_file(config["completed_arrays"]),
            "completed_meta_sha256": sha256_file(config["completed_meta"]),
            "quantisation": {
                "accumulator_bits_calibration": int(
                    qinfo["accumulator_bits_calibration"]),
                "accumulator_absmax_calibration": int(
                    qinfo["accumulator_absmax_calibration"]),
            },
            "structural_certificate": certificate,
        })
        report["success"] = bool(
            certificate["success"]
            and report["identical_logit_vectors"] == config["n_test"]
            and report["post_calibration_identical_logit_vectors"]
                == report["post_calibration_images"]
            and report["max_abs_logit_difference"] == 0)
    except Exception as exc:
        report.update({
            "status": "error", "success": False,
            "error": "%s: %s" % (type(exc).__name__, exc),
            "traceback": traceback.format_exc(),
        })
    finally:
        report["wall_time_sec"] = round(time.time() - started, 3)
        with open(config["evaluator_report"], "w", encoding="utf-8") as handle:
            json.dump(jsonable(report), handle, indent=2, sort_keys=True)
            handle.write("\n")
    raise SystemExit(0 if report.get("success") else 1)


def main():
    started = time.time()
    os.makedirs(OUT_DIR, exist_ok=True)
    legacy_seed = os.environ.get("TDSC_CLASS_SEED")
    attack_seed = int(os.environ.get(
        "TDSC_CLASS_ATTACK_SEED", legacy_seed or "20260923"))
    oracle_seed = int(os.environ.get(
        "TDSC_CLASS_ORACLE_SEED", legacy_seed or "20260925"))
    checkpoint = os.path.join(DATA, "cifar10_resnet20.pt")
    cifar = os.path.join(DATA, "cifar-10-python.tar.gz")
    paths = {
        "oracle_report": os.path.join(OUT_DIR, "oracle.json"),
        "attacker_report": os.path.join(OUT_DIR, "attacker.json"),
        "evaluator_report": os.path.join(OUT_DIR, "evaluator.json"),
        "source_arrays": os.path.join(SOURCE_DIR, "recovered.npz"),
        "source_meta": os.path.join(SOURCE_DIR, "recovered.json"),
        "completed_arrays": os.path.join(OUT_DIR, "completed.npz"),
        "completed_meta": os.path.join(OUT_DIR, "completed.json"),
        "arithmetic_trace": os.path.join(OUT_DIR, "arithmetic.ndjson.gz"),
        "result": os.path.join(OUT_DIR, "result.json"),
    }
    oracle_config = {
        "checkpoint": checkpoint,
        "cifar": cifar,
        "oracle_seed": oracle_seed,
        "w_bits": W_BITS,
        "trace_arithmetic": TRACE_ARITHMETIC,
        "oracle_report": paths["oracle_report"],
        "arithmetic_trace": paths["arithmetic_trace"],
    }
    attacker_config = {
        "attack_seed": attack_seed,
        "forbidden_data_root": DATA,
        "search_limit": SEARCH_LIMIT,
        "min_cover": MIN_MARKER_COVER,
        "shortcut_carrier_repeats": SHORTCUT_CARRIER_REPEATS,
        "control_values": CONTROL_VALUES,
        "attacker_report": paths["attacker_report"],
        "source_arrays": paths["source_arrays"],
        "source_meta": paths["source_meta"],
        "completed_arrays": paths["completed_arrays"],
        "completed_meta": paths["completed_meta"],
    }
    evaluator_config = {
        "checkpoint": checkpoint,
        "cifar": cifar,
        "w_bits": W_BITS,
        "n_test": N_TEST,
        "completed_arrays": paths["completed_arrays"],
        "completed_meta": paths["completed_meta"],
        "evaluator_report": paths["evaluator_report"],
    }

    ctx = mp.get_context("spawn")
    rpc_attacker, rpc_oracle = ctx.Pipe(duplex=True)
    ready_parent, ready_oracle = ctx.Pipe(duplex=True)
    oracle = ctx.Process(
        name="p050-class-oracle", target=oracle_worker,
        args=(rpc_oracle, ready_oracle, oracle_config))
    oracle.start()
    rpc_oracle.close()
    ready_oracle.close()
    if not ready_parent.poll(600):
        raise RuntimeError("oracle did not publish a public manifest")
    ready = json.loads(ready_parent.recv_bytes().decode("utf-8"))
    ready_parent.close()
    if ready.get("status") != "ready":
        raise RuntimeError("oracle setup failed: %s" % ready.get("error"))
    attacker_config["public_manifest_sha256"] = ready[
        "public_manifest_sha256"]

    attacker = ctx.Process(
        name="p050-class-attacker", target=attacker_worker,
        args=(rpc_attacker, ready["manifest"], attacker_config))
    saved_private_environment = remove_private_environment_for_spawn()
    try:
        attacker.start()
    finally:
        restore_private_environment(saved_private_environment)
    rpc_attacker.close()
    attacker.join(1800)
    if attacker.is_alive():
        raise RuntimeError("attacker exceeded orchestration timeout")
    oracle.join(180)
    if oracle.is_alive():
        raise RuntimeError("oracle did not close after attacker exit")

    attacker_report = read_json(paths["attacker_report"])
    oracle_report = read_json(paths["oracle_report"])
    evaluator_exitcode = None
    if SKIP_EVALUATOR:
        evaluator_report = {
            "schema": "p050-class-aware-independent-evaluator-v2",
            "status": "skipped",
            "success": None,
            "reason": "TDSC_CLASS_SKIP_EVAL=1",
        }
        with open(paths["evaluator_report"], "w", encoding="utf-8") as handle:
            json.dump(evaluator_report, handle, indent=2, sort_keys=True)
            handle.write("\n")
    elif attacker.exitcode == 0 and os.path.isfile(paths["completed_meta"]):
        evaluator = ctx.Process(
            name="p050-class-evaluator", target=evaluator_worker,
            args=(evaluator_config,))
        evaluator.start()
        evaluator.join(1800)
        if evaluator.is_alive():
            raise RuntimeError("evaluator exceeded orchestration timeout")
        evaluator_exitcode = evaluator.exitcode
        evaluator_report = read_json(paths["evaluator_report"])
    else:
        evaluator_report = {
            "schema": "p050-class-aware-independent-evaluator-v2",
            "status": "not-run",
            "success": False,
            "reason": "attacker did not produce a completed artifact",
        }
        with open(paths["evaluator_report"], "w", encoding="utf-8") as handle:
            json.dump(evaluator_report, handle, indent=2, sort_keys=True)
            handle.write("\n")
    success = bool(
        attacker.exitcode == 0 and oracle.exitcode == 0
        and attacker_report.get("success")
        and (SKIP_EVALUATOR or (
            evaluator_exitcode == 0 and evaluator_report.get("success")))
        and oracle_report.get("status") == "closed"
        and oracle_report.get("canary_response_leaks") == 0
        and oracle_report.get("oracle_counters", {}).get(
            "inadmissible_entries") == 0
        and oracle_report.get("oracle_counters", {}).get(
            "rounding_failures") == 0)
    if not SKIP_EVALUATOR:
        success = bool(
            success
            and attacker_report.get("completed_model_sha256")
                == evaluator_report.get("completed_model_sha256")
            and oracle_report.get("private_model_sha256")
                == evaluator_report.get("private_model_sha256"))
    result = {
        "schema": "p050-class-aware-completion-v3",
        "run_name": RUN_NAME,
        "status": "complete",
        "success": success,
        "source_sha256": sha256_file(os.path.abspath(__file__)),
        "source_run": SOURCE_RUN,
        "resolved_working_directory": os.path.dirname(
            os.path.dirname(os.path.abspath(__file__))),
        "command": (
            "TDSC_CLASS_RUN=%s TDSC_CLASS_SOURCE=%s "
            "TDSC_CLASS_ATTACK_SEED=%d TDSC_CLASS_ORACLE_SEED=%d "
            "TDSC_CLASS_WBITS=%d TDSC_CLASS_NTEST=%d TDSC_CLASS_SEARCH=%d "
            "TDSC_CLASS_COVER=%d TDSC_CLASS_SREPEATS=%d "
            "TDSC_CLASS_VALUES=%s TDSC_CLASS_TRACE=%d "
            "TDSC_CLASS_SKIP_EVAL=%d PYTHONPATH=scripts:. python3 "
            "scripts/run_completion.py" % (
                RUN_NAME, SOURCE_RUN, attack_seed, oracle_seed,
                W_BITS, N_TEST, SEARCH_LIMIT,
                MIN_MARKER_COVER, SHORTCUT_CARRIER_REPEATS,
                ",".join(str(value) for value in CONTROL_VALUES),
                int(TRACE_ARITHMETIC), int(SKIP_EVALUATOR))),
        "attack_seed": attack_seed,
        "oracle_seed": oracle_seed,
        "seed_disclosed_to_attacker": False,
        "truth_comparison_stage": "independent evaluator after attacker exit",
        "w_bits": W_BITS,
        "n_test": N_TEST,
        "control_values": list(CONTROL_VALUES),
        "search_limit": SEARCH_LIMIT,
        "minimum_marker_cover": MIN_MARKER_COVER,
        "shortcut_carrier_repeats": SHORTCUT_CARRIER_REPEATS,
        "trace_arithmetic": TRACE_ARITHMETIC,
        "skip_evaluator": SKIP_EVALUATOR,
        "process_exit_codes": {
            "attacker": attacker.exitcode,
            "oracle": oracle.exitcode,
            "evaluator": evaluator_exitcode,
        },
        "attacker": attacker_report,
        "oracle": oracle_report,
        "evaluator": evaluator_report,
        "wall_time_sec": round(time.time() - started, 3),
    }
    with open(paths["result"], "w", encoding="utf-8") as handle:
        json.dump(jsonable(result), handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps({
        "output": paths["result"],
        "success": success,
        "dead_channels": attacker_report.get(
            "dead_channels_discovered_from_clone"),
        "anchors_selected": attacker_report.get("anchors_selected"),
        "w1_changed": attacker_report.get(
            "recovery", {}).get("w1_changed_coefficients"),
        "shortcut_changed": attacker_report.get(
            "recovery", {}).get("shortcut_changed_coefficients"),
        "strict_records": evaluator_report.get(
            "structural_certificate", {}).get("parameter_records_successful"),
        "identical_logits": evaluator_report.get("identical_logit_vectors"),
    }, sort_keys=True), flush=True)
    return 0 if success else 1


if __name__ == "__main__":
    raise SystemExit(main())

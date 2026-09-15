#!/usr/bin/env python3
"""Process-isolated P050 whole-network extraction regression.

The orchestrator never loads the victim checkpoint.  A spawn-started oracle
process loads and quantises the victim, then exposes only a raw byte protocol:
start a session, submit integer round messages, read an ordered integer reply,
or read counters.  A second spawn-started process reconstructs an all-zero
public network from a shape/quantisation manifest and runs the attack with
diagnostics disabled.  Only after that process exits does a third process load
the victim and compare the recovered model on the test set.

The RPC uses Connection.send_bytes/recv_bytes only.  It never calls recv(), so
attacker-controlled pickle objects are not deserialised by the oracle.
"""

from __future__ import print_function

import hashlib
import gzip
import json
import multiprocessing as mp
import os
import secrets
import struct
import sys
import time
import traceback

import numpy as np

import _bootstrap
from _bootstrap import DATA, RESULTS


RUN_NAME = os.environ.get(
    "TDSC_RPC_RUN", "tdsc-extraction")
OUT_REL = os.environ.get("TDSC_RPC_OUT", os.path.join("provenance", RUN_NAME))
OUT_DIR = os.path.join(RESULTS, OUT_REL)
W_BITS = int(os.environ.get("TDSC_RPC_WBITS", 8))
N_TEST = int(os.environ.get("TDSC_RPC_NTEST", 10000))
BUDGET = float(os.environ.get("TDSC_RPC_BUDGET", 1800))
T_STEPS = int(os.environ.get("TDSC_RPC_T", 3))
EVAL_TIMEOUT = float(os.environ.get("TDSC_RPC_EVAL_TIMEOUT", 1800))
DIAGNOSTICS = bool(int(os.environ.get("TDSC_RPC_DIAGNOSTICS", 0)))
SKIP_EVALUATOR = bool(int(os.environ.get("TDSC_RPC_SKIP_EVALUATOR", 0)))
TRACE_ARITHMETIC = bool(int(os.environ.get("TDSC_RPC_TRACE_ARITHMETIC", 0)))

PRIVATE_ATTACKER_CONFIG_KEYS = frozenset((
    "checkpoint", "cifar", "oracle_seed", "expected_model_sha256",
    "private_model_sha256", "seed",
))
PRIVATE_ATTACKER_ENV_KEYS = (
    "TDSC_RPC_SEED", "TDSC_RPC_ORACLE_SEED",
    "TDSC_RPC_EXPECT_MODEL_SHA256",
    "TDSC_CLASS_SEED", "TDSC_CLASS_ORACLE_SEED",
)
PRIVATE_ATTACKER_GLOBAL_NAMES = frozenset((
    "CKPT", "CIFAR", "SEED", "ORACLE_SEED", "EXPECTED_MODEL_SHA256",
))

OP_NEW_SESSION = 1
OP_ROUND = 2
OP_STATS = 3
OP_CLOSE = 4

RESP_NONE = 0
RESP_ARRAY = 1
RESP_JSON = 2
RESP_ERROR = 255


def canonical_json(obj):
    return json.dumps(obj, sort_keys=True, separators=(",", ":")).encode("utf-8")


def sha256_bytes(payload):
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            block = handle.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def audit_attacker_view(config, module_globals):
    """Assert that process-spawn inputs expose public attack data only."""
    leaked_config = sorted(PRIVATE_ATTACKER_CONFIG_KEYS.intersection(config))
    leaked_environment = sorted(
        key for key in PRIVATE_ATTACKER_ENV_KEYS if key in os.environ)
    leaked_globals = sorted(
        name for name in PRIVATE_ATTACKER_GLOBAL_NAMES
        if name in module_globals)
    audit = {
        "public_config_keys": sorted(config),
        "private_config_keys_present": leaked_config,
        "private_environment_keys_present": leaked_environment,
        "private_imported_globals_present": leaked_globals,
    }
    if leaked_config or leaked_environment or leaked_globals:
        raise AssertionError("private information entered attacker view: %r" % audit)
    return audit


def remove_private_environment_for_spawn():
    """Temporarily remove private parent variables before attacker spawn."""
    saved = {}
    for key in PRIVATE_ATTACKER_ENV_KEYS:
        if key in os.environ:
            saved[key] = os.environ.pop(key)
    return saved


def restore_private_environment(saved):
    for key, value in saved.items():
        os.environ[key] = value


def write_canonical_gzip_ndjson(path, rows):
    """Write deterministic gzip and hash both file and canonical content."""
    content_digest = hashlib.sha256()
    with open(path, "wb") as raw:
        with gzip.GzipFile(
                filename="", mode="wb", fileobj=raw, mtime=0) as handle:
            for row in rows:
                payload = (
                    json.dumps(row, sort_keys=True, separators=(",", ":"))
                    + "\n"
                ).encode("utf-8")
                content_digest.update(payload)
                handle.write(payload)
    return sha256_file(path), content_digest.hexdigest()


def signed_bits_for_range(minimum, maximum):
    """Smallest two's-complement width containing [minimum, maximum]."""
    minimum, maximum = int(minimum), int(maximum)
    bits = 1
    while minimum < -(1 << (bits - 1)) or maximum > (1 << (bits - 1)) - 1:
        bits += 1
    return bits


def arithmetic_summary(rows):
    widths = (8, 12, 14, 16, 19, 20, 22, 24, 32)
    by_round = {}
    for row in rows:
        item = by_round.setdefault(row["round_name"], {
            "round": int(row["round"]),
            "round_name": row["round_name"],
            "evaluations": 0,
            "cache_hits": 0,
            "input_min": None,
            "input_max": None,
            "input_below_zero": 0,
            "input_above_activation_max": 0,
            "output_min": None,
            "output_max": None,
        })
        item["evaluations"] += 1
        item["cache_hits"] += int(row["cache_hit"])
        item["input_below_zero"] += int(row["input_below_zero"])
        item["input_above_activation_max"] += int(
            row["input_above_activation_max"])
        for key, value, reducer in (
                ("input_min", row["input_min"], min),
                ("input_max", row["input_max"], max),
                ("output_min", row["output_min"], min),
                ("output_max", row["output_max"], max)):
            if value is not None:
                item[key] = value if item[key] is None else reducer(
                    item[key], value)
    ordered = sorted(by_round.values(), key=lambda item: item["round"])
    for item in ordered:
        item["output_absmax"] = max(
            abs(int(item["output_min"])), abs(int(item["output_max"])))
        item["minimum_signed_bits"] = signed_bits_for_range(
            item["output_min"], item["output_max"])
    global_min = min(item["output_min"] for item in ordered)
    global_max = max(item["output_max"] for item in ordered)
    width_checks = {}
    for bits in widths:
        lower, upper = -(1 << (bits - 1)), (1 << (bits - 1)) - 1
        overflow = [
            row for row in rows
            if row["output_min"] < lower or row["output_max"] > upper
        ]
        width_checks[str(bits)] = {
            "signed_min": lower,
            "signed_max": upper,
            "overflow_evaluations": len(overflow),
            "first_overflow": (None if not overflow else {
                "evaluation": overflow[0]["evaluation"],
                "session": overflow[0]["session"],
                "round": overflow[0]["round"],
                "round_name": overflow[0]["round_name"],
                "output_min": overflow[0]["output_min"],
                "output_max": overflow[0]["output_max"],
            }),
        }
    return {
        "trace_rows": len(rows),
        "input_below_zero": sum(row["input_below_zero"] for row in rows),
        "input_above_activation_max": sum(
            row["input_above_activation_max"] for row in rows),
        "global_output_min": global_min,
        "global_output_max": global_max,
        "global_output_absmax": max(abs(global_min), abs(global_max)),
        "minimum_signed_bits": signed_bits_for_range(global_min, global_max),
        "candidate_signed_widths": width_checks,
        "deployment_width_mapping": None,
        "scope_note": (
            "Widths are hypothetical two's-complement accumulator ranges for "
            "the abstract exact-integer simulator; no Safhire arithmetic "
            "mapping is inferred."
        ),
        "per_round": ordered,
    }


def jsonable(value):
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, set):
        return [jsonable(v) for v in sorted(value)]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def model_digest(net):
    digest = hashlib.sha256()
    for key in ("W", "b"):
        digest.update(np.ascontiguousarray(net.stem[key]).tobytes())
    for block in net.blocks:
        for key in ("W1", "b1", "W2", "b2", "S", "bs"):
            digest.update(np.ascontiguousarray(block[key]).tobytes())
    for key in ("W", "b"):
        digest.update(np.ascontiguousarray(net.fc[key]).tobytes())
    return digest.hexdigest()


def public_manifest(net):
    return {
        "schema": "p050-public-zero-network-v1",
        "A": int(net.A),
        "stem": {
            "W_shape": list(net.stem["W"].shape),
            "b_shape": list(net.stem["b"].shape),
            "eta": float(net.stem["eta"]),
        },
        "blocks": [
            {
                "name": str(block["name"]),
                "W1_shape": list(block["W1"].shape),
                "b1_shape": list(block["b1"].shape),
                "W2_shape": list(block["W2"].shape),
                "b2_shape": list(block["b2"].shape),
                "S_shape": list(block["S"].shape),
                "bs_shape": list(block["bs"].shape),
                "eta1": float(block["eta1"]),
                "eta": float(block["eta"]),
                "shortcut": str(block["shortcut"]),
                "C_in": int(block["C_in"]),
                "C_out": int(block["C_out"]),
            }
            for block in net.blocks
        ],
        "fc": {
            "W_shape": list(net.fc["W"].shape),
            "b_shape": list(net.fc["b"].shape),
        },
    }


def zero_network_from_manifest(manifest):
    from lib.resnet20 import IntResNet20

    stem_meta = manifest["stem"]
    stem = {
        "W": np.zeros(stem_meta["W_shape"], dtype=np.int64),
        "b": np.zeros(stem_meta["b_shape"], dtype=np.int64),
        "eta": float(stem_meta["eta"]),
    }
    blocks = []
    for item in manifest["blocks"]:
        blocks.append({
            "name": item["name"],
            "W1": np.zeros(item["W1_shape"], dtype=np.int64),
            "b1": np.zeros(item["b1_shape"], dtype=np.int64),
            "eta1": float(item["eta1"]),
            "W2": np.zeros(item["W2_shape"], dtype=np.int64),
            "b2": np.zeros(item["b2_shape"], dtype=np.int64),
            "S": np.zeros(item["S_shape"], dtype=np.int64),
            "bs": np.zeros(item["bs_shape"], dtype=np.int64),
            "eta": float(item["eta"]),
            "shortcut": item["shortcut"],
            "C_in": int(item["C_in"]),
            "C_out": int(item["C_out"]),
        })
    fc_meta = manifest["fc"]
    fc = {
        "W": np.zeros(fc_meta["W_shape"], dtype=np.int64),
        "b": np.zeros(fc_meta["b_shape"], dtype=np.int64),
    }
    return IntResNet20(stem, blocks, fc, int(manifest["A"]))


def secret_nonzero_count(net):
    count = sum(int(np.count_nonzero(net.stem[k])) for k in ("W", "b"))
    count += sum(
        int(np.count_nonzero(block[k]))
        for block in net.blocks
        for k in ("W1", "b1", "W2", "b2", "S", "bs")
    )
    count += sum(int(np.count_nonzero(net.fc[k])) for k in ("W", "b"))
    return count


def save_network(net, array_path, meta_path):
    arrays = {
        "stem_W": np.ascontiguousarray(net.stem["W"], dtype=np.int64),
        "stem_b": np.ascontiguousarray(net.stem["b"], dtype=np.int64),
        "fc_W": np.ascontiguousarray(net.fc["W"], dtype=np.int64),
        "fc_b": np.ascontiguousarray(net.fc["b"], dtype=np.int64),
    }
    meta = {
        "schema": "p050-recovered-int-resnet20-v1",
        "A": int(net.A),
        "stem_eta": float(net.stem["eta"]),
        "blocks": [],
    }
    for index, block in enumerate(net.blocks):
        prefix = "block_%02d_" % index
        for key in ("W1", "b1", "W2", "b2", "S", "bs"):
            arrays[prefix + key] = np.ascontiguousarray(
                block[key], dtype=np.int64)
        meta["blocks"].append({
            "name": str(block["name"]),
            "eta1": float(block["eta1"]),
            "eta": float(block["eta"]),
            "shortcut": str(block["shortcut"]),
            "C_in": int(block["C_in"]),
            "C_out": int(block["C_out"]),
        })
    np.savez_compressed(array_path, **arrays)
    with open(meta_path, "w", encoding="utf-8") as handle:
        json.dump(meta, handle, indent=2, sort_keys=True)
        handle.write("\n")


def load_network(array_path, meta_path):
    from lib.resnet20 import IntResNet20

    with open(meta_path, "r", encoding="utf-8") as handle:
        meta = json.load(handle)
    with np.load(array_path, allow_pickle=False) as arrays:
        stem = {
            "W": arrays["stem_W"].astype(np.int64, copy=True),
            "b": arrays["stem_b"].astype(np.int64, copy=True),
            "eta": float(meta["stem_eta"]),
        }
        blocks = []
        for index, item in enumerate(meta["blocks"]):
            prefix = "block_%02d_" % index
            block = {
                key: arrays[prefix + key].astype(np.int64, copy=True)
                for key in ("W1", "b1", "W2", "b2", "S", "bs")
            }
            block.update({
                "name": item["name"],
                "eta1": float(item["eta1"]),
                "eta": float(item["eta"]),
                "shortcut": item["shortcut"],
                "C_in": int(item["C_in"]),
                "C_out": int(item["C_out"]),
            })
            blocks.append(block)
        fc = {
            "W": arrays["fc_W"].astype(np.int64, copy=True),
            "b": arrays["fc_b"].astype(np.int64, copy=True),
        }
    return IntResNet20(stem, blocks, fc, int(meta["A"]))


def transcript_update(digest, direction, payload):
    digest.update(direction)
    digest.update(struct.pack("<Q", len(payload)))
    digest.update(payload)


def encode_round(vectors, read):
    payload = bytearray(struct.pack("<BBH", OP_ROUND, int(bool(read)), len(vectors)))
    for vector in vectors:
        arr = np.ascontiguousarray(np.asarray(vector, dtype="<i8").ravel())
        payload.extend(struct.pack("<Q", arr.size))
        payload.extend(arr.tobytes())
    return bytes(payload)


def decode_round(payload):
    if len(payload) < 4:
        raise ValueError("short round request")
    op, read, count = struct.unpack_from("<BBH", payload, 0)
    if op != OP_ROUND:
        raise ValueError("wrong round opcode")
    cursor = 4
    vectors = []
    for _ in range(count):
        if cursor + 8 > len(payload):
            raise ValueError("truncated vector length")
        size = struct.unpack_from("<Q", payload, cursor)[0]
        cursor += 8
        nbytes = int(size) * 8
        if cursor + nbytes > len(payload):
            raise ValueError("truncated vector body")
        vector = np.frombuffer(
            payload, dtype="<i8", count=int(size), offset=cursor
        ).astype(np.int64, copy=True)
        cursor += nbytes
        vectors.append(vector)
    if cursor != len(payload):
        raise ValueError("trailing bytes in round request")
    return bool(read), vectors


def encode_array(array):
    arr = np.ascontiguousarray(np.asarray(array, dtype="<i8").ravel())
    return bytes([RESP_ARRAY]) + struct.pack("<Q", arr.size) + arr.tobytes()


def decode_array(payload):
    if not payload or payload[0] != RESP_ARRAY or len(payload) < 9:
        raise ValueError("invalid array response")
    size = struct.unpack_from("<Q", payload, 1)[0]
    if len(payload) != 9 + int(size) * 8:
        raise ValueError("array response length mismatch")
    return np.frombuffer(payload, dtype="<i8", count=int(size), offset=9).astype(
        np.int64, copy=True)


def encode_json_response(value):
    return bytes([RESP_JSON]) + canonical_json(value)


def decode_json_response(payload):
    if not payload or payload[0] != RESP_JSON:
        raise ValueError("invalid JSON response")
    return json.loads(payload[1:].decode("utf-8"))


def encode_error(exc):
    message = "%s: %s" % (type(exc).__name__, exc)
    return bytes([RESP_ERROR]) + message.encode("utf-8", errors="replace")


def oracle_worker(connection, ready_connection, config):
    started = time.time()
    report = {
        "schema": "p050-rpc-oracle-v1",
        "status": "initialising",
        "start_method": "spawn",
        "transport": "raw send_bytes/recv_bytes; no object deserialisation",
    }
    private_oracle = None
    transcript = hashlib.sha256()
    request_bytes = 0
    response_bytes = 0
    request_messages = 0
    response_messages = 0
    canary = secrets.token_bytes(32)
    canary_response_leaks = 0
    arithmetic_rows = []
    try:
        from lib import cifar10
        from lib.fmap import quantise_resnet20_fmap
        from lib.fmap_partial import PartialOracle, build_r3_rounds
        from lib.lta import noise_fns_bounded
        from lib.models import env_info
        from lib.resnet20 import load_resnet20_cifar10

        model, _ = load_resnet20_cifar10(config["checkpoint"])
        data = cifar10.load(
            config["cifar"], extract_dir=os.path.join(DATA, "cifar10_extract"))
        X, _ = data["test_batch"]
        private_net, qinfo = quantise_resnet20_fmap(
            model,
            config["w_bits"],
            config["w_bits"],
            X[:64],
            normalise=(cifar10.MEAN, cifar10.STD),
        )
        rounds, _ = build_r3_rounds(private_net, 32)
        rng = np.random.default_rng([config["oracle_seed"], 1])
        noise_fn = noise_fns_bounded(rng)["Gaussian"]
        private_oracle = PartialOracle(
            rounds, private_net.A, noise_fn, rng, longdouble=True)
        private_oracle._audit_canary = canary
        if config["trace_arithmetic"]:
            original_eval_round = private_oracle.eval_round

            def traced_eval_round(round_index, true_maps):
                before_cache_hits = private_oracle.cache_hits
                output = original_eval_round(round_index, true_maps)
                maps = [np.asarray(item, dtype=np.int64) for item in true_maps]
                input_min = min(int(item.min()) for item in maps)
                input_max = max(int(item.max()) for item in maps)
                below = sum(int(np.sum(item < 0)) for item in maps)
                above = sum(
                    int(np.sum(item > private_net.A)) for item in maps)
                spec = private_oracle.rounds[round_index]
                arithmetic_rows.append({
                    "evaluation": int(private_oracle.n_round_evaluations + 1),
                    "session": int(private_oracle.n_sessions),
                    "round": int(round_index),
                    "round_name": str(spec.name),
                    "role": str(spec.role),
                    "cache_hit": bool(
                        private_oracle.cache_hits > before_cache_hits),
                    "input_maps": len(maps),
                    "input_entries": int(sum(item.size for item in maps)),
                    "input_min": input_min,
                    "input_max": input_max,
                    "input_below_zero": below,
                    "input_above_activation_max": above,
                    "output_entries": int(output.size),
                    "output_min": int(output.min()),
                    "output_max": int(output.max()),
                })
                return output

            private_oracle.eval_round = traced_eval_round
        manifest = public_manifest(private_net)
        manifest_hash = sha256_bytes(canonical_json(manifest))
        report.update({
            "status": "ready",
            "environment": env_info(),
            "private_model_sha256": model_digest(private_net),
            "public_manifest_sha256": manifest_hash,
            "quantisation": {
                "accumulator_bits_calibration": qinfo[
                    "accumulator_bits_calibration"],
                "accumulator_absmax_calibration": qinfo[
                    "accumulator_absmax_calibration"],
            },
            "canary_sha256": sha256_bytes(canary),
        })
        ready_connection.send_bytes(canonical_json({
            "status": "ready",
            "manifest": manifest,
            "public_manifest_sha256": manifest_hash,
        }))
        ready_connection.close()

        active_session = None
        while True:
            try:
                payload = connection.recv_bytes()
            except EOFError:
                report["status"] = "client_eof"
                break
            op = payload[0] if payload else -1
            tracked = op in (OP_NEW_SESSION, OP_ROUND)
            if tracked:
                transcript_update(transcript, b"Q", payload)
                request_bytes += len(payload)
                request_messages += 1
            try:
                if op == OP_NEW_SESSION:
                    active_session = private_oracle.session()
                    response = bytes([RESP_NONE])
                elif op == OP_ROUND:
                    if active_session is None:
                        raise RuntimeError("round requested before session")
                    read, vectors = decode_round(payload)
                    answer = active_session.round(vectors, read=read)
                    response = (bytes([RESP_NONE]) if answer is None
                                else encode_array(answer))
                elif op == OP_STATS:
                    response = encode_json_response({
                        "n_sessions": int(private_oracle.n_sessions),
                        "n_round_evaluations": int(
                            private_oracle.n_round_evaluations),
                        "n_replies_read": int(private_oracle.n_replies_read),
                        "cache_hits": int(private_oracle.cache_hits),
                        "rounding_exact": bool(private_oracle.rounding_exact),
                        "n_rounding_failures": int(
                            private_oracle.n_rounding_failures),
                        "inadmissible_entries": int(
                            private_oracle.inadmissible_entries),
                    })
                elif op == OP_CLOSE:
                    close_stats = {
                        "status": "closed",
                        "transcript_sha256": transcript.hexdigest(),
                        "request_bytes": int(request_bytes),
                        "response_bytes": int(response_bytes),
                        "request_messages": int(request_messages),
                        "response_messages": int(response_messages),
                    }
                    response = encode_json_response(close_stats)
                    connection.send_bytes(response)
                    report["status"] = "closed"
                    break
                else:
                    raise ValueError("unknown opcode %r" % op)
            except Exception as exc:
                response = encode_error(exc)
            if canary in response:
                canary_response_leaks += 1
            if tracked:
                transcript_update(transcript, b"R", response)
                response_bytes += len(response)
                response_messages += 1
            connection.send_bytes(response)
    except Exception as exc:
        report.update({
            "status": "error",
            "error": "%s: %s" % (type(exc).__name__, exc),
            "traceback": traceback.format_exc(),
        })
        try:
            ready_connection.send_bytes(canonical_json({
                "status": "error", "error": report["error"]}))
        except Exception:
            pass
    finally:
        if private_oracle is not None:
            report["oracle_counters"] = {
                "n_sessions": int(private_oracle.n_sessions),
                "n_round_evaluations": int(
                    private_oracle.n_round_evaluations),
                "n_replies_read": int(private_oracle.n_replies_read),
                "cache_hits": int(private_oracle.cache_hits),
                "rounding_failures": int(private_oracle.n_rounding_failures),
                "inadmissible_entries": int(private_oracle.inadmissible_entries),
            }
        report.update({
            "transcript_sha256": transcript.hexdigest(),
            "request_bytes": int(request_bytes),
            "response_bytes": int(response_bytes),
            "request_messages": int(request_messages),
            "response_messages": int(response_messages),
            "canary_response_leaks": int(canary_response_leaks),
            "wall_time_sec": round(time.time() - started, 3),
        })
        if config["trace_arithmetic"]:
            try:
                file_digest, content_digest = write_canonical_gzip_ndjson(
                    config["arithmetic_trace"], arithmetic_rows)
                report["arithmetic"] = arithmetic_summary(arithmetic_rows)
                report["arithmetic_trace_sha256"] = file_digest
                report["arithmetic_trace_file_sha256"] = file_digest
                report["arithmetic_trace_content_sha256"] = content_digest
                report["arithmetic_trace_canonical_encoding"] = (
                    "UTF-8 canonical NDJSON; gzip filename empty; mtime=0")
            except Exception as exc:
                report["arithmetic_trace_error"] = "%s: %s" % (
                    type(exc).__name__, exc)
        try:
            with open(config["oracle_report"], "w", encoding="utf-8") as handle:
                json.dump(jsonable(report), handle, indent=2, sort_keys=True)
                handle.write("\n")
        finally:
            try:
                connection.close()
            except Exception:
                pass


class RpcSession(object):
    def __init__(self, oracle):
        self.oracle = oracle
        self.r = 0

    def round(self, vectors, read=True):
        payload = encode_round(vectors, read)
        response = self.oracle._exchange(payload, tracked=True)
        self.oracle.n_round_evaluations += 1
        self.r += 1
        if not read:
            if response != bytes([RESP_NONE]):
                raise RuntimeError("expected an empty unread-round response")
            return None
        self.oracle.n_replies_read += 1
        return decode_array(response)


class RpcOracleView(object):
    """PartialOracle-compatible public proxy backed by the raw byte channel."""

    def __init__(self, connection, rounds, A):
        self.connection = connection
        self.rounds = rounds
        self.A = int(A)
        self.R = len(rounds)
        self.n_sessions = 0
        self.n_round_evaluations = 0
        self.n_replies_read = 0
        self.transcript = hashlib.sha256()
        self.closed = False

    def _exchange(self, payload, tracked=False):
        if tracked:
            transcript_update(self.transcript, b"Q", payload)
        self.connection.send_bytes(payload)
        response = self.connection.recv_bytes()
        if tracked:
            transcript_update(self.transcript, b"R", response)
        if response and response[0] == RESP_ERROR:
            raise RuntimeError(response[1:].decode("utf-8", errors="replace"))
        return response

    def _stats(self):
        return decode_json_response(
            self._exchange(bytes([OP_STATS]), tracked=False))

    def session(self):
        response = self._exchange(bytes([OP_NEW_SESSION]), tracked=True)
        if response != bytes([RESP_NONE]):
            raise RuntimeError("invalid new-session acknowledgement")
        self.n_sessions += 1
        return RpcSession(self)

    @property
    def cache_hits(self):
        return int(self._stats()["cache_hits"])

    @property
    def rounding_exact(self):
        return bool(self._stats()["rounding_exact"])

    @property
    def n_rounding_failures(self):
        return int(self._stats()["n_rounding_failures"])

    @property
    def inadmissible_entries(self):
        return int(self._stats()["inadmissible_entries"])

    def close(self):
        if self.closed:
            return {}
        server = decode_json_response(
            self._exchange(bytes([OP_CLOSE]), tracked=False))
        self.closed = True
        server["attacker_transcript_sha256"] = self.transcript.hexdigest()
        server["transcript_hash_match"] = bool(
            server["transcript_sha256"] == self.transcript.hexdigest())
        self.connection.close()
        return server


def install_data_guard(forbidden_root):
    root = os.path.realpath(forbidden_root)
    state = {"forbidden_open_attempts": 0, "sample": []}

    def audit(event, args):
        if event != "open" or not args:
            return
        candidate = args[0]
        if not isinstance(candidate, (str, bytes, os.PathLike)):
            return
        try:
            path = os.path.realpath(os.fsdecode(candidate))
            inside = os.path.commonpath([root, path]) == root
        except Exception:
            return
        if inside:
            state["forbidden_open_attempts"] += 1
            if len(state["sample"]) < 5:
                state["sample"].append(path)
            raise PermissionError("attacker process cannot open data path")

    sys.addaudithook(audit)
    return state


def attacker_worker(connection, manifest, config):
    started = time.time()
    report = {
        "schema": "p050-rpc-attacker-v1",
        "status": "initialising",
        "pid": os.getpid(),
        "start_method": "spawn",
    }
    proxy = None
    original_oracle = None
    try:
        from lib import fmap_partial as fp
        from lib.lta_run import summarize_recovery_records
        from lib.models import env_info

        view_audit = audit_attacker_view(config, globals())

        public_net = zero_network_from_manifest(manifest)
        public_rounds, _ = fp.build_r3_rounds(public_net, 32)
        if secret_nonzero_count(public_net) != 0:
            raise AssertionError("public network contains a nonzero secret entry")
        if sha256_bytes(canonical_json(manifest)) != config[
                "public_manifest_sha256"]:
            raise AssertionError("public manifest hash mismatch")

        guard = install_data_guard(config["forbidden_data_root"])
        original_oracle = fp.PartialOracle
        holder = {}

        def rpc_factory(rounds, A, noise_fn, rng, longdouble=True):
            del noise_fn, rng, longdouble
            if int(A) != int(manifest["A"]):
                raise AssertionError("public activation alphabet changed")
            view = RpcOracleView(connection, rounds, A)
            holder["proxy"] = view
            return view

        fp.PartialOracle = rpc_factory
        try:
            records, extracted, attack_summary = fp.extract_chain_partial(
                public_net,
                T=config["T"],
                seed=config["attack_seed"],
                log=lambda *args: None,
                deadline=time.time() + config["budget"],
                diagnostics=config["diagnostics"],
                repair_attempts=3,
            )
        finally:
            fp.PartialOracle = original_oracle

        proxy = holder.get("proxy")
        if extracted is None:
            raise RuntimeError("attack did not produce a recovered network")
        save_network(extracted, config["recovered_arrays"],
                     config["recovered_meta"])
        recovered_model_sha256 = model_digest(extracted)
        certificate = summarize_recovery_records(records)
        server_close = proxy.close()
        report.update({
            "status": "complete",
            "environment": env_info(),
            "public_manifest_sha256": config["public_manifest_sha256"],
            "public_secret_nonzero_entries": secret_nonzero_count(public_net),
            "forbidden_open_attempts": int(guard["forbidden_open_attempts"]),
            "forbidden_open_sample": guard["sample"],
            "attacker_view_audit": view_audit,
            "records_total": len(records),
            "client_certified_records": certificate["records_certified"],
            "lta_certificate_summary": certificate,
            "diagnostics": bool(config["diagnostics"]),
            "attack_summary": jsonable(attack_summary),
            "records": jsonable(records),
            "recovered_model_sha256": recovered_model_sha256,
            "recovered_arrays_sha256": sha256_file(config["recovered_arrays"]),
            "recovered_meta_sha256": sha256_file(config["recovered_meta"]),
            "server_close": server_close,
        })
        success = bool(
            len(records) == 29
            and certificate["success"]
            and certificate["records_certified"] == 29
            and secret_nonzero_count(public_net) == 0
            and guard["forbidden_open_attempts"] == 0
            and attack_summary["inadmissible_query_entries"] == 0
            and attack_summary["oracle_rounding_failures"] == 0
            and attack_summary["isolation_failures"] == 0
            and attack_summary["probe_multiset_failures"] == 0
            and attack_summary["missing_coordinate_events"] == 0
            and server_close.get("transcript_hash_match")
        )
        report["success"] = success
    except Exception as exc:
        report.update({
            "status": "error",
            "success": False,
            "error": "%s: %s" % (type(exc).__name__, exc),
            "traceback": traceback.format_exc(),
        })
        if original_oracle is not None:
            try:
                from lib import fmap_partial as fp
                fp.PartialOracle = original_oracle
            except Exception:
                pass
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


def evaluator_worker(config):
    started = time.time()
    report = {
        "schema": "p050-rpc-evaluator-v1",
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
        X, labels = data["test_batch"]
        private_net, qinfo = quantise_resnet20_fmap(
            model,
            config["w_bits"],
            config["w_bits"],
            X[:64],
            normalise=(cifar10.MEAN, cifar10.STD),
        )
        recovered = load_network(
            config["recovered_arrays"], config["recovered_meta"])
        truth = forward_fmap(private_net, X[:config["n_test"]])
        candidate = forward_fmap(recovered, X[:config["n_test"]])
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
            "max_abs_logit_difference": int(
                np.abs(truth - candidate).max()),
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
            "private_model_sha256": model_digest(private_net),
            "recovered_model_sha256": model_digest(recovered),
            "quantisation": {
                "accumulator_bits_calibration": qinfo[
                    "accumulator_bits_calibration"],
                "accumulator_absmax_calibration": qinfo[
                    "accumulator_absmax_calibration"],
            },
        })
        report["success"] = bool(
            report["identical_logit_vectors"] == config["n_test"]
            and report["post_calibration_identical_logit_vectors"]
                == report["post_calibration_images"]
            and report["max_abs_logit_difference"] == 0
        )
    except Exception as exc:
        report.update({
            "status": "error",
            "success": False,
            "error": "%s: %s" % (type(exc).__name__, exc),
            "traceback": traceback.format_exc(),
        })
    finally:
        report["wall_time_sec"] = round(time.time() - started, 3)
        with open(config["evaluator_report"], "w", encoding="utf-8") as handle:
            json.dump(jsonable(report), handle, indent=2, sort_keys=True)
            handle.write("\n")
    raise SystemExit(0 if report.get("success") else 1)


def read_json(path):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def main():
    started = time.time()
    os.makedirs(OUT_DIR, exist_ok=True)
    legacy_seed = os.environ.get("TDSC_RPC_SEED")
    attack_seed = int(os.environ.get(
        "TDSC_RPC_ATTACK_SEED", legacy_seed or "20260923"))
    oracle_seed = int(os.environ.get(
        "TDSC_RPC_ORACLE_SEED", legacy_seed or "20260924"))
    expected_model_sha256 = os.environ.get(
        "TDSC_RPC_EXPECT_MODEL_SHA256", "")
    checkpoint = os.path.join(DATA, "cifar10_resnet20.pt")
    cifar = os.path.join(DATA, "cifar-10-python.tar.gz")
    paths = {
        "oracle_report": os.path.join(OUT_DIR, "oracle.json"),
        "attacker_report": os.path.join(OUT_DIR, "attacker.json"),
        "evaluator_report": os.path.join(OUT_DIR, "evaluator.json"),
        "recovered_arrays": os.path.join(OUT_DIR, "recovered.npz"),
        "recovered_meta": os.path.join(OUT_DIR, "recovered.json"),
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
        "budget": BUDGET,
        "T": T_STEPS,
        "forbidden_data_root": DATA,
        "diagnostics": DIAGNOSTICS,
        "attacker_report": paths["attacker_report"],
        "recovered_arrays": paths["recovered_arrays"],
        "recovered_meta": paths["recovered_meta"],
    }
    evaluator_config = {
        "checkpoint": checkpoint,
        "cifar": cifar,
        "w_bits": W_BITS,
        "n_test": N_TEST,
        "recovered_arrays": paths["recovered_arrays"],
        "recovered_meta": paths["recovered_meta"],
        "evaluator_report": paths["evaluator_report"],
    }
    ctx = mp.get_context("spawn")
    rpc_attacker, rpc_oracle = ctx.Pipe(duplex=True)
    ready_parent, ready_oracle = ctx.Pipe(duplex=True)
    oracle = ctx.Process(
        name="p050-oracle",
        target=oracle_worker,
        args=(rpc_oracle, ready_oracle, oracle_config),
    )
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
        name="p050-attacker",
        target=attacker_worker,
        args=(rpc_attacker, ready["manifest"], attacker_config),
    )
    saved_private_environment = remove_private_environment_for_spawn()
    try:
        attacker.start()
    finally:
        restore_private_environment(saved_private_environment)
    rpc_attacker.close()
    attacker.join(BUDGET + 600)
    if attacker.is_alive():
        raise RuntimeError("attacker exceeded orchestration timeout")
    oracle.join(120)
    if oracle.is_alive():
        raise RuntimeError("oracle did not close after attacker exit")

    evaluator_exit = None
    if (not SKIP_EVALUATOR
            and attacker.exitcode == 0 and oracle.exitcode == 0):
        evaluator = ctx.Process(
            name="p050-evaluator", target=evaluator_worker,
            args=(evaluator_config,))
        evaluator.start()
        evaluator.join(EVAL_TIMEOUT)
        if evaluator.is_alive():
            raise RuntimeError("evaluator exceeded orchestration timeout")
        evaluator_exit = evaluator.exitcode

    attacker_report = (read_json(paths["attacker_report"])
                       if os.path.exists(paths["attacker_report"]) else {})
    oracle_report = (read_json(paths["oracle_report"])
                     if os.path.exists(paths["oracle_report"]) else {})
    evaluator_report = (read_json(paths["evaluator_report"])
                        if os.path.exists(paths["evaluator_report"]) else {})
    evaluator_success = bool(
        SKIP_EVALUATOR
        or (evaluator_exit == 0 and evaluator_report.get("success")))
    arithmetic_success = bool(
        not TRACE_ARITHMETIC
        or (
            oracle_report.get("arithmetic_trace_sha256")
            and not oracle_report.get("arithmetic_trace_error")
            and oracle_report.get("arithmetic", {}).get("trace_rows")
                == oracle_report.get("oracle_counters", {}).get(
                    "n_round_evaluations")
            and oracle_report.get("arithmetic", {}).get("input_below_zero") == 0
            and oracle_report.get("arithmetic", {}).get(
                "input_above_activation_max") == 0
        )
    )
    post_attack_expected_digest_match = bool(
        not expected_model_sha256
        or attacker_report.get("recovered_model_sha256")
            == expected_model_sha256
    )
    success = bool(
        attacker.exitcode == 0
        and oracle.exitcode == 0
        and attacker_report.get("success")
        and oracle_report.get("status") == "closed"
        and oracle_report.get("canary_response_leaks") == 0
        and evaluator_success
        and arithmetic_success
        and post_attack_expected_digest_match
        and (SKIP_EVALUATOR
             or attacker_report.get("recovered_model_sha256")
                == evaluator_report.get("recovered_model_sha256"))
        and (SKIP_EVALUATOR
             or oracle_report.get("private_model_sha256")
                == evaluator_report.get("private_model_sha256"))
    )
    result = {
        "schema": "p050-tdsc-rpc-isolated-regression-v2",
        "run_name": RUN_NAME,
        "success": success,
        "start_method": "spawn",
        "parent_loaded_private_model": False,
        "diagnostics": DIAGNOSTICS,
        "evaluation_skipped": SKIP_EVALUATOR,
        "arithmetic_trace_enabled": TRACE_ARITHMETIC,
        "arithmetic_trace_success": arithmetic_success,
        "expected_model_sha256": expected_model_sha256 or None,
        "post_attack_expected_digest_match": post_attack_expected_digest_match,
        "truth_digest_comparison_stage": "parent after attacker exit",
        "truth_digest_disclosed_to_attacker": False,
        "transport": (
            "raw length-prefixed little-endian int64 arrays over "
            "Connection.send_bytes/recv_bytes; no recv/pickle"
        ),
        "attack_seed": attack_seed,
        "oracle_seed": oracle_seed,
        "seed_disclosed_to_attacker": False,
        "w_bits": W_BITS,
        "T": T_STEPS,
        "budget_sec": BUDGET,
        "n_test": N_TEST,
        "process_exit_codes": {
            "oracle": oracle.exitcode,
            "attacker": attacker.exitcode,
            "evaluator": evaluator_exit,
        },
        "public_manifest_sha256": ready["public_manifest_sha256"],
        "source_sha256": sha256_file(os.path.abspath(__file__)),
        "oracle": oracle_report,
        "attacker": attacker_report,
        "evaluator": evaluator_report,
        "wall_time_sec": round(time.time() - started, 3),
    }
    with open(paths["result"], "w", encoding="utf-8") as handle:
        json.dump(jsonable(result), handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps({
        "output": paths["result"],
        "success": success,
        "process_exit_codes": result["process_exit_codes"],
        "client_certified_records": attacker_report.get(
            "client_certified_records"),
        "sessions_total": attacker_report.get(
            "attack_summary", {}).get("sessions_total"),
        "identical_logit_fraction": evaluator_report.get(
            "identical_logit_fraction"),
        "transcript_hash_match": attacker_report.get(
            "server_close", {}).get("transcript_hash_match"),
        "forbidden_open_attempts": attacker_report.get(
            "forbidden_open_attempts"),
        "evaluation_skipped": SKIP_EVALUATOR,
        "recovered_model_sha256": attacker_report.get(
            "recovered_model_sha256"),
        "minimum_signed_bits": oracle_report.get(
            "arithmetic", {}).get("minimum_signed_bits"),
    }, sort_keys=True), flush=True)
    return 0 if success else 1


if __name__ == "__main__":
    raise SystemExit(main())

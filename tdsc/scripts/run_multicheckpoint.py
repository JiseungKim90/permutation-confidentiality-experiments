#!/usr/bin/env python3
"""Fail-closed multi-checkpoint audit for the TDSC whole-network attack.

Every checkpoint is run with one shared configuration.  The driver never
changes attack seeds, search limits, control values, or completion settings in
response to an outcome.  A nonzero process, missing result, or incomplete
certificate is recorded as a fail-closed outcome rather than retried.
"""

import argparse
import concurrent.futures
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

import numpy as np
import torch

import _bootstrap


ROOT = Path(_bootstrap.ROOT)
DATA = Path(_bootstrap.DATA)
RESULTS = Path(_bootstrap.RESULTS)
# Hash the complete library surface used by the spawned extraction/completion
# processes, not only their two entry points.  The former four-file list missed
# changes to the LTA certificate and feature-map assembler, precisely the code
# whose effect this audit is intended to measure.
SOURCE_FILES = tuple(sorted((ROOT / "lib").glob("*.py"))) + (
    ROOT / "scripts" / "run_extraction.py",
    ROOT / "scripts" / "run_completion.py",
    Path(__file__).resolve(),
)


def utc_now():
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat()


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            block = handle.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def canonical_hash(records):
    payload = json.dumps(records, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def read_json(path):
    try:
        with Path(path).open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, ValueError):
        return {}


def nested(record, *keys):
    current = record
    for key in keys:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def sanitised_environment():
    env = dict(os.environ)
    for key in tuple(env):
        if key.startswith("TDSC_"):
            env.pop(key, None)
    env.update({
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "PYTHONPATH": "scripts:.",
    })
    return env


def run_logged(command, env, log_path, timeout):
    started = time.time()
    with Path(log_path).open("wb") as handle:
        completed = subprocess.run(
            command,
            cwd=str(ROOT),
            env=env,
            stdout=handle,
            stderr=subprocess.STDOUT,
            timeout=timeout,
            check=False,
        )
    return completed.returncode, round(time.time() - started, 3)


def extraction_summary(record, returncode):
    attacker = record.get("attacker", {})
    evaluator = record.get("evaluator", {})
    attack_summary = attacker.get("attack_summary", {})
    certificate = attacker.get("lta_certificate_summary", {})
    return {
        "returncode": returncode,
        "success": bool(returncode == 0 and record.get("success") is True),
        "process_exit_codes": record.get("process_exit_codes"),
        "client_certified_records": attacker.get("client_certified_records"),
        "lta_invocations_certified": certificate.get("lta_invocations_certified"),
        "sessions": attack_summary.get("sessions_total"),
        "affine_evaluations": attack_summary.get("round_evaluations_total"),
        "identical_logit_vectors": evaluator.get("identical_logit_vectors"),
        "max_abs_logit_difference": evaluator.get("max_abs_logit_difference"),
        "attacker_status": attacker.get("status"),
        "attacker_error": attacker.get("error"),
        "oracle_status": nested(record, "oracle", "status"),
    }


def completion_summary(record, returncode):
    attacker = record.get("attacker", {})
    evaluator = record.get("evaluator", {})
    structural = evaluator.get("structural_certificate", {})
    recovery = attacker.get("recovery", {})
    oracle_stats = attacker.get("oracle_stats_before_close", {})
    return {
        "returncode": returncode,
        "success": bool(returncode == 0 and record.get("success") is True),
        "process_exit_codes": record.get("process_exit_codes"),
        "observable_maps_checked": structural.get("parameter_records_checked"),
        "observable_maps_successful": structural.get(
            "parameter_records_successful"),
        "identical_logit_vectors": evaluator.get("identical_logit_vectors"),
        "max_abs_logit_difference": evaluator.get("max_abs_logit_difference"),
        "first_conv_singleton_checks": sum(
            int(value) for value in
            (recovery.get("line_candidate_count_histogram") or {}).values()),
        "shortcut_singleton_checks": sum(
            int(value) for value in
            (recovery.get("shortcut_line_candidate_count_histogram") or {}).values()),
        "sessions": oracle_stats.get("n_sessions"),
        "affine_evaluations": oracle_stats.get("n_round_evaluations"),
        "attacker_status": attacker.get("status"),
        "attacker_error": attacker.get("error"),
        "oracle_status": nested(record, "oracle", "status"),
    }


def full_certificate(extraction, completion, n_test):
    return bool(
        extraction.get("success")
        and extraction.get("client_certified_records") == 29
        and extraction.get("lta_invocations_certified") == 34
        and extraction.get("identical_logit_vectors") == n_test
        and extraction.get("max_abs_logit_difference") == 0
        and completion.get("success")
        and completion.get("observable_maps_checked") == 29
        and completion.get("observable_maps_successful") == 29
        and completion.get("identical_logit_vectors") == n_test
        and completion.get("max_abs_logit_difference") == 0
    )


def validate_checkpoint(item):
    name = item["name"]
    path = (ROOT / item["path"]).resolve()
    data_root = DATA.resolve()
    if data_root != path and data_root not in path.parents:
        raise ValueError("%s is outside the guarded data directory" % path)
    if not path.is_file():
        raise ValueError("missing checkpoint %s" % path)
    actual = sha256_file(path)
    expected = str(item["sha256"]).lower()
    if actual != expected:
        raise ValueError("%s digest is %s, expected %s" % (name, actual, expected))
    return name, path, expected


def one_checkpoint(item, common, audit_dir):
    name, checkpoint, digest = validate_checkpoint(item)
    extraction_run = "%s-%s-extraction" % (common["run_name"], name)
    completion_run = "%s-%s-completion" % (common["run_name"], name)
    extraction_dir = RESULTS / "provenance" / extraction_run
    completion_dir = RESULTS / "provenance" / completion_run
    if extraction_dir.exists() or completion_dir.exists():
        raise ValueError("refusing to overwrite an existing run for %s" % name)

    base_env = sanitised_environment()
    base_env.update({
        "TDSC_MODEL_CHECKPOINT": str(checkpoint),
        "TDSC_MODEL_CHECKPOINT_SHA256": digest,
    })
    extraction_env = dict(base_env)
    extraction_env.update({
        "TDSC_RPC_RUN": extraction_run,
        "TDSC_RPC_NTEST": str(common["n_test"]),
        "TDSC_RPC_BUDGET": str(common["budget_sec"]),
        "TDSC_RPC_EVAL_TIMEOUT": str(common["eval_timeout_sec"]),
        "TDSC_RPC_ATTACK_SEED": str(common["attack_seed"]),
        "TDSC_RPC_ORACLE_SEED": str(common["extraction_oracle_seed"]),
        "TDSC_RPC_WBITS": str(common["w_bits"]),
        "TDSC_RPC_T": str(common["t_steps"]),
        "TDSC_RPC_TRACE_ARITHMETIC": str(int(common["trace_arithmetic"])),
    })
    extraction_log = audit_dir / (name + "-extraction.log")
    try:
        extraction_rc, extraction_wall = run_logged(
            [sys.executable, "scripts/run_extraction.py"],
            extraction_env,
            extraction_log,
            common["budget_sec"] + common["eval_timeout_sec"] + 1200,
        )
    except subprocess.TimeoutExpired:
        extraction_rc, extraction_wall = 124, None
    extraction_record = read_json(extraction_dir / "result.json")
    extraction = extraction_summary(extraction_record, extraction_rc)
    extraction["wall_time_sec_driver"] = extraction_wall
    extraction["log"] = str(extraction_log)

    completion = {
        "returncode": None,
        "success": False,
        "not_run_reason": "extraction did not pass its fail-closed checks",
    }
    if extraction["success"]:
        completion_env = dict(base_env)
        completion_env.update({
            "TDSC_CLASS_RUN": completion_run,
            "TDSC_CLASS_SOURCE": extraction_run,
            "TDSC_CLASS_ATTACK_SEED": str(common["attack_seed"]),
            "TDSC_CLASS_ORACLE_SEED": str(common["completion_oracle_seed"]),
            "TDSC_CLASS_WBITS": str(common["w_bits"]),
            "TDSC_CLASS_NTEST": str(common["n_test"]),
            "TDSC_CLASS_SEARCH": str(common["search_limit"]),
            "TDSC_CLASS_COVER": str(common["minimum_marker_cover"]),
            "TDSC_CLASS_SREPEATS": str(common["shortcut_carrier_repeats"]),
            "TDSC_CLASS_VALUES": common["control_values"],
            "TDSC_CLASS_TRACE": str(int(common["trace_arithmetic"])),
            "TDSC_CLASS_SKIP_EVAL": "0",
        })
        completion_log = audit_dir / (name + "-completion.log")
        try:
            completion_rc, completion_wall = run_logged(
                [sys.executable, "scripts/run_completion.py"],
                completion_env,
                completion_log,
                common["budget_sec"] + common["eval_timeout_sec"] + 1200,
            )
        except subprocess.TimeoutExpired:
            completion_rc, completion_wall = 124, None
        completion_record = read_json(completion_dir / "result.json")
        completion = completion_summary(completion_record, completion_rc)
        completion["wall_time_sec_driver"] = completion_wall
        completion["log"] = str(completion_log)

    certified = full_certificate(extraction, completion, common["n_test"])
    return {
        "name": name,
        "checkpoint_sha256": digest,
        "provenance": item.get("provenance"),
        "training_seed": item.get("training_seed"),
        "training_precision": item.get("training_precision"),
        "configuration_changed_for_checkpoint": False,
        "extraction": extraction,
        "completion": completion,
        "full_certificate": certified,
        "outcome": "certified" if certified else "fail_closed",
    }


def git_value(*args):
    try:
        return subprocess.check_output(
            ["git"] + list(args), cwd=str(ROOT.parent), text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--only", default="")
    parser.add_argument("--n-test", type=int, default=10000)
    parser.add_argument("--budget-sec", type=int, default=1800)
    parser.add_argument("--eval-timeout-sec", type=int, default=2400)
    parser.add_argument("--trace-arithmetic", action="store_true")
    args = parser.parse_args()

    manifest = read_json(args.manifest)
    checkpoints = list(manifest.get("checkpoints", []))
    selected = set(filter(None, args.only.split(",")))
    if selected:
        checkpoints = [item for item in checkpoints if item.get("name") in selected]
        missing = selected.difference(item.get("name") for item in checkpoints)
        if missing:
            raise ValueError("unknown --only checkpoint(s): %s" % sorted(missing))
    if not checkpoints:
        raise ValueError("manifest selects no checkpoints")

    audit_dir = RESULTS / "multicheckpoint" / args.run_name
    if audit_dir.exists():
        raise ValueError("refusing to overwrite %s" % audit_dir)
    audit_dir.mkdir(parents=True)
    common = {
        "run_name": args.run_name,
        "attack_seed": 20260923,
        "extraction_oracle_seed": 20260924,
        "completion_oracle_seed": 20260925,
        "w_bits": 8,
        "t_steps": 3,
        "n_test": args.n_test,
        "budget_sec": args.budget_sec,
        "eval_timeout_sec": args.eval_timeout_sec,
        "search_limit": 256,
        "minimum_marker_cover": 2,
        "shortcut_carrier_repeats": 2,
        "control_values": "1,2,4,8,16,32,64,128,255",
        "trace_arithmetic": args.trace_arithmetic,
    }
    source_hashes = {str(path.relative_to(ROOT)): sha256_file(path)
                     for path in SOURCE_FILES}
    launch = {
        "schema": "p050-multicheckpoint-launch-v1",
        "started_utc": utc_now(),
        "host": platform.node(),
        "resolved_working_directory": str(ROOT),
        "exact_command": [sys.executable] + sys.argv,
        "git_commit": git_value("rev-parse", "HEAD"),
        "git_diff_sha256": hashlib.sha256(
            (git_value("diff", "--binary") or "").encode("utf-8")).hexdigest(),
        "source_sha256": source_hashes,
        "source_set_sha256": canonical_hash(source_hashes),
        "environment": {
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "torch": torch.__version__,
            "platform": platform.platform(),
        },
        "global_configuration": common,
        "workers": args.workers,
        "checkpoint_names": [item["name"] for item in checkpoints],
        "success_criteria": (
            "Each certified checkpoint must pass extraction, 29/29 observable-map "
            "completion, and exact logits on n_test images without a per-checkpoint "
            "seed or parameter change.  The paper-inclusion gate requires at least "
            "five certified checkpoints."
        ),
        "failure_criteria": (
            "Any nonzero process, missing output, failed internal check, fewer than "
            "29/29 observable maps, or nonzero logit difference is fail_closed; "
            "there is no automatic retry."
        ),
    }
    with (audit_dir / "launch.json").open("w", encoding="utf-8") as handle:
        json.dump(launch, handle, indent=2, sort_keys=True)
        handle.write("\n")

    records = []
    with concurrent.futures.ThreadPoolExecutor(
            max_workers=max(1, args.workers)) as executor:
        futures = {executor.submit(one_checkpoint, item, common, audit_dir): item
                   for item in checkpoints}
        for future in concurrent.futures.as_completed(futures):
            item = futures[future]
            try:
                record = future.result()
            except Exception as exc:
                record = {
                    "name": item.get("name"),
                    "checkpoint_sha256": item.get("sha256"),
                    "configuration_changed_for_checkpoint": False,
                    "full_certificate": False,
                    "outcome": "driver_error",
                    "driver_error": "%s: %s" % (type(exc).__name__, exc),
                }
            records.append(record)
            records.sort(key=lambda row: row.get("name") or "")
            with (audit_dir / "progress.json").open("w", encoding="utf-8") as handle:
                json.dump(records, handle, indent=2, sort_keys=True)
                handle.write("\n")

    certified = sum(bool(record.get("full_certificate")) for record in records)
    result = {
        "schema": "p050-multicheckpoint-result-v1",
        "completed_utc": utc_now(),
        "launch": launch,
        "attempted_checkpoints": len(records),
        "certified_checkpoints": certified,
        "fail_closed_checkpoints": sum(
            record.get("outcome") == "fail_closed" for record in records),
        "driver_error_checkpoints": sum(
            record.get("outcome") == "driver_error" for record in records),
        "per_checkpoint_manual_tuning": False,
        "paper_inclusion_gate_passed": certified >= 5,
        "records": records,
    }
    with (audit_dir / "result.json").open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps({
        "result": str(audit_dir / "result.json"),
        "attempted": len(records),
        "certified": certified,
        "paper_inclusion_gate_passed": certified >= 5,
    }, sort_keys=True))
    return 0 if certified == len(records) else 1


if __name__ == "__main__":
    raise SystemExit(main())

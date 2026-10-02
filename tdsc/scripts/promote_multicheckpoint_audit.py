#!/usr/bin/env python3
"""Validate and path-sanitise a complete multi-checkpoint audit result."""

import argparse
import hashlib
import json
from pathlib import Path


EXPECTED_NAMES = {"official"}.union(
    "qat-w%d-s%d" % (width, seed)
    for width in range(5, 9)
    for seed in range(2)
)


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def selected(source, fields):
    return {field: source.get(field) for field in fields}


def clean_record(record):
    common = (
        "success",
        "returncode",
        "process_exit_codes",
        "attacker_status",
        "oracle_status",
        "forward_implementation",
        "sessions",
        "affine_evaluations",
        "identical_logit_vectors",
        "post_calibration_identical_logit_vectors",
        "max_abs_logit_difference",
        "parallel_batched_exact",
        "accuracy_private",
    )
    extraction = selected(
        record["extraction"],
        common + (
            "client_certified_records",
            "lta_invocations_certified",
            "lta_passes_total",
            "repairs_triggered",
        ),
    )
    completion = selected(
        record["completion"],
        common + (
            "observable_maps_checked",
            "observable_maps_successful",
            "first_conv_singleton_checks",
            "shortcut_singleton_checks",
        ),
    )
    for phase in (extraction, completion):
        accuracy = phase.get("accuracy_private")
        if isinstance(accuracy, (int, float)):
            phase["accuracy_private"] = round(float(accuracy), 2)
    return {
        "name": record["name"],
        "checkpoint_sha256": record["checkpoint_sha256"],
        "provenance": record.get("provenance"),
        "training_precision": record.get("training_precision"),
        "training_seed": record.get("training_seed"),
        "configuration_changed_for_checkpoint": record.get(
            "configuration_changed_for_checkpoint"),
        "full_certificate": record.get("full_certificate"),
        "outcome": record.get("outcome"),
        "extraction": extraction,
        "completion": completion,
    }


def validate(result):
    if result.get("schema") != "p050-multicheckpoint-result-v1":
        raise ValueError("unexpected result schema")
    records = result.get("records")
    if not isinstance(records, list) or len(records) != 9:
        raise ValueError("expected exactly nine checkpoint records")
    names = [record.get("name") for record in records]
    if len(names) != len(set(names)) or set(names) != EXPECTED_NAMES:
        raise ValueError("checkpoint schedule is incomplete or duplicated")
    if (
        result.get("attempted_checkpoints") != 9
        or result.get("certified_checkpoints") != 9
        or result.get("fail_closed_checkpoints") != 0
        or result.get("driver_error_checkpoints") != 0
        or result.get("all_selected_checkpoints_certified") is not True
        or result.get("per_checkpoint_manual_tuning") is not False
    ):
        raise ValueError("result is not a complete 9/9 fixed-configuration audit")
    launch = result.get("launch", {})
    if set(launch.get("checkpoint_names", [])) != EXPECTED_NAMES:
        raise ValueError("launch record does not select the complete schedule")
    if launch.get("git_diff_sha256") != (
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"):
        raise ValueError("audit was not launched from a clean source tree")
    n_test = launch.get("global_configuration", {}).get("n_test")
    held_out = max(0, n_test - min(64, n_test)) if isinstance(n_test, int) else None
    for record in records:
        if (
            record.get("outcome") != "certified"
            or record.get("full_certificate") is not True
            or record.get("configuration_changed_for_checkpoint") is not False
        ):
            raise ValueError("uncertified checkpoint: %s" % record.get("name"))
        extraction = record.get("extraction", {})
        completion = record.get("completion", {})
        if (
            extraction.get("success") is not True
            or extraction.get("returncode") != 0
            or extraction.get("client_certified_records") != 29
            or extraction.get("lta_invocations_certified") != 34
            or not isinstance(extraction.get("lta_passes_total"), int)
            or extraction.get("lta_passes_total") < 34
            or not isinstance(extraction.get("sessions"), int)
            or extraction.get("sessions") <= 0
            or not isinstance(extraction.get("affine_evaluations"), int)
            or extraction.get("affine_evaluations") <= 0
            or extraction.get("identical_logit_vectors") != n_test
            or extraction.get("post_calibration_identical_logit_vectors")
                != held_out
            or extraction.get("max_abs_logit_difference") != 0
            or extraction.get("parallel_batched_exact") is not True
        ):
            raise ValueError("incomplete extraction certificate: %s" %
                             record.get("name"))
        if (
            completion.get("success") is not True
            or completion.get("returncode") != 0
            or completion.get("observable_maps_checked") != 29
            or completion.get("observable_maps_successful") != 29
            or not isinstance(completion.get("sessions"), int)
            or completion.get("sessions") <= 0
            or not isinstance(completion.get("affine_evaluations"), int)
            or completion.get("affine_evaluations") <= 0
            or completion.get("identical_logit_vectors") != n_test
            or completion.get("post_calibration_identical_logit_vectors")
                != held_out
            or completion.get("max_abs_logit_difference") != 0
            or completion.get("parallel_batched_exact") is not True
        ):
            raise ValueError("incomplete completion certificate: %s" %
                             record.get("name"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--result", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result_path = Path(args.result).resolve()
    output_path = Path(args.output).resolve()
    if output_path.exists():
        raise FileExistsError("refusing to overwrite %s" % output_path)
    launch_path = result_path.with_name("launch.json")
    with result_path.open("r", encoding="utf-8") as handle:
        result = json.load(handle)
    with launch_path.open("r", encoding="utf-8") as handle:
        launch_file = json.load(handle)
    validate(result)
    launch = result.get("launch")
    if launch != launch_file:
        raise ValueError("embedded and standalone launch records differ")
    common = launch["global_configuration"]
    controls = [int(value) for value in common["control_values"].split(",")]
    audit = {
        "schema": "p050-public-multicheckpoint-audit-v2",
        "audit_date_utc": result["completed_utc"][:10],
        "base_commit": launch["git_commit"],
        "claim_scope": (
            "Fixed-configuration weight-set audit of one exact-integer "
            "ResNet-20 simulator architecture; not a population success-rate "
            "estimate or a deployed encrypted-backend exploit."
        ),
        "fixed_configuration": {
            "attack_seed": common["attack_seed"],
            "extraction_oracle_seed": common["extraction_oracle_seed"],
            "completion_oracle_seed": common["completion_oracle_seed"],
            "weight_bits": common["w_bits"],
            "t_steps": common["t_steps"],
            "evaluation_images": common["n_test"],
            "budget_sec": common["budget_sec"],
            "evaluation_timeout_sec": common["eval_timeout_sec"],
            "search_limit": common["search_limit"],
            "minimum_marker_cover": common["minimum_marker_cover"],
            "shortcut_carrier_repeats": common["shortcut_carrier_repeats"],
            "trace_arithmetic": common["trace_arithmetic"],
            "control_values": controls,
            "per_checkpoint_manual_tuning": False,
        },
        "result_summary": {
            "attempted_checkpoints": result["attempted_checkpoints"],
            "certified_checkpoints": result["certified_checkpoints"],
            "fail_closed_checkpoints": result["fail_closed_checkpoints"],
            "driver_error_checkpoints": result["driver_error_checkpoints"],
            "all_checkpoints_certified": result[
                "all_selected_checkpoints_certified"],
            "per_checkpoint_manual_tuning": result[
                "per_checkpoint_manual_tuning"],
        },
        "records": [
            clean_record(record)
            for record in sorted(result["records"], key=lambda row: row["name"])
        ],
        "reproducibility": {
            "driver": "scripts/run_multicheckpoint.py",
            "promotion_script": "scripts/promote_multicheckpoint_audit.py",
            "checkpoint_manifest": "reference/multicheckpoint/checkpoints.json",
            "all_qat_checkpoint_bytes_distributed": True,
            "official_checkpoint_and_cifar_downloaded_by_digest": True,
            "qat_models_independent_from_scratch": False,
        },
        "run_provenance": {
            "run_name": common["run_name"],
            "git_commit": launch["git_commit"],
            "git_diff_sha256": launch["git_diff_sha256"],
            "source_set_sha256": launch["source_set_sha256"],
            "launch_sha256": sha256_file(launch_path),
            "result_sha256": sha256_file(result_path),
            "environment": launch["environment"],
            "workers": launch["workers"],
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("x", encoding="utf-8") as handle:
        json.dump(audit, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps({
        "output": str(output_path),
        "records": len(audit["records"]),
        "result_sha256": audit["run_provenance"]["result_sha256"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

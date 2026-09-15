#!/usr/bin/env python3
"""Verify complete extraction/completion outputs against submission claims."""

import argparse
import hashlib
import json
from pathlib import Path


EXPECTED = {
    "recovered.npz": "72a3a5d7f7fe3344f283d29fe65f9d12dabd87accb400b3c03f936d21854b7dd",
    "recovered.json": "d49345c2820771ede65ccce55a8e1935d3bb8fbad996e522903880bf7034cbf6",
    "extraction_trace": "c5f2aba5affea371a4d50e82c98558c594368ca5d87ae9b1298ed3486f7d3cdb",
    "extraction_trace_content": "1ff92bff62e9a5dccabb640b097664e68124b08e2b02380af4d9a63d187b9951",
    "completed.npz": "f4fc25cb10a1bbe6d4987c7c008bb871faf220b669130f6017e2edbce15bfe4e",
    "completed.json": "d49345c2820771ede65ccce55a8e1935d3bb8fbad996e522903880bf7034cbf6",
    "completion_trace": "40186244aa7566e08def1b260cb63928ecfc178a4f6e8480d5228e4205259b02",
    "completion_trace_content": "ae3961421ff6b15f1d77cc6f65e441384e2a782b1118b6dd341e761a0226fd0e",
    "recovered_model": "b3b291c77186dfbccf406cd3bb361e9dc257788e1886174bc2f3ea6966babcda",
    "completed_model": "c30f2aa072c5a57ab42d6e7e326af109ea8a4e09d99e4de274a38e20d228f075",
    "private_model": "08185fccb93a919a4d16a3232dbf3a6339315d6d0b40ea3aa4a72c56cfbacb6d",
}


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def load_result(directory, failures):
    path = directory / "result.json"
    if not path.is_file():
        failures.append("missing result: %s" % path)
        return None
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def expect(failures, condition, message):
    if not condition:
        failures.append(message)


def empty_private_audit(audit):
    return all(not audit.get(field) for field in (
        "private_config_keys_present",
        "private_environment_keys_present",
        "private_imported_globals_present",
    ))


def check_file_hash(directory, filename, expected_key, failures):
    path = directory / filename
    if not path.is_file():
        failures.append("missing output: %s" % path)
        return
    actual = sha256_file(path)
    expect(failures, actual == EXPECTED[expected_key],
           "%s SHA-256 is %s, expected %s" % (filename, actual, EXPECTED[expected_key]))


def verify_extraction(directory):
    failures = []
    result = load_result(directory, failures)
    if result is None:
        return failures
    attacker = result.get("attacker", {})
    oracle = result.get("oracle", {})
    evaluator = result.get("evaluator", {})
    arithmetic = oracle.get("arithmetic", {})
    summary = attacker.get("attack_summary", {})
    certificate = attacker.get("lta_certificate_summary", {})

    expect(failures, result.get("success") is True, "extraction success is not true")
    expect(failures, result.get("process_exit_codes") == {
        "attacker": 0, "evaluator": 0, "oracle": 0}, "extraction process exit codes differ")
    expect(failures, result.get("parent_loaded_private_model") is False,
           "extraction parent loaded the private model")
    expect(failures, result.get("seed_disclosed_to_attacker") is False,
           "oracle seed was disclosed to the attacker")
    expect(failures, result.get("truth_digest_disclosed_to_attacker") is False,
           "truth digest was disclosed to the attacker")
    expect(failures, result.get("post_attack_expected_digest_match") is True,
           "post-attack expected digest check failed")
    expect(failures, result.get("arithmetic_trace_success") is True,
           "extraction arithmetic trace audit failed")

    expect(failures, attacker.get("success") is True, "attacker did not succeed")
    expect(failures, attacker.get("forbidden_open_attempts") == 0,
           "attacker attempted to open private data")
    expect(failures, attacker.get("public_secret_nonzero_entries") == 0,
           "public attacker model contains secret entries")
    expect(failures, empty_private_audit(attacker.get("attacker_view_audit", {})),
           "attacker view contains private state")
    expect(failures, attacker.get("client_certified_records") == 29,
           "attacker did not certify 29 records")
    expect(failures, certificate.get("success") is True,
           "LTA certificate summary is not successful")
    expect(failures, certificate.get("lta_invocations_certified") == 34,
           "LTA certificate count is not 34")
    expect(failures, summary.get("sessions_total") == 3676,
           "extraction session count is not 3,676")
    expect(failures, summary.get("round_evaluations_total") == 73520,
           "extraction affine-evaluation count is not 73,520")
    for key in ("isolation_failures", "inadmissible_query_entries",
                "oracle_rounding_failures", "probe_multiset_failures"):
        expect(failures, summary.get(key) == 0, "%s is not zero" % key)

    expect(failures, evaluator.get("success") is True, "extraction evaluator failed")
    expect(failures, evaluator.get("identical_logit_vectors") == 10000,
           "extraction does not match all 10,000 logits")
    expect(failures, evaluator.get("post_calibration_identical_logit_vectors") == 9936,
           "extraction does not match all 9,936 post-calibration logits")
    expect(failures, evaluator.get("max_abs_logit_difference") == 0,
           "extraction has a nonzero logit difference")
    expect(failures, evaluator.get("accuracy_private") == evaluator.get("accuracy_recovered") == 92.09,
           "extraction full-set accuracies differ from 92.09")
    expect(failures, evaluator.get("recovered_model_sha256") == EXPECTED["recovered_model"],
           "recovered model digest differs")
    expect(failures, evaluator.get("private_model_sha256") == EXPECTED["private_model"],
           "private model digest differs")
    expect(failures, oracle.get("arithmetic_trace_file_sha256") == EXPECTED["extraction_trace"],
           "extraction trace file digest differs")
    expect(failures, oracle.get("arithmetic_trace_content_sha256") == EXPECTED["extraction_trace_content"],
           "extraction trace content digest differs")
    expect(failures, arithmetic.get("trace_rows") == 73520,
           "extraction arithmetic trace does not contain 73,520 rows")
    expect(failures, arithmetic.get("global_output_min") == -171319,
           "extraction global output minimum differs")
    expect(failures, arithmetic.get("global_output_max") == 158161,
           "extraction global output maximum differs")
    expect(failures, arithmetic.get("minimum_signed_bits") == 19,
           "extraction minimum signed width is not 19 bits")
    expect(failures, arithmetic.get("candidate_signed_widths", {}).get("16", {}).get(
        "overflow_evaluations") == 52044,
        "extraction signed-16 overflow count is not 52,044")

    check_file_hash(directory, "recovered.npz", "recovered.npz", failures)
    check_file_hash(directory, "recovered.json", "recovered.json", failures)
    check_file_hash(directory, "arithmetic.ndjson.gz", "extraction_trace", failures)
    return failures


def verify_completion(directory):
    failures = []
    result = load_result(directory, failures)
    if result is None:
        return failures
    attacker = result.get("attacker", {})
    oracle = result.get("oracle", {})
    evaluator = result.get("evaluator", {})
    arithmetic = oracle.get("arithmetic", {})
    recovery = attacker.get("recovery", {})
    closure = attacker.get("completed_clone_closure_audit", {})
    stats = attacker.get("oracle_stats_before_close", {})
    structural = evaluator.get("structural_certificate", {})

    expect(failures, result.get("success") is True, "completion success is not true")
    expect(failures, result.get("process_exit_codes") == {
        "attacker": 0, "evaluator": 0, "oracle": 0}, "completion process exit codes differ")
    expect(failures, result.get("seed_disclosed_to_attacker") is False,
           "completion oracle seed was disclosed to the attacker")
    expect(failures, attacker.get("success") is True, "completion attacker failed")
    expect(failures, attacker.get("forbidden_open_attempts") == 0,
           "completion attacker attempted to open private data")
    expect(failures, attacker.get("public_secret_nonzero_entries") == 0,
           "completion public model contains secret entries")
    expect(failures, empty_private_audit(attacker.get("attacker_view_audit", {})),
           "completion attacker view contains private state")
    expect(failures, attacker.get("source_arrays_sha256") == EXPECTED["recovered.npz"],
           "completion source-array digest differs")
    expect(failures, attacker.get("source_meta_sha256") == EXPECTED["recovered.json"],
           "completion source-metadata digest differs")
    expect(failures, attacker.get("completed_arrays_sha256") == EXPECTED["completed.npz"],
           "completion array digest differs")
    expect(failures, attacker.get("completed_meta_sha256") == EXPECTED["completed.json"],
           "completion metadata digest differs")
    expect(failures, recovery.get("line_candidate_count_histogram") == {"1": 830},
           "first-convolution singleton check count is not 830")
    expect(failures, recovery.get("shortcut_line_candidate_count_histogram") == {"1": 2048},
           "shortcut singleton check count is not 2,048")
    expect(failures, recovery.get("w1_changed_coefficients") == 3,
           "first-convolution changed coefficient count is not 3")
    expect(failures, recovery.get("shortcut_changed_coefficients") == 2,
           "shortcut changed coefficient count is not 2")
    expect(failures, closure.get("success") is True and closure.get("probes") == 4,
           "four-probe completion closure failed")
    expect(failures, closure.get("round1_equal") == closure.get("round2_equal") == 4,
           "completion closure does not agree at both rounds")
    expect(failures, stats.get("n_sessions") == 97,
           "completion session count is not 97")
    expect(failures, stats.get("n_round_evaluations") == 234,
           "completion affine-evaluation count is not 234")
    expect(failures, stats.get("inadmissible_entries") == 0,
           "completion contains inadmissible entries")
    expect(failures, stats.get("n_rounding_failures") == 0,
           "completion contains rounding failures")

    expect(failures, evaluator.get("success") is True, "completion evaluator failed")
    expect(failures, structural.get("success") is True,
           "completion structural certificate failed")
    expect(failures, structural.get("parameter_records_checked") == 29,
           "completion did not check 29 observable maps")
    expect(failures, structural.get("parameter_records_successful") == 29,
           "completion did not verify all 29 observable maps")
    expect(failures, evaluator.get("identical_logit_vectors") == 10000,
           "completion does not match all 10,000 logits")
    expect(failures, evaluator.get("post_calibration_identical_logit_vectors") == 9936,
           "completion does not match all 9,936 post-calibration logits")
    expect(failures, evaluator.get("max_abs_logit_difference") == 0,
           "completion has a nonzero logit difference")
    expect(failures, evaluator.get("completed_model_sha256") == EXPECTED["completed_model"],
           "completed model digest differs")
    expect(failures, evaluator.get("private_model_sha256") == EXPECTED["private_model"],
           "completion private model digest differs")
    expect(failures, oracle.get("arithmetic_trace_file_sha256") == EXPECTED["completion_trace"],
           "completion trace file digest differs")
    expect(failures, oracle.get("arithmetic_trace_content_sha256") == EXPECTED["completion_trace_content"],
           "completion trace content digest differs")
    expect(failures, arithmetic.get("trace_rows") == 234,
           "completion arithmetic trace does not contain 234 rows")
    expect(failures, arithmetic.get("global_output_min") == -190959,
           "completion global output minimum differs")
    expect(failures, arithmetic.get("global_output_max") == 175569,
           "completion global output maximum differs")
    expect(failures, arithmetic.get("minimum_signed_bits") == 19,
           "completion minimum signed width is not 19 bits")
    expect(failures, arithmetic.get("candidate_signed_widths", {}).get("16", {}).get(
        "overflow_evaluations") == 224,
        "completion signed-16 overflow count is not 224")

    check_file_hash(directory, "completed.npz", "completed.npz", failures)
    check_file_hash(directory, "completed.json", "completed.json", failures)
    check_file_hash(directory, "arithmetic.ndjson.gz", "completion_trace", failures)
    return failures


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--extraction-dir", type=Path)
    parser.add_argument("--completion-dir", type=Path)
    args = parser.parse_args()
    if args.extraction_dir is None and args.completion_dir is None:
        parser.error("provide --extraction-dir and/or --completion-dir")

    checks = {}
    failures = []
    if args.extraction_dir is not None:
        current = verify_extraction(args.extraction_dir)
        checks["extraction"] = {"directory": str(args.extraction_dir), "failures": current}
        failures.extend("extraction: " + item for item in current)
    if args.completion_dir is not None:
        current = verify_completion(args.completion_dir)
        checks["completion"] = {"directory": str(args.completion_dir), "failures": current}
        failures.extend("completion: " + item for item in current)
    print(json.dumps({"success": not failures, "checks": checks, "failures": failures},
                     indent=2, sort_keys=True))
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())

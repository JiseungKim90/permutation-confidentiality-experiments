#!/usr/bin/env python3
"""Verify the bundled full finite-enumeration result against its source and claims."""

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "scripts" / "identifiability_exhaustive.py"
REFERENCE = ROOT / "reference" / "finite_reference.json"


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def main():
    with REFERENCE.open("r", encoding="utf-8") as handle:
        report = json.load(handle)
    failures = []
    source_hash = sha256_file(SOURCE)
    if report.get("source", {}).get("sha256") != source_hash:
        failures.append("finite reference source digest does not match current source")
    if report.get("success") is not True:
        failures.append("finite reference success is not true")
    criteria = report.get("success_criteria", {})
    false_criteria = sorted(key for key, value in criteria.items() if value is not True)
    if false_criteria:
        failures.append("unsatisfied finite criteria: %r" % false_criteria)

    control = report.get("stabilizer_control", {})
    records = control.get("records", [])
    response_target_pairs = sum(item.get("response_target_pairs", 0) for item in records)
    controller_checks = sum(
        item.get("controllable_pairs", 0) * item.get("permutations_per_pair", 0)
        for item in records
    )
    expected = {
        "max_width": 6,
        "condition_mismatches": 0,
        "controller_failures": 0,
        "response_target_pairs": 5460,
        "controller_checks": 197956,
        "fibre_cases": 9,
        "two_round_cases": 3,
    }
    actual = {
        "max_width": control.get("max_width"),
        "condition_mismatches": control.get("total_condition_mismatches"),
        "controller_failures": control.get("total_controller_failures"),
        "response_target_pairs": response_target_pairs,
        "controller_checks": controller_checks,
        "fibre_cases": len(report.get("fibre_cases", [])),
        "two_round_cases": len(report.get("two_round_kernels", [])),
    }
    for key, value in expected.items():
        if actual.get(key) != value:
            failures.append("%s is %r, expected %r" % (key, actual.get(key), value))

    print(json.dumps({
        "success": not failures,
        "failures": failures,
        "source_sha256": source_hash,
        "counts": actual,
    }, indent=2, sort_keys=True))
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())

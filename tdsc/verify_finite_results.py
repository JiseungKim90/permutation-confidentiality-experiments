#!/usr/bin/env python3
"""Compare finite-model enumeration output with the published reference.

This checker reads JSON only. It does not query a model or run an extractor.
Runtime-dependent provenance fields are excluded from the numerical comparison.
"""

import argparse
import json
import sys
from pathlib import Path


FIELDS = (
    "schema",
    "success",
    "finite_domain_witness",
    "fibre_cases",
    "stabilizer_control",
    "two_round_kernels",
    "success_criteria",
)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def compare_reports(observed, reference):
    if not isinstance(observed, dict) or not isinstance(reference, dict):
        raise ValueError("Both reports must be JSON objects.")
    if reference.get("success") is not True:
        raise ValueError("The reference must record a successful complete run.")
    missing = [key for key in FIELDS if key not in observed or key not in reference]
    if missing:
        raise ValueError("Missing required fields: " + ", ".join(missing))
    mismatches = [key for key in FIELDS if canonical(observed[key]) != canonical(reference[key])]
    if observed.get("success") is not True and "success" not in mismatches:
        mismatches.append("success")
    return mismatches


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    args = parser.parse_args()
    try:
        observed = json.loads(args.report.read_text(encoding="utf-8"))
        reference = json.loads(args.reference.read_text(encoding="utf-8"))
        mismatches = compare_reports(observed, reference)
    except (OSError, ValueError, TypeError) as error:
        print(json.dumps({"success": False, "error": str(error)}))
        return 1
    print(json.dumps({"success": not mismatches, "mismatched_fields": mismatches}))
    return 1 if mismatches else 0


if __name__ == "__main__":
    sys.exit(main())


"""Independently verify ordinary-inference evidence, never recovery results.

The optional raw-array check recomputes every comparison from stored logits.
It deliberately cannot certify a trained-network extraction or an encrypted
backend: neither experiment is performed by the ordinary-forward runner.
"""

import argparse
import hashlib
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check_report(report, manifest, expected_names, expected_count=10000):
    require(report.get("schema") == "p050-normal-inference-validation-v1", "wrong experiment schema")
    require(report.get("status") == "complete" and report.get("success") is True, "incomplete or failed run")
    require(report.get("new_extraction_experiments") == 0, "ordinary inference cannot certify extraction")
    require(report.get("dataset_sha256") == "6d958be074577803d12ecdefd02955f39262c83c16fe9348329d7fe0b5c001ce", "dataset digest mismatch")
    launch = report["launch"]
    require(launch.get("dtype") == "float64", "unexpected numerical model")
    require(launch.get("n_test") == expected_count, "wrong planned sample count")
    schedule = launch.get("checkpoint_schedule", [])
    require(len(schedule) == len(set(schedule)), "duplicate planned checkpoint")
    require(set(schedule) == set(expected_names), "incomplete checkpoint schedule")
    records = report.get("records", [])
    names = [record.get("checkpoint") for record in records]
    require(len(names) == len(set(names)), "duplicate completed checkpoint")
    require(set(names) == set(schedule), "missing or unexpected completed checkpoint")
    known = {item["name"]: item for item in manifest["checkpoints"]}
    require(set(expected_names) <= set(known), "unknown checkpoint requested")
    for record in records:
        name = record["checkpoint"]
        require(record.get("success") is True, "failed checkpoint: " + name)
        require(record.get("n_test") == expected_count, "wrong completed sample count: " + name)
        require(record.get("checkpoint_sha256") == known[name]["sha256"], "checkpoint digest mismatch: " + name)
        loader = record["loader"]
        require(loader.get("checkpoint_sha256") == known[name]["sha256"], "loader digest mismatch: " + name)
        require(loader.get("missing_keys") == [] and loader.get("unexpected_keys") == [], "architecture mismatch: " + name)
        require(type(record.get("reference_correct")) is int and 0 <= record["reference_correct"] <= expected_count, "invalid accuracy numerator")
        fixed = record["comparisons"]["repaired"]
        require(set(record["comparisons"]) == {"repaired", "uniform_bias_control"}, "missing or unexpected comparison")
        error = fixed.get("maximum_logit_absolute_difference")
        require(type(error) in (int, float) and math.isfinite(error) and 0 <= error <= 1e-9, "invalid corrected logit error: " + name)
        require(fixed.get("logit_vectors_within_1e-9") == expected_count, "incomplete logit agreement: " + name)
        require(fixed.get("identical_predictions") == expected_count, "prediction mismatch: " + name)
        require(fixed.get("correct_predictions") == record["reference_correct"], "accuracy mismatch despite identical predictions")
        for comparison in record["comparisons"].values():
            value = comparison.get("maximum_logit_absolute_difference")
            require(type(value) in (int, float) and math.isfinite(value) and value >= 0, "invalid comparison error")
            for field in ("logit_vectors_within_1e-9", "identical_predictions", "correct_predictions"):
                count = comparison.get(field)
                require(type(count) is int and 0 <= count <= expected_count, "invalid comparison count: " + field)
    return records


def check_raw_arrays(records, directory):
    import numpy as np

    for record in records:
        path = directory / (record["checkpoint"] + "-ordinary-logits.npz")
        require(digest(path) == record["raw_array_sha256"], "raw-array digest mismatch")
        with np.load(path, allow_pickle=False) as arrays:
            count = record["n_test"]
            require(set(arrays.files) == {"labels", "reference", "repaired", "uniform_bias_control"}, "unexpected raw-array fields")
            labels = arrays["labels"]
            reference = arrays["reference"]
            require(labels.shape == (count,) and np.issubdtype(labels.dtype, np.integer), "invalid labels")
            require(np.all((0 <= labels) & (labels < 10)), "labels outside class range")
            require(reference.shape == (count, 10) and reference.dtype == np.float64 and np.isfinite(reference).all(), "invalid reference logits")
            reference_prediction = reference.argmax(axis=1)
            require(int((reference_prediction == labels).sum()) == record["reference_correct"], "reference accuracy differs from raw arrays")
            for name in ("repaired", "uniform_bias_control"):
                value = arrays[name]
                require(value.shape == reference.shape and value.dtype == np.float64 and np.isfinite(value).all(), "invalid comparison logits")
                difference = np.abs(value - reference)
                recomputed = {
                    "maximum_logit_absolute_difference": float(difference.max()),
                    "logit_vectors_within_1e-9": int((difference.max(axis=1) <= 1e-9).sum()),
                    "identical_predictions": int((value.argmax(axis=1) == reference_prediction).sum()),
                    "correct_predictions": int((value.argmax(axis=1) == labels).sum()),
                }
                require(recomputed == record["comparisons"][name], "summary differs from raw arrays: " + name)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--checkpoints", nargs="+", required=True)
    parser.add_argument("--raw-dir", type=Path)
    parser.add_argument("--check-source", action="store_true")
    args = parser.parse_args()
    report = json.loads(args.report.read_text(encoding="utf-8"))
    manifest = json.loads((ROOT / "reference/multicheckpoint/checkpoints.json").read_text(encoding="utf-8"))
    records = check_report(report, manifest, args.checkpoints)
    sources_checked = 0
    if args.check_source:
        for relative, expected in report["launch"]["source_sha256"].items():
            path = (ROOT / relative).resolve()
            require(ROOT in path.parents, "source path escapes the code root")
            require(digest(path) == expected, "source changed since the recorded run: " + relative)
            sources_checked += 1
        require(sources_checked > 0, "source manifest is empty")
    if args.raw_dir:
        check_raw_arrays(records, args.raw_dir)
    print(json.dumps({
        "success": True, "checkpoints": len(records),
        "ordinary_logit_vectors_checked": sum(record["n_test"] for record in records),
        "raw_arrays_recomputed": args.raw_dir is not None,
        "source_files_checked": sources_checked,
        "report_sha256": digest(args.report),
        "supports": "ordinary float64 inference after normalization",
        "certifies_trained_network_extraction": False,
        "certifies_native_qat": False,
        "certifies_encrypted_backend": False,
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

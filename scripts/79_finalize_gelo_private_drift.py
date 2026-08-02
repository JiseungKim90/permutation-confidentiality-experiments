#!/usr/bin/env python3
"""Validate and summarize the private-prefix drift and 1%-FWER campaign."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import subprocess
import tempfile
from pathlib import Path
from statistics import mean, median
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


EXPECTED = {"calibration": 100, "closed": 100, "partial": 50, "open": 300}
BANK_SIZE = 100_000


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=path.name + ".", dir=str(path.parent))
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def atomic_json(path: Path, value: Dict) -> None:
    atomic_text(path, json.dumps(value, indent=2, sort_keys=True) + "\n")


def read_jsonl(path: Path) -> List[Dict]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise RuntimeError("invalid JSON at {}:{}".format(path, line_number)) from error
    return rows


def metric(row: Dict, field: str = "metrics") -> Optional[Dict]:
    values = row.get(field)
    if not values:
        return None
    matches = [value for value in values if int(value["bank_size"]) == BANK_SIZE]
    if len(matches) != 1:
        raise RuntimeError("expected exactly one 100K metric")
    return matches[0]


def percentile(values: Sequence[float], fraction: float) -> Optional[float]:
    if not values:
        return None
    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def numeric(values: Iterable[Optional[float]]) -> Dict[str, Optional[float]]:
    present = [float(value) for value in values if value is not None]
    if not present:
        return {"mean": None, "median": None, "p05": None, "p95": None, "min": None, "max": None}
    return {
        "mean": mean(present),
        "median": median(present),
        "p05": percentile(present, 0.05),
        "p95": percentile(present, 0.95),
        "min": min(present),
        "max": max(present),
    }


def wilson(successes: int, trials: int, z: float = 1.959963984540054) -> Tuple[float, float]:
    if trials <= 0:
        return (0.0, 1.0)
    proportion = successes / trials
    denominator = 1.0 + z * z / trials
    center = (proportion + z * z / (2.0 * trials)) / denominator
    radius = (
        z
        * math.sqrt(proportion * (1.0 - proportion) / trials + z * z / (4.0 * trials * trials))
        / denominator
    )
    lower = 0.0 if successes == 0 else max(0.0, center - radius)
    upper = 1.0 if successes == trials else min(1.0, center + radius)
    return (lower, upper)


def validate(rows: List[Dict]) -> Dict:
    counts = {split: 0 for split in EXPECTED}
    keys = set()
    duplicates = []
    unexpected = []
    for row in rows:
        key = (
            row["source_model"], int(row["layer"]), row["condition"],
            row["split"], int(row["trial"]),
        )
        if key in keys:
            duplicates.append(list(key))
        keys.add(key)
        if key[0] != "private" or key[1] != 8 or key[2] != "ideal" or key[3] not in EXPECTED:
            unexpected.append(list(key))
        else:
            counts[key[3]] += 1
    complete = not duplicates and not unexpected and counts == EXPECTED
    return {
        "complete": complete,
        "counts": counts,
        "expected": EXPECTED,
        "duplicates": duplicates,
        "unexpected": unexpected,
        "records": len(rows),
    }


def summarize_split(rows: List[Dict], split: str) -> Dict:
    selected = [row for row in rows if row["split"] == split]
    metrics = [metric(row) for row in selected]
    present = [value for value in metrics if value is not None]
    false_positive_trials = sum(bool(value["trial_false_positive"]) for value in present)
    interval = wilson(false_positive_trials, len(present))
    return {
        "split": split,
        "trials": len(selected),
        "mean_recall_at_source_count": numeric(value.get("recall_at_source_count") for value in present),
        "mean_average_precision": numeric(value.get("average_precision") for value in present),
        "exact_source_set_rate": numeric(
            1.0 if value.get("recall_at_source_count") == 1.0 else 0.0 for value in present
        ),
        "threshold_recall": numeric(value.get("threshold_recall") for value in present),
        "threshold_precision": numeric(value.get("threshold_precision") for value in present),
        "candidate_false_positive_rate": numeric(
            value.get("candidate_false_positive_rate") for value in present
        ),
        "trial_false_positive_count": false_positive_trials,
        "trial_false_positive_rate": false_positive_trials / max(1, len(present)),
        "trial_false_positive_wilson95": list(interval),
        "accepted_count": numeric(value.get("accepted_count") for value in present),
        "scoring_seconds": numeric(row.get("scoring_seconds") for row in selected),
        "peak_rss_bytes": numeric(row.get("peak_rss_bytes") for row in selected),
    }


def flatten(prefix: str, value: Dict, output: Dict) -> None:
    for key, child in value.items():
        name = "{}_{}".format(prefix, key) if prefix else key
        if isinstance(child, dict):
            flatten(name, child, output)
        elif isinstance(child, list):
            output[name] = json.dumps(child, separators=(",", ":"))
        else:
            output[name] = child


def write_csv(path: Path, rows: List[Dict]) -> None:
    flattened = []
    fields = set()
    for row in rows:
        current: Dict = {}
        flatten("", row, current)
        flattened.append(current)
        fields.update(current)
    ordered = sorted(fields)
    descriptor, temporary = tempfile.mkstemp(prefix=path.name + ".", dir=str(path.parent))
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=ordered)
            writer.writeheader()
            writer.writerows(flattened)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def git_revision(repo: Path) -> Optional[str]:
    try:
        return subprocess.check_output(
            ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", required=True)
    parser.add_argument("--drift-root", required=True)
    parser.add_argument("--checkpoints", type=int, nargs="+", required=True)
    args = parser.parse_args()

    root = Path(args.drift_root).resolve()
    final = root / "final"
    final.mkdir(parents=True, exist_ok=True)
    validation = {}
    results = []
    input_hashes = {}
    for step in args.checkpoints:
        tag = "step{:04d}".format(step)
        raw_path = root / "evaluation" / tag / "trial_records.jsonl"
        summary_path = root / "evaluation" / tag / "summary.json"
        checkpoint_path = root / "checkpoints" / tag / "checkpoint_manifest.json"
        for path in [raw_path, summary_path, checkpoint_path]:
            if not path.exists():
                raise RuntimeError("missing required result {}".format(path))
            input_hashes[str(path.relative_to(root))] = sha256(path)
        rows = read_jsonl(raw_path)
        validation[tag] = validate(rows)
        checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        results.append(
            {
                "step": step,
                "parameter_drift": checkpoint["parameter_drift"],
                "representation_drift": checkpoint["representation_drift"],
                "utility": checkpoint["utility"],
                "splits": {
                    split: summarize_split(rows, split)
                    for split in ["closed", "partial", "open"]
                },
            }
        )

    if not all(value["complete"] for value in validation.values()):
        raise RuntimeError("drift campaign validation failed: " + json.dumps(validation, sort_keys=True))

    base_perplexity = results[0]["utility"]["evaluation_open"]["perplexity"]
    boundary = []
    first_below_95 = None
    for result in results:
        closed = result["splits"]["closed"]
        recall = closed["mean_recall_at_source_count"]["mean"]
        exact = closed["exact_source_set_rate"]["mean"]
        perplexity = result["utility"]["evaluation_open"]["perplexity"]
        retained = bool(recall is not None and recall >= 0.95 and exact is not None and exact >= 0.90)
        if not retained and first_below_95 is None:
            first_below_95 = result["step"]
        boundary.append(
            {
                "step": result["step"],
                "relative_parameter_l2_drift": result["parameter_drift"]["relative_parameter_l2_drift"],
                "relative_hidden_l2_drift": result["representation_drift"]["relative_hidden_l2_drift"],
                "mean_row_cosine": result["representation_drift"]["mean_row_cosine"],
                "evaluation_open_perplexity": perplexity,
                "perplexity_ratio_to_public": perplexity / base_perplexity,
                "closed_mean_recall": recall,
                "closed_exact_set_rate": exact,
                "open_trial_false_positive_rate": result["splits"]["open"]["trial_false_positive_rate"],
                "open_trial_false_positive_wilson95": result["splits"]["open"]["trial_false_positive_wilson95"],
                "attack_retained_at_prespecified_level": retained,
            }
        )

    results_path = final / "private_drift_results.json"
    boundary_path = final / "private_drift_boundary.csv"
    atomic_json(results_path, {"bank_size": BANK_SIZE, "checkpoints": results})
    write_csv(boundary_path, boundary)
    completion = {
        "status": "complete",
        "bank_size": BANK_SIZE,
        "calibration_trials": EXPECTED["calibration"],
        "nominal_trial_false_positive_alpha": 0.01,
        "open_trials": EXPECTED["open"],
        "paired_checkpoints": args.checkpoints,
        "first_checkpoint_below_prespecified_attack_level": first_below_95,
        "validation": validation,
        "git_revision": git_revision(Path(args.repo_root).resolve()),
        "input_sha256": input_hashes,
        "artifact_sha256": {
            results_path.name: sha256(results_path),
            boundary_path.name: sha256(boundary_path),
        },
        "interpretation_guardrail": (
            "The sweep quantifies public-bank mismatch for one fine-tuning trajectory; "
            "it is not a universal result for arbitrary proprietary prefixes."
        ),
    }
    atomic_json(final / "campaign_complete.json", completion)
    print(json.dumps(completion, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

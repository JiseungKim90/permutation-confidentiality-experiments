#!/usr/bin/env python3
"""Validate and summarize the completed 100K GELO MS MARCO campaigns."""

import argparse
import csv
import hashlib
import json
import math
import os
import subprocess
import tempfile
from collections import defaultdict
from pathlib import Path
from statistics import mean, median
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


CAMPAIGNS = {
    "core": {
        "directory": "evaluation_core_torch",
        "layers": [4, 8, 12],
        "conditions": ["ideal", "gelo_nonorth", "manifold_stress"],
    },
    "robustness": {
        "directory": "evaluation_robustness_torch",
        "layers": [8],
        "conditions": [
            "gelo_gaussian",
            "quantized_gaussian",
            "quantized_nonorth",
            "manifold",
        ],
    },
}
SOURCE_MODELS = ["public", "private"]
EXPECTED_SPLITS = {"calibration": 20, "closed": 50, "partial": 50, "open": 50}
PAPER_BANK_SIZE = 100_000


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


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
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as error:
                    raise RuntimeError(f"invalid JSON at {path}:{line_number}: {error}") from error
    return rows


def metric_at_100k(row: Dict, field: str = "metrics") -> Optional[Dict]:
    values = row.get(field)
    if not values:
        return None
    matches = [value for value in values if int(value["bank_size"]) == PAPER_BANK_SIZE]
    if len(matches) != 1:
        raise RuntimeError(f"expected one {PAPER_BANK_SIZE} metric, found {len(matches)}")
    return matches[0]


def numeric_summary(values: Iterable[Optional[float]]) -> Dict[str, Optional[float]]:
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


def git_revision(repo: Path) -> Optional[str]:
    try:
        return subprocess.check_output(
            ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def validate_campaign(name: str, rows: List[Dict]) -> Dict:
    spec = CAMPAIGNS[name]
    observed: Dict[Tuple, int] = defaultdict(int)
    keys = set()
    duplicates = []
    for row in rows:
        key = (
            row["source_model"], int(row["layer"]), row["condition"],
            row["split"], int(row["trial"]),
        )
        if key in keys:
            duplicates.append(key)
        keys.add(key)
        observed[key[:4]] += 1

    missing_or_extra = []
    for source_model in SOURCE_MODELS:
        for layer in spec["layers"]:
            for condition in spec["conditions"]:
                for split, expected in EXPECTED_SPLITS.items():
                    key = (source_model, layer, condition, split)
                    actual = observed.get(key, 0)
                    if actual != expected:
                        missing_or_extra.append({
                            "source_model": source_model,
                            "layer": layer,
                            "condition": condition,
                            "split": split,
                            "expected": expected,
                            "actual": actual,
                        })

    expected_total = (
        len(SOURCE_MODELS) * len(spec["layers"]) * len(spec["conditions"])
        * sum(EXPECTED_SPLITS.values())
    )
    unexpected = [
        key for key in observed
        if key[0] not in SOURCE_MODELS
        or key[1] not in spec["layers"]
        or key[2] not in spec["conditions"]
        or key[3] not in EXPECTED_SPLITS
    ]
    complete = not duplicates and not missing_or_extra and not unexpected and len(rows) == expected_total
    return {
        "complete": complete,
        "expected_records": expected_total,
        "actual_records": len(rows),
        "duplicate_keys": [list(key) for key in duplicates],
        "missing_or_extra_groups": missing_or_extra,
        "unexpected_groups": [list(key) for key in unexpected],
    }


def aggregate_performance(campaign: str, rows: List[Dict]) -> List[Dict]:
    groups: Dict[Tuple, List[Dict]] = defaultdict(list)
    control_groups: Dict[Tuple, List[Dict]] = defaultdict(list)
    for row in rows:
        key = (row["source_model"], int(row["layer"]), row["condition"], row["split"])
        groups[key].append(row)
        if row.get("control_metrics"):
            control_groups[(row["source_model"], int(row["layer"]), row["condition"], "closed_random_control")].append(row)

    result = []
    metric_fields = [
        "recall_at_source_count", "recall_at_known_count", "average_precision",
        "threshold_recall", "threshold_precision", "candidate_false_positive_rate",
        "trial_false_positive", "accepted_count",
    ]
    for key, values in sorted(groups.items()):
        record = {
            "campaign": campaign,
            "source_model": key[0],
            "layer": key[1],
            "condition": key[2],
            "split": key[3],
            "trials": len(values),
            "scoring_seconds": numeric_summary(value.get("scoring_seconds") for value in values),
            "peak_rss_bytes": numeric_summary(value.get("peak_rss_bytes") for value in values),
        }
        metrics = [metric_at_100k(value) for value in values]
        present_metrics = [value for value in metrics if value is not None]
        for field in metric_fields:
            record[field] = numeric_summary(value.get(field) for value in present_metrics)
        result.append(record)

    for key, values in sorted(control_groups.items()):
        record = {
            "campaign": campaign,
            "source_model": key[0],
            "layer": key[1],
            "condition": key[2],
            "split": key[3],
            "trials": len(values),
            "scoring_seconds": numeric_summary([]),
            "peak_rss_bytes": numeric_summary([]),
        }
        metrics = [metric_at_100k(value, "control_metrics") for value in values]
        for field in metric_fields:
            record[field] = numeric_summary(value.get(field) for value in metrics if value is not None)
        result.append(record)
    return result


def aggregate_rank_pr(campaign: str, rows: List[Dict]) -> List[Dict]:
    points: Dict[Tuple, List[float]] = defaultdict(list)
    for row in rows:
        metric = metric_at_100k(row)
        if not metric or not metric.get("known_ranks"):
            continue
        ranks = sorted(int(rank) for rank in metric["known_ranks"].values())
        count = len(ranks)
        for recovered, rank in enumerate(ranks, start=1):
            recall = recovered / count
            precision = recovered / rank
            key = (
                row["source_model"], int(row["layer"]), row["condition"],
                row["split"], recall,
            )
            points[key].append(precision)
    result = []
    for key, values in sorted(points.items()):
        result.append({
            "campaign": campaign,
            "source_model": key[0],
            "layer": key[1],
            "condition": key[2],
            "split": key[3],
            "recall": key[4],
            "trials": len(values),
            "precision": numeric_summary(values),
        })
    return result


def flatten(prefix: str, value: Dict, output: Dict) -> None:
    for key, child in value.items():
        name = f"{prefix}_{key}" if prefix else key
        if isinstance(child, dict):
            flatten(name, child, output)
        else:
            output[name] = child


def write_csv(path: Path, rows: List[Dict]) -> None:
    flattened = []
    fieldnames = set()
    for row in rows:
        current: Dict = {}
        flatten("", row, current)
        flattened.append(current)
        fieldnames.update(current)
    ordered = sorted(fieldnames)
    path.parent.mkdir(parents=True, exist_ok=True)
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--repo-root", required=True)
    args = parser.parse_args()

    run_root = Path(args.run_root).resolve()
    repo_root = Path(args.repo_root).resolve()
    final_dir = run_root / "final"
    rows_by_campaign = {}
    validation = {}
    input_hashes = {}
    performance = []
    rank_pr = []

    for campaign, spec in CAMPAIGNS.items():
        raw_path = run_root / spec["directory"] / "trial_records.jsonl"
        summary_path = run_root / spec["directory"] / "summary.json"
        if not raw_path.exists() or not summary_path.exists():
            raise RuntimeError(f"missing completed output for {campaign}: {raw_path}")
        rows = read_jsonl(raw_path)
        rows_by_campaign[campaign] = rows
        validation[campaign] = validate_campaign(campaign, rows)
        input_hashes[str(raw_path.relative_to(run_root))] = sha256(raw_path)
        input_hashes[str(summary_path.relative_to(run_root))] = sha256(summary_path)
        performance.extend(aggregate_performance(campaign, rows))
        rank_pr.extend(aggregate_rank_pr(campaign, rows))

    if not all(value["complete"] for value in validation.values()):
        raise RuntimeError("campaign completeness validation failed: " + json.dumps(validation, sort_keys=True))

    provenance_files = [
        run_root / "data" / "manifest.json",
        run_root / "data" / "trials.json",
        run_root / "cache" / "public_manifest.json",
        run_root / "cache" / "private_full_manifest.json",
        run_root / "private_prefix" / "private_prefix_manifest.json",
    ]
    for path in provenance_files:
        if not path.exists():
            raise RuntimeError(f"missing provenance file: {path}")
        input_hashes[str(path.relative_to(run_root))] = sha256(path)

    performance_path = final_dir / "performance_100k.json"
    rank_pr_path = final_dir / "rank_pr_100k.json"
    atomic_json(performance_path, {"bank_size": PAPER_BANK_SIZE, "rows": performance})
    atomic_json(rank_pr_path, {"bank_size": PAPER_BANK_SIZE, "rows": rank_pr})
    write_csv(final_dir / "performance_100k.csv", performance)
    write_csv(final_dir / "rank_pr_100k.csv", rank_pr)

    artifact_hashes = {
        path.name: sha256(path)
        for path in [
            performance_path,
            rank_pr_path,
            final_dir / "performance_100k.csv",
            final_dir / "rank_pr_100k.csv",
        ]
    }
    completion = {
        "status": "complete",
        "bank_size": PAPER_BANK_SIZE,
        "git_revision": git_revision(repo_root),
        "validation": validation,
        "input_sha256": input_hashes,
        "artifact_sha256": artifact_hashes,
        "notes": {
            "rank_pr": "Exact per-trial precision at every positive-rank recall event, aggregated across trials.",
            "open_set": "Open-set trials have no positive class; report calibrated candidate- and trial-level false-positive rates instead of PR.",
            "runtime": "scoring_seconds covers one maximum-bank attack scan; random-control timing was not persisted by the evaluator.",
        },
    }
    atomic_json(final_dir / "campaign_complete.json", completion)
    print(json.dumps(completion, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Validate and summarize the 100K full-bank public-prefix campaign."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


CONDITIONS = ["ideal", "gelo_nonorth", "manifold_stress"]
EXPECTED = {"calibration": 100, "closed": 100, "partial": 50, "open": 300}
BANK_SIZE = 100_000


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=path.name + ".", dir=str(path.parent))
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


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


def numeric(values: Iterable[Optional[float]]) -> Dict[str, Optional[float]]:
    present = [float(value) for value in values if value is not None]
    if not present:
        return {"mean": None, "min": None, "max": None}
    return {"mean": sum(present) / len(present), "min": min(present), "max": max(present)}


def wilson(successes: int, trials: int, z: float = 1.959963984540054) -> Tuple[float, float]:
    if trials <= 0:
        return (0.0, 1.0)
    proportion = successes / trials
    denominator = 1.0 + z * z / trials
    center = (proportion + z * z / (2.0 * trials)) / denominator
    radius = z * math.sqrt(
        proportion * (1.0 - proportion) / trials + z * z / (4.0 * trials * trials)
    ) / denominator
    return (max(0.0, center - radius), min(1.0, center + radius))


def summarize(rows: List[Dict], split: str, control: bool = False) -> Dict:
    selected = [row for row in rows if row["split"] == split]
    field = "control_metrics" if control else "metrics"
    metrics = [metric(row, field) for row in selected]
    present = [value for value in metrics if value is not None]
    false_positive_trials = sum(bool(value.get("trial_false_positive")) for value in present)
    result = {
        "split": split + ("_random_control" if control else ""),
        "trials": len(present),
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
        "trial_false_positive_wilson95": list(wilson(false_positive_trials, len(present))),
    }
    if not control:
        result["scoring_seconds"] = numeric(row.get("scoring_seconds") for row in selected)
        result["peak_rss_bytes"] = numeric(row.get("peak_rss_bytes") for row in selected)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", required=True)
    parser.add_argument("--campaign-root", required=True)
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    root = Path(args.campaign_root).resolve()
    raw_path = root / "evaluation" / "trial_records.jsonl"
    summary_path = root / "evaluation" / "summary.json"
    manifest_path = root / "data" / "manifest.json"
    for path in [raw_path, summary_path, manifest_path]:
        if not path.exists():
            raise RuntimeError("missing required result {}".format(path))

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("candidate_source_pool") != [0, 100_000]:
        raise RuntimeError("positive-placement manifest does not cover the full bank")

    rows = read_jsonl(raw_path)
    counts = defaultdict(int)
    keys = set()
    duplicates = []
    unexpected = []
    by_condition = defaultdict(list)
    for row in rows:
        key = (
            row["source_model"], int(row["layer"]), row["condition"],
            row["split"], int(row["trial"]),
        )
        if key in keys:
            duplicates.append(list(key))
        keys.add(key)
        if key[0] != "public" or key[1] != 8 or key[2] not in CONDITIONS or key[3] not in EXPECTED:
            unexpected.append(list(key))
        else:
            counts[(key[2], key[3])] += 1
            by_condition[key[2]].append(row)

    count_errors = []
    for condition in CONDITIONS:
        for split, expected in EXPECTED.items():
            actual = counts[(condition, split)]
            if actual != expected:
                count_errors.append(
                    {"condition": condition, "split": split, "expected": expected, "actual": actual}
                )
    if duplicates or unexpected or count_errors:
        raise RuntimeError(
            "campaign completeness validation failed: "
            + json.dumps(
                {"duplicates": duplicates, "unexpected": unexpected, "count_errors": count_errors},
                sort_keys=True,
            )
        )

    results = []
    gates = []
    for condition in CONDITIONS:
        condition_rows = by_condition[condition]
        splits = {
            split: summarize(condition_rows, split)
            for split in ["closed", "partial", "open"]
        }
        control = summarize(condition_rows, "closed", control=True)
        closed_recall = splits["closed"]["mean_recall_at_source_count"]["mean"]
        closed_ap = splits["closed"]["mean_average_precision"]["mean"]
        partial_recall = splits["partial"]["mean_recall_at_source_count"]["mean"]
        partial_ap = splits["partial"]["mean_average_precision"]["mean"]
        open_upper = splits["open"]["trial_false_positive_wilson95"][1]
        control_recall = control["mean_recall_at_source_count"]["mean"]
        gate = {
            "condition": condition,
            "closed_recall_at_least_0_95": closed_recall is not None and closed_recall >= 0.95,
            "closed_ap_at_least_0_95": closed_ap is not None and closed_ap >= 0.95,
            "partial_recall_at_least_0_90": partial_recall is not None and partial_recall >= 0.90,
            "partial_ap_at_least_0_90": partial_ap is not None and partial_ap >= 0.90,
            "open_trial_fpr_wilson_upper_at_most_0_05": open_upper <= 0.05,
            "random_control_recall_at_most_0_05": control_recall is not None and control_recall <= 0.05,
        }
        gate["pass"] = all(value for key, value in gate.items() if key != "condition")
        gates.append(gate)
        results.append({"condition": condition, "splits": splits, "closed_random_control": control})

    final = root / "final"
    results_path = final / "fullbank_results.json"
    atomic_json(results_path, {"bank_size": BANK_SIZE, "conditions": results})
    completion = {
        "status": "complete",
        "bank_size": BANK_SIZE,
        "positive_placement": "uniform over candidate indices [0,100000)",
        "calibration_trials": EXPECTED["calibration"],
        "nominal_trial_false_positive_alpha": 0.01,
        "open_trials": EXPECTED["open"],
        "validation": {
            "records": len(rows),
            "expected_records": len(CONDITIONS) * sum(EXPECTED.values()),
            "conditions": CONDITIONS,
            "counts": {"{}:{}".format(*key): value for key, value in sorted(counts.items())},
        },
        "prespecified_gates": gates,
        "all_prespecified_gates_pass": all(gate["pass"] for gate in gates),
        "git_revision": subprocess.check_output(
            ["git", "-C", str(repo_root), "rev-parse", "HEAD"], text=True
        ).strip(),
        "input_sha256": {
            str(path.relative_to(root)): sha256(path)
            for path in [raw_path, summary_path, manifest_path]
        },
        "artifact_sha256": {results_path.name: sha256(results_path)},
    }
    atomic_json(final / "campaign_complete.json", completion)
    print(json.dumps(completion, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

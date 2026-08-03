#!/usr/bin/env python3
"""Validate counts, gates, controls, bindings, and resource evidence for KV-Cloak."""

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
from typing import Dict, Iterable, List, Tuple


PINNED_COMMIT = "6b40f36edb2f337557543e7e60b10022308883d4"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
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


def read_json(path: Path) -> Dict:
    return json.loads(path.read_text(encoding="utf-8"))


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


def mean(values: Iterable[float]) -> float:
    present = [float(value) for value in values]
    return sum(present) / max(1, len(present))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", required=True)
    parser.add_argument("--campaign-root", required=True)
    parser.add_argument("--smoke", required=True)
    parser.add_argument("--trials", required=True)
    parser.add_argument("--main-evaluation", required=True)
    parser.add_argument("--additional-evaluations", nargs="*", default=[])
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    campaign_root = Path(args.campaign_root).resolve()
    smoke_path = Path(args.smoke).resolve()
    trials_path = Path(args.trials).resolve()
    main_root = Path(args.main_evaluation).resolve()
    raw_path = main_root / "trial_records.jsonl"
    summary_path = main_root / "summary.json"
    codebook_path = main_root / "codebook" / "manifest.json"
    required = [smoke_path, trials_path, raw_path, summary_path, codebook_path]
    for path in required:
        if not path.exists():
            raise RuntimeError("missing required campaign artifact {}".format(path))

    smoke = read_json(smoke_path)
    trials = read_json(trials_path)
    summary = read_json(summary_path)
    codebook = read_json(codebook_path)
    rows = read_jsonl(raw_path)
    metadata = summary["metadata"]
    if smoke.get("official_commit") != PINNED_COMMIT or metadata.get("official_commit") != PINNED_COMMIT:
        raise RuntimeError("official implementation binding mismatch")
    if not smoke.get("all_gates_pass"):
        raise RuntimeError("official-code invariant smoke did not pass")
    if metadata.get("conditions") != ["official_reuse"]:
        raise RuntimeError("headline evaluation must contain only official_reuse")
    if sorted(metadata.get("kv_types", [])) != ["key", "value"]:
        raise RuntimeError("headline evaluation must contain K and V")
    if metadata.get("precision") != "bfloat16" or int(metadata.get("block_size")) != 16:
        raise RuntimeError("headline evaluation must be bfloat16 with b=16")

    bank_sizes = [int(value) for value in trials["bank_sizes"]]
    expected_calibration = int(trials["calibration_trials"])
    expected_closed = int(trials["closed_trials_per_bank"])
    expected_open = int(trials["open_trials"])
    expected_total = 2 * (expected_calibration + len(bank_sizes) * expected_closed + expected_open)
    counts = defaultdict(int)
    keys = set()
    duplicates = []
    for row in rows:
        bank_size = int(row.get("bank_size", 0))
        key = (row["kv_type"], row["condition"], row["split"], bank_size, int(row["trial"]))
        if key in keys:
            duplicates.append(list(key))
        keys.add(key)
        counts[(row["kv_type"], row["split"], bank_size)] += 1
    count_errors = []
    for kv_type in ["key", "value"]:
        for split, bank_size, expected in [
            ("calibration", 0, expected_calibration),
            ("open", 0, expected_open),
        ]:
            actual = counts[(kv_type, split, bank_size)]
            if actual != expected:
                count_errors.append({"kv_type": kv_type, "split": split, "bank_size": bank_size, "expected": expected, "actual": actual})
        for bank_size in bank_sizes:
            actual = counts[(kv_type, "closed", bank_size)]
            if actual != expected_closed:
                count_errors.append({"kv_type": kv_type, "split": "closed", "bank_size": bank_size, "expected": expected_closed, "actual": actual})
    if duplicates or count_errors or len(rows) != expected_total:
        raise RuntimeError("headline completeness failure: " + json.dumps({"duplicates": duplicates, "count_errors": count_errors, "records": len(rows), "expected": expected_total}, sort_keys=True))

    results = []
    gates = []
    for kv_type in ["key", "value"]:
        for bank_size in bank_sizes:
            closed = [row for row in rows if row["kv_type"] == kv_type and row["split"] == "closed" and int(row["bank_size"]) == bank_size]
            open_metrics = [
                metric
                for row in rows if row["kv_type"] == kv_type and row["split"] == "open"
                for metric in row["metrics"] if int(metric["bank_size"]) == bank_size
            ]
            top1 = mean(row["top1_hit"] for row in closed)
            threshold_recall = mean(row["accepted_true"] for row in closed)
            random_recall = mean(row["random_label_hit"] for row in closed)
            false_trials = sum(bool(metric["trial_false_positive"]) for metric in open_metrics)
            interval = wilson(false_trials, len(open_metrics))
            result = {
                "kv_type": kv_type, "bank_size": bank_size,
                "closed_trials": len(closed), "top1_recall": top1,
                "threshold_recall": threshold_recall,
                "random_label_top1_recall": random_recall,
                "open_trials": len(open_metrics),
                "open_trial_false_positive_count": false_trials,
                "open_trial_false_positive_rate": false_trials / max(1, len(open_metrics)),
                "open_trial_false_positive_wilson95": list(interval),
                "search_seconds": {
                    "mean": mean(row["search_seconds"] for row in closed),
                    "max": max(float(row["search_seconds"]) for row in closed),
                },
                "peak_rss_bytes": max(int(row["peak_rss_bytes"]) for row in closed),
            }
            gate = {
                "kv_type": kv_type, "bank_size": bank_size,
                "closed_top1_recall_at_least_0_95": top1 >= 0.95,
                "threshold_recall_at_least_0_90": threshold_recall >= 0.90,
                "random_label_recall_at_most_0_05": random_recall <= 0.05,
                "open_trial_fpr_wilson_upper_at_most_0_05": interval[1] <= 0.05,
            }
            gate["pass"] = all(value for key, value in gate.items() if key not in ["kv_type", "bank_size"])
            results.append(result)
            gates.append(gate)

    placement_ok = all(
        int(values["occupied_deciles"]) == 10
        for values in trials["candidate_placement"].values()
    )
    additional = []
    additional_paths = []
    for value in args.additional_evaluations:
        root = Path(value).resolve()
        path = root / "summary.json"
        raw = root / "trial_records.jsonl"
        if not path.exists() or not raw.exists():
            raise RuntimeError("missing additional evaluation {}".format(root))
        payload = read_json(path)
        additional.append({
            "path": str(root), "metadata": payload["metadata"],
            "summary": payload["summary"], "raw_records": payload["raw_records"],
        })
        additional_paths.extend([path, raw, root / "codebook" / "manifest.json"])

    final_dir = campaign_root / "final"
    results_path = final_dir / "kvcloak_results.json"
    atomic_json(results_path, {"headline": results, "additional": additional})
    input_paths = required + additional_paths
    completion = {
        "status": "complete",
        "decision_surface": "STIP + KV-Cloak + GELO independent NDSS evidence",
        "headline": {
            "dataset": "MS MARCO Passage Ranking collection",
            "model": metadata.get("cache_manifest"),
            "block_size": 16, "precision": "bfloat16",
            "candidate_banks": bank_sizes,
            "positive_placement_covers_all_deciles": placement_ok,
            "records": len(rows), "expected_records": expected_total,
            "results": results,
        },
        "official_code_smoke_all_gates_pass": smoke["all_gates_pass"],
        "prespecified_gates": gates,
        "all_prespecified_gates_pass": placement_ok and all(gate["pass"] for gate in gates),
        "official_repository": "https://github.com/SiO-2/kvcloak",
        "official_commit": PINNED_COMMIT,
        "repo_revision": subprocess.check_output(["git", "-C", str(repo_root), "rev-parse", "HEAD"], text=True).strip(),
        "codebook": codebook,
        "additional_evaluation_count": len(additional),
        "input_sha256": {str(path): sha256(path) for path in input_paths},
        "artifact_sha256": {results_path.name: sha256(results_path)},
    }
    atomic_json(final_dir / "campaign_complete.json", completion)
    print(json.dumps(completion, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

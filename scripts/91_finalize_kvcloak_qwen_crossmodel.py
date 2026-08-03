#!/usr/bin/env python3
"""Finalize the prespecified Qwen2 KV-Cloak dimension-boundary campaign."""

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


def validate_evaluation(root: Path, trials_path: Path, expected_block: int) -> Dict:
    summary_path = root / "summary.json"
    raw_path = root / "trial_records.jsonl"
    codebook_path = root / "codebook" / "manifest.json"
    for path in [summary_path, raw_path, codebook_path, trials_path]:
        if not path.exists():
            raise RuntimeError("missing campaign input {}".format(path))
    trials = read_json(trials_path)
    summary = read_json(summary_path)
    codebook = read_json(codebook_path)
    rows = read_jsonl(raw_path)
    metadata = summary["metadata"]
    if int(metadata["block_size"]) != expected_block:
        raise RuntimeError("unexpected block size in {}".format(root))
    if metadata.get("official_commit") != PINNED_COMMIT:
        raise RuntimeError("official implementation binding mismatch")
    if metadata.get("conditions") != ["official_reuse"]:
        raise RuntimeError("cross-model evaluations must use the official reuse condition")
    if sorted(metadata.get("kv_types", [])) != ["key", "value"]:
        raise RuntimeError("cross-model evaluations must include K and V")
    if metadata.get("precision") != "bfloat16":
        raise RuntimeError("cross-model evaluations must use BF16 storage")

    bank_sizes = [int(value) for value in trials["bank_sizes"]]
    expected_calibration = int(trials["calibration_trials"])
    expected_closed = int(trials["closed_trials_per_bank"])
    expected_open = int(trials["open_trials"])
    expected_total = 2 * (expected_calibration + len(bank_sizes) * expected_closed + expected_open)
    counts = defaultdict(int)
    keys = set()
    for row in rows:
        bank_size = int(row.get("bank_size", 0))
        key = (row["kv_type"], row["condition"], row["split"], bank_size, int(row["trial"]))
        if key in keys:
            raise RuntimeError("duplicate evaluation record {}".format(key))
        keys.add(key)
        counts[(row["kv_type"], row["split"], bank_size)] += 1
    if len(rows) != expected_total:
        raise RuntimeError("record count mismatch in {}: {} != {}".format(root, len(rows), expected_total))
    for kv_type in ["key", "value"]:
        if counts[(kv_type, "calibration", 0)] != expected_calibration:
            raise RuntimeError("calibration count mismatch")
        if counts[(kv_type, "open", 0)] != expected_open:
            raise RuntimeError("open-set count mismatch")
        for bank_size in bank_sizes:
            if counts[(kv_type, "closed", bank_size)] != expected_closed:
                raise RuntimeError("closed-set count mismatch")

    results = []
    for kv_type in ["key", "value"]:
        for bank_size in bank_sizes:
            closed = [
                row for row in rows
                if row["kv_type"] == kv_type and row["split"] == "closed" and int(row["bank_size"]) == bank_size
            ]
            open_metrics = [
                metric
                for row in rows if row["kv_type"] == kv_type and row["split"] == "open"
                for metric in row["metrics"] if int(metric["bank_size"]) == bank_size
            ]
            false_trials = sum(bool(metric["trial_false_positive"]) for metric in open_metrics)
            results.append(
                {
                    "kv_type": kv_type,
                    "bank_size": bank_size,
                    "closed_trials": len(closed),
                    "top1_recall": mean(row["top1_hit"] for row in closed),
                    "threshold_recall": mean(row["accepted_true"] for row in closed),
                    "random_label_top1_recall": mean(row["random_label_hit"] for row in closed),
                    "open_trials": len(open_metrics),
                    "open_trial_false_positive_count": false_trials,
                    "open_trial_false_positive_rate": false_trials / max(1, len(open_metrics)),
                    "open_trial_false_positive_wilson95": list(wilson(false_trials, len(open_metrics))),
                    "mean_search_seconds": mean(row["search_seconds"] for row in closed),
                    "peak_rss_bytes": max(int(row["peak_rss_bytes"]) for row in closed),
                }
            )
    return {
        "root": str(root),
        "metadata": metadata,
        "codebook": codebook,
        "results": results,
        "records": len(rows),
        "expected_records": expected_total,
        "placement_covers_all_deciles": all(
            int(values["occupied_deciles"]) == 10 for values in trials["candidate_placement"].values()
        ),
        "paths": [summary_path, raw_path, codebook_path, trials_path],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", required=True)
    parser.add_argument("--campaign-root", required=True)
    parser.add_argument("--smoke", required=True)
    parser.add_argument("--cache-manifest", required=True)
    parser.add_argument("--indices-manifest", required=True)
    parser.add_argument("--trials-b64", required=True)
    parser.add_argument("--main-b64", required=True)
    parser.add_argument("--trials-b128", required=True)
    parser.add_argument("--boundary-b128", required=True)
    parser.add_argument("--trials-head1", required=True)
    parser.add_argument("--head1-b64", required=True)
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    campaign_root = Path(args.campaign_root).resolve()
    smoke_path = Path(args.smoke).resolve()
    cache_path = Path(args.cache_manifest).resolve()
    indices_path = Path(args.indices_manifest).resolve()
    smoke = read_json(smoke_path)
    cache = read_json(cache_path)
    indices = read_json(indices_path)
    if smoke.get("official_commit") != PINNED_COMMIT or not smoke.get("all_gates_pass"):
        raise RuntimeError("official KV-Cloak smoke binding failed")
    if cache.get("model_type") != "qwen2" or int(cache.get("head_dim", 0)) != 128:
        raise RuntimeError("the prespecified cross-model campaign requires Qwen2 with d=128")
    if not cache.get("forward_pass_fidelity", {}).get("pass"):
        raise RuntimeError("formula-derived Qwen2 cache did not pass the forward comparison")
    if not indices.get("strictly_increasing"):
        raise RuntimeError("cache index manifest is invalid")

    main_b64 = validate_evaluation(Path(args.main_b64).resolve(), Path(args.trials_b64).resolve(), 64)
    boundary_b128 = validate_evaluation(Path(args.boundary_b128).resolve(), Path(args.trials_b128).resolve(), 128)
    head1_b64 = validate_evaluation(Path(args.head1_b64).resolve(), Path(args.trials_head1).resolve(), 64)

    gates = []
    for result in main_b64["results"]:
        gate = {
            "surface": "qwen2_b64_attack",
            "kv_type": result["kv_type"],
            "bank_size": result["bank_size"],
            "closed_top1_recall_at_least_0_95": result["top1_recall"] >= 0.95,
            "threshold_recall_at_least_0_90": result["threshold_recall"] >= 0.90,
            "random_label_recall_at_most_0_05": result["random_label_top1_recall"] <= 0.05,
            "open_trial_fpr_wilson_upper_at_most_0_05": result["open_trial_false_positive_wilson95"][1] <= 0.05,
        }
        gate["pass"] = all(
            value for key, value in gate.items()
            if key not in ["surface", "kv_type", "bank_size"]
        )
        gates.append(gate)
    for result in boundary_b128["results"]:
        gate = {
            "surface": "qwen2_b128_full_rowspace_boundary",
            "kv_type": result["kv_type"],
            "bank_size": result["bank_size"],
            "closed_top1_recall_at_most_0_05": result["top1_recall"] <= 0.05,
            "random_label_recall_at_most_0_05": result["random_label_top1_recall"] <= 0.05,
        }
        gate["pass"] = gate["closed_top1_recall_at_most_0_05"] and gate["random_label_recall_at_most_0_05"]
        gates.append(gate)
    for result in head1_b64["results"]:
        gate = {
            "surface": "qwen2_second_kv_head_b64",
            "kv_type": result["kv_type"],
            "bank_size": result["bank_size"],
            "closed_top1_recall_at_least_0_95": result["top1_recall"] >= 0.95,
            "random_label_recall_at_most_0_05": result["random_label_top1_recall"] <= 0.05,
        }
        gate["pass"] = gate["closed_top1_recall_at_least_0_95"] and gate["random_label_recall_at_most_0_05"]
        gates.append(gate)

    final_dir = campaign_root / "final"
    results_path = final_dir / "kvcloak_qwen_crossmodel_results.json"
    result_payload = {
        "qwen2_b64": main_b64,
        "qwen2_b128": boundary_b128,
        "qwen2_second_kv_head_b64": head1_b64,
    }
    for evaluation in result_payload.values():
        evaluation["paths"] = [str(path) for path in evaluation["paths"]]
    atomic_json(results_path, result_payload)

    input_paths = [smoke_path, cache_path, indices_path]
    for evaluation in [main_b64, boundary_b128, head1_b64]:
        input_paths.extend(Path(path) for path in evaluation["paths"])
    input_paths = list(dict.fromkeys(path.resolve() for path in input_paths))
    completion = {
        "status": "complete",
        "decision_surface": "cryptographic dimension boundary of official KV-Cloak on modern GQA K/V caches",
        "prespecified_hypothesis": "rowspace recognition survives at standard b=64 when d=128 and collapses only at b=d=128",
        "model": {
            "path": cache["model"],
            "model_type": cache["model_type"],
            "architecture": cache["architecture"],
            "hidden_dimension": cache["hidden_dimension"],
            "attention_heads": cache["attention_heads"],
            "key_value_heads": cache["key_value_heads"],
            "head_dimension": cache["head_dim"],
            "projection_sha256": cache["projection_sha256"],
            "forward_pass_fidelity": cache["forward_pass_fidelity"],
        },
        "candidate_index_count": indices["index_count"],
        "qwen2_b64": {key: value for key, value in main_b64.items() if key != "paths"},
        "qwen2_b128": {key: value for key, value in boundary_b128.items() if key != "paths"},
        "qwen2_second_kv_head_b64": {key: value for key, value in head1_b64.items() if key != "paths"},
        "prespecified_gates": gates,
        "all_prespecified_gates_pass": (
            main_b64["placement_covers_all_deciles"]
            and boundary_b128["placement_covers_all_deciles"]
            and head1_b64["placement_covers_all_deciles"]
            and all(gate["pass"] for gate in gates)
        ),
        "official_repository": "https://github.com/SiO-2/kvcloak",
        "official_commit": PINNED_COMMIT,
        "repo_revision": subprocess.check_output(
            ["git", "-C", str(repo_root), "rev-parse", "HEAD"], text=True
        ).strip(),
        "input_sha256": {str(path): sha256(path) for path in input_paths},
        "artifact_sha256": {results_path.name: sha256(results_path)},
    }
    atomic_json(final_dir / "campaign_complete.json", completion)
    print(json.dumps(completion, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

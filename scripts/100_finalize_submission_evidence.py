#!/usr/bin/env python3
"""Independently validate the final GELO and KV-Cloak submission evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


OFFICIAL_COMMIT = "6b40f36edb2f337557543e7e60b10022308883d4"
OFFICIAL_SOURCE_SHA256 = "246a72d5760a3f1b7c61800ad778cc14200d58e85bead89b8e7d4c8d2e86051d"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def close(actual: float, expected: float, name: str, tolerance: float = 1e-10) -> None:
    require(math.isclose(float(actual), float(expected), rel_tol=tolerance, abs_tol=tolerance),
            f"{name}: {actual} != {expected}")


def wilson(successes: int, trials: int) -> list[float]:
    z = 1.959963984540054
    proportion = successes / trials
    denominator = 1.0 + z * z / trials
    center = (proportion + z * z / (2.0 * trials)) / denominator
    radius = z * math.sqrt(
        (proportion * (1.0 - proportion) + z * z / (4.0 * trials)) / trials
    ) / denominator
    return [max(0.0, center - radius), min(1.0, center + radius)]


def verify_hash(path: Path, expected: str, label: str) -> None:
    require(path.is_file(), f"missing {label}: {path}")
    require(sha256(path) == expected, f"SHA-256 mismatch for {label}: {path}")


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def verify_gelo(manifest_path: Path) -> dict:
    manifest = load_json(manifest_path)
    require(manifest.get("status") == "complete", f"incomplete GELO manifest: {manifest_path}")
    final_dir = manifest_path.parent
    campaign_root = final_dir.parent
    for relative, expected in manifest["artifact_sha256"].items():
        verify_hash(final_dir / relative, expected, f"GELO artifact {relative}")
    for relative, expected in manifest["input_sha256"].items():
        verify_hash(campaign_root / relative, expected, f"GELO input {relative}")
    validation = manifest.get("validation", {})
    if "records" in validation:
        require(validation["records"] == validation["expected_records"],
                f"GELO record mismatch: {manifest_path}")
    for name, entry in validation.items():
        if isinstance(entry, dict) and "complete" in entry:
            require(entry["complete"], f"GELO validation incomplete: {name}")
            if "actual_records" in entry:
                require(entry["actual_records"] == entry["expected_records"],
                        f"GELO record mismatch: {name}")
            if "records" in entry and "expected" in entry:
                require(entry["records"] == sum(int(value) for value in entry["expected"].values()),
                        f"GELO split-count mismatch: {name}")
            require(not entry.get("duplicates"), f"GELO duplicate records: {name}")
            require(not entry.get("duplicate_keys"), f"GELO duplicate keys: {name}")
            require(not entry.get("unexpected"), f"GELO unexpected records: {name}")
            require(not entry.get("unexpected_groups"), f"GELO unexpected groups: {name}")
            require(not entry.get("missing_or_extra_groups"), f"GELO group mismatch: {name}")
    return {
        "path": str(manifest_path),
        "sha256": sha256(manifest_path),
        "bank_size": int(manifest["bank_size"]),
        "all_prespecified_gates_pass": manifest.get("all_prespecified_gates_pass"),
        "validation": validation,
    }


def stable_seed(manifest_path: Path) -> int:
    label = manifest_path.parent.name
    return {"qwen_b128": 20261328, "gpt2_b64": 20261064, "qwen_b64": 20261264}[label]


def verify_normalized_gram(manifest_path: Path) -> dict:
    manifest = load_json(manifest_path)
    require(manifest.get("status") == "complete", f"incomplete Gram manifest: {manifest_path}")
    require(manifest["official_commit"] == OFFICIAL_COMMIT, "official commit mismatch")
    require(manifest["official_source_sha256"] == OFFICIAL_SOURCE_SHA256,
            "official source hash mismatch")
    verify_hash(Path(manifest["cache_manifest"]), manifest["cache_manifest_sha256"], "cache manifest")
    trials_path = Path(manifest["trials"])
    verify_hash(trials_path, manifest["trials_sha256"], "trial manifest")
    trials = load_json(trials_path)
    candidate_path = Path(trials["candidate_indices"])
    verify_hash(candidate_path, manifest["candidate_indices_sha256"], "candidate indices")
    output = manifest_path.parent
    verify_hash(output / "trial_records.jsonl", manifest["trial_records_sha256"], "trial records")
    for kv_type in ("key", "value"):
        codebook_path = output / f"{kv_type}_gram_features.npy"
        verify_hash(codebook_path, manifest["codebook_sha256"][kv_type], f"{kv_type} codebook")
        codebook = np.load(codebook_path, mmap_mode="r", allow_pickle=False)
        require(codebook.shape == (manifest["candidate_count"], manifest["feature_dimension"]),
                f"{kv_type} codebook shape mismatch")
        require(np.isfinite(codebook).all(), f"non-finite {kv_type} codebook")

    records = [json.loads(line) for line in (output / "trial_records.jsonl").read_text(
        encoding="utf-8").splitlines() if line]
    require(len(records) == manifest["record_count"], "Gram record-count mismatch")
    keys = [(row["condition"], row["kv_type"], row["split"], row["trial"]) for row in records]
    require(len(keys) == len(set(keys)), "duplicate Gram trial records")
    expected_split_counts = {
        "calibration": len(trials["records"]["calibration"]),
        "closed": len(trials["records"]["closed"]["10000"]),
        "open": len(trials["records"]["open"]),
    }
    grouped: dict[tuple[str, str], dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
    for row in records:
        grouped[(row["condition"], row["kv_type"])][row["split"]].append(row)
        require(row["accepted"] == (row["best_distance"] <= row["threshold"]),
                "accepted flag does not match frozen threshold")
    require(set(grouped) == {(condition, kv) for condition in manifest["conditions"]
                              for kv in ("key", "value")}, "Gram group mismatch")

    result_map = {(row["condition"], row["kv_type"]): row for row in manifest["results"]}
    gate_map = {(row["condition"], row["kv_type"]): row for row in manifest["prespecified_gates"]}
    seed = stable_seed(manifest_path)
    verified_results = []
    for group, splits in grouped.items():
        for split, count in expected_split_counts.items():
            require(len(splits[split]) == count, f"{group} {split} count mismatch")
        calibration = sorted(float(row["best_distance"]) for row in splits["calibration"])
        threshold = calibration[0]
        closed = splits["closed"]
        opened = splits["open"]
        hits = np.asarray([row["predicted_position"] == row["true_position"] for row in closed])
        accepted_hits = np.asarray([row["accepted"] and hit for row, hit in zip(closed, hits)])
        false_accepts = sum(bool(row["accepted"]) for row in opened)
        kv_index = 0 if group[1] == "key" else 1
        random_labels = np.random.default_rng(seed + 20000 + kv_index).permutation(10000)
        random_hits = np.asarray([
            random_labels[int(row["predicted_position"])] == int(row["true_position"])
            for row in closed
        ])
        expected = result_map[group]
        close(expected["threshold"], threshold, f"{group} threshold")
        close(expected["top1_recall"], hits.mean(), f"{group} top1")
        close(expected["threshold_recall"], accepted_hits.mean(), f"{group} threshold recall")
        close(expected["random_label_top1_recall"], random_hits.mean(), f"{group} random control")
        require(expected["open_trial_false_positive_count"] == false_accepts,
                f"{group} false-positive count mismatch")
        interval = wilson(false_accepts, len(opened))
        close(expected["open_trial_false_positive_wilson95"][0], interval[0], f"{group} Wilson lower")
        close(expected["open_trial_false_positive_wilson95"][1], interval[1], f"{group} Wilson upper")
        close(expected["closed_distance_mean"], np.mean([row["best_distance"] for row in closed]),
              f"{group} closed mean")
        close(expected["open_distance_mean"], np.mean([row["best_distance"] for row in opened]),
              f"{group} open mean")
        gate = gate_map[group]
        require(gate["top1_recall_at_least_0_95"] == (expected["top1_recall"] >= 0.95),
                f"{group} top1 gate mismatch")
        require(gate["threshold_recall_at_least_0_90"] == (expected["threshold_recall"] >= 0.90),
                f"{group} threshold gate mismatch")
        require(gate["random_label_recall_at_most_0_05"] ==
                (expected["random_label_top1_recall"] <= 0.05), f"{group} random gate mismatch")
        require(gate["open_wilson_upper_at_most_0_05"] == (interval[1] <= 0.05),
                f"{group} Wilson gate mismatch")
        require(gate["pass"] == all(value for key, value in gate.items()
                                     if key not in {"condition", "kv_type", "pass"}),
                f"{group} combined gate mismatch")
        verified_results.append(expected)
    require(manifest["all_prespecified_gates_pass"] == all(row["pass"] for row in gate_map.values()),
            "all-gates field mismatch")
    return {
        "path": str(manifest_path),
        "sha256": sha256(manifest_path),
        "record_count": len(records),
        "runtime_seconds": manifest["elapsed_seconds"],
        "peak_rss_bytes": manifest["peak_rss_bytes"],
        "results": verified_results,
        "gates": manifest["prespecified_gates"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    repo = Path(args.repo_root).resolve()
    official_source = repo.parent / "third_party" / "kvcloak" / "defense" / "core" / "kvcloak.py"
    verify_hash(official_source, OFFICIAL_SOURCE_SHA256, "official KV-Cloak source")
    official_repo = official_source.parents[2]
    commit = subprocess.check_output(["git", "-C", str(official_repo), "rev-parse", "HEAD"], text=True).strip()
    require(commit == OFFICIAL_COMMIT, "official KV-Cloak checkout mismatch")

    gelo_root = repo / "outputs" / "gelo_msmarco_100k_dedup_20260802"
    gelo = {
        "main": verify_gelo(gelo_root / "final" / "campaign_complete.json"),
        "private_drift": verify_gelo(gelo_root / "private_drift_followup" / "final" / "campaign_complete.json"),
        "fullbank": verify_gelo(gelo_root / "fullbank_validation" / "final" / "campaign_complete.json"),
    }
    gram_root = repo / "outputs" / "kvcloak_normgram_stable_20260804"
    gram = {
        label: verify_normalized_gram(gram_root / label / "campaign_complete.json")
        for label in ("qwen_b128", "gpt2_b64", "qwen_b64")
    }
    preserved_failures = {}
    for campaign in ("kvcloak_gram_followup_20260804", "kvcloak_fullgram_stable_20260804"):
        preserved_failures[campaign] = {}
        for label in ("qwen_b128", "gpt2_b64", "qwen_b64"):
            path = repo / "outputs" / campaign / label / "campaign_complete.json"
            manifest = load_json(path)
            require(manifest.get("status") == "complete", f"incomplete preserved campaign: {path}")
            preserved_failures[campaign][label] = {
                "path": str(path), "sha256": sha256(path),
                "all_prespecified_gates_pass": manifest["all_prespecified_gates_pass"],
                "results": manifest["results"],
            }

    result = {
        "status": "complete",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "validator_sha256": sha256(Path(__file__).resolve()),
        "repo_revision": subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip(),
        "official_kvcloak": {"commit": commit, "source_sha256": OFFICIAL_SOURCE_SHA256},
        "gelo": gelo,
        "kvcloak_normalized_gram_stable": gram,
        "preserved_adaptive_failures": preserved_failures,
    }
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, output)
    print(json.dumps({"status": "complete", "output": str(output), "sha256": sha256(output)}, indent=2))


if __name__ == "__main__":
    main()


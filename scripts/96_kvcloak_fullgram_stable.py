#!/usr/bin/env python3
"""Evaluate the orthogonal-left Gram invariant in official KV-Cloak."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import resource
import subprocess
import time
from pathlib import Path
from typing import Sequence

import numpy as np
import torch


PINNED_COMMIT = "6b40f36edb2f337557543e7e60b10022308883d4"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def load_evaluator(path: Path):
    spec = importlib.util.spec_from_file_location("kv_eval", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load rowspace evaluator")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def wilson(successes: int, trials: int) -> list[float]:
    if trials <= 0:
        return [0.0, 1.0]
    z = 1.959963984540054
    p = successes / trials
    denominator = 1.0 + z * z / trials
    center = (p + z * z / (2.0 * trials)) / denominator
    radius = z * math.sqrt((p * (1.0 - p) + z * z / (4.0 * trials)) / trials) / denominator
    return [max(0.0, center - radius), min(1.0, center + radius)]


def calibration_order(values: Sequence[float], alpha: float) -> float:
    ordered = sorted(float(value) for value in values)
    one_based = max(1, int(math.floor(alpha * (len(ordered) + 1))))
    return ordered[min(len(ordered) - 1, one_based - 1)]


def upper_tail_threshold(values: Sequence[float], alpha: float) -> float:
    """Finite-sample threshold for accepting unusually large membership scores."""
    ordered = sorted(float(value) for value in values)
    one_based = int(math.ceil((1.0 - alpha) * (len(ordered) + 1)))
    return ordered[min(len(ordered) - 1, max(0, one_based - 1))]


class GramFeatures:
    def __init__(self, dimension: int, pair_count: int, seed: int):
        rng = np.random.default_rng(seed)
        all_left, all_right = np.triu_indices(dimension, k=1)
        if pair_count <= 0 or pair_count >= all_left.shape[0]:
            selected = np.arange(all_left.shape[0])
        else:
            selected = np.sort(rng.choice(all_left.shape[0], size=pair_count, replace=False))
        self.left = torch.from_numpy(all_left[selected].astype(np.int64))
        self.right = torch.from_numpy(all_right[selected].astype(np.int64))
        self.pairs = np.stack([all_left[selected], all_right[selected]], axis=1).astype("<i8")
        self.dimension = dimension

    def __call__(self, observed: torch.Tensor) -> np.ndarray:
        value = observed.to(torch.float32)
        gram = torch.matmul(value.transpose(-1, -2), value)
        diagonal = torch.diagonal(gram, dim1=-2, dim2=-1)
        off_diagonal = gram[:, self.left, self.right] * math.sqrt(2.0)
        feature = torch.cat([diagonal, off_diagonal], dim=-1)
        return feature.cpu().numpy().astype(np.float32)


def feature_batch(evaluator, cache, cloak, indices, head_position, block_size, precision, mapper):
    key = cache.rows("key", indices, head_position, block_size)
    value = cache.rows("value", indices, head_position, block_size)
    protected_key, protected_value = evaluator.protect_batch(cloak, key, value, precision)
    return mapper(protected_key), mapper(protected_value)


def enroll(evaluator, cache, cloak, indices, head_position, block_size, precision, mapper, batch_size):
    outputs = {"key": [], "value": []}
    for start in range(0, len(indices), batch_size):
        key, value = feature_batch(
            evaluator, cache, cloak, indices[start : start + batch_size],
            head_position, block_size, precision, mapper,
        )
        outputs["key"].append(key)
        outputs["value"].append(value)
    return {name: np.concatenate(parts, axis=0) for name, parts in outputs.items()}


def query_records(evaluator, cache, cloak, records, head_position, block_size, precision, mapper):
    outputs = {"key": [], "value": []}
    for record in records:
        key, value = feature_batch(
            evaluator, cache, cloak, [int(record["source_index"])],
            head_position, block_size, precision, mapper,
        )
        outputs["key"].append(key[0])
        outputs["value"].append(value[0])
    return {name: np.stack(parts, axis=0) for name, parts in outputs.items()}


class StableSearch:
    """Centered float64 Euclidean search, avoiding cancellation near cosine one."""

    def __init__(self, bank: np.ndarray):
        self.center = bank.mean(axis=0, dtype=np.float64)
        self.bank = bank.astype(np.float64) - self.center
        self.bank_sq = np.einsum("ij,ij->i", self.bank, self.bank)
        self.transposed = self.bank.T

    def __call__(self, queries: np.ndarray, chunk: int = 32):
        predictions, best_distances, second_distances = [], [], []
        for start in range(0, queries.shape[0], chunk):
            centered = queries[start : start + chunk].astype(np.float64) - self.center
            query_sq = np.einsum("ij,ij->i", centered, centered)
            squared = query_sq[:, None] + self.bank_sq[None, :] - 2.0 * (centered @ self.transposed)
            squared = np.maximum(squared, 0.0)
            top_two = np.argpartition(squared, kth=1, axis=1)[:, :2]
            values = np.take_along_axis(squared, top_two, axis=1)
            order = np.argsort(values, axis=1)
            top_two = np.take_along_axis(top_two, order, axis=1)
            values = np.take_along_axis(values, order, axis=1)
            predictions.extend(int(value) for value in top_two[:, 0])
            best_distances.extend(float(value) for value in np.sqrt(values[:, 0]))
            second_distances.extend(float(value) for value in np.sqrt(values[:, 1]))
        return np.asarray(predictions), np.asarray(best_distances), np.asarray(second_distances)


def separation_score(best: np.ndarray, second: np.ndarray) -> np.ndarray:
    """Relative nearest-neighbor gap in [0,1], with zero for unresolved ties."""
    return np.maximum(second - best, 0.0) / np.maximum(second, 1e-30)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", required=True)
    parser.add_argument("--cache-manifest", required=True)
    parser.add_argument("--trials", required=True)
    parser.add_argument("--official-repo", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--head-position", type=int, default=0)
    parser.add_argument("--precision", default="bfloat16")
    parser.add_argument("--evaluation-seed", type=int, required=True)
    parser.add_argument("--conditions", nargs="+", default=["official_reuse", "refresh_left"])
    parser.add_argument("--pair-count", type=int, default=-1,
                        help="off-diagonal Gram entries; <=0 uses the complete upper triangle")
    parser.add_argument("--enrollment-batch-size", type=int, default=64)
    parser.add_argument("--calibration-alpha", type=float, default=0.01)
    parser.add_argument("--threads", type=int, default=20)
    args = parser.parse_args()

    os.environ["OPENBLAS_NUM_THREADS"] = str(args.threads)
    os.environ["OMP_NUM_THREADS"] = str(args.threads)
    os.environ["MKL_NUM_THREADS"] = str(args.threads)
    torch.set_num_threads(args.threads)
    started = time.time()
    repo_root = Path(args.repo_root).resolve()
    evaluator_path = repo_root / "scripts" / "86_kvcloak_codebook_evaluate.py"
    evaluator = load_evaluator(evaluator_path)
    cache_manifest = Path(args.cache_manifest).resolve()
    trials_path = Path(args.trials).resolve()
    trials = json.loads(trials_path.read_text(encoding="utf-8"))
    block_size = int(trials["block_size"])
    if [int(value) for value in trials["bank_sizes"]] != [10000]:
        raise RuntimeError("Gram follow-up is prespecified for one 10K bank")
    candidate_path = Path(trials["candidate_indices"]).resolve()
    candidate_indices = np.load(candidate_path, allow_pickle=False).astype(np.int64)
    if candidate_indices.shape != (10000,):
        raise RuntimeError("candidate bank must contain exactly 10000 indices")
    cache = evaluator.PlainCache(str(cache_manifest))
    dimension = int(cache.manifest["head_dim"])
    mapper = GramFeatures(dimension, args.pair_count, args.evaluation_seed + 701)
    official_repo = Path(args.official_repo).resolve()
    official, official_source = evaluator.load_official(official_repo)
    official_commit = subprocess.check_output(
        ["git", "-C", str(official_repo), "rev-parse", "HEAD"], text=True
    ).strip()
    if official_commit != PINNED_COMMIT:
        raise RuntimeError("official commit mismatch")
    base_config, base_cloak, _ = evaluator.make_base_cloak(
        official, cache, block_size, args.precision, args.evaluation_seed + 2
    )

    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    codebook = enroll(
        evaluator, cache, base_cloak, candidate_indices, args.head_position,
        block_size, args.precision, mapper, args.enrollment_batch_size,
    )
    for name, value in codebook.items():
        np.save(output / f"{name}_gram_features.npy", value, allow_pickle=False)
    searchers = {name: StableSearch(value) for name, value in codebook.items()}

    calibration_records = trials["records"]["calibration"]
    closed_records = trials["records"]["closed"]["10000"]
    open_records = trials["records"]["open"]
    all_results = []
    raw_records = []
    gates = []
    for condition_index, condition in enumerate(args.conditions):
        cloak = evaluator.make_variant_cloak(
            official, base_config, base_cloak, condition,
            args.evaluation_seed + 10000 + condition_index,
        )
        split_features = {
            "calibration": query_records(evaluator, cache, cloak, calibration_records, args.head_position, block_size, args.precision, mapper),
            "closed": query_records(evaluator, cache, cloak, closed_records, args.head_position, block_size, args.precision, mapper),
            "open": query_records(evaluator, cache, cloak, open_records, args.head_position, block_size, args.precision, mapper),
        }
        for kv_index, kv_type in enumerate(["key", "value"]):
            calibration_prediction, calibration_distance, calibration_second = searchers[kv_type](
                split_features["calibration"][kv_type]
            )
            distance_threshold = calibration_order(calibration_distance, args.calibration_alpha)
            calibration_gap = separation_score(calibration_distance, calibration_second)
            gap_threshold = upper_tail_threshold(calibration_gap, args.calibration_alpha)
            closed_prediction, closed_distance, closed_second = searchers[kv_type](
                split_features["closed"][kv_type]
            )
            open_prediction, open_distance, open_second = searchers[kv_type](
                split_features["open"][kv_type]
            )
            closed_gap = separation_score(closed_distance, closed_second)
            open_gap = separation_score(open_distance, open_second)
            true_position = np.asarray([int(record["candidate_position"]) for record in closed_records])
            top1_hit = closed_prediction == true_position
            accepted_true = top1_hit & (closed_gap > gap_threshold)
            random_labels = np.random.default_rng(args.evaluation_seed + 20000 + kv_index).permutation(10000)
            random_hit = random_labels[closed_prediction] == true_position
            false_accept = open_gap > gap_threshold
            interval = wilson(int(false_accept.sum()), int(false_accept.shape[0]))
            result = {
                "condition": condition,
                "kv_type": kv_type,
                "bank_size": 10000,
                "closed_trials": int(top1_hit.shape[0]),
                "top1_recall": float(top1_hit.mean()),
                "gap_threshold_recall": float(accepted_true.mean()),
                "random_label_top1_recall": float(random_hit.mean()),
                "open_trials": int(false_accept.shape[0]),
                "open_trial_false_positive_count": int(false_accept.sum()),
                "open_trial_false_positive_rate": float(false_accept.mean()),
                "open_trial_false_positive_wilson95": interval,
                "gap_threshold": float(gap_threshold),
                "distance_threshold_diagnostic": float(distance_threshold),
                "closed_distance_mean": float(closed_distance.mean()),
                "open_distance_mean": float(open_distance.mean()),
                "closed_gap_mean": float(closed_gap.mean()),
                "open_gap_mean": float(open_gap.mean()),
            }
            all_results.append(result)
            gate = {
                "condition": condition,
                "kv_type": kv_type,
                "top1_recall_at_least_0_95": result["top1_recall"] >= 0.95,
                "gap_threshold_recall_at_least_0_90": result["gap_threshold_recall"] >= 0.90,
                "random_label_recall_at_most_0_05": result["random_label_top1_recall"] <= 0.05,
                "open_wilson_upper_at_most_0_05": interval[1] <= 0.05,
            }
            gate["pass"] = all(value for key, value in gate.items() if key not in {"condition", "kv_type", "pass"})
            gates.append(gate)
            for split, records, predictions, distances, seconds, gaps in [
                ("calibration", calibration_records, calibration_prediction, calibration_distance, calibration_second, calibration_gap),
                ("closed", closed_records, closed_prediction, closed_distance, closed_second, closed_gap),
                ("open", open_records, open_prediction, open_distance, open_second, open_gap),
            ]:
                for index, record in enumerate(records):
                    raw_records.append({
                        "condition": condition,
                        "kv_type": kv_type,
                        "split": split,
                        "trial": int(record["trial"]),
                        "source_index": int(record["source_index"]),
                        "true_position": int(record["candidate_position"]) if split == "closed" else None,
                        "predicted_position": int(predictions[index]),
                        "best_distance": float(distances[index]),
                        "second_distance": float(seconds[index]),
                        "membership_gap": float(gaps[index]),
                        "gap_threshold": float(gap_threshold),
                        "distance_threshold_diagnostic": float(distance_threshold),
                        "accepted": bool(gaps[index] > gap_threshold),
                    })

    raw_path = output / "trial_records.jsonl"
    raw_path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in raw_records), encoding="utf-8")
    result = {
        "status": "complete",
        "adaptive_followup": True,
        "motivation": "the official paper and implementation restrict S and P to the orthogonal group, so Y^T Y remains nontrivial when rowspace becomes full",
        "prespecified_followup_hypothesis": "the complete raw Gram invariant, stable Euclidean search, and an out-of-bank calibrated nearest-neighbor gap identify enrolled blocks under official reuse and refreshed orthogonal left transforms",
        "official_commit": official_commit,
        "official_source_sha256": sha256(official_source),
        "repo_revision": subprocess.check_output(["git", "-C", str(repo_root), "rev-parse", "HEAD"], text=True).strip(),
        "cache_manifest": str(cache_manifest),
        "cache_manifest_sha256": sha256(cache_manifest),
        "trials": str(trials_path),
        "trials_sha256": sha256(trials_path),
        "candidate_indices_sha256": sha256(candidate_path),
        "block_size": block_size,
        "head_dimension": dimension,
        "head_position": args.head_position,
        "model_head": int(cache.manifest["heads"][args.head_position]),
        "precision": args.precision,
        "conditions": args.conditions,
        "calibration_alpha": args.calibration_alpha,
        "calibration_rule": "membership threshold is the ceil((1-alpha)(n+1)) upper-tail order statistic with strict acceptance; n=100 and alpha=.01",
        "gram_feature": "raw Frobenius-compatible Gram coordinates: diagonal plus sqrt(2)-weighted selected off-diagonal entries",
        "distance": "mean-centered float64 Euclidean distance; centering preserves pairwise distances and prevents cosine-one cancellation",
        "membership_score": "relative nearest-neighbor gap (d2-d1)/max(d2,1e-30)",
        "feature_dimension": dimension + int(mapper.pairs.shape[0]),
        "complete_upper_triangle": bool(mapper.pairs.shape[0] == dimension * (dimension - 1) // 2),
        "pairs_sha256": hashlib.sha256(mapper.pairs.tobytes()).hexdigest(),
        "candidate_count": 10000,
        "record_count": len(raw_records),
        "trial_records_sha256": sha256(raw_path),
        "codebook_sha256": {
            name: sha256(output / f"{name}_gram_features.npy") for name in ["key", "value"]
        },
        "results": all_results,
        "prespecified_gates": gates,
        "all_prespecified_gates_pass": all(gate["pass"] for gate in gates),
        "elapsed_seconds": time.time() - started,
        "peak_rss_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024),
    }
    atomic_json(output / "campaign_complete.json", result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

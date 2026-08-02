#!/usr/bin/env python3
"""Evaluate GELO row-space leakage at 10K--100K and open-set conditions."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
import os
import random
import socket
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import psutil
import torch


CONDITIONS = {
    "ideal": {
        "mix": "orthogonal", "shield": "gaussian", "fraction": 0.0,
        "scale": 0.0, "precision": "float32", "mix_condition": 1.0,
    },
    "gelo_gaussian": {
        "mix": "orthogonal", "shield": "gaussian", "fraction": 0.05,
        "scale": 10.0, "precision": "float32", "mix_condition": 1.0,
    },
    "gelo_nonorth": {
        "mix": "non_orthogonal", "shield": "gaussian", "fraction": 0.05,
        "scale": 10.0, "precision": "float32", "mix_condition": 50.0,
    },
    "quantized_gaussian": {
        "mix": "orthogonal", "shield": "gaussian", "fraction": 0.12,
        "scale": 25.0, "precision": "bfloat16", "mix_condition": 1.0,
    },
    "quantized_nonorth": {
        "mix": "non_orthogonal", "shield": "gaussian", "fraction": 0.12,
        "scale": 25.0, "precision": "bfloat16", "mix_condition": 50.0,
    },
    "manifold": {
        "mix": "orthogonal", "shield": "manifold", "fraction": 0.05,
        "scale": 10.0, "precision": "float32", "mix_condition": 1.0,
    },
    "manifold_stress": {
        "mix": "non_orthogonal", "shield": "manifold", "fraction": 0.14,
        "scale": 30.0, "precision": "bfloat16", "mix_condition": 50.0,
    },
}


def atomic_json(path: Path, value: Dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(str(temporary), str(path))


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


class HiddenStore:
    def __init__(self, manifest_path: str):
        self.manifest_path = Path(manifest_path)
        self.manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        self.shape = tuple(int(x) for x in self.manifest["shape"])
        indices_path = self.manifest.get("indices")
        if indices_path:
            indices = np.load(indices_path, allow_pickle=False).astype(np.int64)
            self.global_to_local = {int(value): offset for offset, value in enumerate(indices)}
        else:
            self.global_to_local = None
        self.layers: Dict[int, np.memmap] = {}
        for layer, path in self.manifest["files"].items():
            self.layers[int(layer)] = np.memmap(
                path, mode="r", dtype=np.float16, shape=self.shape
            )

    def rows(self, layer: int, global_index: int) -> np.ndarray:
        if self.global_to_local is None:
            local_index = global_index
        else:
            if global_index not in self.global_to_local:
                raise KeyError("global index {} absent from source cache".format(global_index))
            local_index = self.global_to_local[global_index]
        return np.asarray(self.layers[layer][local_index], dtype=np.float32)


def quantize(observed: torch.Tensor, precision: str) -> np.ndarray:
    if precision == "float32":
        return observed.detach().cpu().numpy().astype(np.float64)
    if precision == "bfloat16":
        return observed.to(torch.bfloat16).to(torch.float32).cpu().numpy().astype(np.float64)
    if precision == "float16":
        return observed.to(torch.float16).to(torch.float32).cpu().numpy().astype(np.float64)
    raise ValueError(precision)


def rowspace_basis(observed: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    _, singular, vt = np.linalg.svd(observed, full_matrices=False)
    threshold = float(singular[0]) * max(observed.shape) * np.finfo(np.float64).eps
    rank = int(np.sum(singular > threshold))
    return vt[:rank], singular[:rank]


def random_basis(rank: int, dim: int, rng: np.random.Generator) -> np.ndarray:
    q, _ = np.linalg.qr(rng.standard_normal((dim, rank)))
    return q.T


def score_bank(
    bank: np.memmap,
    bank_size: int,
    basis: np.ndarray,
    sampled_rows: int,
    chunk_size: int,
    score_dtype: str,
) -> Tuple[np.ndarray, float, int]:
    dtype = np.float32 if score_dtype.startswith("float32") else np.float64
    space = np.asarray(basis, dtype=dtype)
    scores = np.empty(bank_size, dtype=np.float64)
    process = psutil.Process()
    peak = process.memory_info().rss
    started = time.time()
    for start in range(0, bank_size, chunk_size):
        stop = min(start + chunk_size, bank_size)
        rows = np.asarray(bank[start:stop], dtype=dtype)
        norms_sq = np.sum(rows * rows, axis=-1)
        flat_coordinates = rows.reshape(-1, rows.shape[-1]) @ space.T
        coordinates = flat_coordinates.reshape(
            rows.shape[0], rows.shape[1], space.shape[0]
        )
        denominator = np.maximum(norms_sq, np.finfo(dtype).tiny)
        if score_dtype == "float32_stable":
            flat_rows = rows.reshape(-1, rows.shape[-1])
            projected_rows = flat_coordinates @ space
            delta = flat_rows - projected_rows
            residual = np.sqrt(
                np.sum(delta * delta, axis=-1)
                / denominator.reshape(-1)
            ).reshape(rows.shape[0], rows.shape[1])
        else:
            projected_sq = np.sum(coordinates * coordinates, axis=-1)
            residual_sq = np.maximum(norms_sq - projected_sq, 0.0)
            residual = np.sqrt(residual_sq / denominator)
        smallest = np.partition(residual, sampled_rows - 1, axis=1)[:, :sampled_rows]
        scores[start:stop] = np.mean(smallest, axis=1, dtype=np.float64)
        peak = max(peak, process.memory_info().rss)
    return scores, time.time() - started, peak


def make_observation(
    record: Dict,
    source_store: HiddenStore,
    public_store: HiddenStore,
    layer: int,
    condition: Dict,
    sampled_rows: int,
    max_bank: int,
    seed: int,
    official: object,
) -> Tuple[np.ndarray, List[int], List[int], Dict]:
    rng = random.Random(seed)
    np_rng = np.random.default_rng(seed + 17)
    torch.manual_seed(seed + 31)
    known = [int(x) for x in record["known"]]
    unknown = [int(x) for x in record["unknown"]]
    chunks = []
    for index in known + unknown:
        rows = source_store.rows(layer, index)
        chosen = rng.sample(range(rows.shape[0]), k=sampled_rows)
        chunks.append(rows[np.asarray(chosen, dtype=np.int64)])
    clean = np.concatenate(chunks, axis=0).astype(np.float32)
    x = clean
    shield_ids: List[int] = []
    shield_rows = int(condition["fraction"] * clean.shape[0])
    if shield_rows > 0 and condition["shield"] == "gaussian":
        shield = np_rng.standard_normal((shield_rows, clean.shape[1])).astype(np.float32)
        mean_norm = float(np.mean(np.linalg.norm(clean, axis=1)))
        scale = float(condition["scale"]) * mean_norm / (math.sqrt(clean.shape[1]) + 1e-12)
        shield *= np.float32(scale)
        x = np.concatenate([clean, shield], axis=0)
    elif shield_rows > 0 and condition["shield"] == "manifold":
        remaining = shield_rows
        pieces = []
        exclude = set(known)
        while remaining > 0:
            candidate = rng.randrange(max_bank)
            if candidate in exclude:
                continue
            rows = public_store.rows(layer, candidate)
            take = min(remaining, rows.shape[0])
            chosen = rng.sample(range(rows.shape[0]), k=take)
            pieces.append(rows[np.asarray(chosen, dtype=np.int64)])
            shield_ids.append(candidate)
            remaining -= take
        shield = np.concatenate(pieces, axis=0).astype(np.float32)
        mean_norm = float(np.mean(np.linalg.norm(clean, axis=1)))
        shield_norm = float(np.mean(np.linalg.norm(shield, axis=1)))
        shield *= np.float32(float(condition["scale"]) * mean_norm / (shield_norm + 1e-12))
        x = np.concatenate([clean, shield], axis=0)
    h = torch.from_numpy(x)
    if condition["mix"] == "orthogonal":
        mixing = official.random_orthogonal(h.shape[0], h.device, h.dtype)
    elif condition["mix"] == "non_orthogonal":
        mixing = official.random_invertible_with_condition(
            h.shape[0], float(condition["mix_condition"]), h.device, h.dtype
        )
    else:
        raise ValueError(condition["mix"])
    observed = quantize(mixing @ h, str(condition["precision"]))
    basis, singular = rowspace_basis(observed)
    diagnostics = {
        "observation_rows": int(observed.shape[0]),
        "rowspace_rank": int(basis.shape[0]),
        "observed_condition": float(singular[0] / singular[-1]),
        "shield_rows": shield_rows,
        "shield_candidate_ids": sorted(set(shield_ids)),
    }
    return basis, known, sorted(set(known + shield_ids)), diagnostics


def average_precision(scores: np.ndarray, true_ids: Sequence[int]) -> Optional[float]:
    if not true_ids:
        return None
    order = np.argsort(scores, kind="stable")
    truth = np.zeros(scores.shape[0], dtype=bool)
    truth[np.asarray(true_ids, dtype=np.int64)] = True
    positions = np.flatnonzero(truth[order]) + 1
    return float(np.mean(np.arange(1, positions.shape[0] + 1) / positions))


def metrics_at_size(
    scores: np.ndarray,
    size: int,
    known: Sequence[int],
    union_ids: Sequence[int],
    source_count: int,
    threshold: float,
) -> Dict:
    current = scores[:size]
    known_here = sorted(x for x in known if x < size)
    union_here = sorted(x for x in union_ids if x < size)
    order = np.argsort(current, kind="stable")
    top_source = set(int(x) for x in order[:source_count])
    top_known = set(int(x) for x in order[: max(1, len(known_here))])
    accepted = np.flatnonzero(current <= threshold)
    known_set = set(known_here)
    union_set = set(union_here)
    accepted_set = set(int(x) for x in accepted)
    true_accepted = len(known_set & accepted_set)
    false_accepted = len(accepted_set - known_set)
    union_true_accepted = len(union_set & accepted_set)
    union_false_accepted = len(accepted_set - union_set)
    false_denominator = max(1, size - len(known_here))
    ranks = {str(index): int(np.flatnonzero(order == index)[0] + 1) for index in known_here}
    return {
        "bank_size": size,
        "known_count": len(known_here),
        "candidate_union_count": len(union_here),
        "top_candidates": [
            {
                "id": int(index),
                "score": float(current[index]),
                "is_known": int(index) in known_set,
                "is_union": int(index) in union_set,
            }
            for index in order[: min(20, size)]
        ],
        "recall_at_source_count": len(known_set & top_source) / max(1, len(known_here)),
        "recall_at_known_count": len(known_set & top_known) / max(1, len(known_here)),
        "candidate_union_recall_at_union_count": (
            len(set(union_here) & set(int(x) for x in order[: max(1, len(union_here))]))
            / max(1, len(union_here))
        ),
        "average_precision": average_precision(current, known_here),
        "threshold": threshold,
        "accepted_count": int(accepted.shape[0]),
        "true_accepted": true_accepted,
        "false_accepted": false_accepted,
        "threshold_recall": true_accepted / max(1, len(known_here)),
        "threshold_precision": true_accepted / max(1, accepted.shape[0]),
        "candidate_false_positive_rate": false_accepted / false_denominator,
        "trial_false_positive": bool(false_accepted > 0),
        "union_true_accepted": union_true_accepted,
        "union_false_accepted": union_false_accepted,
        "union_threshold_recall": union_true_accepted / max(1, len(union_here)),
        "union_threshold_precision": union_true_accepted / max(1, accepted.shape[0]),
        "union_candidate_false_positive_rate": union_false_accepted / max(1, size - len(union_here)),
        "union_trial_false_positive": bool(union_false_accepted > 0),
        "known_ranks": ranks,
        "min_score": float(np.min(current)),
        "min_known_score": float(np.min(current[known_here])) if known_here else None,
        "max_known_score": float(np.max(current[known_here])) if known_here else None,
    }


def load_existing(path: Path) -> Tuple[List[Dict], set]:
    rows: List[Dict] = []
    keys = set()
    if not path.exists():
        return rows, keys
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        rows.append(row)
        keys.add((row["source_model"], row["layer"], row["condition"], row["split"], row["trial"]))
    return rows, keys


def summarize(rows: List[Dict], metadata: Dict) -> Dict:
    groups: Dict[Tuple, List[Dict]] = defaultdict(list)
    for row in rows:
        if row["split"] == "calibration":
            continue
        for metric in row["metrics"]:
            key = (
                row["source_model"], row["layer"], row["condition"],
                row["split"], metric["bank_size"],
            )
            groups[key].append(metric)
    summary_rows = []
    numeric_fields = [
        "recall_at_source_count", "recall_at_known_count",
        "candidate_union_recall_at_union_count", "average_precision",
        "accepted_count", "threshold_recall", "threshold_precision",
        "candidate_false_positive_rate", "trial_false_positive",
        "union_threshold_recall", "union_threshold_precision",
        "union_candidate_false_positive_rate", "union_trial_false_positive",
    ]
    for key, values in sorted(groups.items()):
        record = {
            "source_model": key[0], "layer": key[1], "condition": key[2],
            "split": key[3], "bank_size": key[4], "trials": len(values),
        }
        for field in numeric_fields:
            present = [float(value[field]) for value in values if value.get(field) is not None]
            record["mean_" + field] = float(np.mean(present)) if present else None
        summary_rows.append(record)
    return {"metadata": metadata, "summary": summary_rows, "raw_records": len(rows)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", required=True)
    parser.add_argument("--public-cache", required=True)
    parser.add_argument("--private-cache")
    parser.add_argument("--official-repo", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--layers", type=int, nargs="+", default=[4, 8, 12])
    parser.add_argument("--source-models", nargs="+", choices=["public", "private"], default=["public", "private"])
    parser.add_argument("--conditions", nargs="+", choices=sorted(CONDITIONS), default=list(CONDITIONS))
    parser.add_argument("--bank-sizes", type=int, nargs="+", default=[10_000, 50_000, 100_000])
    parser.add_argument("--sampled-rows", type=int, default=16)
    parser.add_argument("--calibration-trials", type=int, default=20)
    parser.add_argument("--evaluation-trials", type=int, default=50)
    parser.add_argument("--chunk-size", type=int, default=128)
    parser.add_argument("--score-dtype", choices=["float32", "float32_stable", "float64"], default="float64")
    parser.add_argument("--control-trials", type=int, default=5)
    parser.add_argument("--threads", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260805)
    args = parser.parse_args()

    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    trial_manifest = json.loads(Path(args.trials).read_text(encoding="utf-8"))
    public_store = HiddenStore(args.public_cache)
    stores = {"public": public_store}
    if "private" in args.source_models:
        if not args.private_cache:
            raise ValueError("--private-cache is required")
        stores["private"] = HiddenStore(args.private_cache)
    max_bank = max(args.bank_sizes)
    if max_bank > public_store.shape[0]:
        raise ValueError("candidate bank exceeds public cache")

    official_repo = Path(args.official_repo).resolve()
    sys.path.insert(0, str(official_repo / "gelo"))
    official = importlib.import_module("train_learned_attack")
    official_commit = subprocess.run(
        ["git", "-C", str(official_repo), "rev-parse", "HEAD"],
        check=True, capture_output=True, text=True,
    ).stdout.strip()
    official_code = official_repo / "gelo" / "train_learned_attack.py"
    official_sha = hashlib.sha256(official_code.read_bytes()).hexdigest()

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    raw_path = output / "trial_records.jsonl"
    rows, completed = load_existing(raw_path)
    source_count = int(trial_manifest["source_count"])
    metadata = {
        "attack": "training-free row-space candidate containment",
        "dataset": "MS MARCO Passage Ranking collection",
        "trials_manifest": str(Path(args.trials).resolve()),
        "public_cache": str(Path(args.public_cache).resolve()),
        "private_cache": str(Path(args.private_cache).resolve()) if args.private_cache else None,
        "official_repository": "https://github.com/noskill/gelo",
        "official_commit": official_commit,
        "official_batch_code_sha256": official_sha,
        "official_functions_reused": ["random_orthogonal", "random_invertible_with_condition"],
        "conditions": {name: CONDITIONS[name] for name in args.conditions},
        "bank_sizes": args.bank_sizes,
        "sampled_rows": args.sampled_rows,
        "source_count": source_count,
        "score_dtype": args.score_dtype,
        "chunk_size": args.chunk_size,
        "calibration_trials": args.calibration_trials,
        "evaluation_trials": args.evaluation_trials,
        "control_trials": args.control_trials,
        "seed": args.seed,
        "hostname": socket.gethostname(),
        "numpy_version": np.__version__,
        "torch_version": torch.__version__,
    }

    with raw_path.open("a", encoding="utf-8") as raw:
        for source_model in args.source_models:
            source_store = stores[source_model]
            for layer in args.layers:
                candidate_bank = public_store.layers[layer]
                for condition_index, condition_name in enumerate(args.conditions):
                    condition = CONDITIONS[condition_name]
                    calibration_min: Dict[int, List[float]] = {size: [] for size in args.bank_sizes}
                    for existing in rows:
                        if (
                            existing["source_model"] == source_model
                            and existing["layer"] == layer
                            and existing["condition"] == condition_name
                            and existing["split"] == "calibration"
                        ):
                            for size, value in existing["min_scores"].items():
                                calibration_min[int(size)].append(float(value))
                    calibration_records = trial_manifest["records"]["calibration"][: args.calibration_trials]
                    for record in calibration_records:
                        key = (source_model, layer, condition_name, "calibration", int(record["trial"]))
                        if key in completed:
                            continue
                        seed = args.seed + 10_000_000 * (1 if source_model == "private" else 0) + 100_000 * layer + 1000 * condition_index + int(record["trial"])
                        basis, known, union_ids, diagnostics = make_observation(
                            record, source_store, public_store, layer, condition,
                            args.sampled_rows, max_bank, seed, official,
                        )
                        scores, scoring_seconds, peak_rss = score_bank(
                            candidate_bank, max_bank, basis, args.sampled_rows,
                            args.chunk_size, args.score_dtype,
                        )
                        min_scores = {str(size): float(np.min(scores[:size])) for size in args.bank_sizes}
                        for size in args.bank_sizes:
                            calibration_min[size].append(min_scores[str(size)])
                        row = {
                            "source_model": source_model, "layer": layer,
                            "condition": condition_name, "split": "calibration",
                            "trial": int(record["trial"]), "min_scores": min_scores,
                            "diagnostics": diagnostics, "scoring_seconds": scoring_seconds,
                            "peak_rss_bytes": int(peak_rss), "seed": seed,
                        }
                        raw.write(json.dumps(row, sort_keys=True) + "\n")
                        raw.flush()
                        rows.append(row)
                        completed.add(key)
                        print(json.dumps({k: v for k, v in row.items() if k != "diagnostics"}, sort_keys=True), flush=True)
                    thresholds: Dict[int, float] = {}
                    for size in args.bank_sizes:
                        values = sorted(calibration_min[size])
                        if len(values) < args.calibration_trials:
                            raise RuntimeError("incomplete calibration records")
                        index = max(0, int(math.floor(0.05 * len(values))) - 1)
                        thresholds[size] = float(values[index])

                    for split in ["closed", "partial", "open"]:
                        for record in trial_manifest["records"][split][: args.evaluation_trials]:
                            key = (source_model, layer, condition_name, split, int(record["trial"]))
                            if key in completed:
                                continue
                            split_index = {"closed": 1, "partial": 2, "open": 3}[split]
                            seed = args.seed + 10_000_000 * (1 if source_model == "private" else 0) + 100_000 * layer + 1000 * condition_index + 100 * split_index + int(record["trial"])
                            basis, known, union_ids, diagnostics = make_observation(
                                record, source_store, public_store, layer, condition,
                                args.sampled_rows, max_bank, seed, official,
                            )
                            scores, scoring_seconds, peak_rss = score_bank(
                                candidate_bank, max_bank, basis, args.sampled_rows,
                                args.chunk_size, args.score_dtype,
                            )
                            metrics = [
                                metrics_at_size(
                                    scores, size, known, union_ids, source_count, thresholds[size]
                                )
                                for size in args.bank_sizes
                            ]
                            control_metrics = None
                            if split == "closed" and int(record["trial"]) < args.control_trials:
                                rng = np.random.default_rng(seed + 7919)
                                control_space = random_basis(basis.shape[0], basis.shape[1], rng)
                                control_scores, control_seconds, _ = score_bank(
                                    candidate_bank, max_bank, control_space, args.sampled_rows,
                                    args.chunk_size, args.score_dtype,
                                )
                                control_metrics = [
                                    metrics_at_size(
                                        control_scores, size, known, union_ids, source_count, thresholds[size]
                                    )
                                    for size in args.bank_sizes
                                ]
                            row = {
                                "source_model": source_model, "layer": layer,
                                "condition": condition_name, "split": split,
                                "trial": int(record["trial"]), "metrics": metrics,
                                "control_metrics": control_metrics, "diagnostics": diagnostics,
                                "scoring_seconds": scoring_seconds,
                                "peak_rss_bytes": int(peak_rss), "seed": seed,
                            }
                            raw.write(json.dumps(row, sort_keys=True) + "\n")
                            raw.flush()
                            rows.append(row)
                            completed.add(key)
                            print(
                                json.dumps(
                                    {
                                        "source_model": source_model, "layer": layer,
                                        "condition": condition_name, "split": split,
                                        "trial": int(record["trial"]),
                                        "recall100k": metrics[-1]["recall_at_source_count"],
                                        "ap100k": metrics[-1]["average_precision"],
                                        "fpr100k": metrics[-1]["candidate_false_positive_rate"],
                                        "seconds": scoring_seconds,
                                    },
                                    sort_keys=True,
                                ),
                                flush=True,
                            )
                    atomic_json(output / "summary.json", summarize(rows, metadata))
    metadata["trial_records_sha256"] = sha256_file(raw_path)
    atomic_json(output / "summary.json", summarize(rows, metadata))
    print("completed evaluation {}".format(output), flush=True)


if __name__ == "__main__":
    main()

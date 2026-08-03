#!/usr/bin/env python3
"""Run scalable chosen-prompt codebook recovery against official KV-Cloak."""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import math
import os
import random
import socket
import subprocess
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import psutil
import torch
from transformers.cache_utils import DynamicCache


PINNED_COMMIT = "6b40f36edb2f337557543e7e60b10022308883d4"
ALLOWED_CONDITIONS = [
    "official_reuse", "refresh_left", "refresh_a_row",
    "refresh_a_vector", "refresh_m", "fresh_all",
]


def atomic_json(path: Path, value: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(str(temporary), str(path))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_official(repo: Path):
    source = repo / "defense" / "core" / "kvcloak.py"
    spec = importlib.util.spec_from_file_location("official_kvcloak", source)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot import official KV-Cloak implementation")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, source


class PlainCache:
    def __init__(self, manifest_path: str):
        self.manifest_path = Path(manifest_path).resolve()
        self.manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        self.shape = tuple(int(value) for value in self.manifest["shape"])
        self.maps = {
            name: np.memmap(path, mode="r", dtype=np.float16, shape=self.shape)
            for name, path in self.manifest["files"].items()
        }
        indices_path = self.manifest.get("source_indices")
        if indices_path:
            indices = np.load(indices_path, allow_pickle=False).astype(np.int64)
            self.lookup = {int(value): offset for offset, value in enumerate(indices)}
        else:
            self.lookup = None

    def rows(self, name: str, global_indices: Sequence[int], head_position: int, block_size: int) -> np.ndarray:
        if self.lookup is None:
            local = np.asarray(global_indices, dtype=np.int64)
        else:
            local = np.asarray([self.lookup[int(value)] for value in global_indices], dtype=np.int64)
        return np.asarray(
            self.maps[name][local, head_position, :block_size, :], dtype=np.float32
        )


def storage_dtype(precision: str) -> torch.dtype:
    return {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[precision]


def quantize(value: torch.Tensor, precision: str) -> torch.Tensor:
    return value.detach().cpu().to(storage_dtype(precision)).to(torch.float32)


def protect_batch(cloak, key: np.ndarray, value: np.ndarray, precision: str) -> Tuple[torch.Tensor, torch.Tensor]:
    cache = DynamicCache.from_legacy_cache(
        past_key_values=((torch.from_numpy(key)[:, None], torch.from_numpy(value)[:, None]),)
    )
    protected = cloak.obfuscate(cache)
    return quantize(protected[0][0][:, 0], precision), quantize(protected[0][1][:, 0], precision)


def orthonormal_basis(observed: torch.Tensor) -> torch.Tensor:
    q, _ = torch.linalg.qr(observed.transpose(-1, -2), mode="reduced")
    return q


def projector_features(basis: torch.Tensor, pairs: np.ndarray) -> torch.Tensor:
    diagonal = torch.sum(basis * basis, dim=-1)
    left = basis[:, torch.from_numpy(pairs[:, 0]).long(), :]
    right = basis[:, torch.from_numpy(pairs[:, 1]).long(), :]
    off_diagonal = torch.sum(left * right, dim=-1)
    return torch.cat([diagonal, off_diagonal], dim=-1)


def select_pairs(dimension: int, count: int, seed: int) -> np.ndarray:
    all_pairs = np.asarray(
        [(left, right) for left in range(dimension) for right in range(left + 1, dimension)],
        dtype=np.int64,
    )
    if count >= all_pairs.shape[0]:
        return all_pairs
    rng = np.random.default_rng(seed)
    chosen = rng.choice(all_pairs.shape[0], size=count, replace=False)
    return all_pairs[np.sort(chosen)]


def make_base_cloak(official, cache: PlainCache, block_size: int, precision: str, seed: int):
    torch.manual_seed(seed)
    theta = [
        float(cache.manifest["max_abs"]["key"]) * 2.0,
        float(cache.manifest["max_abs"]["value"]) * 2.0,
    ]
    dimension = int(cache.manifest["head_dim"])
    config = official.create_test_kv_config(
        1, 1, theta, 1.0, 1.0, block_size, dimension, "cpu", storage_dtype(precision)
    )
    cloak = official.KVCloak(
        config, storage_dtype(precision), fused=False, need_ratio=False, add_a=True
    )
    cloak._prepare_device_tensors(torch.device("cpu"))
    return config, cloak, theta


def make_variant_cloak(official, base_config, base_cloak, condition: str, seed: int):
    if condition == "official_reuse":
        return base_cloak
    torch.manual_seed(seed)
    config = copy.deepcopy(base_config)
    block_size = int(config[0][0][0]["S"].shape[0])
    for kv_index in [0, 1]:
        entry = config[0][0][kv_index]
        if condition in ["refresh_left", "fresh_all"]:
            entry["S"] = official.random_orthogonal_matrix(block_size)
        if condition in ["refresh_m", "fresh_all"]:
            entry["M_angles"] = torch.rand_like(entry["M_angles"]) * 2 * torch.pi
        if condition in ["refresh_a_vector", "fresh_all"]:
            threshold = float(entry["theta"]) * float(entry["M_ratio"]) * 1.42
            entry["a"] = (torch.rand_like(entry["a"]) + 3.0) * threshold
    cloak = official.KVCloak(
        config, base_cloak.dtype, fused=False, need_ratio=False, add_a=True
    )
    cloak._prepare_device_tensors(torch.device("cpu"))
    base = base_cloak._device_cache[torch.device("cpu")]["obf"][0]
    current = cloak._device_cache[torch.device("cpu")]["obf"][0]
    if condition in ["refresh_left", "refresh_m"]:
        current["A_k_batch"].copy_(base["A_k_batch"])
        current["A_v_batch"].copy_(base["A_v_batch"])
    elif condition == "refresh_a_row":
        for name, kv_index in [("A_k_batch", 0), ("A_v_batch", 1)]:
            target = current[name]
            target.zero_()
            base_rows = torch.sum(torch.abs(base[name][0, 0, 0]), dim=-1)
            base_row = int(torch.argmax(base_rows).item())
            new_row = (base_row + 1 + seed % max(1, block_size - 1)) % block_size
            target[0, 0, 0, new_row] = config[0][0][kv_index]["a"].to(target.dtype)
    return cloak


def enroll_codebook(
    cache: PlainCache,
    cloak,
    candidate_indices: np.ndarray,
    head_position: int,
    block_size: int,
    precision: str,
    pairs: np.ndarray,
    output: Path,
    batch_size: int,
) -> Dict:
    codebook = output / "codebook"
    codebook.mkdir(parents=True, exist_ok=True)
    dimension = int(cache.manifest["head_dim"])
    feature_dim = dimension + pairs.shape[0]
    shape_basis = (candidate_indices.shape[0], dimension, block_size)
    shape_features = (candidate_indices.shape[0], feature_dim)
    paths = {
        "key_basis": codebook / "key_basis.f32",
        "value_basis": codebook / "value_basis.f32",
        "key_features": codebook / "key_features.f32",
        "value_features": codebook / "value_features.f32",
    }
    manifest_path = codebook / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (
            manifest["candidate_count"] != int(candidate_indices.shape[0])
            or manifest["block_size"] != block_size
            or manifest["precision"] != precision
            or manifest["head_position"] != head_position
        ):
            raise RuntimeError("existing codebook metadata does not match request")
        return manifest

    maps = {
        "key_basis": np.memmap(paths["key_basis"], mode="r+" if paths["key_basis"].exists() else "w+", dtype=np.float32, shape=shape_basis),
        "value_basis": np.memmap(paths["value_basis"], mode="r+" if paths["value_basis"].exists() else "w+", dtype=np.float32, shape=shape_basis),
        "key_features": np.memmap(paths["key_features"], mode="r+" if paths["key_features"].exists() else "w+", dtype=np.float32, shape=shape_features),
        "value_features": np.memmap(paths["value_features"], mode="r+" if paths["value_features"].exists() else "w+", dtype=np.float32, shape=shape_features),
    }
    state_path = codebook / "state.json"
    completed = 0
    if state_path.exists():
        state = json.loads(state_path.read_text(encoding="utf-8"))
        completed = int(state["completed"])
    started = time.time()
    process = psutil.Process()
    peak = process.memory_info().rss
    for start in range(completed, candidate_indices.shape[0], batch_size):
        stop = min(start + batch_size, candidate_indices.shape[0])
        current = candidate_indices[start:stop]
        key = cache.rows("key", current, head_position, block_size)
        value = cache.rows("value", current, head_position, block_size)
        protected_key, protected_value = protect_batch(cloak, key, value, precision)
        for name, protected in [("key", protected_key), ("value", protected_value)]:
            basis = orthonormal_basis(protected)
            features = projector_features(basis, pairs)
            maps[name + "_basis"][start:stop] = basis.numpy().astype(np.float32)
            maps[name + "_features"][start:stop] = features.numpy().astype(np.float32)
        peak = max(peak, process.memory_info().rss)
        if stop % 2048 < batch_size or stop == candidate_indices.shape[0]:
            for mapping in maps.values():
                mapping.flush()
            atomic_json(
                state_path,
                {"completed": int(stop), "elapsed_current_run_seconds": time.time() - started, "peak_rss_bytes": int(peak)},
            )
            print("enroll {}/{} elapsed={:.1f}s".format(stop, candidate_indices.shape[0], time.time() - started), flush=True)
    manifest = {
        "status": "complete",
        "candidate_count": int(candidate_indices.shape[0]),
        "block_size": block_size,
        "dimension": dimension,
        "feature_dimension": feature_dim,
        "precision": precision,
        "head_position": head_position,
        "basis_shape": list(shape_basis),
        "feature_shape": list(shape_features),
        "paths": {key: str(value.resolve()) for key, value in paths.items()},
        "sha256": {key: sha256(value) for key, value in paths.items()},
        "pairs_sha256": hashlib.sha256(pairs.tobytes()).hexdigest(),
        "elapsed_current_run_seconds": time.time() - started,
        "peak_rss_bytes": int(peak),
    }
    atomic_json(manifest_path, manifest)
    return manifest


class Searcher:
    def __init__(self, codebook: Dict, kv_type: str, shortlist: int):
        self.count = int(codebook["candidate_count"])
        self.dimension = int(codebook["dimension"])
        self.block_size = int(codebook["block_size"])
        self.feature_dim = int(codebook["feature_dimension"])
        self.features = np.memmap(
            codebook["paths"][kv_type + "_features"], mode="r", dtype=np.float32,
            shape=(self.count, self.feature_dim),
        )
        self.bases = np.memmap(
            codebook["paths"][kv_type + "_basis"], mode="r", dtype=np.float32,
            shape=(self.count, self.dimension, self.block_size),
        )
        self.norms = np.sum(np.asarray(self.features) ** 2, axis=1, dtype=np.float32)
        self.shortlist = shortlist

    def search(self, victim_basis: np.ndarray, victim_features: np.ndarray, bank_size: int) -> Dict:
        started = time.time()
        coarse = self.norms[:bank_size] + float(np.dot(victim_features, victim_features))
        coarse = coarse - 2.0 * (self.features[:bank_size] @ victim_features)
        take = min(self.shortlist, bank_size)
        shortlist = np.argpartition(coarse, take - 1)[:take]
        candidate_basis = np.asarray(self.bases[shortlist], dtype=np.float64)
        victim = np.asarray(victim_basis, dtype=np.float64)
        overlap = np.einsum("ndb,dk->nbk", candidate_basis, victim, optimize=True)
        exact = np.sqrt(
            np.maximum(0.0, self.block_size - np.sum(overlap * overlap, axis=(1, 2)))
            / self.block_size
        )
        order = np.argsort(exact, kind="stable")
        ranked = shortlist[order]
        return {
            "predicted_position": int(ranked[0]),
            "best_exact_chordal": float(exact[order[0]]),
            "second_exact_chordal": float(exact[order[1]]) if take > 1 else None,
            "shortlist": [int(value) for value in ranked[: min(10, take)]],
            "coarse_best_position": int(np.argmin(coarse)),
            "search_seconds": time.time() - started,
        }


def victim_observation(
    cache: PlainCache,
    cloak,
    source_index: int,
    head_position: int,
    block_size: int,
    precision: str,
    kv_type: str,
    pairs: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    key = cache.rows("key", [source_index], head_position, block_size)
    value = cache.rows("value", [source_index], head_position, block_size)
    protected = protect_batch(cloak, key, value, precision)[0 if kv_type == "key" else 1]
    basis = orthonormal_basis(protected)
    features = projector_features(basis, pairs)
    return basis[0].numpy().astype(np.float32), features[0].numpy().astype(np.float32)


def calibration_order(values: Sequence[float], alpha: float) -> float:
    ordered = sorted(float(value) for value in values)
    one_based = max(1, int(math.floor(alpha * (len(ordered) + 1))))
    return ordered[min(len(ordered) - 1, one_based - 1)]


def load_rows(path: Path) -> Tuple[List[Dict], set]:
    rows: List[Dict] = []
    keys = set()
    if not path.exists():
        return rows, keys
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        rows.append(row)
        keys.add((row["kv_type"], row["condition"], row["split"], row.get("bank_size", 0), row["trial"]))
    return rows, keys


def summarize(rows: Iterable[Dict], metadata: Dict) -> Dict:
    groups = defaultdict(list)
    for row in rows:
        if row["split"] == "calibration":
            continue
        if row["split"] == "open":
            for metric in row["metrics"]:
                groups[(row["kv_type"], row["condition"], "open", metric["bank_size"])].append(metric)
        else:
            groups[(row["kv_type"], row["condition"], "closed", row["bank_size"])].append(row)
    output = []
    for key, values in sorted(groups.items()):
        if key[2] == "closed":
            output.append({
                "kv_type": key[0], "condition": key[1], "split": key[2], "bank_size": key[3],
                "trials": len(values),
                "top1_recall": float(np.mean([value["top1_hit"] for value in values])),
                "threshold_recall": float(np.mean([value["accepted_true"] for value in values])),
                "random_label_top1_recall": float(np.mean([value["random_label_hit"] for value in values])),
                "mean_search_seconds": float(np.mean([value["search_seconds"] for value in values])),
                "max_peak_rss_bytes": int(max(value["peak_rss_bytes"] for value in values)),
            })
        else:
            false_trials = sum(bool(value["trial_false_positive"]) for value in values)
            output.append({
                "kv_type": key[0], "condition": key[1], "split": key[2], "bank_size": key[3],
                "trials": len(values),
                "trial_false_positive_count": false_trials,
                "trial_false_positive_rate": false_trials / max(1, len(values)),
                "mean_best_exact_chordal": float(np.mean([value["best_exact_chordal"] for value in values])),
                "mean_search_seconds": float(np.mean([value["search_seconds"] for value in values])),
            })
    return {"metadata": metadata, "summary": output, "raw_records": len(list(rows)) if not isinstance(rows, list) else len(rows)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-manifest", required=True)
    parser.add_argument("--trials", required=True)
    parser.add_argument("--official-repo", required=True)
    parser.add_argument("--repo-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--head-position", type=int, default=0)
    parser.add_argument("--kv-types", nargs="+", choices=["key", "value"], default=["key", "value"])
    parser.add_argument("--conditions", nargs="+", choices=ALLOWED_CONDITIONS, default=["official_reuse"])
    parser.add_argument("--precision", choices=["float32", "float16", "bfloat16"], default="bfloat16")
    parser.add_argument("--sketch-pairs", type=int, default=512)
    parser.add_argument("--shortlist", type=int, default=64)
    parser.add_argument("--enrollment-batch-size", type=int, default=256)
    parser.add_argument("--calibration-alpha", type=float, default=0.01)
    parser.add_argument("--threads", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260814)
    args = parser.parse_args()

    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    cache = PlainCache(args.cache_manifest)
    trial_manifest = json.loads(Path(args.trials).read_text(encoding="utf-8"))
    candidate_indices = np.load(trial_manifest["candidate_indices"], allow_pickle=False).astype(np.int64)
    block_size = int(trial_manifest["block_size"])
    bank_sizes = [int(value) for value in trial_manifest["bank_sizes"]]
    if max(bank_sizes) != candidate_indices.shape[0]:
        raise ValueError("candidate index count must equal the maximum bank size")
    dimension = int(cache.manifest["head_dim"])
    pairs = select_pairs(dimension, args.sketch_pairs, args.seed + 1)
    official_repo = Path(args.official_repo).resolve()
    official, official_source = load_official(official_repo)
    official_commit = subprocess.check_output(
        ["git", "-C", str(official_repo), "rev-parse", "HEAD"], text=True
    ).strip()
    if official_commit != PINNED_COMMIT:
        raise RuntimeError("official KV-Cloak repository is not pinned")
    base_config, base_cloak, theta = make_base_cloak(
        official, cache, block_size, args.precision, args.seed + 2
    )
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    codebook = enroll_codebook(
        cache, base_cloak, candidate_indices, args.head_position, block_size,
        args.precision, pairs, output, args.enrollment_batch_size,
    )
    raw_path = output / "trial_records.jsonl"
    rows, completed = load_rows(raw_path)
    process = psutil.Process()
    searchers = {kv_type: Searcher(codebook, kv_type, args.shortlist) for kv_type in args.kv_types}
    random_labels = {}
    for size in bank_sizes:
        rng = np.random.default_rng(args.seed + 1000 + size)
        random_labels[size] = rng.permutation(size)

    metadata = {
        "attack": "chosen-prompt protected-block codebook via row-space projector",
        "threat_model": "attacker can observe protected KV blocks and enroll chosen prompts under a fixed deployed KV-Cloak configuration",
        "cache_manifest": str(Path(args.cache_manifest).resolve()),
        "trials": str(Path(args.trials).resolve()),
        "official_repository": "https://github.com/SiO-2/kvcloak",
        "official_commit": official_commit,
        "official_source_sha256": sha256(official_source),
        "repo_revision": subprocess.check_output(
            ["git", "-C", str(Path(args.repo_root).resolve()), "rev-parse", "HEAD"], text=True
        ).strip(),
        "block_size": block_size,
        "dimension": dimension,
        "head_position": args.head_position,
        "model_head": cache.manifest["heads"][args.head_position],
        "layer": cache.manifest["layer"],
        "kv_types": args.kv_types,
        "conditions": args.conditions,
        "precision": args.precision,
        "bank_sizes": bank_sizes,
        "sketch_pairs": int(pairs.shape[0]),
        "shortlist": args.shortlist,
        "calibration_alpha": args.calibration_alpha,
        "theta": theta,
        "seed": args.seed,
        "hostname": socket.gethostname(),
        "torch_version": torch.__version__,
        "numpy_version": np.__version__,
    }

    with raw_path.open("a", encoding="utf-8") as raw:
        for condition_index, condition in enumerate(args.conditions):
            for kv_index, kv_type in enumerate(args.kv_types):
                searcher = searchers[kv_type]
                calibration_scores = {size: [] for size in bank_sizes}
                for existing in rows:
                    if existing["kv_type"] == kv_type and existing["condition"] == condition and existing["split"] == "calibration":
                        for metric in existing["metrics"]:
                            calibration_scores[int(metric["bank_size"])].append(float(metric["best_exact_chordal"]))
                for record in trial_manifest["records"]["calibration"]:
                    key = (kv_type, condition, "calibration", 0, int(record["trial"]))
                    if key in completed:
                        continue
                    seed = args.seed + 10_000_000 * condition_index + 100_000 * kv_index + int(record["trial"])
                    cloak = make_variant_cloak(official, base_config, base_cloak, condition, seed)
                    victim_basis, victim_features = victim_observation(
                        cache, cloak, int(record["source_index"]), args.head_position,
                        block_size, args.precision, kv_type, pairs,
                    )
                    metrics = []
                    for size in bank_sizes:
                        result = searcher.search(victim_basis, victim_features, size)
                        result["bank_size"] = size
                        metrics.append(result)
                        calibration_scores[size].append(result["best_exact_chordal"])
                    row = {"kv_type": kv_type, "condition": condition, "split": "calibration", "trial": int(record["trial"]), "source_index": int(record["source_index"]), "metrics": metrics, "seed": seed}
                    raw.write(json.dumps(row, sort_keys=True) + "\n"); raw.flush()
                    rows.append(row); completed.add(key)
                    print(json.dumps({"kv_type": kv_type, "condition": condition, "split": "calibration", "trial": record["trial"]}), flush=True)
                thresholds = {}
                for size in bank_sizes:
                    expected = int(trial_manifest["calibration_trials"])
                    if len(calibration_scores[size]) != expected:
                        raise RuntimeError("incomplete calibration records")
                    thresholds[size] = calibration_order(calibration_scores[size], args.calibration_alpha)

                for size in bank_sizes:
                    for record in trial_manifest["records"]["closed"][str(size)]:
                        key = (kv_type, condition, "closed", size, int(record["trial"]))
                        if key in completed:
                            continue
                        seed = args.seed + 10_000_000 * condition_index + 100_000 * kv_index + 10_000 + size + int(record["trial"])
                        cloak = make_variant_cloak(official, base_config, base_cloak, condition, seed)
                        victim_basis, victim_features = victim_observation(
                            cache, cloak, int(record["source_index"]), args.head_position,
                            block_size, args.precision, kv_type, pairs,
                        )
                        result = searcher.search(victim_basis, victim_features, size)
                        true_position = int(record["candidate_position"])
                        top1 = result["predicted_position"] == true_position
                        accepted = result["best_exact_chordal"] <= thresholds[size]
                        row = {
                            "kv_type": kv_type, "condition": condition, "split": "closed",
                            "bank_size": size, "trial": int(record["trial"]),
                            "source_index": int(record["source_index"]), "true_position": true_position,
                            "predicted_position": result["predicted_position"], "top1_hit": top1,
                            "accepted": accepted, "accepted_true": bool(accepted and top1),
                            "random_label_hit": bool(random_labels[size][result["predicted_position"]] == true_position),
                            "best_exact_chordal": result["best_exact_chordal"],
                            "second_exact_chordal": result["second_exact_chordal"],
                            "true_in_shortlist": true_position in result["shortlist"],
                            "coarse_best_position": result["coarse_best_position"],
                            "threshold": thresholds[size], "search_seconds": result["search_seconds"],
                            "peak_rss_bytes": int(process.memory_info().rss), "seed": seed,
                        }
                        raw.write(json.dumps(row, sort_keys=True) + "\n"); raw.flush()
                        rows.append(row); completed.add(key)
                        print(json.dumps({"kv_type": kv_type, "condition": condition, "split": "closed", "bank_size": size, "trial": record["trial"], "hit": top1}), flush=True)

                for record in trial_manifest["records"]["open"]:
                    key = (kv_type, condition, "open", 0, int(record["trial"]))
                    if key in completed:
                        continue
                    seed = args.seed + 10_000_000 * condition_index + 100_000 * kv_index + 20_000 + int(record["trial"])
                    cloak = make_variant_cloak(official, base_config, base_cloak, condition, seed)
                    victim_basis, victim_features = victim_observation(
                        cache, cloak, int(record["source_index"]), args.head_position,
                        block_size, args.precision, kv_type, pairs,
                    )
                    metrics = []
                    for size in bank_sizes:
                        result = searcher.search(victim_basis, victim_features, size)
                        result.update({
                            "bank_size": size, "threshold": thresholds[size],
                            "trial_false_positive": result["best_exact_chordal"] <= thresholds[size],
                        })
                        metrics.append(result)
                    row = {"kv_type": kv_type, "condition": condition, "split": "open", "trial": int(record["trial"]), "source_index": int(record["source_index"]), "metrics": metrics, "seed": seed}
                    raw.write(json.dumps(row, sort_keys=True) + "\n"); raw.flush()
                    rows.append(row); completed.add(key)
                    print(json.dumps({"kv_type": kv_type, "condition": condition, "split": "open", "trial": record["trial"], "fp100k": metrics[-1]["trial_false_positive"]}), flush=True)
                atomic_json(output / "summary.json", summarize(rows, metadata))
    metadata["trial_records_sha256"] = sha256(raw_path)
    atomic_json(output / "summary.json", summarize(rows, metadata))
    print("completed KV-Cloak evaluation {}".format(output), flush=True)


if __name__ == "__main__":
    main()

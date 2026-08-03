#!/usr/bin/env python3
"""Audit the full-row-rank premise behind the Qwen KV-Cloak boundary."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import subprocess
import time
from pathlib import Path

import numpy as np
import torch


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_module(path: Path):
    spec = importlib.util.spec_from_file_location("kv_eval", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load evaluator")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def atomic_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def summarize(matrices: torch.Tensor) -> dict:
    values = matrices.detach().cpu().numpy().astype(np.float64)
    singular = np.linalg.svd(values, compute_uv=False)
    maximum = singular[:, 0]
    minimum = singular[:, -1]
    dimension = values.shape[-1]
    exact_tolerance = maximum * dimension * np.finfo(np.float64).eps
    stable_tolerance = maximum * dimension * np.finfo(np.float32).eps
    exact_rank = np.sum(singular > exact_tolerance[:, None], axis=1)
    stable_rank = np.sum(singular > stable_tolerance[:, None], axis=1)

    q, _ = np.linalg.qr(np.swapaxes(values, -1, -2), mode="reduced")
    projectors = q @ np.swapaxes(q, -1, -2)
    identity = np.eye(dimension, dtype=np.float64)[None]
    projector_error = np.linalg.norm(projectors - identity, axis=(1, 2))
    return {
        "sample_count": int(values.shape[0]),
        "dimension": int(dimension),
        "float64_default_rank_min": int(exact_rank.min()),
        "float64_default_rank_max": int(exact_rank.max()),
        "float32_stable_rank_min": int(stable_rank.min()),
        "float32_stable_rank_max": int(stable_rank.max()),
        "smallest_singular_value_min": float(minimum.min()),
        "smallest_singular_value_median": float(np.median(minimum)),
        "condition_number_max": float(np.max(maximum / minimum)),
        "condition_number_median": float(np.median(maximum / minimum)),
        "projector_frobenius_error_from_identity_max": float(projector_error.max()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", required=True)
    parser.add_argument("--cache-manifest", required=True)
    parser.add_argument("--trials", required=True)
    parser.add_argument("--official-repo", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--evaluation-seed", type=int, default=20261328)
    parser.add_argument("--sample-count", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=10)
    args = parser.parse_args()

    started = time.time()
    repo_root = Path(args.repo_root).resolve()
    evaluator_path = repo_root / "scripts" / "86_kvcloak_codebook_evaluate.py"
    evaluator = load_module(evaluator_path)
    cache_manifest = Path(args.cache_manifest).resolve()
    trials_path = Path(args.trials).resolve()
    official_repo = Path(args.official_repo).resolve()
    trials = json.loads(trials_path.read_text(encoding="utf-8"))
    candidate_indices_path = Path(trials["candidate_indices"]).resolve()
    candidate_indices = np.load(candidate_indices_path, allow_pickle=False).astype(np.int64)
    offsets = np.linspace(0, candidate_indices.shape[0] - 1, args.sample_count, dtype=np.int64)
    sampled = candidate_indices[offsets]

    cache = evaluator.PlainCache(str(cache_manifest))
    official, official_source = evaluator.load_official(official_repo)
    official_commit = subprocess.check_output(
        ["git", "-C", str(official_repo), "rev-parse", "HEAD"], text=True
    ).strip()
    if official_commit != evaluator.PINNED_COMMIT:
        raise RuntimeError("official repository commit mismatch")
    _, cloak, _ = evaluator.make_base_cloak(
        official, cache, 128, "bfloat16", args.evaluation_seed + 2
    )

    protected = {"key": [], "value": []}
    for start in range(0, sampled.shape[0], args.batch_size):
        current = sampled[start : start + args.batch_size]
        key = cache.rows("key", current, 0, 128)
        value = cache.rows("value", current, 0, 128)
        key_observed, value_observed = evaluator.protect_batch(cloak, key, value, "bfloat16")
        protected["key"].append(key_observed)
        protected["value"].append(value_observed)

    summaries = {
        name: summarize(torch.cat(protected[name], dim=0))
        for name in ["key", "value"]
    }
    result = {
        "status": "complete",
        "purpose": "post-campaign audit of the full-row-rank premise at b=d=128",
        "evaluation_seed": args.evaluation_seed,
        "block_size": 128,
        "head_dimension": int(cache.manifest["head_dim"]),
        "head_position": 0,
        "sample_count": int(sampled.shape[0]),
        "sample_selection": "100 evenly spaced positions in the prespecified 10K candidate index array",
        "sampled_source_indices_sha256": hashlib.sha256(sampled.astype("<i8").tobytes()).hexdigest(),
        "official_commit": official_commit,
        "official_source_sha256": sha256(official_source),
        "input_sha256": {
            str(cache_manifest): sha256(cache_manifest),
            str(trials_path): sha256(trials_path),
            str(candidate_indices_path): sha256(candidate_indices_path),
            str(evaluator_path): sha256(evaluator_path),
        },
        "key": summaries["key"],
        "value": summaries["value"],
        "all_sampled_blocks_float64_full_rank": all(
            summaries[name]["float64_default_rank_min"] == 128
            for name in ["key", "value"]
        ),
        "all_sampled_blocks_float32_stable_full_rank": all(
            summaries[name]["float32_stable_rank_min"] == 128
            for name in ["key", "value"]
        ),
        "elapsed_seconds": time.time() - started,
    }
    if result["head_dimension"] != 128:
        raise RuntimeError("rank audit requires d=128")
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(output, result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

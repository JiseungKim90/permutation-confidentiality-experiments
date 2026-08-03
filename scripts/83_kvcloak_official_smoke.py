#!/usr/bin/env python3
"""Verify KV-Cloak's row-space invariant against the pinned official code."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import subprocess
import time
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import psutil
import torch
from transformers.cache_utils import DynamicCache


PINNED_COMMIT = "6b40f36edb2f337557543e7e60b10022308883d4"


def atomic_json(path: Path, value: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(str(temporary), str(path))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
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


def quantize(value: torch.Tensor, precision: str) -> np.ndarray:
    if precision == "float32":
        return value.detach().cpu().to(torch.float32).numpy().astype(np.float64)
    dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16}[precision]
    return value.detach().cpu().to(dtype).to(torch.float32).numpy().astype(np.float64)


def basis(value: np.ndarray) -> np.ndarray:
    q, r = np.linalg.qr(value.T, mode="reduced")
    rank = int(np.linalg.matrix_rank(r))
    return q[:, :rank]


def chordal(left: np.ndarray, right: np.ndarray) -> float:
    rank = min(left.shape[1], right.shape[1])
    overlap = np.linalg.norm(left.T @ right, ord="fro") ** 2
    return math.sqrt(max(0.0, rank - overlap) / max(1, rank))


def protect(cloak, key: torch.Tensor, value: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    cache = DynamicCache.from_legacy_cache(
        past_key_values=((key[None, None], value[None, None]),)
    )
    protected = cloak.obfuscate(cache)
    return protected[0][0][0, 0], protected[0][1][0, 0]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--official-repo", required=True)
    parser.add_argument("--repo-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--repetitions", type=int, default=32)
    parser.add_argument("--different-blocks", type=int, default=32)
    parser.add_argument("--seed", type=int, default=20260812)
    args = parser.parse_args()

    started = time.time()
    process = psutil.Process()
    peak_rss = process.memory_info().rss
    official_repo = Path(args.official_repo).resolve()
    official, official_source = load_official(official_repo)
    official_commit = subprocess.check_output(
        ["git", "-C", str(official_repo), "rev-parse", "HEAD"], text=True
    ).strip()
    if official_commit != PINNED_COMMIT:
        raise RuntimeError("official repository is not pinned to {}".format(PINNED_COMMIT))

    results = []
    for block_size in [16, 32, 64]:
        torch.manual_seed(args.seed + block_size)
        key = torch.randn(block_size, 64, dtype=torch.float32)
        value = torch.randn(block_size, 64, dtype=torch.float32)
        theta = [float(key.abs().max()) * 2.0, float(value.abs().max()) * 2.0]
        config = official.create_test_kv_config(
            1, 1, theta, 1.0, 1.0, block_size, 64, "cpu", torch.float32
        )
        cloak = official.KVCloak(
            config, torch.float32, fused=False, need_ratio=False, add_a=True
        )
        protected = [protect(cloak, key, value) for _ in range(args.repetitions)]
        alternatives = []
        for _ in range(args.different_blocks):
            alternatives.append(protect(cloak, torch.randn_like(key), torch.randn_like(value)))
        for kv_index, kv_type in enumerate(["key", "value"]):
            for precision in ["float32", "float16", "bfloat16"]:
                reference = basis(quantize(protected[0][kv_index], precision))
                same = [
                    chordal(reference, basis(quantize(row[kv_index], precision)))
                    for row in protected[1:]
                ]
                different = [
                    chordal(reference, basis(quantize(row[kv_index], precision)))
                    for row in alternatives
                ]
                tolerance = 2e-5 if precision == "float32" else 1e-2
                result = {
                    "block_size": block_size,
                    "kv_type": kv_type,
                    "precision": precision,
                    "same_block_max_chordal": max(same),
                    "same_block_mean_chordal": float(np.mean(same)),
                    "different_block_min_chordal": min(different),
                    "different_block_median_chordal": float(np.median(different)),
                    "same_invariant_gate": max(same) <= tolerance,
                    "separation_gate": max(same) * 10.0 < min(different),
                }
                result["pass"] = result["same_invariant_gate"] and result["separation_gate"]
                results.append(result)
        peak_rss = max(peak_rss, process.memory_info().rss)

    completion = {
        "status": "complete",
        "attack_surface": "stable row space of each protected KV block",
        "official_repository": "https://github.com/SiO-2/kvcloak",
        "official_commit": official_commit,
        "official_source": str(official_source),
        "official_source_sha256": sha256(official_source),
        "repo_revision": subprocess.check_output(
            ["git", "-C", str(Path(args.repo_root).resolve()), "rev-parse", "HEAD"], text=True
        ).strip(),
        "repetitions": args.repetitions,
        "different_blocks": args.different_blocks,
        "results": results,
        "all_gates_pass": all(row["pass"] for row in results),
        "elapsed_seconds": time.time() - started,
        "peak_rss_bytes": int(peak_rss),
        "torch_version": torch.__version__,
        "numpy_version": np.__version__,
    }
    atomic_json(Path(args.output), completion)
    print(json.dumps(completion, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

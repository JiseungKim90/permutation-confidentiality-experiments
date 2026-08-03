#!/usr/bin/env python3
"""Prepare disjoint full-bank, calibration, and open-set KV-Cloak trials."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Dict, List

import numpy as np


def atomic_json(path: Path, value: Dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(str(temporary), str(path))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--block-size", type=int, required=True)
    parser.add_argument("--bank-sizes", type=int, nargs="+", default=[10_000, 50_000, 100_000])
    parser.add_argument("--calibration-trials", type=int, default=100)
    parser.add_argument("--closed-trials", type=int, default=100)
    parser.add_argument("--open-trials", type=int, default=300)
    parser.add_argument("--seed", type=int, default=20260813)
    args = parser.parse_args()

    tokens_path = Path(args.tokens).resolve()
    tokens = np.load(tokens_path, mmap_mode="r")
    if tokens.ndim != 2 or tokens.shape[1] < args.block_size:
        raise ValueError("token array is shorter than requested block size")
    if sorted(set(args.bank_sizes)) != sorted(args.bank_sizes):
        raise ValueError("bank sizes must be unique")
    max_bank = max(args.bank_sizes)
    required_absent = args.calibration_trials + args.open_trials

    unique: List[int] = []
    seen = set()
    duplicates = 0
    for index in range(tokens.shape[0]):
        key = np.asarray(tokens[index, : args.block_size], dtype="<i4").tobytes()
        if key in seen:
            duplicates += 1
            continue
        seen.add(key)
        unique.append(index)
    if len(unique) < max_bank + required_absent:
        raise RuntimeError("only {} unique prefixes; need {}".format(len(unique), max_bank + required_absent))

    rng = np.random.default_rng(args.seed)
    shuffled = np.asarray(unique, dtype=np.int64)
    rng.shuffle(shuffled)
    candidates = shuffled[:max_bank]
    calibration_pool = shuffled[max_bank : max_bank + args.calibration_trials]
    open_pool = shuffled[
        max_bank + args.calibration_trials : max_bank + args.calibration_trials + args.open_trials
    ]

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    candidate_path = output / "candidate_indices.npy"
    np.save(candidate_path, candidates, allow_pickle=False)

    closed: Dict[str, List[Dict]] = {}
    placement = {}
    for bank_size in args.bank_sizes:
        offsets = (np.arange(args.closed_trials) + rng.random(args.closed_trials)) / args.closed_trials
        positions = np.minimum((offsets * bank_size).astype(np.int64), bank_size - 1)
        rng.shuffle(positions)
        closed[str(bank_size)] = [
            {
                "trial": trial,
                "bank_size": bank_size,
                "candidate_position": int(position),
                "source_index": int(candidates[position]),
            }
            for trial, position in enumerate(positions)
        ]
        placement[str(bank_size)] = {
            "min": int(positions.min()), "max": int(positions.max()),
            "occupied_deciles": len(set(int(value * 10 // bank_size) for value in positions)),
        }

    records = {
        "calibration": [
            {"trial": trial, "source_index": int(index)}
            for trial, index in enumerate(calibration_pool)
        ],
        "closed": closed,
        "open": [
            {"trial": trial, "source_index": int(index)}
            for trial, index in enumerate(open_pool)
        ],
    }
    manifest = {
        "status": "complete",
        "dataset": "MS MARCO Passage Ranking collection",
        "tokens": str(tokens_path),
        "tokens_sha256": sha256(tokens_path),
        "block_size": args.block_size,
        "unique_prefixes": len(unique),
        "duplicate_prefixes_skipped": duplicates,
        "candidate_indices": str(candidate_path.resolve()),
        "candidate_indices_sha256": sha256(candidate_path),
        "candidate_count": int(candidates.shape[0]),
        "candidate_source_index_min": int(candidates.min()),
        "candidate_source_index_max": int(candidates.max()),
        "bank_sizes": args.bank_sizes,
        "calibration_trials": args.calibration_trials,
        "closed_trials_per_bank": args.closed_trials,
        "open_trials": args.open_trials,
        "candidate_placement": placement,
        "disjoint_pools": True,
        "seed": args.seed,
        "records": records,
    }
    manifest_path = output / "trials.json"
    atomic_json(manifest_path, manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

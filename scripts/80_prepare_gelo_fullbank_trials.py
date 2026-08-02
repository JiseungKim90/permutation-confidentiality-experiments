#!/usr/bin/env python3
"""Create a 100K full-bank placement and open-set validation manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Dict, List

import numpy as np


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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--trials", type=int, default=300)
    parser.add_argument("--source-count", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260810)
    args = parser.parse_args()

    tokens_path = Path(args.tokens).resolve()
    tokens = np.load(str(tokens_path), mmap_mode="r", allow_pickle=False)
    if tokens.shape[0] < 120_000:
        raise ValueError("full-bank validation requires at least 120000 token rows")
    if args.source_count < 2:
        raise ValueError("source-count must be at least two")

    rng = np.random.default_rng(args.seed)
    candidate = np.arange(0, 100_000, dtype=np.int64)
    calibration = np.arange(100_000, 110_000, dtype=np.int64)
    open_set = np.arange(110_000, 120_000, dtype=np.int64)

    def choice(pool: np.ndarray, count: int) -> List[int]:
        return [int(value) for value in rng.choice(pool, size=count, replace=False)]

    records = {name: [] for name in ["calibration", "closed", "partial", "open"]}
    half = args.source_count // 2
    for trial in range(args.trials):
        records["calibration"].append(
            {"trial": trial, "known": [], "unknown": choice(calibration, args.source_count)}
        )
        records["closed"].append(
            {"trial": trial, "known": choice(candidate, args.source_count), "unknown": []}
        )
        records["partial"].append(
            {
                "trial": trial,
                "known": choice(candidate, half),
                "unknown": choice(open_set, args.source_count - half),
            }
        )
        records["open"].append(
            {"trial": trial, "known": [], "unknown": choice(open_set, args.source_count)}
        )

    source_indices = sorted(
        {
            index
            for split_records in records.values()
            for record in split_records
            for index in record["known"] + record["unknown"]
        }
    )
    trials = {
        "seed": args.seed,
        "trials_per_split": args.trials,
        "source_count": args.source_count,
        "candidate_bank_sizes": [100_000],
        "candidate_source_pool": [0, 100_000],
        "calibration_pool": [100_000, 110_000],
        "open_pool": [110_000, 120_000],
        "placement_control": "known positives sampled uniformly from the entire 100K bank",
        "records": records,
    }

    output = Path(args.output_dir).resolve()
    trials_path = output / "trials_fullbank.json"
    indices_path = output / "source_indices_fullbank.npy"
    atomic_json(trials_path, trials)
    np.save(str(indices_path), np.asarray(source_indices, dtype=np.int64), allow_pickle=False)
    atomic_json(
        output / "manifest.json",
        {
            "tokens": str(tokens_path),
            "tokens_sha256": sha256(tokens_path),
            "tokens_shape": list(tokens.shape),
            "seed": args.seed,
            "trials_per_split": args.trials,
            "source_count": args.source_count,
            "candidate_source_pool": [0, 100_000],
            "calibration_pool": [100_000, 110_000],
            "open_pool": [110_000, 120_000],
            "source_index_count": len(source_indices),
            "trials": str(trials_path),
            "trials_sha256": sha256(trials_path),
            "source_indices_sha256": sha256(indices_path),
        },
    )
    print(
        "prepared {} full-bank trials with {} unique source indices".format(
            args.trials, len(source_indices)
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()

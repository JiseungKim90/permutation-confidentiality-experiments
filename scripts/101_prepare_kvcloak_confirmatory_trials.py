#!/usr/bin/env python3
"""Prepare candidate and trial pools disjoint from all metric-development records."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np


FROZEN_METRIC_COMMIT = "f6f76a5c8f8f93c677741c57e479883313e9bef5"


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


def trial_sources(path: Path) -> set[int]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    sources = set(int(value) for value in np.load(payload["candidate_indices"], allow_pickle=False))
    sources.update(int(row["source_index"]) for row in payload["records"]["calibration"])
    sources.update(int(row["source_index"]) for rows in payload["records"]["closed"].values() for row in rows)
    sources.update(int(row["source_index"]) for row in payload["records"]["open"])
    return sources


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", required=True)
    parser.add_argument("--exclude-trials", required=True, nargs="+")
    parser.add_argument("--allowed-indices",
                        help="optional cache-backed source-index array")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--block-size", type=int, required=True)
    parser.add_argument("--bank-size", type=int, default=10_000)
    parser.add_argument("--calibration-trials", type=int, default=100)
    parser.add_argument("--closed-trials", type=int, default=100)
    parser.add_argument("--open-trials", type=int, default=300)
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()

    tokens_path = Path(args.tokens).resolve()
    tokens = np.load(tokens_path, mmap_mode="r")
    if tokens.ndim != 2 or tokens.shape[1] < args.block_size:
        raise RuntimeError("token cache is shorter than the requested block")
    excluded_paths = [Path(value).resolve() for value in args.exclude_trials]
    excluded = set()
    for path in excluded_paths:
        excluded.update(trial_sources(path))
    allowed = None
    allowed_path = None
    if args.allowed_indices:
        allowed_path = Path(args.allowed_indices).resolve()
        allowed = set(int(value) for value in np.load(allowed_path, allow_pickle=False))

    unique = []
    seen = set()
    for index in range(tokens.shape[0]):
        if allowed is not None and index not in allowed:
            continue
        if index in excluded:
            continue
        key = np.asarray(tokens[index, : args.block_size], dtype="<i4").tobytes()
        if key in seen:
            continue
        seen.add(key)
        unique.append(index)
    required = args.bank_size + args.calibration_trials + args.open_trials
    if len(unique) < required:
        raise RuntimeError(f"only {len(unique)} unused unique prefixes; need {required}")

    rng = np.random.default_rng(args.seed)
    shuffled = np.asarray(unique, dtype=np.int64)
    rng.shuffle(shuffled)
    candidates = shuffled[: args.bank_size]
    calibration = shuffled[args.bank_size : args.bank_size + args.calibration_trials]
    opened = shuffled[
        args.bank_size + args.calibration_trials :
        args.bank_size + args.calibration_trials + args.open_trials
    ]
    offsets = (np.arange(args.closed_trials) + rng.random(args.closed_trials)) / args.closed_trials
    positions = np.minimum((offsets * args.bank_size).astype(np.int64), args.bank_size - 1)
    rng.shuffle(positions)

    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    candidate_path = output / "candidate_indices.npy"
    np.save(candidate_path, candidates, allow_pickle=False)
    records = {
        "calibration": [
            {"trial": trial, "source_index": int(index)}
            for trial, index in enumerate(calibration)
        ],
        "closed": {
            str(args.bank_size): [
                {
                    "trial": trial,
                    "bank_size": args.bank_size,
                    "candidate_position": int(position),
                    "source_index": int(candidates[position]),
                }
                for trial, position in enumerate(positions)
            ]
        },
        "open": [
            {"trial": trial, "source_index": int(index)}
            for trial, index in enumerate(opened)
        ],
    }
    used = set(int(value) for value in candidates)
    used.update(int(value) for value in calibration)
    used.update(int(value) for value in opened)
    if used & excluded:
        raise RuntimeError("confirmatory pool overlaps metric-development records")
    manifest = {
        "status": "complete",
        "confirmatory_holdout": True,
        "metric_frozen_before_holdout": True,
        "frozen_metric_commit": FROZEN_METRIC_COMMIT,
        "tokens": str(tokens_path),
        "tokens_sha256": sha256(tokens_path),
        "block_size": args.block_size,
        "bank_sizes": [args.bank_size],
        "candidate_count": args.bank_size,
        "candidate_indices": str(candidate_path),
        "candidate_indices_sha256": sha256(candidate_path),
        "candidate_placement": {
            str(args.bank_size): {
                "min": int(positions.min()),
                "max": int(positions.max()),
                "occupied_deciles": len(set(int(value * 10 // args.bank_size) for value in positions)),
            }
        },
        "calibration_trials": args.calibration_trials,
        "closed_trials_per_bank": args.closed_trials,
        "open_trials": args.open_trials,
        "excluded_source_count": len(excluded),
        "excluded_trial_manifests": {
            str(path): sha256(path) for path in excluded_paths
        },
        "allowed_indices": str(allowed_path) if allowed_path else None,
        "allowed_indices_sha256": sha256(allowed_path) if allowed_path else None,
        "disjoint_from_metric_development": True,
        "seed": args.seed,
        "records": records,
    }
    manifest_path = output / "trials.json"
    atomic_json(manifest_path, manifest)
    print(json.dumps({
        "status": "complete",
        "manifest": str(manifest_path),
        "candidate_sha256": manifest["candidate_indices_sha256"],
        "excluded_source_count": len(excluded),
        "remaining_unique_prefixes": len(unique),
    }, indent=2))


if __name__ == "__main__":
    main()

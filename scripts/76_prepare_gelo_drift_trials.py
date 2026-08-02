#!/usr/bin/env python3
"""Create the paired long-run trial manifest used by the private-drift study."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path

import numpy as np


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(str(temporary), str(path))


def load_prepare(repo_root: Path):
    path = repo_root / "scripts" / "69_gelo_msmarco_prepare.py"
    spec = importlib.util.spec_from_file_location("gelo_prepare", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load {}".format(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", required=True)
    parser.add_argument("--tokens", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--trials", type=int, default=300)
    parser.add_argument("--source-count", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260807)
    args = parser.parse_args()

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    tokens_path = Path(args.tokens).resolve()
    tokens = np.load(str(tokens_path), mmap_mode="r", allow_pickle=False)
    prepare = load_prepare(Path(args.repo_root).resolve())
    trials = prepare.build_trials(
        int(tokens.shape[0]), args.seed, args.trials, args.source_count
    )
    source_indices = np.asarray(trials.pop("source_indices"), dtype=np.int64)
    trials_path = output / "trials_drift.json"
    indices_path = output / "source_indices_drift.npy"
    np.save(str(indices_path), source_indices, allow_pickle=False)
    atomic_json(trials_path, trials)
    atomic_json(
        output / "manifest.json",
        {
            "tokens": str(tokens_path),
            "tokens_sha256": sha256(tokens_path),
            "tokens_shape": list(tokens.shape),
            "seed": args.seed,
            "trials_per_split": args.trials,
            "source_count": args.source_count,
            "source_indices": str(indices_path.resolve()),
            "source_index_count": int(source_indices.shape[0]),
            "trials": str(trials_path.resolve()),
            "trials_sha256": sha256(trials_path),
            "source_indices_sha256": sha256(indices_path),
        },
    )
    print(
        "prepared {} paired trials with {} unique source indices".format(
            args.trials, source_indices.shape[0]
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Collect every candidate and evaluation row needed by KV-Cloak campaigns."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Dict, Set

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
    parser.add_argument("--trials", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    selected: Set[int] = set()
    input_hashes = {}
    summaries = []
    for value in args.trials:
        path = Path(value).resolve()
        payload = json.loads(path.read_text(encoding="utf-8"))
        candidate_path = Path(payload["candidate_indices"]).resolve()
        candidates = np.load(candidate_path, allow_pickle=False).astype(np.int64)
        selected.update(int(index) for index in candidates)
        split_counts = {}
        for split, records in payload["records"].items():
            split_counts[split] = len(records) if not isinstance(records, dict) else sum(len(rows) for rows in records.values())
            iterable = records.values() if isinstance(records, dict) else [records]
            for group in iterable:
                for record in group:
                    selected.add(int(record["source_index"]))
        input_hashes[str(path)] = sha256(path)
        input_hashes[str(candidate_path)] = sha256(candidate_path)
        summaries.append(
            {
                "trials": str(path),
                "block_size": int(payload["block_size"]),
                "candidate_count": int(candidates.shape[0]),
                "split_counts": split_counts,
            }
        )

    ordered = np.asarray(sorted(selected), dtype=np.int64)
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    np.save(output, ordered, allow_pickle=False)
    manifest = {
        "status": "complete",
        "indices": str(output),
        "indices_sha256": sha256(output),
        "index_count": int(ordered.shape[0]),
        "index_min": int(ordered.min()),
        "index_max": int(ordered.max()),
        "strictly_increasing": bool(np.all(ordered[1:] > ordered[:-1])),
        "inputs": summaries,
        "input_sha256": input_hashes,
    }
    atomic_json(output.with_suffix(".manifest.json"), manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

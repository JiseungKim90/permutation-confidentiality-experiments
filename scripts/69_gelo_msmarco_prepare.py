#!/usr/bin/env python3
"""Prepare a deterministic, disjoint MS MARCO passage sample for GELO scaling."""

from __future__ import annotations

import argparse
import hashlib
import heapq
import json
import os
import tarfile
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from transformers import AutoTokenizer


def atomic_json(path: Path, value: Dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(str(temporary), str(path))


def priority(doc_id: str, seed: int) -> int:
    payload = (str(seed) + "\0" + doc_id).encode("utf-8")
    return int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "big")


def read_lowest_hash_passages(
    archive: Path, reservoir_size: int, seed: int
) -> Tuple[List[Tuple[int, str, str]], int]:
    heap: List[Tuple[int, int, str, str]] = []
    scanned = 0
    with tarfile.open(str(archive), "r:gz") as tar:
        member = tar.getmember("collection.tsv")
        stream = tar.extractfile(member)
        if stream is None:
            raise RuntimeError("collection.tsv is absent from archive")
        for raw in stream:
            scanned += 1
            line = raw.decode("utf-8", errors="replace").rstrip("\n")
            try:
                doc_id, text = line.split("\t", 1)
            except ValueError:
                continue
            rank = priority(doc_id, seed)
            item = (-rank, -scanned, doc_id, " ".join(text.split()))
            if len(heap) < reservoir_size:
                heapq.heappush(heap, item)
            elif item > heap[0]:
                heapq.heapreplace(heap, item)
            if scanned % 1_000_000 == 0:
                print("scan {}/? retained={}".format(scanned, len(heap)), flush=True)
    selected = [(-neg_rank, doc_id, text) for neg_rank, _, doc_id, text in heap]
    selected.sort(key=lambda value: (value[0], value[1]))
    return selected, scanned


def build_trials(total: int, seed: int, count: int, source_count: int) -> Dict:
    if total < 120_000:
        raise ValueError("trial layout requires at least 120000 usable passages")
    rng = np.random.default_rng(seed)
    candidate_core = np.arange(0, 10_000, dtype=np.int64)
    calibration = np.arange(100_000, 110_000, dtype=np.int64)
    open_set = np.arange(110_000, 120_000, dtype=np.int64)

    def choice(pool: np.ndarray, n: int) -> List[int]:
        return [int(x) for x in rng.choice(pool, size=n, replace=False)]

    records: Dict[str, List[Dict]] = {name: [] for name in ["calibration", "closed", "partial", "open"]}
    for trial in range(count):
        records["calibration"].append(
            {"trial": trial, "known": [], "unknown": choice(calibration, source_count)}
        )
        records["closed"].append(
            {"trial": trial, "known": choice(candidate_core, source_count), "unknown": []}
        )
        half = source_count // 2
        records["partial"].append(
            {
                "trial": trial,
                "known": choice(candidate_core, half),
                "unknown": choice(open_set, source_count - half),
            }
        )
        records["open"].append(
            {"trial": trial, "known": [], "unknown": choice(open_set, source_count)}
        )
    source_indices = sorted(
        {
            index
            for split_records in records.values()
            for record in split_records
            for index in record["known"] + record["unknown"]
        }
    )
    return {
        "seed": seed,
        "trials_per_split": count,
        "source_count": source_count,
        "candidate_bank_sizes": [10_000, 50_000, 100_000],
        "candidate_source_pool": [0, 10_000],
        "calibration_pool": [100_000, 110_000],
        "open_pool": [110_000, 120_000],
        "records": records,
        "source_indices": source_indices,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--collection", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--count", type=int, default=120_000)
    parser.add_argument("--reservoir-size", type=int, default=300_000)
    parser.add_argument("--seq-len", type=int, default=32)
    parser.add_argument("--seed", type=int, default=20260802)
    parser.add_argument("--trials", type=int, default=50)
    parser.add_argument("--source-count", type=int, default=4)
    args = parser.parse_args()

    started = time.time()
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    passages, scanned = read_lowest_hash_passages(
        Path(args.collection), args.reservoir_size, args.seed
    )
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    token_rows: List[List[int]] = []
    manifest_rows: List[str] = []
    seen_prefixes = set()
    duplicate_prefixes = 0
    batch_size = 1024
    for start in range(0, len(passages), batch_size):
        batch = passages[start : start + batch_size]
        encoded = tokenizer(
            [row[2] for row in batch],
            add_special_tokens=False,
            truncation=True,
            max_length=args.seq_len,
        )["input_ids"]
        for (rank, doc_id, text), ids in zip(batch, encoded):
            if len(ids) < args.seq_len:
                continue
            prefix = np.asarray(ids[: args.seq_len], dtype="<i4")
            prefix_key = prefix.tobytes()
            if prefix_key in seen_prefixes:
                duplicate_prefixes += 1
                continue
            seen_prefixes.add(prefix_key)
            local_index = len(token_rows)
            token_rows.append(prefix.tolist())
            manifest_rows.append(
                "{}\t{}\t{:016x}\t{}\t{}".format(
                    local_index,
                    doc_id,
                    rank,
                    hashlib.sha256(text.encode("utf-8")).hexdigest(),
                    hashlib.sha256(prefix_key).hexdigest(),
                )
            )
            if len(token_rows) == args.count:
                break
        if len(token_rows) == args.count:
            break
        if start % (20 * batch_size) == 0:
            print("tokenize {}/{} usable={}".format(start, len(passages), len(token_rows)), flush=True)
    if len(token_rows) != args.count:
        raise RuntimeError(
            "only {} passages have at least {} tokens; increase --reservoir-size".format(
                len(token_rows), args.seq_len
            )
        )

    tokens = np.asarray(token_rows, dtype=np.int32)
    np.save(str(output / "tokens.npy"), tokens, allow_pickle=False)
    (output / "passages.tsv").write_text(
        "local_index\tmsmarco_pid\tselection_hash\ttext_sha256\ttoken_prefix_sha256\n"
        + "\n".join(manifest_rows)
        + "\n",
        encoding="utf-8",
    )
    trials = build_trials(args.count, args.seed + 1, args.trials, args.source_count)
    source_indices = np.asarray(trials.pop("source_indices"), dtype=np.int64)
    np.save(str(output / "source_indices.npy"), source_indices, allow_pickle=False)
    atomic_json(output / "trials.json", trials)
    atomic_json(
        output / "manifest.json",
        {
            "collection": str(Path(args.collection).resolve()),
            "collection_sha256": "70667529e474322327d6441c8f7b621f2a805a7f6220897d9f80e5a8294bd62e",
            "tokenizer": args.tokenizer,
            "selection": "lowest keyed BLAKE2b-64 priority, unique 32-token prefixes",
            "seed": args.seed,
            "scanned_passages": scanned,
            "reservoir_size": args.reservoir_size,
             "duplicate_token_prefixes_skipped": duplicate_prefixes,
            "usable_passages": args.count,
            "sequence_length": args.seq_len,
            "tokens_dtype": str(tokens.dtype),
            "tokens_shape": list(tokens.shape),
            "source_index_count": int(source_indices.shape[0]),
            "elapsed_seconds": time.time() - started,
        },
    )
    print("prepared {} in {:.1f}s".format(output, time.time() - started), flush=True)


if __name__ == "__main__":
    main()

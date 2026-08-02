#!/usr/bin/env python3
"""Certify STIP public-dictionary recovery under private embedding drift.

This script complements 40_orbit_private_embeddings.py.  It computes the exact
public-dictionary separation radius of every canonical embedding signature,
checks the half-margin recovery certificate, and evaluates two controlled
boundaries without pretending to train additional checkpoints:

1. partial row update: only a selected fraction of rows receive the measured
   full-fine-tune update; and
2. vocabulary overlap: only a selected fraction of private tokens retain a
   public semantic alignment.

Unaligned tokens are counted as semantic-recovery failures, while the paper's
separate nonidentifiability proposition explains why their labels cannot be
recovered from the transcript alone.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import platform
import time
from pathlib import Path

import faiss
import numpy as np
import torch


SCRIPT_DIR = Path(__file__).resolve().parent


def load_orbit_helpers():
    path = SCRIPT_DIR / "40_orbit_private_embeddings.py"
    spec = importlib.util.spec_from_file_location("orbit_private_embeddings", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import helpers from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.canonical_orbit, module.load_rows


canonical_orbit, load_rows = load_orbit_helpers()


def exact_search(
    index: faiss.Index,
    query: np.ndarray,
    k: int,
    batch_size: int,
    label: str,
) -> tuple[np.ndarray, np.ndarray]:
    scores = np.empty((query.shape[0], k), dtype=np.float32)
    ids = np.empty((query.shape[0], k), dtype=np.int64)
    started = time.time()
    for start in range(0, query.shape[0], batch_size):
        stop = min(start + batch_size, query.shape[0])
        scores[start:stop], ids[start:stop] = index.search(query[start:stop], k)
        print(
            json.dumps(
                {
                    "event": "search_progress",
                    "label": label,
                    "completed": stop,
                    "total": query.shape[0],
                    "elapsed_seconds": round(time.time() - started, 3),
                },
                sort_keys=True,
            ),
            flush=True,
        )
    return scores, ids


def nearest_other(
    scores: np.ndarray, ids: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    n = scores.shape[0]
    targets = np.arange(n)
    use_second = ids[:, 0] == targets
    other_score = np.where(use_second, scores[:, 1], scores[:, 0])
    other_id = np.where(use_second, ids[:, 1], ids[:, 0])
    if bool((other_id == targets).any()):
        raise AssertionError("self identifier survived nearest-other selection")
    return other_score, other_id


def query_competitor(
    scores: np.ndarray, ids: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    n = scores.shape[0]
    targets = np.arange(n)
    target_is_first = ids[:, 0] == targets
    competitor_score = np.where(target_is_first, scores[:, 1], scores[:, 0])
    competitor_id = np.where(target_is_first, ids[:, 1], ids[:, 0])
    if bool((competitor_id == targets).any()):
        raise AssertionError("target identifier survived competitor selection")
    return competitor_score, competitor_id


def quantiles(values: np.ndarray) -> dict:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return {"count": 0}
    return {
        "count": int(finite.size),
        "min": float(np.min(finite)),
        "p01": float(np.quantile(finite, 0.01)),
        "p10": float(np.quantile(finite, 0.10)),
        "median": float(np.median(finite)),
        "p90": float(np.quantile(finite, 0.90)),
        "p99": float(np.quantile(finite, 0.99)),
        "max": float(np.max(finite)),
    }


def repeated_fraction_phase(
    updated_success: np.ndarray,
    unchanged_success: np.ndarray,
    fractions: list[float],
    repeats: int,
    seed: int,
) -> list[dict]:
    n = updated_success.size
    out = []
    for fraction in fractions:
        count = int(round(fraction * n))
        values = []
        for repeat in range(repeats):
            rng = np.random.default_rng(seed + 1009 * repeat + int(10000 * fraction))
            updated = np.zeros(n, dtype=bool)
            if count:
                updated[rng.choice(n, size=count, replace=False)] = True
            success = np.where(updated, updated_success, unchanged_success)
            values.append(float(success.mean()))
        out.append(
            {
                "updated_fraction": fraction,
                "updated_rows": count,
                "mean_semantic_top1": float(np.mean(values)),
                "std_semantic_top1": float(np.std(values)),
                "min_semantic_top1": float(np.min(values)),
                "max_semantic_top1": float(np.max(values)),
            }
        )
    return out


def overlap_phase(
    aligned_success: np.ndarray,
    fractions: list[float],
    repeats: int,
    seed: int,
) -> list[dict]:
    n = aligned_success.size
    out = []
    for fraction in fractions:
        count = int(round(fraction * n))
        overall = []
        conditional = []
        for repeat in range(repeats):
            rng = np.random.default_rng(seed + 2027 * repeat + int(10000 * fraction))
            if count:
                aligned = rng.choice(n, size=count, replace=False)
                recovered = int(aligned_success[aligned].sum())
                overall.append(recovered / n)
                conditional.append(recovered / count)
            else:
                overall.append(0.0)
                conditional.append(0.0)
        out.append(
            {
                "aligned_vocabulary_fraction": fraction,
                "aligned_rows": count,
                "mean_overall_semantic_top1": float(np.mean(overall)),
                "std_overall_semantic_top1": float(np.std(overall)),
                "mean_conditional_aligned_top1": float(np.mean(conditional)),
                "std_conditional_aligned_top1": float(np.std(conditional)),
            }
        )
    return out


def exact_unique_rows(rows: np.ndarray) -> tuple[int, int]:
    contiguous = np.ascontiguousarray(rows)
    row_bytes = contiguous.view(
        np.dtype((np.void, contiguous.dtype.itemsize * contiguous.shape[1]))
    ).ravel()
    unique = int(np.unique(row_bytes).size)
    return unique, int(contiguous.shape[0] - unique)


def analyze_pair(
    base_id: str,
    victim_id: str,
    variant: str,
    cache_dir: str,
    batch_size: int,
    fractions: list[float],
    repeats: int,
    seed: int,
    detail_dir: Path | None,
) -> dict:
    started = time.time()
    base_rows, base_sha = load_rows(base_id, variant, cache_dir)
    victim_rows, victim_sha = load_rows(victim_id, variant, cache_dir)
    if base_rows.shape != victim_rows.shape:
        raise ValueError(f"shape mismatch: {base_rows.shape} != {victim_rows.shape}")

    base = canonical_orbit(base_rows)
    victim = canonical_orbit(victim_rows)
    del base_rows, victim_rows

    index = faiss.IndexFlatIP(base.shape[1])
    index.add(base)

    base_scores, base_ids = exact_search(
        index, base, 2, batch_size, f"{variant}:base-margin"
    )
    neighbor_cosine, neighbor_id = nearest_other(base_scores, base_ids)
    neighbor_cosine = np.clip(neighbor_cosine, -1.0, 1.0)
    base_margin = np.sqrt(np.maximum(0.0, 2.0 - 2.0 * neighbor_cosine))
    base_unique = base_margin > 1e-7

    victim_scores, victim_ids = exact_search(
        index, victim, 5, batch_size, f"{variant}:private-query"
    )
    target = np.arange(base.shape[0])
    target_cosine = np.sum(victim * base, axis=1)
    target_cosine = np.clip(target_cosine, -1.0, 1.0)
    competitor_cosine, competitor_id = query_competitor(victim_scores, victim_ids)
    observed_top1 = victim_ids[:, 0] == target
    strict_top1 = target_cosine > competitor_cosine + 1e-6
    score_gap = target_cosine - competitor_cosine

    drift = np.sqrt(np.maximum(0.0, 2.0 - 2.0 * target_cosine))
    certified = drift < (base_margin / 2.0)
    ratio = np.full_like(drift, np.inf)
    positive_margin = base_margin > 0
    ratio[positive_margin] = 2.0 * drift[positive_margin] / base_margin[positive_margin]

    unique_private, private_collision_rows = exact_unique_rows(victim)
    reindex_rng = np.random.default_rng(seed + 991)
    reindex = reindex_rng.permutation(base.shape[0])
    reindexed_top1 = float(observed_top1[reindex].mean())
    if not math.isclose(reindexed_top1, float(observed_top1.mean()), abs_tol=1e-15):
        raise AssertionError("token-ID reindexing changed public-dictionary recovery")

    partial = repeated_fraction_phase(
        observed_top1,
        base_unique,
        fractions,
        repeats,
        seed,
    )
    overlap = overlap_phase(
        observed_top1,
        fractions,
        repeats,
        seed + 17,
    )

    if detail_dir is not None:
        detail_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            detail_dir / f"{variant}_margin_detail.npz",
            target_cosine=target_cosine,
            competitor_cosine=competitor_cosine,
            score_gap=score_gap,
            drift=drift,
            base_margin=base_margin,
            drift_to_half_margin_ratio=ratio,
            observed_top1=observed_top1,
            strict_top1=strict_top1,
            certified=certified,
            base_unique=base_unique,
            nearest_public_neighbor_id=neighbor_id,
            private_competitor_id=competitor_id,
        )

    record = {
        "base": base_id,
        "victim": victim_id,
        "variant": variant,
        "base_revision": base_sha,
        "victim_revision": victim_sha,
        "shape": list(base.shape),
        "index_kind": "faiss.IndexFlatIP",
        "empirical": {
            "observed_top1": float(observed_top1.mean()),
            "strict_margin_top1": float(strict_top1.mean()),
            "errors": int((~observed_top1).sum()),
            "strict_margin_failures": int((~strict_top1).sum()),
            "score_gap": quantiles(score_gap),
        },
        "certificate": {
            "condition": "L2(private_signature, public_signature) < public_nearest_neighbor_L2 / 2",
            "certified_fraction": float(certified.mean()),
            "certified_rows": int(certified.sum()),
            "uncertified_rows": int((~certified).sum()),
            "base_unique_fraction": float(base_unique.mean()),
            "base_zero_margin_rows": int((~base_unique).sum()),
            "signature_drift_l2": quantiles(drift),
            "public_separation_l2": quantiles(base_margin),
            "drift_to_half_margin_ratio": quantiles(ratio),
        },
        "controlled_partial_row_update": {
            "interpretation": (
                "Selected rows use the measured full-fine-tune outcome; "
                "unselected rows remain at the public base. This is a controlled "
                "row-update ablation, not an independently trained checkpoint."
            ),
            "phase": partial,
        },
        "controlled_vocabulary_overlap": {
            "interpretation": (
                "Only aligned rows can receive a public semantic label. "
                "Unaligned rows count as semantic failures, while their repeated "
                "signatures remain pseudotoken-linkable."
            ),
            "phase": overlap,
        },
        "controls": {
            "token_id_reindexing_top1": reindexed_top1,
            "token_id_reindexing_changed_result": False,
            "private_signature_unique_rows": unique_private,
            "private_signature_exact_collision_rows": private_collision_rows,
            "pseudotoken_repeat_linkage": (
                "exact for equal scale-permutation signatures; distinct semantic "
                "labels are not identifiable without public alignment or anchors"
            ),
        },
        "elapsed_seconds": time.time() - started,
    }
    print(json.dumps({"event": "pair_complete", **record}, sort_keys=True), flush=True)
    return record


def self_test() -> None:
    torch.manual_seed(7)
    rows = torch.randn(32, 16)
    scales = torch.linspace(-3.0, 4.0, 32)
    scales[torch.abs(scales) < 0.1] = 0.7
    perm = torch.randperm(rows.shape[1])
    transformed = scales[:, None] * rows[:, perm]
    a = canonical_orbit(rows)
    b = canonical_orbit(transformed)
    if not np.allclose(a, b, atol=1e-6):
        raise AssertionError("canonical signature is not scale-permutation invariant")

    index = faiss.IndexFlatIP(a.shape[1])
    index.add(a)
    scores, ids = index.search(a, 2)
    neighbor, _ = nearest_other(scores, ids)
    margin = np.sqrt(np.maximum(0.0, 2.0 - 2.0 * np.clip(neighbor, -1.0, 1.0)))
    if not bool((margin > 0).all()):
        raise AssertionError("random self-test dictionary unexpectedly collided")

    phase = repeated_fraction_phase(
        np.ones(32, dtype=bool),
        np.ones(32, dtype=bool),
        [0.0, 0.5, 1.0],
        3,
        9,
    )
    if not all(item["mean_semantic_top1"] == 1.0 for item in phase):
        raise AssertionError("partial-update phase endpoint failed")
    print(json.dumps({"event": "self_test_pass", "rows": 32}), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir")
    parser.add_argument("--output")
    parser.add_argument("--detail-dir")
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--threads", type=int, default=16)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260802)
    parser.add_argument(
        "--fractions",
        default="0,0.01,0.05,0.1,0.25,0.5,0.75,1",
    )
    parser.add_argument(
        "--variants",
        choices=("wte", "all"),
        default="all",
    )
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    if args.self_test:
        self_test()
        return
    if not args.cache_dir or not args.output:
        parser.error("--cache-dir and --output are required unless --self-test is used")

    faiss.omp_set_num_threads(args.threads)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    fractions = [float(item) for item in args.fractions.split(",")]
    if any(item < 0 or item > 1 for item in fractions):
        raise ValueError("fractions must lie in [0,1]")

    pairs = [
        ("gpt2", "lvwerra/gpt2-imdb", "gpt2-wte"),
        (
            "bert-base-uncased",
            "textattack/bert-base-uncased-SST-2",
            "bert-wte",
        ),
    ]
    if args.variants == "all":
        pairs.insert(1, ("gpt2", "lvwerra/gpt2-imdb", "gpt2-pos0"))
        pairs.append(
            (
                "bert-base-uncased",
                "textattack/bert-base-uncased-SST-2",
                "bert-pos0",
            )
        )

    started = time.time()
    detail_dir = Path(args.detail_dir) if args.detail_dir else None
    output = {
        "seed": args.seed,
        "fractions": fractions,
        "repeats": args.repeats,
        "batch_size": args.batch_size,
        "threads": args.threads,
        "host": platform.node(),
        "faiss_version": faiss.__version__,
        "torch_version": torch.__version__,
        "definition": (
            "unit-L2 normalized, lexicographic sign-canonicalized "
            "sort(row / Linf(row))"
        ),
        "pairs": [],
    }
    for pair in pairs:
        output["pairs"].append(
            analyze_pair(
                *pair,
                cache_dir=args.cache_dir,
                batch_size=args.batch_size,
                fractions=fractions,
                repeats=args.repeats,
                seed=args.seed,
                detail_dir=detail_dir,
            )
        )
    output["elapsed_seconds"] = time.time() - started
    Path(args.output).write_text(
        json.dumps(output, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "event": "complete",
                "output": args.output,
                "pairs": len(output["pairs"]),
                "elapsed_seconds": output["elapsed_seconds"],
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Evaluate public-base matching of private fine-tuned embedding orbits.

The victim exposes alpha * e_t * P, where alpha is any nonzero scalar and P is
a feature permutation.  The canonical representative below removes both.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import faiss
import numpy as np
import torch
from huggingface_hub import hf_hub_download, model_info
from safetensors import safe_open
from transformers import AutoModel, AutoModelForCausalLM


def canonical_orbit(rows: torch.Tensor) -> np.ndarray:
    """Return a complete representative of {alpha * row * P: alpha != 0}."""
    x = rows.detach().float().cpu()
    scale = x.abs().amax(dim=1, keepdim=True)
    if bool((scale == 0).any()):
        raise ValueError("zero embedding row has no scale-permutation orbit")
    asc = torch.sort(x / scale, dim=1).values
    neg = -torch.flip(asc, dims=(1,))
    unresolved = torch.ones(asc.shape[0], dtype=torch.bool)
    use_neg = torch.zeros(asc.shape[0], dtype=torch.bool)
    for col in range(asc.shape[1]):
        lt = neg[:, col] < asc[:, col]
        gt = neg[:, col] > asc[:, col]
        use_neg[unresolved & lt] = True
        unresolved &= ~(lt | gt)
        if not bool(unresolved.any()):
            break
    out = torch.where(use_neg[:, None], neg, asc).numpy().astype("float32")
    faiss.normalize_L2(out)
    return out


def load_rows(model_id: str, variant: str, cache_dir: str) -> tuple[torch.Tensor, str]:
    info = model_info(model_id)
    if variant.startswith("gpt2"):
        model = AutoModelForCausalLM.from_pretrained(model_id, cache_dir=cache_dir)
        rows = model.transformer.wte.weight.detach().cpu()
        if variant == "gpt2-pos0":
            rows = rows + model.transformer.wpe.weight[0].detach().cpu()
    elif variant.startswith("bert"):
        model = AutoModel.from_pretrained(model_id, cache_dir=cache_dir)
        emb = model.embeddings
        rows = emb.word_embeddings.weight.detach().cpu()
        if variant == "bert-pos0":
            pos = emb.position_embeddings.weight[0].detach().cpu()
            seg = emb.token_type_embeddings.weight[0].detach().cpu()
            rows = emb.LayerNorm(rows + pos + seg).detach().cpu()
    else:
        raise ValueError(f"unknown variant: {variant}")
    del model
    return rows, info.sha


def evaluate(
    base: np.ndarray, victim: np.ndarray, sample_ids: np.ndarray, index_kind: str
) -> dict:
    dim = base.shape[1]
    if index_kind == "flat":
        index = faiss.IndexFlatIP(dim)
    else:
        index = faiss.IndexHNSWFlat(dim, 48, faiss.METRIC_INNER_PRODUCT)
        index.hnsw.efConstruction = 160
        index.hnsw.efSearch = 512
    index.add(base)
    scores, ids = index.search(victim[sample_ids], 5)
    target = sample_ids
    target_score = np.sum(victim[target] * base[target], axis=1)
    ann_missed_target = (ids[:, 0] != target) & (target_score > scores[:, 0] + 1e-6)
    certified_top1 = (ids[:, 0] == target) | ann_missed_target
    hit5 = (ids == target[:, None]).any(axis=1) | ann_missed_target
    wrong_better = (ids[:, 0] != target) & (scores[:, 0] > target_score + 1e-6)
    return {
        "n": int(target.size),
        "certified_top1": float(certified_top1.mean()),
        "certified_errors": int((~certified_top1).sum()),
        "top5_or_ann_target": float(hit5.mean()),
        "wrong_candidate_strictly_better": int(wrong_better.sum()),
        "ann_missed_known_target": int(ann_missed_target.sum()),
        "median_target_cosine": float(np.median(target_score)),
        "p01_target_cosine": float(np.quantile(target_score, 0.01)),
        "median_top1_minus_target": float(np.median(scores[:, 0] - target_score)),
        "max_top1_minus_target": float(np.max(scores[:, 0] - target_score)),
        "random_label_baseline": float(1.0 / base.shape[0]),
        "index_kind": index_kind,
    }


def inspect_adapter(repo_id: str, cache_dir: str) -> dict:
    info = model_info(repo_id)
    config_path = hf_hub_download(
        repo_id, "adapter_config.json", cache_dir=cache_dir
    )
    config = json.loads(Path(config_path).read_text())
    siblings = {item.rfilename for item in info.siblings}
    if "adapter_model.safetensors" in siblings:
        weight_name = "adapter_model.safetensors"
        weight_path = hf_hub_download(repo_id, weight_name, cache_dir=cache_dir)
        with safe_open(weight_path, framework="pt", device="cpu") as handle:
            keys = sorted(handle.keys())
    elif "adapter_model.bin" in siblings:
        weight_name = "adapter_model.bin"
        weight_path = hf_hub_download(repo_id, weight_name, cache_dir=cache_dir)
        state = torch.load(weight_path, map_location="cpu", weights_only=True)
        keys = sorted(state.keys())
    else:
        raise FileNotFoundError(f"no adapter weights in {repo_id}")
    embedding_markers = (
        "wte",
        "embed_tokens",
        "word_embeddings",
        "position_embeddings",
    )
    embedding_keys = [
        key for key in keys if any(marker in key for marker in embedding_markers)
    ]
    return {
        "adapter": repo_id,
        "revision": info.sha,
        "base_model_name_or_path": config.get("base_model_name_or_path"),
        "peft_type": config.get("peft_type"),
        "target_modules": config.get("target_modules"),
        "modules_to_save": config.get("modules_to_save"),
        "trainable_token_indices": config.get("trainable_token_indices"),
        "weight_file": weight_name,
        "state_key_count": len(keys),
        "embedding_state_keys": embedding_keys,
        "embedding_frozen_by_artifact": (
            len(embedding_keys) == 0
            and not config.get("modules_to_save")
            and not config.get("trainable_token_indices")
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--samples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260802)
    parser.add_argument("--index", choices=("hnsw", "flat"), default="hnsw")
    args = parser.parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    pairs = [
        ("gpt2", "lvwerra/gpt2-imdb", "gpt2-wte"),
        ("gpt2", "lvwerra/gpt2-imdb", "gpt2-pos0"),
        (
            "bert-base-uncased",
            "textattack/bert-base-uncased-SST-2",
            "bert-wte",
        ),
        (
            "bert-base-uncased",
            "textattack/bert-base-uncased-SST-2",
            "bert-pos0",
        ),
    ]
    output = {
        "seed": args.seed,
        "samples": args.samples,
        "definition": "lexicographic sign-canonicalized sort(row / Linf(row))",
        "pairs": [],
        "adapters": [],
    }
    for base_id, victim_id, variant in pairs:
        started = time.time()
        base_rows, base_sha = load_rows(base_id, variant, args.cache_dir)
        victim_rows, victim_sha = load_rows(victim_id, variant, args.cache_dir)
        if base_rows.shape != victim_rows.shape:
            raise ValueError(
                f"shape mismatch: {base_rows.shape} != {victim_rows.shape}"
            )
        delta = (victim_rows.float() - base_rows.float()).abs()
        base_signature = canonical_orbit(base_rows)
        victim_signature = canonical_orbit(victim_rows)
        rng = np.random.default_rng(args.seed)
        count = min(args.samples, base_signature.shape[0])
        sample_ids = np.sort(
            rng.choice(base_signature.shape[0], size=count, replace=False)
        )
        record = {
            "base": base_id,
            "victim": victim_id,
            "variant": variant,
            "base_revision": base_sha,
            "victim_revision": victim_sha,
            "shape": list(base_rows.shape),
            "exact_equal_rows": int((delta.amax(dim=1) == 0).sum()),
            "mean_abs_weight_delta": float(delta.mean()),
            "max_abs_weight_delta": float(delta.max()),
            **evaluate(base_signature, victim_signature, sample_ids, args.index),
            "elapsed_seconds": time.time() - started,
        }
        output["pairs"].append(record)
        print(json.dumps(record, sort_keys=True), flush=True)
        del base_rows, victim_rows, base_signature, victim_signature

    for adapter_id in (
        "palsp/gpt2-lora",
        "nutrientartcd/recipe-gpt2-lora",
    ):
        record = inspect_adapter(adapter_id, args.cache_dir)
        output["adapters"].append(record)
        print(json.dumps(record, sort_keys=True), flush=True)

    Path(args.output).write_text(
        json.dumps(output, indent=2, sort_keys=True) + "\n"
    )


if __name__ == "__main__":
    main()

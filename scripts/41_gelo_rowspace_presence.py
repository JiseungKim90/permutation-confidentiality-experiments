#!/usr/bin/env python3
"""Test exact candidate-presence leakage from GELO's observable row space.

GELO reveals U=A H_full for fresh invertible A.  Therefore row(U) equals
row(H_full).  Candidate hidden states can be tested by their projection
residual without estimating A or reconstructing H.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from huggingface_hub import model_info
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_texts(model_id: str, cache_dir: str, count: int, seq_len: int) -> tuple:
    tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=cache_dir)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    data = load_dataset(
        "wikitext",
        "wikitext-2-raw-v1",
        split="validation",
        cache_dir=cache_dir,
    )
    texts = []
    for text in data["text"]:
        text = " ".join(text.strip().split())
        if not text or text.startswith("="):
            continue
        if len(tokenizer.encode(text, add_special_tokens=False)) < seq_len:
            continue
        texts.append(text)
        if len(texts) == count:
            break
    if len(texts) < count:
        raise RuntimeError(f"only found {len(texts)} texts")
    encoded = tokenizer(
        texts,
        padding="max_length",
        truncation=True,
        max_length=seq_len,
        return_tensors="pt",
    )
    return tokenizer, texts, encoded


def hidden_dictionary(
    model_id: str,
    cache_dir: str,
    encoded: dict,
    layers: list[int],
    batch_size: int,
) -> tuple[dict[int, np.ndarray], str]:
    info = model_info(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        cache_dir=cache_dir,
        torch_dtype=torch.float32,
    )
    model.eval()
    outputs = {layer: [] for layer in layers}
    with torch.no_grad():
        total = encoded["input_ids"].shape[0]
        for start in range(0, total, batch_size):
            stop = min(start + batch_size, total)
            batch = {key: value[start:stop] for key, value in encoded.items()}
            result = model(**batch, output_hidden_states=True, use_cache=False)
            for layer in layers:
                outputs[layer].append(
                    result.hidden_states[layer].detach().float().cpu().numpy()
                )
    del model
    return {
        layer: np.concatenate(chunks, axis=0).astype("float64")
        for layer, chunks in outputs.items()
    }, info.sha


def mixing_matrix(
    size: int, kind: str, rng: np.random.Generator
) -> tuple[np.ndarray, float]:
    q1, _ = np.linalg.qr(rng.standard_normal((size, size)))
    if kind == "orthogonal":
        return q1, 1.0
    if kind == "nonorthogonal":
        q2, _ = np.linalg.qr(rng.standard_normal((size, size)))
        singular = np.geomspace(1.0, 50.0, size)
        return q1 @ np.diag(singular) @ q2.T, 50.0
    raise ValueError(kind)


def observed_row_basis(u: np.ndarray) -> tuple[np.ndarray, int, float]:
    _, singular, vt = np.linalg.svd(u, full_matrices=False)
    threshold = singular[0] * max(u.shape) * np.finfo(u.dtype).eps
    rank = int(np.sum(singular > threshold))
    return vt[:rank], rank, float(singular[0] / singular[rank - 1])


def residuals(dictionary: np.ndarray, basis: np.ndarray) -> np.ndarray:
    projected = np.einsum(
        "cld,rd->clr",
        dictionary,
        basis,
        optimize=True,
    )
    reconstructed = np.einsum(
        "clr,rd->cld",
        projected,
        basis,
        optimize=True,
    )
    numerator = np.linalg.norm(dictionary - reconstructed, axis=(1, 2))
    denominator = np.linalg.norm(dictionary, axis=(1, 2))
    return numerator / denominator


def one_trial(
    dictionary: np.ndarray,
    source_count: int,
    shield_fraction: float,
    shield_scale: float,
    mixing: str,
    precision: str,
    rng: np.random.Generator,
) -> dict:
    candidate_count, seq_len, dim = dictionary.shape
    source_ids = np.sort(
        rng.choice(candidate_count, size=source_count, replace=False)
    )
    data_rows = dictionary[source_ids].reshape(source_count * seq_len, dim)
    shield_count = int(np.ceil(data_rows.shape[0] * shield_fraction))
    if shield_count:
        typical_norm = np.median(np.linalg.norm(data_rows, axis=1))
        shields = rng.standard_normal((shield_count, dim))
        shields /= np.linalg.norm(shields, axis=1, keepdims=True)
        shields *= shield_scale * typical_norm
        full = np.concatenate([data_rows, shields], axis=0)
    else:
        full = data_rows
    a, intended_condition = mixing_matrix(full.shape[0], mixing, rng)
    u = a @ full
    if precision == "float32":
        u = u.astype("float32").astype("float64")
    elif precision == "bfloat16":
        u = (
            torch.from_numpy(u.astype("float32"))
            .to(torch.bfloat16)
            .to(torch.float32)
            .numpy()
            .astype("float64")
        )
    elif precision != "float64":
        raise ValueError(precision)
    basis, rank, observed_condition = observed_row_basis(u)
    score = residuals(dictionary, basis)
    predicted = np.argsort(score)[:source_count]
    source_set = set(source_ids.tolist())
    predicted_set = set(predicted.tolist())
    true_score = score[source_ids]
    false_mask = np.ones(candidate_count, dtype=bool)
    false_mask[source_ids] = False
    false_score = score[false_mask]
    return {
        "recall": len(source_set & predicted_set) / source_count,
        "exact_set": source_set == predicted_set,
        "max_true_residual": float(np.max(true_score)),
        "min_false_residual": float(np.min(false_score)),
        "gap": float(np.min(false_score) - np.max(true_score)),
        "observed_rank": rank,
        "row_count": int(full.shape[0]),
        "intended_condition": intended_condition,
        "observed_condition": observed_condition,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="gpt2")
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--candidates", type=int, default=128)
    parser.add_argument("--seq-len", type=int, default=16)
    parser.add_argument("--source-count", type=int, default=4)
    parser.add_argument("--trials", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260802)
    parser.add_argument("--layers", type=int, nargs="+", default=[2, 4, 8, 12])
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)

    _, texts, encoded = load_texts(
        args.model, args.cache_dir, args.candidates, args.seq_len
    )
    dictionaries, revision = hidden_dictionary(
        args.model,
        args.cache_dir,
        encoded,
        args.layers,
        args.batch_size,
    )
    conditions = [
        ("orthogonal", 0.0, 1.0, "float32"),
        ("nonorthogonal", 0.0, 1.0, "float32"),
        ("orthogonal", 0.12, 25.0, "float32"),
        ("nonorthogonal", 0.12, 25.0, "float32"),
        ("orthogonal", 0.12, 25.0, "bfloat16"),
        ("nonorthogonal", 0.12, 25.0, "bfloat16"),
    ]
    output = {
        "model": args.model,
        "model_revision": revision,
        "dataset": "Salesforce/wikitext:wikitext-2-raw-v1:validation",
        "seed": args.seed,
        "candidate_count": args.candidates,
        "seq_len": args.seq_len,
        "source_count": args.source_count,
        "trials": args.trials,
        "layers": args.layers,
        "text_sha256_note": "texts stored by index only; dataset revision is cached",
        "results": [],
    }
    for layer, dictionary in dictionaries.items():
        for mixing, fraction, scale, precision in conditions:
            started = time.time()
            trials = [
                one_trial(
                    dictionary,
                    args.source_count,
                    fraction,
                    scale,
                    mixing,
                    precision,
                    rng,
                )
                for _ in range(args.trials)
            ]
            record = {
                "layer": layer,
                "mixing": mixing,
                "shield_fraction": fraction,
                "shield_scale": scale,
                "precision": precision,
                "mean_recall": float(np.mean([t["recall"] for t in trials])),
                "exact_sets": int(sum(t["exact_set"] for t in trials)),
                "min_gap": float(min(t["gap"] for t in trials)),
                "max_true_residual": float(
                    max(t["max_true_residual"] for t in trials)
                ),
                "min_false_residual": float(
                    min(t["min_false_residual"] for t in trials)
                ),
                "rank_min": int(min(t["observed_rank"] for t in trials)),
                "rank_max": int(max(t["observed_rank"] for t in trials)),
                "elapsed_seconds": time.time() - started,
            }
            output["results"].append(record)
            print(json.dumps(record, sort_keys=True), flush=True)
    output["candidate_texts"] = texts
    Path(args.output).write_text(
        json.dumps(output, indent=2, sort_keys=True) + "\n"
    )


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Attack GELO's official MixedBatchDataset using its exact row-space invariant."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import random
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from huggingface_hub import model_info
from transformers import AutoModelForCausalLM, AutoTokenizer

from lib.maximal_invariants import candidate_presence_scores, rowspace_basis


def load_public_texts(
    model_id: str,
    cache_dir: str,
    count: int,
    seq_len: int,
) -> tuple[list[str], dict[str, torch.Tensor], object]:
    tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=cache_dir)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    dataset = load_dataset(
        "wikitext",
        "wikitext-2-raw-v1",
        split="validation",
        cache_dir=cache_dir,
    )
    texts: list[str] = []
    for raw in dataset["text"]:
        text = " ".join(raw.strip().split())
        if not text or text.startswith("="):
            continue
        if len(tokenizer.encode(text, add_special_tokens=False)) < seq_len:
            continue
        texts.append(text)
        if len(texts) == count:
            break
    if len(texts) != count:
        raise RuntimeError(f"found {len(texts)} usable texts, need {count}")
    encoded = tokenizer(
        texts,
        padding="max_length",
        truncation=True,
        max_length=seq_len,
        return_tensors="pt",
    )
    return texts, encoded, tokenizer


def extract_hidden_bank(
    model_id: str,
    cache_dir: str,
    encoded: dict[str, torch.Tensor],
    layers: list[int],
    batch_size: int,
    model_dtype: str,
) -> tuple[dict[int, np.ndarray], str]:
    dtype = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[model_dtype]
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        cache_dir=cache_dir,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
    ).eval()
    model.to("cpu")
    outputs: dict[int, list[np.ndarray]] = {layer: [] for layer in layers}
    with torch.no_grad():
        total = encoded["input_ids"].shape[0]
        for start in range(0, total, batch_size):
            stop = min(start + batch_size, total)
            batch = {key: value[start:stop] for key, value in encoded.items()}
            result = model(
                **batch,
                output_hidden_states=True,
                return_dict=True,
                use_cache=False,
            )
            if max(layers) >= len(result.hidden_states):
                raise RuntimeError(
                    f"layer {max(layers)} requested, model exposes "
                    f"{len(result.hidden_states)} hidden-state tensors"
                )
            for layer in layers:
                outputs[layer].append(
                    result.hidden_states[layer].detach().float().cpu().numpy()
                )
            print(f"hidden extraction {stop}/{total}", flush=True)
    revision = model_info(model_id).sha
    del model
    return {
        layer: np.concatenate(chunks).astype(np.float16)
        for layer, chunks in outputs.items()
    }, revision


def build_official_entries(
    bank: np.ndarray,
    store: Path,
    text_entry_class: type,
) -> list[object]:
    store.mkdir(parents=True, exist_ok=True)
    entries: list[object] = []
    seq_len = bank.shape[1]
    for index, rows in enumerate(bank):
        text_id = f"candidate_{index:05d}"
        path = store / f"{text_id}.npz"
        np.savez_compressed(
            path,
            token_ids=np.arange(seq_len, dtype=np.int32),
            attention_mask=np.ones(seq_len, dtype=np.int8),
            embeddings=rows.astype(np.float16),
        )
        entries.append(
            text_entry_class(
                text_id=text_id,
                npz_path=path,
                token_count=seq_len,
                has_embeddings=True,
                has_embeddings_rms=False,
            )
        )
    return entries


def quantize_observation(observed: np.ndarray, precision: str) -> np.ndarray:
    tensor = torch.from_numpy(observed.astype(np.float32))
    if precision == "float32":
        return tensor.numpy().astype(np.float64)
    if precision == "bfloat16":
        return tensor.to(torch.bfloat16).to(torch.float32).numpy().astype(np.float64)
    if precision == "float16":
        return tensor.to(torch.float16).to(torch.float32).numpy().astype(np.float64)
    raise ValueError(precision)


def random_basis(rank: int, dim: int, rng: np.random.Generator) -> np.ndarray:
    q, _ = np.linalg.qr(rng.standard_normal((dim, rank)))
    return q.T


def trial_metrics(
    sample: dict,
    bank: np.ndarray,
    sampled_rows: int,
    source_count: int,
    precision: str,
    rng: np.random.Generator,
) -> dict:
    observed = quantize_observation(sample["u"].numpy(), precision)
    basis, singular = rowspace_basis(observed)
    scores = candidate_presence_scores(
        bank.astype(np.float64),
        basis,
        sampled_rows=sampled_rows,
    )
    source_ids = np.array(
        [int(value.rsplit("_", 1)[1]) for value in sample["positive_ids"]],
        dtype=np.int64,
    )
    predicted = np.argsort(scores)[:source_count]
    false_mask = np.ones(bank.shape[0], dtype=bool)
    false_mask[source_ids] = False

    control = candidate_presence_scores(
        bank.astype(np.float64),
        random_basis(basis.shape[0], bank.shape[-1], rng),
        sampled_rows=sampled_rows,
    )
    control_predicted = np.argsort(control)[:source_count]

    source_set = set(source_ids.tolist())
    predicted_set = set(predicted.tolist())
    control_set = set(control_predicted.tolist())
    row_ids = {
        value
        for value in sample.get("row_text_ids", [])
        if value and value != "__shield__"
    }
    observed_candidate_union = {
        int(value.rsplit("_", 1)[1]) for value in row_ids
    }
    union_k = max(source_count, len(observed_candidate_union))
    union_prediction = set(np.argsort(scores)[:union_k].tolist())
    return {
        "recall": len(source_set & predicted_set) / source_count,
        "exact_set": source_set == predicted_set,
        "control_recall": len(source_set & control_set) / source_count,
        "max_true_score": float(np.max(scores[source_ids])),
        "min_false_score": float(np.min(scores[false_mask])),
        "gap": float(np.min(scores[false_mask]) - np.max(scores[source_ids])),
        "rank": int(basis.shape[0]),
        "rows": int(observed.shape[0]),
        "condition": float(singular[0] / singular[-1]),
        "candidate_union_size": len(observed_candidate_union),
        "candidate_union_exact": (
            observed_candidate_union == union_prediction
        ),
        "candidate_union_recall": (
            len(observed_candidate_union & union_prediction)
            / max(1, len(observed_candidate_union))
        ),
        "candidate_union_excess": len(observed_candidate_union - source_set),
    }


def run_condition(
    *,
    official: object,
    entries: list[object],
    bank: np.ndarray,
    trials: int,
    source_count: int,
    sampled_rows: int,
    mix_mode: str,
    shield_kind: str,
    shield_fraction: float,
    shield_scale: float,
    precision: str,
    mix_condition: float,
    seed: int,
) -> dict:
    torch.manual_seed(seed)
    dataset = official.MixedBatchDataset(
        entries=entries,
        steps_per_epoch=trials,
        positives_per_example=source_count,
        positives_per_example_min=source_count,
        positives_per_example_max=source_count,
        tokens_per_text=sampled_rows,
        token_key="embeddings",
        mix_mode=mix_mode,
        shield_kind=shield_kind,
        gauss_rows=0,
        gauss_frac=shield_fraction,
        gauss_scale=shield_scale,
        gauss_randomization="none",
        gauss_frac_min=shield_fraction,
        gauss_frac_max=shield_fraction,
        gauss_scale_min=shield_scale,
        gauss_scale_max=shield_scale,
        student_t_params=None,
        seed=seed,
        mix_cond_min=mix_condition,
        mix_cond_max=mix_condition,
        mix_cond_randomization="none",
        tokens_per_text_min=sampled_rows,
        tokens_per_text_max=sampled_rows,
        tokens_per_text_randomization="none",
        return_clean_rows=False,
        return_row_text_ids=True,
    )
    rng = np.random.default_rng(seed + 7919)
    metrics = [
        trial_metrics(
            dataset[index],
            bank,
            sampled_rows,
            source_count,
            precision,
            rng,
        )
        for index in range(trials)
    ]
    return {
        "mixing": mix_mode,
        "shield_kind": shield_kind,
        "shield_fraction": shield_fraction,
        "shield_scale": shield_scale,
        "precision": precision,
        "mix_condition": mix_condition if mix_mode == "non_orthogonal" else 1.0,
        "mean_recall": float(np.mean([x["recall"] for x in metrics])),
        "exact_sets": int(sum(x["exact_set"] for x in metrics)),
        "mean_control_recall": float(
            np.mean([x["control_recall"] for x in metrics])
        ),
        "min_gap": float(min(x["gap"] for x in metrics)),
        "max_true_score": float(max(x["max_true_score"] for x in metrics)),
        "min_false_score": float(min(x["min_false_score"] for x in metrics)),
        "rank_min": int(min(x["rank"] for x in metrics)),
        "rank_max": int(max(x["rank"] for x in metrics)),
        "row_min": int(min(x["rows"] for x in metrics)),
        "row_max": int(max(x["rows"] for x in metrics)),
        "max_observed_condition": float(max(x["condition"] for x in metrics)),
        "mean_candidate_union_size": float(
            np.mean([x["candidate_union_size"] for x in metrics])
        ),
        "exact_candidate_unions": int(
            sum(x["candidate_union_exact"] for x in metrics)
        ),
        "mean_candidate_union_recall": float(
            np.mean([x["candidate_union_recall"] for x in metrics])
        ),
        "mean_candidate_union_excess": float(
            np.mean([x["candidate_union_excess"] for x in metrics])
        ),
        "trial_metrics": metrics,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--official-repo", required=True)
    parser.add_argument("--model", default="gpt2")
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--candidates", type=int, default=128)
    parser.add_argument("--seq-len", type=int, default=32)
    parser.add_argument("--source-count", type=int, default=4)
    parser.add_argument("--sampled-rows", type=int, default=16)
    parser.add_argument("--trials", type=int, default=20)
    parser.add_argument("--layers", type=int, nargs="+", default=[4, 8, 12])
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument(
        "--model-dtype",
        choices=["float32", "float16", "bfloat16"],
        default="float32",
    )
    parser.add_argument("--seed", type=int, default=20260802)
    args = parser.parse_args()
    if args.sampled_rows > args.seq_len:
        raise ValueError("sampled rows cannot exceed sequence length")

    official_repo = Path(args.official_repo).resolve()
    sys.path.insert(0, str(official_repo / "gelo"))
    official = importlib.import_module("train_learned_attack")
    commit = subprocess.run(
        ["git", "-C", str(official_repo), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    code_path = official_repo / "gelo" / "train_learned_attack.py"
    code_sha = hashlib.sha256(code_path.read_bytes()).hexdigest()

    texts, encoded, _ = load_public_texts(
        args.model,
        args.cache_dir,
        args.candidates,
        args.seq_len,
    )
    banks, model_revision = extract_hidden_bank(
        args.model,
        args.cache_dir,
        encoded,
        args.layers,
        args.batch_size,
        args.model_dtype,
    )
    conditions = [
        ("orthogonal", "gaussian", 0.05, 10.0, "float32", 1.0),
        ("non_orthogonal", "gaussian", 0.05, 10.0, "float32", 50.0),
        ("orthogonal", "gaussian", 0.12, 25.0, "bfloat16", 1.0),
        ("non_orthogonal", "gaussian", 0.12, 25.0, "bfloat16", 50.0),
        ("orthogonal", "manifold", 0.05, 10.0, "float32", 1.0),
        ("non_orthogonal", "manifold", 0.14, 30.0, "bfloat16", 50.0),
    ]
    output = {
        "attack": "training-free row-space candidate containment",
        "official_repository": "https://github.com/noskill/gelo",
        "official_commit": commit,
        "official_batch_code": "gelo/train_learned_attack.py:MixedBatchDataset",
        "official_batch_code_sha256": code_sha,
        "model": args.model,
        "model_revision": model_revision,
        "model_dtype": args.model_dtype,
        "dataset": "Salesforce/wikitext:wikitext-2-raw-v1:validation",
        "candidate_count": args.candidates,
        "sequence_rows_per_candidate": args.seq_len,
        "sampled_rows_per_source": args.sampled_rows,
        "source_count": args.source_count,
        "trials": args.trials,
        "layers": args.layers,
        "seed": args.seed,
        "candidate_text_sha256": [
            hashlib.sha256(text.encode("utf-8")).hexdigest() for text in texts
        ],
        "results": [],
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    started_all = time.time()
    with tempfile.TemporaryDirectory(
        prefix="gelo_official_bank_",
        dir=output_path.parent,
    ) as temporary:
        for layer, raw_bank in banks.items():
            bank = raw_bank.astype(np.float16).astype(np.float64)
            entries = build_official_entries(
                raw_bank,
                Path(temporary) / f"layer_{layer}",
                official.TextEntry,
            )
            for index, condition in enumerate(conditions):
                started = time.time()
                record = run_condition(
                    official=official,
                    entries=entries,
                    bank=bank,
                    trials=args.trials,
                    source_count=args.source_count,
                    sampled_rows=args.sampled_rows,
                    mix_mode=condition[0],
                    shield_kind=condition[1],
                    shield_fraction=condition[2],
                    shield_scale=condition[3],
                    precision=condition[4],
                    mix_condition=condition[5],
                    seed=args.seed + 1000 * layer + 37 * index,
                )
                record["layer"] = layer
                record["elapsed_seconds"] = time.time() - started
                output["results"].append(record)
                print(json.dumps({k: v for k, v in record.items() if k != "trial_metrics"}, sort_keys=True), flush=True)
    output["elapsed_seconds"] = time.time() - started_all
    output_path.write_text(
        json.dumps(output, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {output_path}", flush=True)


if __name__ == "__main__":
    main()

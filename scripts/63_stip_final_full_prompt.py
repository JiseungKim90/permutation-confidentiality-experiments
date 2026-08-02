#!/usr/bin/env python3
"""Replay and attack the final STIP embedding-to-cloud boundary.

This experiment implements the exact boundary formula in Section IV-E of the
NDSS 2026 paper, ``alpha @ Emb(A) @ P``.  The public repository is pinned and
audited for provenance, but it does not contain this final TEE path.  We report
that distinction in every output rather than calling the wrapper official code.
"""

from __future__ import annotations

import argparse
import difflib
import hashlib
import json
import math
import os
import random
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from lib.stip_orbit import (
    apply_final_boundary,
    canonical_orbit,
    cosine_topk,
    invert_final_boundary,
)


FINAL_PAPER_SHA256 = "8e7ac8f1e6e728de8328abad35e66c012dff82a7d059cf8456e73aa65a6c0a77"
OFFICIAL_REPOSITORY_COMMIT = "d8e8b876280979efd996ace5d9ed4b3a50bb271f"


def emit(event: str, **fields: Any) -> None:
    print(json.dumps({"event": event, **fields}, sort_keys=True), flush=True)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def repository_provenance(path: Path) -> dict[str, Any]:
    commit = subprocess.check_output(
        ["git", "-C", str(path), "rev-parse", "HEAD"], text=True
    ).strip()
    tracked = subprocess.check_output(
        ["git", "-C", str(path), "ls-files"], text=True
    ).splitlines()
    searched = []
    for name in tracked:
        candidate = path / name
        if candidate.suffix.lower() not in {".py", ".md", ".txt"}:
            continue
        text = candidate.read_text(encoding="utf-8", errors="ignore").lower()
        if "one-time random diagonal" in text or "αemb" in text or "alphaemb" in text:
            searched.append(name)
    return {
        "path": str(path.resolve()),
        "commit": commit,
        "expected_commit": OFFICIAL_REPOSITORY_COMMIT,
        "matches_expected_commit": commit == OFFICIAL_REPOSITORY_COMMIT,
        "files_containing_final_boundary_terms": searched,
        "final_tee_path_present_in_public_repository": bool(searched),
    }


def load_documents(paths: list[Path], seed: int, max_documents: int) -> list[str]:
    documents: list[str] = []
    for path in paths:
        dataset = Dataset.from_file(str(path))
        columns = set(dataset.column_names)
        for row in dataset:
            if "text" in columns and isinstance(row["text"], str):
                text = row["text"]
            elif "title" in columns and "text" in columns:
                text = f"{row.get('title', '')} {row.get('text', '')}"
            else:
                values = [value for value in row.values() if isinstance(value, str)]
                text = " ".join(values)
            text = text.strip()
            if text:
                documents.append(text)
    random.Random(seed).shuffle(documents)
    if max_documents > 0:
        documents = documents[:max_documents]
    if not documents:
        raise RuntimeError("no nonempty documents were loaded")
    return documents


def collect_prompts(
    tokenizer,
    documents: list[str],
    prompt_count: int,
    prompt_length: int,
    seed: int,
) -> tuple[torch.Tensor, list[str]]:
    generator = random.Random(seed)
    candidates: list[tuple[list[int], str]] = []
    for document in documents:
        ids = tokenizer.encode(document, add_special_tokens=False, verbose=False)
        if len(ids) < prompt_length:
            continue
        start = generator.randrange(0, len(ids) - prompt_length + 1)
        selected = ids[start : start + prompt_length]
        candidates.append((selected, tokenizer.decode(selected)))
    generator.shuffle(candidates)
    if len(candidates) < prompt_count:
        raise RuntimeError(
            f"only {len(candidates)} documents contain {prompt_length} tokens; "
            f"need {prompt_count}"
        )
    chosen = candidates[:prompt_count]
    return torch.tensor([item[0] for item in chosen], dtype=torch.long), [item[1] for item in chosen]


def install_private_rows(base_rows: torch.Tensor, checkpoint: Path | None) -> tuple[torch.Tensor, dict[str, Any]]:
    victim = base_rows.clone()
    if checkpoint is None:
        return victim, {"kind": "public_frozen", "checkpoint": None}
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    selected = payload["selected_ids"].long()
    trained = payload["trained_rows"].to(dtype=victim.dtype)
    if trained.shape != victim[selected].shape:
        raise ValueError("private checkpoint rows do not match the base embedding")
    victim[selected] = trained
    return victim, {
        "kind": "independently_trained_partial_embedding",
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": sha256_file(checkpoint),
        "selected_rows": int(selected.numel()),
        "seed": int(payload["seed"]),
        "selection": payload["selection"],
        "train_fraction_of_active": float(payload["train_fraction_of_active"]),
    }


def prompt_metrics(target: torch.Tensor, prediction: torch.Tensor, tokenizer) -> dict[str, Any]:
    correct = prediction.eq(target)
    exact = correct.all(dim=1)
    decoded_target = [tokenizer.decode(row.tolist()) for row in target]
    decoded_prediction = [tokenizer.decode(row.tolist()) for row in prediction]
    text_exact = [left == right for left, right in zip(decoded_target, decoded_prediction)]
    similarities = [
        difflib.SequenceMatcher(a=left, b=right).ratio()
        for left, right in zip(decoded_target, decoded_prediction)
    ]
    return {
        "tokens": int(target.numel()),
        "token_top1": float(correct.float().mean().item()),
        "token_errors": int((~correct).sum().item()),
        "prompts": int(target.shape[0]),
        "full_prompt_exact": float(exact.float().mean().item()),
        "full_prompt_errors": int((~exact).sum().item()),
        "decoded_text_exact": float(np.mean(text_exact)),
        "decoded_character_similarity_mean": float(np.mean(similarities)),
        "decoded_character_similarity_min": float(np.min(similarities)),
        "examples": [
            {
                "target": decoded_target[index],
                "prediction": decoded_prediction[index],
                "token_exact": bool(exact[index]),
                "character_similarity": similarities[index],
            }
            for index in range(min(12, target.shape[0]))
        ],
    }


def run_trial(
    trial: int,
    token_ids: torch.Tensor,
    public_words: torch.Tensor,
    public_positions: torch.Tensor,
    victim_words: torch.Tensor,
    victim_positions: torch.Tensor,
    tokenizer,
    seed: int,
    representation: str,
    scale_log_min: float,
    scale_log_max: float,
    negative_scale_probability: float,
    dictionary_batch_size: int,
) -> dict[str, Any]:
    generator = torch.Generator().manual_seed(seed + 1009 * trial)
    positions = torch.arange(token_ids.shape[1])
    clean = victim_words[token_ids]
    if representation == "token_position":
        clean = clean + victim_positions[positions][None, :, :]

    permutation = torch.randperm(clean.shape[-1], generator=generator)
    row_scales = torch.empty(token_ids.shape, dtype=torch.float32)
    row_scales.uniform_(scale_log_min, scale_log_max, generator=generator).exp_()
    if negative_scale_probability > 0:
        signs = torch.rand(token_ids.shape, generator=generator) < negative_scale_probability
        row_scales[signs] *= -1
    transcript = apply_final_boundary(clean, row_scales, permutation)
    trusted_inverse = invert_final_boundary(transcript, row_scales, permutation)
    inverse_max_error = float((trusted_inverse - clean).abs().max().item())
    invariant_max_error = float(
        (canonical_orbit(transcript) - canonical_orbit(clean)).abs().max().item()
    )

    prediction = torch.empty_like(token_ids)
    top5_hit = torch.zeros_like(token_ids, dtype=torch.bool)
    raw_prediction = torch.empty_like(token_ids)
    position_times: list[float] = []
    for position in range(token_ids.shape[1]):
        started = time.time()
        dictionary_rows = public_words
        if representation == "token_position":
            dictionary_rows = dictionary_rows + public_positions[position]
        dictionary_signature = canonical_orbit(dictionary_rows)
        query_signature = canonical_orbit(transcript[:, position, :])
        _, indices = cosine_topk(
            query_signature,
            dictionary_signature,
            k=5,
            query_chunk=dictionary_batch_size,
        )
        prediction[:, position] = indices[:, 0]
        top5_hit[:, position] = indices.eq(token_ids[:, position, None]).any(dim=1)

        # Deliberately omit orbit cancellation for a negative control.
        _, raw_indices = cosine_topk(
            transcript[:, position, :],
            dictionary_rows,
            k=1,
            query_chunk=dictionary_batch_size,
        )
        raw_prediction[:, position] = raw_indices[:, 0]
        position_times.append(time.time() - started)
        emit(
            "position_complete",
            trial=trial,
            position=position,
            positions=token_ids.shape[1],
            seconds=round(position_times[-1], 3),
        )

    metrics = prompt_metrics(token_ids, prediction, tokenizer)
    metrics.update(
        {
            "trial": trial,
            "top5": float(top5_hit.float().mean().item()),
            "raw_permuted_token_top1_control": float(
                raw_prediction.eq(token_ids).float().mean().item()
            ),
            "trusted_inverse_max_abs_error": inverse_max_error,
            "canonical_invariance_max_abs_error": invariant_max_error,
            "feature_permutation_sha256": hashlib.sha256(
                permutation.numpy().tobytes()
            ).hexdigest(),
            "row_scale_abs_min": float(row_scales.abs().min().item()),
            "row_scale_abs_max": float(row_scales.abs().max().item()),
            "position_seconds_total": float(sum(position_times)),
        }
    )
    return metrics


def aggregate(trials: list[dict[str, Any]]) -> dict[str, Any]:
    names = (
        "token_top1",
        "top5",
        "full_prompt_exact",
        "decoded_text_exact",
        "decoded_character_similarity_mean",
        "raw_permuted_token_top1_control",
    )
    summary: dict[str, Any] = {}
    for name in names:
        values = np.asarray([trial[name] for trial in trials], dtype=np.float64)
        summary[name] = {
            "mean": float(values.mean()),
            "min": float(values.min()),
            "max": float(values.max()),
        }
    summary["all_replays_exact"] = all(
        trial["trusted_inverse_max_abs_error"] <= 1e-6 for trial in trials
    )
    summary["all_invariance_checks_pass"] = all(
        trial["canonical_invariance_max_abs_error"] <= 2e-6 for trial in trials
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="gpt2")
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--arrow", action="append", required=True)
    parser.add_argument("--official-repo", required=True)
    parser.add_argument("--private-checkpoint")
    parser.add_argument("--output", required=True)
    parser.add_argument("--prompt-count", type=int, default=64)
    parser.add_argument("--prompt-length", type=int, default=16)
    parser.add_argument("--max-documents", type=int, default=0)
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--representation", choices=("token", "token_position"), default="token_position")
    parser.add_argument("--scale-log-min", type=float, default=-8.0)
    parser.add_argument("--scale-log-max", type=float, default=8.0)
    parser.add_argument("--negative-scale-probability", type=float, default=0.0)
    parser.add_argument("--dictionary-batch-size", type=int, default=128)
    parser.add_argument("--threads", type=int, default=16)
    parser.add_argument("--seed", type=int, default=20260802)
    args = parser.parse_args()

    if args.prompt_count <= 0 or args.prompt_length <= 0 or args.trials <= 0:
        parser.error("prompt count, prompt length, and trials must be positive")
    if args.prompt_length > 1024:
        parser.error("GPT-2 position embeddings support at most 1024 positions")
    if not 0 <= args.negative_scale_probability <= 1:
        parser.error("negative scale probability must lie in [0,1]")

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)

    arrows = [Path(item) for item in args.arrow]
    for path in arrows:
        if not path.is_file():
            raise FileNotFoundError(path)
    official_repo = Path(args.official_repo)
    provenance = repository_provenance(official_repo)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model, cache_dir=args.cache_dir, local_files_only=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model, cache_dir=args.cache_dir, local_files_only=True
    )
    model.eval()
    public_words = model.transformer.wte.weight.detach().float().cpu()
    public_positions = model.transformer.wpe.weight.detach().float().cpu()
    victim_words, private = install_private_rows(
        public_words, Path(args.private_checkpoint) if args.private_checkpoint else None
    )
    victim_positions = public_positions.clone()
    documents = load_documents(arrows, args.seed, args.max_documents)
    token_ids, prompt_text = collect_prompts(
        tokenizer,
        documents,
        args.prompt_count,
        args.prompt_length,
        args.seed + 17,
    )
    emit(
        "inputs_ready",
        prompts=args.prompt_count,
        prompt_length=args.prompt_length,
        vocabulary=public_words.shape[0],
        hidden=public_words.shape[1],
        private_kind=private["kind"],
    )

    started = time.time()
    trials = [
        run_trial(
            trial,
            token_ids,
            public_words,
            public_positions,
            victim_words,
            victim_positions,
            tokenizer,
            args.seed,
            args.representation,
            args.scale_log_min,
            args.scale_log_max,
            args.negative_scale_probability,
            args.dictionary_batch_size,
        )
        for trial in range(args.trials)
    ]
    result = {
        "experiment": "STIP NDSS 2026 final-paper embedding boundary full-prompt recovery",
        "paper_boundary_formula": "f(A) = alpha Emb(A) pi",
        "paper_section": "IV-E, TEE Integration for Enhanced Protection",
        "paper_sha256": FINAL_PAPER_SHA256,
        "implementation_scope": "paper-specification boundary replay; not a claim that the public repository contains the final TEE path",
        "official_repository": provenance,
        "model": args.model,
        "private_embedding": private,
        "representation": args.representation,
        "arrow_files": [str(path.resolve()) for path in arrows],
        "seed": args.seed,
        "trials": args.trials,
        "prompt_count": args.prompt_count,
        "prompt_length": args.prompt_length,
        "prompt_text_sha256": hashlib.sha256(
            "\n".join(prompt_text).encode("utf-8")
        ).hexdigest(),
        "scale_log_range": [args.scale_log_min, args.scale_log_max],
        "negative_scale_probability": args.negative_scale_probability,
        "single_transcript_per_prompt_per_trial": True,
        "trials_detail": trials,
        "aggregate": aggregate(trials),
        "elapsed_seconds": time.time() - started,
        "torch_version": torch.__version__,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    emit("complete", output=str(output), aggregate=result["aggregate"])


if __name__ == "__main__":
    main()

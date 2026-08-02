#!/usr/bin/env python3
"""Chosen-prompt codebook attack on STIP's final TEE boundary.

The final STIP paper argues that per-query row scaling prevents a party that
submits known prompts from accumulating a stable embedding dataset.  This
experiment tests that exact claim.  The attacker never reads the victim
embedding weights.  It submits known prompts, canonicalizes the observed
``alpha Emb(A) P`` rows, and stores a position-specific token codebook.  A
disjoint set of target transcripts is then decoded using only that codebook.

This script assumes the tokenizer is known, as in STIP's GPT-2 evaluation.  It
does not claim plaintext labels for a completely private, unaligned tokenizer.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from lib.stip_orbit import canonical_orbit, cosine_topk


def emit(event: str, **fields: Any) -> None:
    print(json.dumps({"event": event, **fields}, sort_keys=True), flush=True)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_documents(paths: list[Path], seed: int) -> list[str]:
    documents: list[str] = []
    for path in paths:
        dataset = Dataset.from_file(str(path))
        for record in dataset:
            title = str(record.get("title") or "").strip()
            body = str(record.get("text") or "").strip()
            joined = (title + "\n" + body).strip()
            if joined:
                documents.append(joined)
    random.Random(seed).shuffle(documents)
    if not documents:
        raise ValueError("no nonempty documents loaded")
    return documents


def sample_prompts(
    tokenizer,
    documents: list[str],
    count: int,
    length: int,
    seed: int,
) -> tuple[torch.Tensor, list[str]]:
    rng = random.Random(seed)
    order = list(range(len(documents)))
    rng.shuffle(order)
    token_rows: list[list[int]] = []
    texts: list[str] = []
    rounds = 0
    while len(token_rows) < count:
        added = 0
        for index in order:
            ids = tokenizer.encode(
                documents[index], add_special_tokens=False, verbose=False
            )
            if len(ids) < length:
                continue
            high = len(ids) - length
            start = rng.randint(0, high) if high else 0
            row = ids[start : start + length]
            token_rows.append(row)
            texts.append(tokenizer.decode(row))
            added += 1
            if len(token_rows) == count:
                break
        if added == 0:
            raise RuntimeError(f"could not sample {count} prompts of length {length}")
        rounds += 1
        rng.shuffle(order)
        if rounds > 100:
            raise RuntimeError("prompt sampler exceeded 100 passes")
    return torch.tensor(token_rows, dtype=torch.long), texts


def install_private_rows(
    base_rows: torch.Tensor, checkpoint: Path | None
) -> tuple[torch.Tensor, dict[str, Any]]:
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
        "kind": "unknown_to_attacker_partial_private_embedding",
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": sha256_file(checkpoint),
        "selected_rows": int(selected.numel()),
        "seed": int(payload["seed"]),
        "selection": str(payload["selection"]),
        "train_fraction_of_active": float(payload["train_fraction_of_active"]),
    }


def transformed_signatures(
    clean: torch.Tensor,
    generator: torch.Generator,
) -> tuple[torch.Tensor, float]:
    """Apply fresh row scales and a fresh feature permutation, then canonicalize."""
    permutation = torch.randperm(clean.shape[-1], generator=generator)
    scales = torch.empty(clean.shape[:-1], dtype=clean.dtype)
    scales.uniform_(-1.5, 1.5, generator=generator).exp_()
    signs = torch.rand(clean.shape[:-1], generator=generator) < 0.5
    scales[signs] *= -1
    transcript = clean[..., permutation] * scales[..., None]
    observed = canonical_orbit(transcript)
    reference = canonical_orbit(clean)
    error = float((observed - reference).abs().max().item())
    return observed, error


def observe_known_prompts(
    token_ids: torch.Tensor,
    start: int,
    stop: int,
    victim_words: torch.Tensor,
    victim_positions: torch.Tensor,
    codebooks: list[dict[int, torch.Tensor]],
    generator: torch.Generator,
    batch_size: int,
) -> dict[str, float | int]:
    max_invariance_error = 0.0
    max_repeat_error = 0.0
    duplicate_rows = 0
    positions = torch.arange(token_ids.shape[1])
    for batch_start in range(start, stop, batch_size):
        batch_stop = min(batch_start + batch_size, stop)
        batch_ids = token_ids[batch_start:batch_stop]
        clean = victim_words[batch_ids] + victim_positions[positions][None, :, :]
        observed, error = transformed_signatures(clean, generator)
        max_invariance_error = max(max_invariance_error, error)
        for row in range(batch_ids.shape[0]):
            for position in range(batch_ids.shape[1]):
                token = int(batch_ids[row, position])
                signature = observed[row, position].clone()
                previous = codebooks[position].get(token)
                if previous is None:
                    codebooks[position][token] = signature
                else:
                    duplicate_rows += 1
                    max_repeat_error = max(
                        max_repeat_error,
                        float((previous - signature).abs().max().item()),
                    )
        emit(
            "known_prompt_progress",
            completed=batch_stop,
            total=stop,
            codebook_entries=sum(len(book) for book in codebooks),
        )
    return {
        "max_invariance_error": max_invariance_error,
        "max_repeat_error": max_repeat_error,
        "duplicate_rows": duplicate_rows,
    }


def evaluate_codebook(
    target_ids: torch.Tensor,
    target_signatures: torch.Tensor,
    target_texts: list[str],
    codebooks: list[dict[int, torch.Tensor]],
    fallback_prediction: torch.Tensor,
    dictionary_batch_size: int,
    membership_threshold: float,
) -> dict[str, Any]:
    codebook_prediction = torch.full_like(target_ids, -1)
    hybrid_prediction = fallback_prediction.clone()
    coverage = torch.zeros_like(target_ids, dtype=torch.bool)
    exact_membership = torch.zeros_like(target_ids, dtype=torch.bool)
    dictionary_sizes: list[int] = []
    for position, book in enumerate(codebooks):
        labels = torch.tensor(list(book.keys()), dtype=torch.long)
        dictionary = torch.stack(list(book.values()))
        dictionary_sizes.append(int(labels.numel()))
        scores, indices = cosine_topk(
            target_signatures[:, position, :],
            dictionary,
            k=1,
            query_chunk=dictionary_batch_size,
        )
        labels_at_top1 = labels[indices[:, 0]]
        codebook_prediction[:, position] = labels_at_top1
        members = scores[:, 0] >= membership_threshold
        exact_membership[:, position] = members
        hybrid_prediction[members, position] = labels_at_top1[members]
        known = set(book)
        coverage[:, position] = torch.tensor(
            [int(token) in known for token in target_ids[:, position]],
            dtype=torch.bool,
        )

    codebook_correct = codebook_prediction.eq(target_ids)
    fallback_correct = fallback_prediction.eq(target_ids)
    hybrid_correct = hybrid_prediction.eq(target_ids)
    full_covered = coverage.all(dim=1)
    full_codebook_exact = codebook_correct.all(dim=1)
    full_fallback_exact = fallback_correct.all(dim=1)
    full_hybrid_exact = hybrid_correct.all(dim=1)
    covered_count = int(coverage.sum().item())
    examples = []
    error_indices = torch.nonzero(~full_hybrid_exact, as_tuple=False).flatten().tolist()
    for index in error_indices[:8]:
        examples.append(
            {
                "target": target_texts[index],
                "prediction": "<contains-unmapped-token>",
                "covered_tokens": int(coverage[index].sum().item()),
                "tokens": int(target_ids.shape[1]),
            }
        )
    return {
        "dictionary_entries_by_position": dictionary_sizes,
        "dictionary_entries_mean": float(np.mean(dictionary_sizes)),
        "token_coverage": float(coverage.float().mean().item()),
        "exact_membership_rate": float(exact_membership.float().mean().item()),
        "membership_coverage_agreement": float(
            exact_membership.eq(coverage).float().mean().item()
        ),
        "codebook_token_top1": float(codebook_correct.float().mean().item()),
        "fallback_token_top1": float(fallback_correct.float().mean().item()),
        "hybrid_token_top1": float(hybrid_correct.float().mean().item()),
        "conditional_accuracy_given_coverage": (
            float(codebook_correct[coverage].float().mean().item())
            if covered_count
            else None
        ),
        "full_prompt_covered": float(full_covered.float().mean().item()),
        "codebook_full_prompt_exact": float(full_codebook_exact.float().mean().item()),
        "fallback_full_prompt_exact": float(full_fallback_exact.float().mean().item()),
        "hybrid_full_prompt_exact": float(full_hybrid_exact.float().mean().item()),
        "tokens": int(target_ids.numel()),
        "prompts": int(target_ids.shape[0]),
        "examples": examples,
    }


def public_fallback_decode(
    target_signatures: torch.Tensor,
    public_words: torch.Tensor,
    public_positions: torch.Tensor,
    dictionary_batch_size: int,
) -> torch.Tensor:
    prediction = torch.empty(target_signatures.shape[:2], dtype=torch.long)
    for position in range(target_signatures.shape[1]):
        dictionary = canonical_orbit(public_words + public_positions[position])
        _, indices = cosine_topk(
            target_signatures[:, position, :],
            dictionary,
            k=1,
            query_chunk=dictionary_batch_size,
        )
        prediction[:, position] = indices[:, 0]
        emit(
            "public_fallback_progress",
            position=position,
            positions=target_signatures.shape[1],
        )
    return prediction


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="gpt2")
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--arrow", action="append", required=True)
    parser.add_argument("--private-checkpoint")
    parser.add_argument("--known-budgets", default="16,64,256,1024,4096")
    parser.add_argument("--target-count", type=int, default=256)
    parser.add_argument("--prompt-length", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--dictionary-batch-size", type=int, default=256)
    parser.add_argument("--membership-threshold", type=float, default=0.99999)
    parser.add_argument("--threads", type=int, default=16)
    parser.add_argument("--seed", type=int, default=20260802)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    started = time.time()
    torch.set_num_threads(args.threads)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    budgets = sorted({int(value) for value in args.known_budgets.split(",")})
    if not budgets or budgets[0] <= 0:
        parser.error("known budgets must be positive")

    tokenizer = AutoTokenizer.from_pretrained(
        args.model, cache_dir=args.cache_dir, local_files_only=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model, cache_dir=args.cache_dir, local_files_only=True
    )
    public_words = model.get_input_embeddings().weight.detach().cpu().float()
    public_positions = model.transformer.wpe.weight.detach().cpu().float()
    victim_words, private = install_private_rows(
        public_words,
        Path(args.private_checkpoint) if args.private_checkpoint else None,
    )
    if args.prompt_length > public_positions.shape[0]:
        parser.error("prompt length exceeds model position table")

    documents = load_documents([Path(path) for path in args.arrow], args.seed)
    split = max(1, int(0.7 * len(documents)))
    known_docs = documents[:split]
    target_docs = documents[split:]
    if not target_docs:
        raise RuntimeError("document split left no target documents")
    known_ids, _ = sample_prompts(
        tokenizer, known_docs, budgets[-1], args.prompt_length, args.seed + 11
    )
    target_ids, target_texts = sample_prompts(
        tokenizer, target_docs, args.target_count, args.prompt_length, args.seed + 29
    )
    overlap = set(map(tuple, known_ids.tolist())) & set(map(tuple, target_ids.tolist()))
    if overlap:
        raise AssertionError("known and target prompt token sequences overlap")

    positions = torch.arange(args.prompt_length)
    target_clean = victim_words[target_ids] + public_positions[positions][None, :, :]
    generator = torch.Generator().manual_seed(args.seed + 101)
    target_signatures, target_invariance_error = transformed_signatures(
        target_clean, generator
    )
    fallback_prediction = public_fallback_decode(
        target_signatures,
        public_words,
        public_positions,
        args.dictionary_batch_size,
    )

    codebooks: list[dict[int, torch.Tensor]] = [
        {} for _ in range(args.prompt_length)
    ]
    results = []
    previous_budget = 0
    max_known_invariance_error = 0.0
    max_repeat_error = 0.0
    duplicate_rows = 0
    for budget in budgets:
        audit = observe_known_prompts(
            known_ids,
            previous_budget,
            budget,
            victim_words,
            public_positions,
            codebooks,
            generator,
            args.batch_size,
        )
        max_known_invariance_error = max(
            max_known_invariance_error, float(audit["max_invariance_error"])
        )
        max_repeat_error = max(max_repeat_error, float(audit["max_repeat_error"]))
        duplicate_rows += int(audit["duplicate_rows"])
        metrics = evaluate_codebook(
            target_ids,
            target_signatures,
            target_texts,
            codebooks,
            fallback_prediction,
            args.dictionary_batch_size,
            args.membership_threshold,
        )
        metrics["known_prompt_queries"] = budget
        results.append(metrics)
        emit(
            "budget_complete",
            known_prompt_queries=budget,
            token_coverage=metrics["token_coverage"],
            fallback_token_top1=metrics["fallback_token_top1"],
            hybrid_token_top1=metrics["hybrid_token_top1"],
            hybrid_full_prompt_exact=metrics["hybrid_full_prompt_exact"],
        )
        previous_budget = budget

    output = {
        "experiment": "chosen-prompt orbit codebook attack on STIP final TEE boundary",
        "paper_boundary_formula": "f(A) = alpha Emb(A) pi",
        "attack_scope": (
            "known tokenizer and chosen-prompt transcript access; victim embedding "
            "weights are not read by the attacker"
        ),
        "non_claim": (
            "does not assign plaintext labels for a completely private, unaligned tokenizer"
        ),
        "model": args.model,
        "private_embedding": private,
        "corpus_arrow_files": [str(Path(path).resolve()) for path in args.arrow],
        "document_split": {
            "known_documents": len(known_docs),
            "target_documents": len(target_docs),
            "exact_prompt_overlap": 0,
        },
        "prompt_length": args.prompt_length,
        "target_prompts": args.target_count,
        "known_budgets": budgets,
        "membership_threshold": args.membership_threshold,
        "results": results,
        "invariance_audit": {
            "target_max_abs_error": target_invariance_error,
            "known_max_abs_error": max_known_invariance_error,
            "repeat_max_abs_error": max_repeat_error,
            "duplicate_observations": duplicate_rows,
        },
        "seed": args.seed,
        "runtime": {
            "host": platform.node(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "threads": args.threads,
            "elapsed_seconds": time.time() - started,
        },
    }
    write_json(Path(args.output), output)
    emit("complete", output=str(Path(args.output).resolve()))


if __name__ == "__main__":
    main()

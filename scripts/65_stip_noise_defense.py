#!/usr/bin/env python3
"""Evaluate additive-noise mitigation against adaptive STIP orbit recovery.

The embedding TEE releases alpha @ (Emb(A) + noise) @ P.  Noise is fresh for
every authorized request.  We measure exact prompt recovery after averaging
1/2/4/8/16 independently noised transcripts and jointly measure GPT-2 loss,
next-token agreement, and KL divergence.  This is a mitigation study, not a
claim that additive noise provides a cryptographic privacy guarantee.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import random
import time
from pathlib import Path
from typing import Any

import faiss
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from lib.stip_orbit import apply_final_boundary, canonical_orbit


def load_prompt_helpers():
    path = Path(__file__).with_name("63_stip_final_full_prompt.py")
    spec = importlib.util.spec_from_file_location("stip_prompt_helpers", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import prompt helpers from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


PROMPTS = load_prompt_helpers()


def emit(event: str, **fields: Any) -> None:
    print(json.dumps({"event": event, **fields}, sort_keys=True), flush=True)


def exact_search(queries: torch.Tensor, dictionary: torch.Tensor) -> torch.Tensor:
    q = queries.detach().float().cpu().numpy().astype("float32")
    d = dictionary.detach().float().cpu().numpy().astype("float32")
    faiss.normalize_L2(q)
    faiss.normalize_L2(d)
    index = faiss.IndexFlatIP(d.shape[1])
    index.add(d)
    _, ids = index.search(q, 1)
    return torch.from_numpy(ids[:, 0].copy()).long()


def utility_logits(model, token_ids, word_rows, noise, batch_size):
    logits = []
    weighted_loss = 0.0
    token_count = 0
    with torch.no_grad():
        for start in range(0, token_ids.shape[0], batch_size):
            ids = token_ids[start : start + batch_size]
            embedding = word_rows[ids]
            if noise is not None:
                embedding = embedding + noise[start : start + ids.shape[0]]
            output = model(inputs_embeds=embedding, labels=ids, use_cache=False)
            logits.append(output.logits.detach().half().cpu())
            count = int(ids.shape[0] * (ids.shape[1] - 1))
            weighted_loss += float(output.loss.item()) * count
            token_count += count
    return torch.cat(logits), weighted_loss / token_count


def utility_metrics(clean_logits, noisy_logits, loss):
    clean = clean_logits[:, :-1].float()
    noisy = noisy_logits[:, :-1].float()
    clean_logp = torch.log_softmax(clean, dim=-1)
    noisy_logp = torch.log_softmax(noisy, dim=-1)
    kl = torch.sum(clean_logp.exp() * (clean_logp - noisy_logp), dim=-1)
    agreement = clean.argmax(dim=-1).eq(noisy.argmax(dim=-1))
    return {
        "cross_entropy": loss,
        "perplexity": float(math.exp(min(loss, 50))),
        "clean_top1_agreement": float(agreement.float().mean()),
        "clean_to_noisy_kl_mean": float(kl.mean()),
        "clean_to_noisy_kl_p95": float(torch.quantile(kl.flatten(), 0.95)),
    }


def evaluate_utility(
    model,
    token_ids,
    word_rows,
    sigmas,
    trials,
    batch_size,
    seed,
):
    model.eval()
    clean_logits, clean_loss = utility_logits(
        model, token_ids, word_rows, None, batch_size
    )
    rows = []
    for sigma in sigmas:
        for trial in range(trials):
            generator = torch.Generator().manual_seed(
                seed + 100003 * trial + int(round(sigma * 1e8))
            )
            noise = torch.randn(
                token_ids.shape[0],
                token_ids.shape[1],
                word_rows.shape[1],
                generator=generator,
            ) * sigma
            noisy_logits, loss = utility_logits(
                model, token_ids, word_rows, noise, batch_size
            )
            row = utility_metrics(clean_logits, noisy_logits, loss)
            row.update({"sigma": sigma, "trial": trial})
            rows.append(row)
            emit(
                "utility_complete",
                sigma=sigma,
                trial=trial,
                agreement=round(row["clean_top1_agreement"], 6),
                loss=round(loss, 6),
            )
    aggregate = []
    for sigma in sigmas:
        selected = [row for row in rows if row["sigma"] == sigma]
        aggregate.append(
            {
                "sigma": sigma,
                "cross_entropy_mean": float(
                    np.mean([row["cross_entropy"] for row in selected])
                ),
                "perplexity_mean": float(
                    np.mean([row["perplexity"] for row in selected])
                ),
                "clean_top1_agreement_mean": float(
                    np.mean([row["clean_top1_agreement"] for row in selected])
                ),
                "clean_top1_agreement_min": float(
                    np.min([row["clean_top1_agreement"] for row in selected])
                ),
                "clean_to_noisy_kl_mean": float(
                    np.mean([row["clean_to_noisy_kl_mean"] for row in selected])
                ),
            }
        )
    return {
        "clean_cross_entropy": clean_loss,
        "clean_perplexity": float(math.exp(min(clean_loss, 50))),
        "trials": rows,
        "aggregate": aggregate,
    }


def evaluate_attack(
    token_ids,
    word_rows,
    positions,
    sigmas,
    repeats,
    trials,
    seed,
):
    prompt_count, prompt_length = token_ids.shape
    hidden = word_rows.shape[1]
    max_repeats = max(repeats)
    keys = [
        (sigma, trial, repeat)
        for sigma in sigmas
        for trial in range(trials)
        for repeat in repeats
    ]
    correct_tokens = {key: 0 for key in keys}
    prompt_correct = {
        key: torch.ones(prompt_count, dtype=torch.bool) for key in keys
    }
    invariant_error = 0.0
    started = time.time()

    for position in range(prompt_length):
        clean_dictionary = canonical_orbit(word_rows + positions[position])
        queries = []
        metadata = []
        clean_rows = word_rows[token_ids[:, position]] + positions[position]
        for sigma in sigmas:
            for trial in range(trials):
                generator = torch.Generator().manual_seed(
                    seed
                    + 1000003 * trial
                    + 1009 * position
                    + int(round(sigma * 1e8))
                )
                permutation = torch.randperm(hidden, generator=generator)
                running = torch.zeros_like(clean_rows)
                for observation in range(1, max_repeats + 1):
                    noise = (
                        torch.randn(
                            clean_rows.shape, generator=generator
                        )
                        * sigma
                    )
                    scales = torch.empty(prompt_count).uniform_(
                        -8, 8, generator=generator
                    ).exp_()
                    transcript = apply_final_boundary(
                        (clean_rows + noise)[:, None, :],
                        scales[:, None],
                        permutation,
                    )[:, 0, :]
                    signature = canonical_orbit(transcript)
                    if sigma == 0:
                        invariant_error = max(
                            invariant_error,
                            float(
                                (
                                    signature
                                    - canonical_orbit(clean_rows)
                                )
                                .abs()
                                .max()
                            ),
                        )
                    running += signature
                    if observation in repeats:
                        averaged = torch.nn.functional.normalize(
                            running / observation, dim=1
                        )
                        queries.append(averaged)
                        metadata.append((sigma, trial, observation))
        predictions = exact_search(
            torch.cat(queries, dim=0), clean_dictionary
        )
        offset = 0
        for key in metadata:
            block = predictions[offset : offset + prompt_count]
            offset += prompt_count
            correct = block.eq(token_ids[:, position])
            correct_tokens[key] += int(correct.sum())
            prompt_correct[key] &= correct
        emit(
            "attack_position_complete",
            position=position,
            positions=prompt_length,
            elapsed=round(time.time() - started, 3),
        )

    details = []
    for sigma, trial, repeat in keys:
        key = (sigma, trial, repeat)
        details.append(
            {
                "sigma": sigma,
                "trial": trial,
                "repeat_queries": repeat,
                "token_top1": correct_tokens[key]
                / float(prompt_count * prompt_length),
                "full_prompt_exact": float(
                    prompt_correct[key].float().mean()
                ),
                "full_prompt_exact_count": int(prompt_correct[key].sum()),
            }
        )
    aggregate = []
    for sigma in sigmas:
        for repeat in repeats:
            selected = [
                row
                for row in details
                if row["sigma"] == sigma
                and row["repeat_queries"] == repeat
            ]
            aggregate.append(
                {
                    "sigma": sigma,
                    "repeat_queries": repeat,
                    "token_top1_mean": float(
                        np.mean([row["token_top1"] for row in selected])
                    ),
                    "token_top1_min": float(
                        np.min([row["token_top1"] for row in selected])
                    ),
                    "full_prompt_exact_mean": float(
                        np.mean(
                            [row["full_prompt_exact"] for row in selected]
                        )
                    ),
                }
            )
    return {
        "fresh_independent_noise_per_authorized_query": True,
        "details": details,
        "aggregate": aggregate,
        "zero_noise_invariance_max_abs_error": invariant_error,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="gpt2")
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--arrow", action="append", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--sigmas",
        type=float,
        nargs="+",
        default=[0.0, 0.001, 0.0025, 0.005, 0.01, 0.02, 0.05],
    )
    parser.add_argument(
        "--repeat-queries",
        type=int,
        nargs="+",
        default=[1, 2, 4, 8, 16],
    )
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--prompt-count", type=int, default=32)
    parser.add_argument("--prompt-length", type=int, default=16)
    parser.add_argument("--utility-batch-size", type=int, default=4)
    parser.add_argument("--max-documents", type=int, default=0)
    parser.add_argument("--threads", type=int, default=16)
    parser.add_argument("--seed", type=int, default=20260802)
    args = parser.parse_args()

    if any(value < 0 for value in args.sigmas):
        parser.error("sigmas must be nonnegative")
    if any(value <= 0 for value in args.repeat_queries):
        parser.error("repeat query counts must be positive")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)

    tokenizer = AutoTokenizer.from_pretrained(
        args.model, cache_dir=args.cache_dir, local_files_only=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model, cache_dir=args.cache_dir, local_files_only=True
    )
    model.eval()
    paths = [Path(value) for value in args.arrow]
    documents = PROMPTS.load_documents(
        paths, args.seed, args.max_documents
    )
    token_ids, prompt_text = PROMPTS.collect_prompts(
        tokenizer,
        documents,
        args.prompt_count,
        args.prompt_length,
        args.seed + 17,
    )
    word_rows = (
        model.transformer.wte.weight.detach().float().cpu().clone()
    )
    positions = (
        model.transformer.wpe.weight.detach().float().cpu().clone()
    )
    embedding_std = float(word_rows.std())
    emit(
        "inputs_ready",
        prompts=args.prompt_count,
        length=args.prompt_length,
        embedding_std=embedding_std,
    )

    utility = evaluate_utility(
        model,
        token_ids,
        word_rows,
        args.sigmas,
        args.trials,
        args.utility_batch_size,
        args.seed + 100,
    )
    attack = evaluate_attack(
        token_ids,
        word_rows,
        positions,
        args.sigmas,
        args.repeat_queries,
        args.trials,
        args.seed + 200,
    )
    result = {
        "experiment": "STIP additive-noise defense under repeated-query orbit recovery",
        "model": args.model,
        "seed": args.seed,
        "sigmas_absolute_embedding_units": args.sigmas,
        "sigma_to_embedding_std": [
            value / embedding_std for value in args.sigmas
        ],
        "repeat_queries": args.repeat_queries,
        "trials": args.trials,
        "prompt_count": args.prompt_count,
        "prompt_length": args.prompt_length,
        "prompt_text_sha256": __import__("hashlib").sha256(
            "\n".join(prompt_text).encode()
        ).hexdigest(),
        "utility": utility,
        "attack": attack,
        "scope": {
            "noise_location": "inside embedding TEE before alpha and P",
            "downstream_semantics": "noise remains in the model input after trusted alpha/P correction",
            "claim": "empirical privacy-utility and repeated-query evaluation, not a proof of security",
        },
        "torch_version": torch.__version__,
        "faiss_version": faiss.__version__,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    emit("complete", output=str(output))


if __name__ == "__main__":
    main()

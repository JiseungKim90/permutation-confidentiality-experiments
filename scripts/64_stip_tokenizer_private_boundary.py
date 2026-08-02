#!/usr/bin/env python3
"""Train and attack STIP with a changed private tokenizer.

Extension mode preserves GPT-2's public vocabulary and trains added domain
rows. BPE mode trains an independently resegmented byte-level vocabulary and
adapts all embedding rows while the Transformer is frozen. Both modes measure
semantic recovery separately from pseudotoken linkage.
"""

from __future__ import annotations

import argparse
import collections
import copy
import hashlib
import importlib.util
import json
import os
import random
import re
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from tokenizers import ByteLevelBPETokenizer
from transformers import AutoModelForCausalLM, AutoTokenizer

from lib.stip_orbit import apply_final_boundary, canonical_orbit, cosine_topk


def load_training_helpers():
    path = Path(__file__).with_name("60_stip_independent_embedding_training.py")
    spec = importlib.util.spec_from_file_location("stip_embedding_training", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import training helpers from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


TRAINING = load_training_helpers()


def emit(event: str, **fields: Any) -> None:
    print(json.dumps({"event": event, **fields}, sort_keys=True), flush=True)


def encode(tokenizer, mode: str, text: str) -> list[int]:
    if mode == "extension":
        return tokenizer.encode(text, add_special_tokens=False, verbose=False)
    return tokenizer.encode(text).ids


def decode(tokenizer, mode: str, ids: list[int]) -> str:
    if mode == "extension":
        return tokenizer.decode(ids, clean_up_tokenization_spaces=False)
    return tokenizer.decode(ids)


def vocab_size(tokenizer, mode: str) -> int:
    return len(tokenizer) if mode == "extension" else tokenizer.get_vocab_size()


def choose_extension_tokens(documents, base_tokenizer, count: int) -> list[str]:
    frequency: collections.Counter[str] = collections.Counter()
    for document in documents:
        frequency.update(re.findall(r"[A-Za-z][A-Za-z0-9-]{5,}", document.lower()))
    selected: list[str] = []
    for word, _ in frequency.most_common():
        candidate = " " + word
        pieces = base_tokenizer.encode(
            candidate, add_special_tokens=False, verbose=False
        )
        if len(pieces) >= 2:
            selected.append(candidate)
        if len(selected) == count:
            break
    if len(selected) < count:
        raise RuntimeError(f"found only {len(selected)} extension candidates")
    return selected


def make_tokenizer(mode, documents, base_tokenizer, extension_tokens, bpe_size):
    if mode == "extension":
        tokenizer = copy.deepcopy(base_tokenizer)
        tokens = choose_extension_tokens(documents, base_tokenizer, extension_tokens)
        added = tokenizer.add_tokens(tokens)
        if added != len(tokens):
            raise AssertionError("not every selected extension token was added")
        return tokenizer, {"added_tokens": tokens, "added_count": added}
    tokenizer = ByteLevelBPETokenizer()
    tokenizer.train_from_iterator(
        documents,
        vocab_size=bpe_size,
        min_frequency=2,
        special_tokens=[base_tokenizer.eos_token],
    )
    return tokenizer, {"requested_bpe_vocabulary": bpe_size}


def compositional_rows(private_tokenizer, mode, base_tokenizer, base_rows):
    size = vocab_size(private_tokenizer, mode)
    rows = torch.empty((size, base_rows.shape[1]), dtype=base_rows.dtype)
    decompositions: list[list[int]] = []
    strings: list[str] = []
    start = 0
    if mode == "extension":
        base_size = len(base_tokenizer)
        rows[:base_size] = base_rows
        decompositions.extend([[token_id] for token_id in range(base_size)])
        strings.extend(
            str(value)
            for value in base_tokenizer.convert_ids_to_tokens(
                list(range(base_size))
            )
        )
        start = base_size
    for token_id in range(start, size):
        text = decode(private_tokenizer, mode, [token_id])
        strings.append(text)
        pieces = base_tokenizer.encode(
            text, add_special_tokens=False, verbose=False
        )
        if not pieces:
            pieces = [int(base_tokenizer.eos_token_id)]
        rows[token_id] = base_rows[torch.tensor(pieces)].mean(dim=0)
        decompositions.append(pieces)
    return rows, decompositions, strings



def tokenize_stream(tokenizer, mode, documents, max_tokens, eos_id):
    stream: list[int] = []
    for document in documents:
        stream.extend(encode(tokenizer, mode, document))
        stream.append(eos_id)
        if len(stream) >= max_tokens:
            break
    if len(stream) < 1000:
        raise RuntimeError("private tokenizer produced too few tokens")
    return torch.tensor(stream[:max_tokens], dtype=torch.long)


def collect_prompts(
    tokenizer, mode, documents, count, length, seed, required_ids
):
    generator = random.Random(seed)
    required = set(required_ids.tolist())
    hits: list[list[int]] = []
    others: list[list[int]] = []
    for document in documents:
        ids = encode(tokenizer, mode, document)
        if len(ids) < length:
            continue
        locations = [index for index, value in enumerate(ids) if value in required]
        if locations:
            anchor = generator.choice(locations)
            low = max(0, anchor - length + 1)
            high = min(anchor, len(ids) - length)
            start = generator.randint(low, high)
        else:
            start = generator.randrange(len(ids) - length + 1)
        window = ids[start : start + length]
        (hits if any(value in required for value in window) else others).append(window)
    generator.shuffle(hits)
    generator.shuffle(others)
    wanted_hits = count if mode == "bpe" else count // 2
    chosen = hits[:wanted_hits]
    chosen.extend(others[: count - len(chosen)])
    if len(chosen) < count:
        chosen.extend(hits[len(chosen) : len(chosen) + count - len(chosen)])
    if len(chosen) < count:
        raise RuntimeError(
            f"only {len(chosen)} prompts available; hits={len(hits)}, others={len(others)}"
        )
    generator.shuffle(chosen)
    hit_count = sum(any(value in required for value in row) for row in chosen)
    return (
        torch.tensor(chosen),
        [decode(tokenizer, mode, ids) for ids in chosen],
        hit_count,
    )



def train_rows(
    model, initial_rows, selected_ids, train_stream, validation_stream, args
):
    device = torch.device(args.device)
    model = model.to(device)
    model.config.use_cache = False
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    embedding = model.get_input_embeddings().weight
    embedding.data.copy_(initial_rows.to(device))
    model.tie_weights()
    embedding.requires_grad_(True)
    mask = torch.zeros(
        (embedding.shape[0], 1), dtype=embedding.dtype, device=device
    )
    mask[selected_ids.to(device)] = 1
    hook = embedding.register_hook(lambda gradient: gradient * mask)
    optimizer = torch.optim.AdamW(
        [embedding], lr=args.learning_rate, weight_decay=0
    )
    before = TRAINING.evaluate_loss(
        model,
        validation_stream,
        args.seed + 101,
        args.eval_batches,
        args.batch_size,
        args.sequence_length,
        device,
    )
    generator = torch.Generator().manual_seed(args.seed + 202)
    losses: list[float] = []
    started = time.time()
    model.train()
    for step in range(1, args.steps + 1):
        x, y = TRAINING.sample_batch(
            train_stream,
            generator,
            args.batch_size,
            args.sequence_length,
            device,
        )
        optimizer.zero_grad(set_to_none=True)
        loss = model(input_ids=x, labels=y, use_cache=False).loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_([embedding], 1)
        optimizer.step()
        losses.append(float(loss.item()))
        if step == 1 or step % args.log_every == 0 or step == args.steps:
            emit(
                "train_progress",
                step=step,
                loss=losses[-1],
                recent=float(np.mean(losses[-args.log_every :])),
                elapsed=round(time.time() - started, 3),
            )
    hook.remove()
    after = TRAINING.evaluate_loss(
        model,
        validation_stream,
        args.seed + 101,
        args.eval_batches,
        args.batch_size,
        args.sequence_length,
        device,
    )
    trained = embedding.detach().float().cpu().clone()
    delta = (trained - initial_rows).abs().amax(dim=1)
    selected_mask = torch.zeros(initial_rows.shape[0], dtype=torch.bool)
    selected_mask[selected_ids] = True
    if bool((delta[~selected_mask] > 0).any()):
        raise AssertionError("an unselected embedding row changed")
    return trained, {
        "validation_loss_before": before,
        "validation_loss_after": after,
        "training_loss_first": losses[0],
        "training_loss_last": losses[-1],
        "selected_rows": int(selected_ids.numel()),
        "changed_selected_rows": int((delta[selected_ids] > 0).sum()),
        "changed_unselected_rows": int((delta[~selected_mask] > 0).sum()),
        "selected_delta_linf_median": float(delta[selected_ids].median()),
        "elapsed_seconds": time.time() - started,
    }


def evaluate_coverage(
    token_ids, transcript, priors, positions, known_ids
):
    if known_ids.numel() == 0:
        return {
            "known_rows": 0,
            "known_prompt_tokens": 0,
            "token_top1": 0.0,
            "known_token_top1": 0.0,
            "full_prompt_exact": 0.0,
        }
    prediction = torch.full_like(token_ids, -1)
    for position in range(token_ids.shape[1]):
        dictionary = canonical_orbit(priors[known_ids] + positions[position])
        query = canonical_orbit(transcript[:, position, :])
        _, local = cosine_topk(query, dictionary, k=1)
        prediction[:, position] = known_ids[local[:, 0]]
    correct = prediction.eq(token_ids)
    known = torch.isin(token_ids, known_ids)
    return {
        "known_rows": int(known_ids.numel()),
        "known_prompt_tokens": int(known.sum()),
        "token_top1": float(correct.float().mean()),
        "known_token_top1": (
            float(correct[known].float().mean()) if bool(known.any()) else 0.0
        ),
        "full_prompt_exact": float(correct.all(dim=1).float().mean()),
        "token_errors": int((~correct).sum()),
    }


def attack(
    token_ids, trained, priors, positions, always_known, frequency_order, seed
):
    clean = trained[token_ids] + positions[: token_ids.shape[1]][None, :, :]
    generator = torch.Generator().manual_seed(seed)
    permutation = torch.randperm(clean.shape[-1], generator=generator)
    scales = torch.empty(token_ids.shape).uniform_(
        -8, 8, generator=generator
    ).exp_()
    transcript = apply_final_boundary(clean, scales, permutation)
    permutation2 = torch.randperm(clean.shape[-1], generator=generator)
    scales2 = torch.empty(token_ids.shape).uniform_(
        -8, 8, generator=generator
    ).exp_()
    transcript2 = apply_final_boundary(clean, scales2, permutation2)
    linkage_error = float(
        (
            canonical_orbit(transcript) - canonical_orbit(transcript2)
        ).abs().max()
    )
    fixed = set(always_known.tolist())
    additional = [
        value for value in frequency_order.tolist() if value not in fixed
    ]
    evaluations = []
    for coverage in [0.0, 0.25, 0.5, 0.75, 1.0]:
        revealed = int(round(coverage * len(additional)))
        known = torch.tensor(
            sorted(fixed | set(additional[:revealed])), dtype=torch.long
        )
        row = evaluate_coverage(
            token_ids, transcript, priors, positions, known
        )
        row.update(
            {
                "private_vocabulary_coverage": coverage,
                "additional_rows_revealed": revealed,
            }
        )
        evaluations.append(row)
    rounded = torch.round(
        canonical_orbit(trained) * 1_000_000
    ).to(torch.int32)
    unique = torch.unique(rounded, dim=0).shape[0]
    return {
        "single_transcript_per_prompt": True,
        "coverage": evaluations,
        "pseudotoken_linkage_max_abs_error": linkage_error,
        "pseudotoken_linkage_exact_within_tolerance": linkage_error <= 2e-6,
        "private_signature_unique_rows_at_1e-6": int(unique),
        "private_signature_collision_rows_at_1e-6": int(
            trained.shape[0] - unique
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("extension", "bpe"), required=True)
    parser.add_argument("--model", default="gpt2")
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--arrow", action="append", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--extension-tokens", type=int, default=256)
    parser.add_argument("--bpe-vocab-size", type=int, default=4096)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--sequence-length", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--eval-batches", type=int, default=10)
    parser.add_argument("--max-documents", type=int, default=0)
    parser.add_argument("--max-tokens", type=int, default=300000)
    parser.add_argument("--prompt-count", type=int, default=64)
    parser.add_argument("--prompt-length", type=int, default=16)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--threads", type=int, default=16)
    parser.add_argument("--seed", type=int, default=20260802)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)
    paths = [Path(item) for item in args.arrow]
    documents = TRAINING.load_documents(
        paths, args.seed, args.max_documents
    )
    base_tokenizer = AutoTokenizer.from_pretrained(
        args.model, cache_dir=args.cache_dir, local_files_only=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model, cache_dir=args.cache_dir, local_files_only=True
    )
    base_rows = (
        model.get_input_embeddings().weight.detach().float().cpu().clone()
    )
    positions = (
        model.transformer.wpe.weight.detach().float().cpu().clone()
    )
    private_tokenizer, tokenizer_meta = make_tokenizer(
        args.mode,
        documents,
        base_tokenizer,
        args.extension_tokens,
        args.bpe_vocab_size,
    )
    priors, decompositions, token_strings = compositional_rows(
        private_tokenizer, args.mode, base_tokenizer, base_rows
    )
    private_size = priors.shape[0]
    model.resize_token_embeddings(private_size, mean_resizing=False)
    model.get_input_embeddings().weight.data.copy_(priors)
    model.tie_weights()
    if args.mode == "extension":
        eos_id = int(base_tokenizer.eos_token_id)
        selected_ids = torch.arange(len(base_tokenizer), private_size)
        always_known = torch.arange(len(base_tokenizer))
    else:
        eos_id = int(
            private_tokenizer.token_to_id(base_tokenizer.eos_token)
        )
        selected_ids = torch.arange(private_size)
        always_known = torch.empty(0, dtype=torch.long)
    stream = tokenize_stream(
        private_tokenizer,
        args.mode,
        documents,
        args.max_tokens,
        eos_id,
    )
    split = int(stream.numel() * 0.9)
    counts = torch.bincount(stream[:split], minlength=private_size)
    frequency_order = torch.argsort(
        counts, descending=True, stable=True
    )
    emit(
        "tokenizer_ready",
        mode=args.mode,
        vocabulary=private_size,
        selected_rows=int(selected_ids.numel()),
        tokens=int(stream.numel()),
    )
    trained, training = train_rows(
        model,
        priors,
        selected_ids,
        stream[:split],
        stream[split:],
        args,
    )
    prompt_ids, prompt_texts, prompts_with_selected = collect_prompts(
        private_tokenizer,
        args.mode,
        documents,
        args.prompt_count,
        args.prompt_length,
        args.seed + 303,
        selected_ids,
    )
    attack_result = attack(
        prompt_ids,
        trained,
        priors,
        positions,
        always_known,
        frequency_order,
        args.seed + 404,
    )

    checkpoint = Path(args.checkpoint_dir)
    checkpoint.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "mode": args.mode,
            "model": args.model,
            "seed": args.seed,
            "trained_rows": trained,
            "initial_rows": priors,
            "selected_ids": selected_ids,
            "token_strings": token_strings,
            "decompositions": decompositions,
        },
        checkpoint / "private_embedding_rows.pt",
    )
    if args.mode == "extension":
        private_tokenizer.save_pretrained(checkpoint / "tokenizer")
    else:
        private_tokenizer.save(str(checkpoint / "tokenizer.json"))
    reindex = torch.randperm(
        private_size,
        generator=torch.Generator().manual_seed(args.seed + 505),
    )
    result = {
        "experiment": (
            "STIP private-tokenizer semantic-to-pseudotoken boundary"
        ),
        "mode": args.mode,
        "model": args.model,
        "seed": args.seed,
        "tokenizer": tokenizer_meta,
        "private_vocabulary": private_size,
        "vocabulary_sha256": hashlib.sha256(
            "\n".join(token_strings).encode()
        ).hexdigest(),
        "training": training,
        "attack": attack_result,
        "prompt_count": args.prompt_count,
        "prompts_with_private_changed_rows": prompts_with_selected,
        "prompt_length": args.prompt_length,
        "prompt_text_sha256": hashlib.sha256(
            "\n".join(prompt_texts).encode()
        ).hexdigest(),
        "secret_id_reindex_fixed_point_fraction": float(
            reindex.eq(torch.arange(private_size)).float().mean()
        ),
        "scope": {
            "semantic_recovery_requires_candidate_strings_or_alignment": True,
            "zero_coverage_claim": "pseudotoken linkage only",
            "bpe_is_frozen_transformer_adaptation_not_production_retraining": (
                args.mode == "bpe"
            ),
        },
        "checkpoint_dir": str(checkpoint),
        "torch_version": torch.__version__,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    emit(
        "complete",
        output=str(output),
        coverage=attack_result["coverage"],
    )


if __name__ == "__main__":
    main()

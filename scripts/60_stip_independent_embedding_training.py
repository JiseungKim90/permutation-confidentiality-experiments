#!/usr/bin/env python3
"""Independently train a selected subset of GPT-2 embedding rows.

The experiment replaces the controlled row-replacement ablation with an actual
language-model adaptation run.  All transformer parameters are frozen.  Only a
predeclared subset of the tied input/output embedding rows may receive a
gradient.  The output records training utility, exact changed-row accounting,
and public-dictionary scale--permutation orbit recovery.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import random
import time
from collections import Counter
from pathlib import Path

import faiss
import numpy as np
import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer


def emit(event: str, **fields) -> None:
    print(json.dumps({"event": event, **fields}, sort_keys=True), flush=True)


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes((json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_documents(paths: list[Path], seed: int, max_docs: int) -> list[str]:
    documents: list[str] = []
    for path in paths:
        dataset = Dataset.from_file(str(path))
        for record in dataset:
            title = str(record.get("title") or "").strip()
            text = str(record.get("text") or "").strip()
            joined = (title + "\n" + text).strip()
            if joined:
                documents.append(joined)
    rng = random.Random(seed)
    rng.shuffle(documents)
    if max_docs > 0:
        documents = documents[:max_docs]
    if not documents:
        raise ValueError("no nonempty documents loaded")
    return documents


def tokenize_documents(
    tokenizer,
    documents: list[str],
    max_tokens: int,
) -> torch.Tensor:
    token_ids: list[int] = []
    eos = int(tokenizer.eos_token_id)
    for index, document in enumerate(documents, 1):
        token_ids.extend(tokenizer.encode(document, add_special_tokens=False))
        token_ids.append(eos)
        if max_tokens > 0 and len(token_ids) >= max_tokens:
            token_ids = token_ids[:max_tokens]
            break
        if index % 1000 == 0:
            emit("tokenize_progress", documents=index, tokens=len(token_ids))
    if len(token_ids) < 1024:
        raise ValueError(f"corpus is too small: {len(token_ids)} tokens")
    return torch.tensor(token_ids, dtype=torch.long)


def sample_batch(
    stream: torch.Tensor,
    generator: torch.Generator,
    batch_size: int,
    sequence_length: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    high = stream.numel() - sequence_length - 1
    if high <= 0:
        raise ValueError("token stream is shorter than one training sequence")
    starts = torch.randint(0, high, (batch_size,), generator=generator)
    x = torch.stack([stream[start : start + sequence_length] for start in starts.tolist()])
    y = torch.stack(
        [stream[start + 1 : start + sequence_length + 1] for start in starts.tolist()]
    )
    return x.to(device), y.to(device)


def evaluate_loss(
    model,
    stream: torch.Tensor,
    seed: int,
    batches: int,
    batch_size: int,
    sequence_length: int,
    device: torch.device,
) -> float:
    generator = torch.Generator().manual_seed(seed)
    values = []
    model.eval()
    with torch.no_grad():
        for _ in range(batches):
            x, y = sample_batch(stream, generator, batch_size, sequence_length, device)
            values.append(float(model(input_ids=x, labels=y, use_cache=False).loss.item()))
    return float(np.mean(values))


def canonical_orbit(rows: torch.Tensor) -> np.ndarray:
    x = rows.detach().float().cpu()
    scale = x.abs().amax(dim=1, keepdim=True)
    if bool((scale == 0).any()):
        raise ValueError("zero embedding row has no scale-permutation orbit")
    ascending = torch.sort(x / scale, dim=1).values
    negative = -torch.flip(ascending, dims=(1,))
    unresolved = torch.ones(ascending.shape[0], dtype=torch.bool)
    use_negative = torch.zeros(ascending.shape[0], dtype=torch.bool)
    for column in range(ascending.shape[1]):
        lower = negative[:, column] < ascending[:, column]
        greater = negative[:, column] > ascending[:, column]
        use_negative[unresolved & lower] = True
        unresolved &= ~(lower | greater)
        if not bool(unresolved.any()):
            break
    output = torch.where(use_negative[:, None], negative, ascending).numpy().astype("float32")
    faiss.normalize_L2(output)
    return output


def search_batches(
    index: faiss.Index,
    queries: np.ndarray,
    k: int,
    batch_size: int,
    label: str,
) -> tuple[np.ndarray, np.ndarray]:
    scores = np.empty((queries.shape[0], k), dtype=np.float32)
    identifiers = np.empty((queries.shape[0], k), dtype=np.int64)
    started = time.time()
    for start in range(0, queries.shape[0], batch_size):
        stop = min(start + batch_size, queries.shape[0])
        scores[start:stop], identifiers[start:stop] = index.search(queries[start:stop], k)
        emit(
            "orbit_search_progress",
            label=label,
            completed=stop,
            total=queries.shape[0],
            elapsed_seconds=round(time.time() - started, 3),
        )
    return scores, identifiers


def summarize_boolean(values: np.ndarray, masks: dict[str, np.ndarray]) -> dict:
    result = {}
    for name, mask in masks.items():
        count = int(mask.sum())
        result[name] = {
            "rows": count,
            "rate": float(values[mask].mean()) if count else None,
            "successes": int(values[mask].sum()) if count else 0,
        }
    return result


def orbit_evaluation(
    base_rows: torch.Tensor,
    trained_rows: torch.Tensor,
    selected: np.ndarray,
    active: np.ndarray,
    batch_size: int,
    threads: int,
) -> dict:
    faiss.omp_set_num_threads(threads)
    started = time.time()
    base = canonical_orbit(base_rows)
    trained = canonical_orbit(trained_rows)
    index = faiss.IndexFlatIP(base.shape[1])
    index.add(base)

    trained_scores, trained_ids = search_batches(
        index, trained, 5, batch_size, "trained-to-public"
    )
    base_scores, base_ids = search_batches(index, base, 2, batch_size, "public-margin")
    target = np.arange(base.shape[0])
    top1 = trained_ids[:, 0] == target
    top5 = np.any(trained_ids == target[:, None], axis=1)

    competitor = np.empty(base.shape[0], dtype=np.float32)
    for row in range(base.shape[0]):
        alternatives = trained_scores[row][trained_ids[row] != row]
        if alternatives.size == 0:
            raise AssertionError("no non-target competitor returned")
        competitor[row] = alternatives[0]
    target_cosine = np.sum(trained * base, axis=1)
    drift = np.sqrt(np.maximum(0.0, 2.0 - 2.0 * np.clip(target_cosine, -1.0, 1.0)))

    nearest_other = np.where(base_ids[:, 0] == target, base_scores[:, 1], base_scores[:, 0])
    public_margin = np.sqrt(
        np.maximum(0.0, 2.0 - 2.0 * np.clip(nearest_other, -1.0, 1.0))
    )
    certified = drift < public_margin / 2.0
    masks = {
        "all": np.ones(base.shape[0], dtype=bool),
        "active": active,
        "selected": selected,
        "active_unselected": active & ~selected,
        "inactive": ~active,
    }
    return {
        "dictionary_rows": int(base.shape[0]),
        "signature_dimension": int(base.shape[1]),
        "index": "faiss.IndexFlatIP",
        "top1": summarize_boolean(top1, masks),
        "top5": summarize_boolean(top5, masks),
        "half_margin_certificate": summarize_boolean(certified, masks),
        "selected_score_gap": {
            "min": float(np.min(target_cosine[selected] - competitor[selected])),
            "median": float(np.median(target_cosine[selected] - competitor[selected])),
            "max": float(np.max(target_cosine[selected] - competitor[selected])),
        },
        "elapsed_seconds": time.time() - started,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="gpt2")
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--arrow", action="append", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--selection", choices=("frequency", "random"), default="frequency")
    parser.add_argument("--train-fraction", type=float, default=0.25)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--sequence-length", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--eval-batches", type=int, default=10)
    parser.add_argument("--max-docs", type=int, default=0)
    parser.add_argument("--max-tokens", type=int, default=500000)
    parser.add_argument("--orbit-batch-size", type=int, default=512)
    parser.add_argument("--threads", type=int, default=16)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260802)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--skip-orbit", action="store_true")
    args = parser.parse_args()

    if not 0 < args.train_fraction <= 1:
        parser.error("--train-fraction must lie in (0,1]")
    if args.steps <= 0:
        parser.error("--steps must be positive")

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)
    device = torch.device(args.device)

    arrows = [Path(item) for item in args.arrow]
    for path in arrows:
        if not path.is_file():
            raise FileNotFoundError(path)
    documents = load_documents(arrows, args.seed, args.max_docs)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model, cache_dir=args.cache_dir, local_files_only=True
    )
    stream = tokenize_documents(tokenizer, documents, args.max_tokens)
    split = int(stream.numel() * 0.9)
    train_stream = stream[:split]
    validation_stream = stream[split:]

    counts = torch.bincount(train_stream, minlength=len(tokenizer))
    active_ids = torch.nonzero(counts > 0, as_tuple=False).flatten()
    active_ids = active_ids[active_ids != int(tokenizer.eos_token_id)]
    selected_count = max(1, int(round(args.train_fraction * active_ids.numel())))
    if args.selection == "frequency":
        order = torch.argsort(counts[active_ids], descending=True, stable=True)
        selected_ids = active_ids[order[:selected_count]]
    else:
        generator = torch.Generator().manual_seed(args.seed + 11)
        selected_ids = active_ids[torch.randperm(active_ids.numel(), generator=generator)[:selected_count]]
    selected_ids = torch.sort(selected_ids).values

    emit(
        "corpus_ready",
        documents=len(documents),
        tokens=int(stream.numel()),
        train_tokens=int(train_stream.numel()),
        validation_tokens=int(validation_stream.numel()),
        active_rows=int(active_ids.numel()),
        selected_rows=int(selected_ids.numel()),
        selection=args.selection,
    )

    model = AutoModelForCausalLM.from_pretrained(
        args.model, cache_dir=args.cache_dir, local_files_only=True
    ).to(device)
    model.config.use_cache = False
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    embedding = model.get_input_embeddings().weight
    embedding.requires_grad_(True)
    base_rows = embedding.detach().cpu().clone()

    gradient_mask = torch.zeros((embedding.shape[0], 1), dtype=embedding.dtype, device=device)
    gradient_mask[selected_ids.to(device)] = 1
    hook = embedding.register_hook(lambda gradient: gradient * gradient_mask)
    optimizer = torch.optim.AdamW([embedding], lr=args.learning_rate, weight_decay=0.0)

    before_loss = evaluate_loss(
        model,
        validation_stream,
        args.seed + 101,
        args.eval_batches,
        args.batch_size,
        args.sequence_length,
        device,
    )
    train_generator = torch.Generator().manual_seed(args.seed + 202)
    losses = []
    started = time.time()
    model.train()
    for step in range(1, args.steps + 1):
        x, y = sample_batch(
            train_stream,
            train_generator,
            args.batch_size,
            args.sequence_length,
            device,
        )
        optimizer.zero_grad(set_to_none=True)
        loss = model(input_ids=x, labels=y, use_cache=False).loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_([embedding], 1.0)
        optimizer.step()
        losses.append(float(loss.item()))
        if step == 1 or step % args.log_every == 0 or step == args.steps:
            emit(
                "train_progress",
                step=step,
                steps=args.steps,
                loss=losses[-1],
                mean_recent=float(np.mean(losses[-args.log_every :])),
                elapsed_seconds=round(time.time() - started, 3),
            )
    hook.remove()

    after_loss = evaluate_loss(
        model,
        validation_stream,
        args.seed + 101,
        args.eval_batches,
        args.batch_size,
        args.sequence_length,
        device,
    )
    trained_rows = embedding.detach().cpu().clone()
    delta = trained_rows - base_rows
    per_row_max = delta.abs().amax(dim=1)
    selected_mask = np.zeros(embedding.shape[0], dtype=bool)
    selected_mask[selected_ids.numpy()] = True
    active_mask = np.zeros(embedding.shape[0], dtype=bool)
    active_mask[active_ids.numpy()] = True
    changed_mask = per_row_max.numpy() > 0
    if bool(changed_mask[~selected_mask].any()):
        raise AssertionError("an unselected embedding row changed")

    checkpoint = Path(args.checkpoint)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": args.model,
            "seed": args.seed,
            "selection": args.selection,
            "train_fraction_of_active": args.train_fraction,
            "selected_ids": selected_ids,
            "trained_rows": trained_rows[selected_ids],
        },
        checkpoint,
    )

    orbit = None
    if not args.skip_orbit:
        orbit = orbit_evaluation(
            base_rows,
            trained_rows,
            selected_mask,
            active_mask,
            args.orbit_batch_size,
            args.threads,
        )

    output = {
        "experiment": "independently trained partial GPT-2 embedding rows",
        "not_synthetic_row_replacement": True,
        "model": args.model,
        "cache_dir": args.cache_dir,
        "corpus_arrow_files": [str(path) for path in arrows],
        "seed": args.seed,
        "selection": args.selection,
        "train_fraction_of_active": args.train_fraction,
        "documents": len(documents),
        "tokens": int(stream.numel()),
        "active_rows": int(active_ids.numel()),
        "selected_rows": int(selected_ids.numel()),
        "selected_ids": selected_ids.tolist(),
        "steps": args.steps,
        "batch_size": args.batch_size,
        "sequence_length": args.sequence_length,
        "learning_rate": args.learning_rate,
        "validation_loss_before": before_loss,
        "validation_loss_after": after_loss,
        "training_loss_first": losses[0],
        "training_loss_last": losses[-1],
        "changed_rows": int(changed_mask.sum()),
        "changed_selected_rows": int(changed_mask[selected_mask].sum()),
        "changed_unselected_rows": int(changed_mask[~selected_mask].sum()),
        "selected_delta_linf": {
            "min": float(per_row_max[selected_ids].min().item()),
            "median": float(per_row_max[selected_ids].median().item()),
            "max": float(per_row_max[selected_ids].max().item()),
        },
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
        "orbit_recovery": orbit,
        "runtime": {
            "host": platform.node(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "faiss": faiss.__version__,
            "device": str(device),
            "threads": args.threads,
            "elapsed_seconds": time.time() - started,
        },
    }
    write_json(Path(args.output), output)
    emit(
        "complete",
        output=args.output,
        checkpoint=args.checkpoint,
        selected_rows=int(selected_ids.numel()),
        changed_rows=int(changed_mask.sum()),
        validation_loss_before=before_loss,
        validation_loss_after=after_loss,
    )


if __name__ == "__main__":
    main()
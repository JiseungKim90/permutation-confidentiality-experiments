#!/usr/bin/env python3
"""Create a controlled private GPT-2 prefix by fine-tuning early blocks."""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from pathlib import Path
from typing import Dict

import numpy as np
import psutil
import torch
from transformers import AutoModelForCausalLM


def atomic_json(path: Path, value: Dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(str(temporary), str(path))


def mean_loss(model: torch.nn.Module, tokens: np.ndarray, indices: np.ndarray, batches: int, batch_size: int) -> float:
    model.eval()
    losses = []
    with torch.inference_mode():
        for start in range(batches):
            chosen = indices[start * batch_size : (start + 1) * batch_size]
            ids = torch.from_numpy(np.asarray(tokens[chosen], dtype=np.int64))
            losses.append(float(model(input_ids=ids, labels=ids, use_cache=False).loss))
    model.train()
    return float(np.mean(losses))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--prefix-depth", type=int, default=8)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--threads", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260804)
    parser.add_argument("--train-start", type=int, default=90_000)
    parser.add_argument("--train-stop", type=int, default=98_000)
    parser.add_argument("--validation-start", type=int, default=98_000)
    parser.add_argument("--validation-stop", type=int, default=100_000)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    tokens = np.load(args.tokens, mmap_mode="r")
    if not (
        0 <= args.train_start < args.train_stop <= args.validation_start
        < args.validation_stop <= tokens.shape[0]
    ):
        raise ValueError("training and validation ranges must be ordered and disjoint")
    train_indices = np.arange(args.train_start, args.train_stop, dtype=np.int64)
    validation_indices = np.arange(args.validation_start, args.validation_stop, dtype=np.int64)
    rng = np.random.default_rng(args.seed)
    rng.shuffle(train_indices)

    model = AutoModelForCausalLM.from_pretrained(
        args.model, local_files_only=True, torch_dtype=torch.float32, low_cpu_mem_usage=True
    ).to("cpu")
    model.config.use_cache = False
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    prefixes = tuple("transformer.h.{}.".format(layer) for layer in range(args.prefix_depth))
    trainable = []
    initial = {}
    for name, parameter in model.named_parameters():
        if name.startswith(prefixes):
            parameter.requires_grad_(True)
            trainable.append(parameter)
            initial[name] = parameter.detach().cpu().clone()
    if not trainable:
        raise RuntimeError("no prefix parameters selected")

    optimizer = torch.optim.AdamW(trainable, lr=args.learning_rate, weight_decay=0.01)
    eval_indices = validation_indices[: 20 * args.batch_size]
    before_loss = mean_loss(model, tokens, eval_indices, 20, args.batch_size)
    process = psutil.Process()
    peak_rss = process.memory_info().rss
    started = time.time()
    log_rows = []
    model.train()
    for step in range(args.steps):
        begin = (step * args.batch_size) % train_indices.shape[0]
        chosen = train_indices[begin : begin + args.batch_size]
        if chosen.shape[0] < args.batch_size:
            rng.shuffle(train_indices)
            chosen = train_indices[: args.batch_size]
        ids = torch.from_numpy(np.asarray(tokens[chosen], dtype=np.int64))
        optimizer.zero_grad(set_to_none=True)
        loss = model(input_ids=ids, labels=ids, use_cache=False).loss
        loss.backward()
        grad_norm = float(torch.nn.utils.clip_grad_norm_(trainable, 1.0))
        optimizer.step()
        peak_rss = max(peak_rss, process.memory_info().rss)
        if (step + 1) % 10 == 0 or step == 0:
            row = {
                "step": step + 1,
                "train_loss": float(loss.detach()),
                "gradient_norm": grad_norm,
                "elapsed_seconds": time.time() - started,
                "rss_bytes": int(process.memory_info().rss),
            }
            log_rows.append(row)
            print(json.dumps(row, sort_keys=True), flush=True)

    after_loss = mean_loss(model, tokens, eval_indices, 20, args.batch_size)
    delta_sq = 0.0
    initial_sq = 0.0
    for name, parameter in model.named_parameters():
        if name in initial:
            current = parameter.detach().cpu()
            delta_sq += float(torch.sum((current - initial[name]) ** 2))
            initial_sq += float(torch.sum(initial[name] ** 2))
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(output), safe_serialization=True)
    (output / "training_log.jsonl").write_text(
        "\n".join(json.dumps(row, sort_keys=True) for row in log_rows) + "\n",
        encoding="utf-8",
    )
    atomic_json(
        output / "private_prefix_manifest.json",
        {
            "base_model": args.model,
            "scope": "full parameters of transformer blocks [0,{})".format(args.prefix_depth),
            "prefix_depth": args.prefix_depth,
            "steps": args.steps,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "seed": args.seed,
            "train_index_range": [args.train_start, args.train_stop],
            "validation_index_range": [args.validation_start, args.validation_stop],
            "validation_batches": 20,
            "validation_loss_before": before_loss,
            "validation_loss_after": after_loss,
            "validation_perplexity_before": math.exp(min(before_loss, 20.0)),
            "validation_perplexity_after": math.exp(min(after_loss, 20.0)),
            "trainable_parameters": int(sum(p.numel() for p in trainable)),
            "relative_parameter_l2_drift": math.sqrt(delta_sq / max(initial_sq, 1e-30)),
            "elapsed_seconds": time.time() - started,
            "peak_rss_bytes": int(peak_rss),
            "torch_version": torch.__version__,
        },
    )
    print("saved private prefix to {}".format(output), flush=True)


if __name__ == "__main__":
    main()

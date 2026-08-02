#!/usr/bin/env python3
"""Train paired private-prefix checkpoints and measure utility/representation drift."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import time
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import numpy as np
import psutil
import torch
from transformers import AutoModelForCausalLM


def atomic_json(path: Path, value: Dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(str(temporary), str(path))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def selected_parameter_names(model: torch.nn.Module, prefix_depth: int) -> List[str]:
    prefixes = tuple("transformer.h.{}.".format(layer) for layer in range(prefix_depth))
    return [name for name, _ in model.named_parameters() if name.startswith(prefixes)]


def freeze_except(model: torch.nn.Module, names: Sequence[str]) -> List[torch.nn.Parameter]:
    selected = set(names)
    trainable = []
    for name, parameter in model.named_parameters():
        enabled = name in selected
        parameter.requires_grad_(enabled)
        if enabled:
            trainable.append(parameter)
    if not trainable:
        raise RuntimeError("no prefix parameters selected")
    return trainable


def collect_initial(model: torch.nn.Module, names: Sequence[str]) -> Dict[str, torch.Tensor]:
    selected = set(names)
    return {
        name: parameter.detach().cpu().clone()
        for name, parameter in model.named_parameters()
        if name in selected
    }


def mean_loss(
    model: torch.nn.Module,
    tokens: np.ndarray,
    indices: np.ndarray,
    batch_size: int,
) -> float:
    was_training = model.training
    model.eval()
    losses = []
    with torch.inference_mode():
        for start in range(0, indices.shape[0], batch_size):
            chosen = indices[start : start + batch_size]
            ids = torch.from_numpy(np.asarray(tokens[chosen], dtype=np.int64))
            losses.append(float(model(input_ids=ids, labels=ids, use_cache=False).loss))
    model.train(was_training)
    return float(np.mean(losses))


def hidden_probe(
    model: torch.nn.Module,
    tokens: np.ndarray,
    indices: np.ndarray,
    layer: int,
    batch_size: int,
) -> np.ndarray:
    was_training = model.training
    model.eval()
    rows = []
    with torch.inference_mode():
        for start in range(0, indices.shape[0], batch_size):
            chosen = indices[start : start + batch_size]
            ids = torch.from_numpy(np.asarray(tokens[chosen], dtype=np.int64))
            result = model(
                input_ids=ids,
                attention_mask=torch.ones_like(ids),
                output_hidden_states=True,
                return_dict=True,
                use_cache=False,
            )
            rows.append(result.hidden_states[layer].detach().cpu().numpy().astype(np.float32))
    model.train(was_training)
    return np.concatenate(rows, axis=0)


def representation_metrics(public: np.ndarray, private: np.ndarray) -> Dict:
    public64 = np.asarray(public, dtype=np.float64)
    private64 = np.asarray(private, dtype=np.float64)
    difference = private64 - public64
    public_flat = public64.reshape(-1, public64.shape[-1])
    private_flat = private64.reshape(-1, private64.shape[-1])
    numerator = np.sum(public_flat * private_flat, axis=1)
    denominator = np.linalg.norm(public_flat, axis=1) * np.linalg.norm(private_flat, axis=1)
    cosine = numerator / np.maximum(denominator, np.finfo(np.float64).tiny)

    public_to_private_residuals = []
    private_to_public_residuals = []
    for public_rows, private_rows in zip(public64, private64):
        _, _, private_vt = np.linalg.svd(private_rows, full_matrices=False)
        _, _, public_vt = np.linalg.svd(public_rows, full_matrices=False)
        public_norm_sq = np.sum(public_rows * public_rows, axis=1)
        private_norm_sq = np.sum(private_rows * private_rows, axis=1)
        public_projected = np.sum((public_rows @ private_vt.T) ** 2, axis=1)
        private_projected = np.sum((private_rows @ public_vt.T) ** 2, axis=1)
        public_to_private_residuals.extend(
            np.sqrt(
                np.maximum(public_norm_sq - public_projected, 0.0)
                / np.maximum(public_norm_sq, np.finfo(np.float64).tiny)
            )
        )
        private_to_public_residuals.extend(
            np.sqrt(
                np.maximum(private_norm_sq - private_projected, 0.0)
                / np.maximum(private_norm_sq, np.finfo(np.float64).tiny)
            )
        )
    p2q = np.asarray(public_to_private_residuals, dtype=np.float64)
    q2p = np.asarray(private_to_public_residuals, dtype=np.float64)
    return {
        "relative_hidden_l2_drift": float(
            np.linalg.norm(difference) / max(np.linalg.norm(public64), 1e-30)
        ),
        "mean_row_cosine": float(np.mean(cosine)),
        "p05_row_cosine": float(np.quantile(cosine, 0.05)),
        "minimum_row_cosine": float(np.min(cosine)),
        "mean_public_to_private_rowspace_residual": float(np.mean(p2q)),
        "p95_public_to_private_rowspace_residual": float(np.quantile(p2q, 0.95)),
        "mean_private_to_public_rowspace_residual": float(np.mean(q2p)),
        "p95_private_to_public_rowspace_residual": float(np.quantile(q2p, 0.95)),
        "probe_sequences": int(public.shape[0]),
        "probe_rows": int(public_flat.shape[0]),
    }


def parameter_drift(
    model: torch.nn.Module,
    initial: Dict[str, torch.Tensor],
) -> Dict[str, float]:
    delta_sq = 0.0
    initial_sq = 0.0
    max_abs = 0.0
    for name, parameter in model.named_parameters():
        if name not in initial:
            continue
        current = parameter.detach().cpu()
        difference = current - initial[name]
        delta_sq += float(torch.sum(difference * difference))
        initial_sq += float(torch.sum(initial[name] * initial[name]))
        max_abs = max(max_abs, float(torch.max(torch.abs(difference))))
    return {
        "relative_parameter_l2_drift": math.sqrt(delta_sq / max(initial_sq, 1e-30)),
        "maximum_absolute_parameter_drift": max_abs,
    }


def deterministic_schedule(
    train_start: int,
    train_stop: int,
    batch_size: int,
    steps: int,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    pool = np.arange(train_start, train_stop, dtype=np.int64)
    required = steps * batch_size
    if required == 0:
        return np.empty((0, batch_size), dtype=np.int64)
    pieces = []
    available = 0
    while available < required:
        permutation = rng.permutation(pool)
        pieces.append(permutation)
        available += permutation.shape[0]
    return np.concatenate(pieces)[:required].reshape(steps, batch_size)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--checkpoints", type=int, nargs="+", default=[0, 100, 400, 1600, 3200])
    parser.add_argument("--prefix-depth", type=int, default=8)
    parser.add_argument("--probe-layer", type=int, default=8)
    parser.add_argument("--probe-sequences", type=int, default=128)
    parser.add_argument("--utility-sequences", type=int, default=320)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--eval-batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--threads", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260808)
    parser.add_argument("--train-start", type=int, default=90_000)
    parser.add_argument("--train-stop", type=int, default=98_000)
    args = parser.parse_args()

    checkpoints = sorted(set(args.checkpoints))
    if not checkpoints or checkpoints[0] != 0:
        raise ValueError("checkpoints must include step zero")
    if args.probe_layer > args.prefix_depth:
        raise ValueError("probe layer must not exceed the trained prefix depth")
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    completion = output / "drift_sweep_manifest.json"
    if completion.exists():
        existing = json.loads(completion.read_text(encoding="utf-8"))
        if existing.get("status") == "complete" and existing.get("checkpoints") == checkpoints:
            print("drift sweep already complete", flush=True)
            return

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    tokens_path = Path(args.tokens).resolve()
    tokens = np.load(str(tokens_path), mmap_mode="r", allow_pickle=False)
    if tokens.shape[0] < 120_000:
        raise ValueError("drift study requires the 120K token pool")

    validation_pools = {
        "heldout_candidate": np.arange(98_000, 98_000 + args.utility_sequences, dtype=np.int64),
        "calibration_open": np.arange(100_000, 100_000 + args.utility_sequences, dtype=np.int64),
        "evaluation_open": np.arange(110_000, 110_000 + args.utility_sequences, dtype=np.int64),
    }
    probe_indices = np.arange(110_000, 110_000 + args.probe_sequences, dtype=np.int64)

    model = AutoModelForCausalLM.from_pretrained(
        args.model, local_files_only=True, torch_dtype=torch.float32, low_cpu_mem_usage=True
    ).to("cpu")
    model.config.use_cache = False
    names = selected_parameter_names(model, args.prefix_depth)
    initial = collect_initial(model, names)
    trainable = freeze_except(model, names)
    public_probe = hidden_probe(
        model, tokens, probe_indices, args.probe_layer, args.eval_batch_size
    )
    optimizer = torch.optim.AdamW(trainable, lr=args.learning_rate, weight_decay=0.01)
    schedule = deterministic_schedule(
        args.train_start, args.train_stop, args.batch_size, checkpoints[-1], args.seed
    )
    process = psutil.Process()
    peak_rss = process.memory_info().rss
    started = time.time()
    training_log = []
    checkpoint_manifests = []

    def save_checkpoint(step: int) -> None:
        nonlocal peak_rss
        checkpoint_dir = output / "step{:04d}".format(step)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        if step == 0:
            model_path = str(Path(args.model).resolve())
        else:
            model.save_pretrained(str(checkpoint_dir), safe_serialization=True)
            model_path = str(checkpoint_dir.resolve())
        private_probe = hidden_probe(
            model, tokens, probe_indices, args.probe_layer, args.eval_batch_size
        )
        utility = {}
        for name, indices in validation_pools.items():
            loss = mean_loss(model, tokens, indices, args.eval_batch_size)
            utility[name] = {
                "loss": loss,
                "perplexity": math.exp(min(loss, 20.0)),
                "sequences": int(indices.shape[0]),
                "index_range": [int(indices[0]), int(indices[-1]) + 1],
            }
        manifest = {
            "step": step,
            "model_path": model_path,
            "model_files_sha256": {
                path.name: sha256(path)
                for path in checkpoint_dir.glob("*.safetensors")
            },
            "parameter_drift": parameter_drift(model, initial),
            "representation_drift": representation_metrics(public_probe, private_probe),
            "utility": utility,
            "elapsed_seconds": time.time() - started,
            "peak_rss_bytes": int(max(peak_rss, process.memory_info().rss)),
        }
        atomic_json(checkpoint_dir / "checkpoint_manifest.json", manifest)
        checkpoint_manifests.append(manifest)
        print(json.dumps(manifest, sort_keys=True), flush=True)

    save_checkpoint(0)
    model.train()
    for step in range(1, checkpoints[-1] + 1):
        chosen = schedule[step - 1]
        ids = torch.from_numpy(np.asarray(tokens[chosen], dtype=np.int64))
        optimizer.zero_grad(set_to_none=True)
        loss = model(input_ids=ids, labels=ids, use_cache=False).loss
        loss.backward()
        gradient_norm = float(torch.nn.utils.clip_grad_norm_(trainable, 1.0))
        optimizer.step()
        peak_rss = max(peak_rss, process.memory_info().rss)
        if step == 1 or step % 50 == 0:
            row = {
                "step": step,
                "train_loss": float(loss.detach()),
                "gradient_norm": gradient_norm,
                "elapsed_seconds": time.time() - started,
                "rss_bytes": int(process.memory_info().rss),
            }
            training_log.append(row)
            print(json.dumps(row, sort_keys=True), flush=True)
        if step in checkpoints:
            save_checkpoint(step)

    log_path = output / "training_log.jsonl"
    log_path.write_text(
        "\n".join(json.dumps(row, sort_keys=True) for row in training_log) + "\n",
        encoding="utf-8",
    )
    atomic_json(
        completion,
        {
            "status": "complete",
            "base_model": str(Path(args.model).resolve()),
            "tokens": str(tokens_path),
            "tokens_sha256": sha256(tokens_path),
            "checkpoints": checkpoints,
            "checkpoint_manifests": checkpoint_manifests,
            "prefix_depth": args.prefix_depth,
            "probe_layer": args.probe_layer,
            "probe_index_range": [int(probe_indices[0]), int(probe_indices[-1]) + 1],
            "learning_rate": args.learning_rate,
            "batch_size": args.batch_size,
            "train_index_range": [args.train_start, args.train_stop],
            "trainable_parameters": int(sum(parameter.numel() for parameter in trainable)),
            "seed": args.seed,
            "elapsed_seconds": time.time() - started,
            "peak_rss_bytes": int(peak_rss),
            "training_log": str(log_path.resolve()),
            "training_log_sha256": sha256(log_path),
            "torch_version": torch.__version__,
        },
    )
    print("completed private-prefix drift sweep", flush=True)


if __name__ == "__main__":
    main()

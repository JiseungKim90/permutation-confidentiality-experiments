#!/usr/bin/env python3
"""Build resumable real GPT-2 K/V block caches from tokens or cached hidden states."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Dict, Optional

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
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def tensor_sha256(value: torch.Tensor) -> str:
    return hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--hidden-manifest")
    source.add_argument("--tokens")
    parser.add_argument("--model", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--heads", type=int, nargs="+", default=[0, 5, 11])
    parser.add_argument("--indices")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--threads", type=int, default=20)
    parser.add_argument("--checkpoint-every", type=int, default=2048)
    args = parser.parse_args()

    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, local_files_only=True, torch_dtype=torch.float32, low_cpu_mem_usage=True
    ).eval().to("cpu")
    config = model.config
    hidden_dim = int(config.n_embd)
    num_heads = int(config.n_head)
    head_dim = hidden_dim // num_heads
    if args.layer < 0 or args.layer >= int(config.n_layer):
        raise ValueError("layer is outside GPT-2 transformer blocks")
    if len(set(args.heads)) != len(args.heads) or min(args.heads) < 0 or max(args.heads) >= num_heads:
        raise ValueError("invalid or duplicate head selection")

    hidden_manifest: Optional[Dict] = None
    hidden_map: Optional[np.memmap] = None
    tokens: Optional[np.ndarray] = None
    if args.hidden_manifest:
        hidden_manifest = json.loads(Path(args.hidden_manifest).read_text(encoding="utf-8"))
        shape = tuple(int(x) for x in hidden_manifest["shape"])
        hidden_path = Path(hidden_manifest["files"][str(args.layer)])
        hidden_map = np.memmap(str(hidden_path), mode="r", dtype=np.float16, shape=shape)
        total, seq_len = shape[0], shape[1]
    else:
        if args.layer != 0:
            raise ValueError("token-derived cache currently supports layer 0 only")
        tokens = np.load(args.tokens, mmap_mode="r")
        total, seq_len = int(tokens.shape[0]), int(tokens.shape[1])
    indices = (
        np.load(args.indices, allow_pickle=False).astype(np.int64)
        if args.indices else np.arange(total, dtype=np.int64)
    )

    block = model.transformer.h[args.layer]
    weight = block.attn.c_attn.weight.detach().cpu().to(torch.float32)
    bias = block.attn.c_attn.bias.detach().cpu().to(torch.float32)
    selected_columns = []
    for offset in [hidden_dim, 2 * hidden_dim]:
        for head in args.heads:
            selected_columns.extend(range(offset + head * head_dim, offset + (head + 1) * head_dim))
    columns = torch.tensor(selected_columns, dtype=torch.long)
    selected_weight = weight.index_select(1, columns).contiguous()
    selected_bias = bias.index_select(0, columns).contiguous()

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    shape = (int(indices.shape[0]), len(args.heads), seq_len, head_dim)
    paths = {
        "key": output / (args.tag + "_key.f16"),
        "value": output / (args.tag + "_value.f16"),
    }
    maps = {
        name: np.memmap(str(path), mode="r+" if path.exists() else "w+", dtype=np.float16, shape=shape)
        for name, path in paths.items()
    }
    state_path = output / (args.tag + "_state.json")
    completed = 0
    max_abs = {"key": 0.0, "value": 0.0}
    if state_path.exists():
        state = json.loads(state_path.read_text(encoding="utf-8"))
        if state["shape"] != list(shape) or state["layer"] != args.layer or state["heads"] != args.heads:
            raise RuntimeError("resume metadata does not match requested cache")
        completed = int(state["completed"])
        max_abs.update({key: float(value) for key, value in state.get("max_abs", {}).items()})

    process = psutil.Process()
    peak_rss = process.memory_info().rss
    started = time.time()
    last_checkpoint = completed
    positions = torch.arange(seq_len, dtype=torch.long).unsqueeze(0)
    with torch.inference_mode():
        for start in range(completed, indices.shape[0], args.batch_size):
            stop = min(start + args.batch_size, indices.shape[0])
            current = indices[start:stop]
            if hidden_map is not None:
                hidden = torch.from_numpy(np.asarray(hidden_map[current], dtype=np.float32))
            else:
                ids = torch.from_numpy(np.asarray(tokens[current], dtype=np.int64))
                hidden = model.transformer.wte(ids) + model.transformer.wpe(positions[:, :seq_len])
                hidden = model.transformer.drop(hidden)
            normalized = block.ln_1(hidden)
            projected = normalized @ selected_weight + selected_bias
            projected = projected.view(stop - start, seq_len, 2, len(args.heads), head_dim)
            key = projected[:, :, 0].permute(0, 2, 1, 3).contiguous()
            value = projected[:, :, 1].permute(0, 2, 1, 3).contiguous()
            for name, tensor in [("key", key), ("value", value)]:
                arrays = tensor.cpu().numpy().astype(np.float16)
                maps[name][start:stop] = arrays
                max_abs[name] = max(max_abs[name], float(tensor.abs().max()))
            peak_rss = max(peak_rss, process.memory_info().rss)
            if stop - last_checkpoint >= args.checkpoint_every or stop == indices.shape[0]:
                for mapping in maps.values():
                    mapping.flush()
                atomic_json(
                    state_path,
                    {
                        "completed": int(stop), "shape": list(shape), "layer": args.layer,
                        "heads": args.heads, "max_abs": max_abs,
                        "elapsed_current_run_seconds": time.time() - started,
                        "peak_rss_bytes": int(peak_rss),
                    },
                )
                last_checkpoint = stop
                print("kv-cache {}/{} elapsed={:.1f}s".format(stop, indices.shape[0], time.time() - started), flush=True)

    manifest = {
        "status": "complete",
        "model": args.model,
        "model_type": config.model_type,
        "source_hidden_manifest": str(Path(args.hidden_manifest).resolve()) if args.hidden_manifest else None,
        "source_tokens": str(Path(args.tokens).resolve()) if args.tokens else None,
        "source_indices": str(Path(args.indices).resolve()) if args.indices else None,
        "layer": args.layer,
        "heads": args.heads,
        "head_dim": head_dim,
        "sequence_length": seq_len,
        "shape": list(shape),
        "storage_dtype": "float16",
        "files": {key: str(value.resolve()) for key, value in paths.items()},
        "sha256": {key: sha256(value) for key, value in paths.items()},
        "selected_projection_weight_sha256": tensor_sha256(selected_weight),
        "selected_projection_bias_sha256": tensor_sha256(selected_bias),
        "max_abs": max_abs,
        "elapsed_current_run_seconds": time.time() - started,
        "peak_rss_bytes": int(peak_rss),
        "torch_version": torch.__version__,
    }
    atomic_json(output / (args.tag + "_manifest.json"), manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

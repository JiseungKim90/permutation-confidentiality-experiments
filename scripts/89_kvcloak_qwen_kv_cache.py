#!/usr/bin/env python3
"""Build formula-faithful Qwen2 K/V caches for KV-Cloak evaluation.

The script evaluates only layer 0, where K/V can be derived without running
the remaining decoder blocks.  It checks that derivation against one ordinary
Hugging Face forward pass before emitting a completion manifest.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Dict, Sequence, Tuple

import numpy as np
import psutil
import torch
from transformers import AutoModelForCausalLM
from transformers.models.qwen2.modeling_qwen2 import rotate_half


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
    array = value.detach().cpu().contiguous().numpy()
    return hashlib.sha256(array.tobytes()).hexdigest()


def project_layer0(
    model,
    ids: torch.Tensor,
    heads: Sequence[int],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return post-RoPE K and unrotated V in cache layout [B,H,L,D]."""
    layer = model.model.layers[0]
    attention = layer.self_attn
    hidden = model.model.embed_tokens(ids)
    normalized = layer.input_layernorm(hidden)
    batch, sequence, _ = normalized.shape
    head_dim = int(model.config.hidden_size) // int(model.config.num_attention_heads)
    kv_heads = int(model.config.num_key_value_heads)

    key = attention.k_proj(normalized).view(batch, sequence, kv_heads, head_dim)
    value = attention.v_proj(normalized).view(batch, sequence, kv_heads, head_dim)
    key = key.transpose(1, 2).contiguous()
    value = value.transpose(1, 2).contiguous()

    position_ids = torch.arange(sequence, device=ids.device, dtype=torch.long).unsqueeze(0)
    position_ids = position_ids.expand(batch, -1)
    cos, sin = model.model.rotary_emb(normalized, position_ids)
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    key = key * cos + rotate_half(key) * sin

    selected = torch.tensor(list(heads), device=ids.device, dtype=torch.long)
    return key.index_select(1, selected).contiguous(), value.index_select(1, selected).contiguous()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--heads", type=int, nargs="+", default=[0, 1])
    parser.add_argument("--indices")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--threads", type=int, default=20)
    parser.add_argument("--checkpoint-every", type=int, default=1024)
    args = parser.parse_args()

    if args.layer != 0:
        raise ValueError("the Qwen2 cache derivation is deliberately restricted to layer 0")
    if len(set(args.heads)) != len(args.heads):
        raise ValueError("duplicate head selection")

    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        local_files_only=True,
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True,
        attn_implementation="eager",
    ).eval().to("cpu")
    config = model.config
    if config.model_type != "qwen2":
        raise ValueError("expected a Qwen2-family checkpoint")
    hidden_dim = int(config.hidden_size)
    attention_heads = int(config.num_attention_heads)
    kv_heads = int(config.num_key_value_heads)
    head_dim = hidden_dim // attention_heads
    if min(args.heads) < 0 or max(args.heads) >= kv_heads:
        raise ValueError("invalid Qwen2 key/value head selection")

    tokens_path = Path(args.tokens).resolve()
    tokens = np.load(tokens_path, mmap_mode="r")
    if tokens.ndim != 2:
        raise ValueError("tokens must be a two-dimensional integer array")
    total, sequence = int(tokens.shape[0]), int(tokens.shape[1])
    indices = (
        np.load(args.indices, allow_pickle=False).astype(np.int64)
        if args.indices else np.arange(total, dtype=np.int64)
    )
    if indices.ndim != 1 or np.any(indices < 0) or np.any(indices >= total):
        raise ValueError("cache indices fall outside the token archive")
    if len(np.unique(indices)) != len(indices):
        raise ValueError("cache indices must be unique")

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    shape = (int(indices.shape[0]), len(args.heads), sequence, head_dim)
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
        if state["shape"] != list(shape) or state["layer"] != 0 or state["heads"] != args.heads:
            raise RuntimeError("resume metadata does not match requested cache")
        completed = int(state["completed"])
        max_abs.update({key: float(value) for key, value in state.get("max_abs", {}).items()})

    process = psutil.Process()
    peak_rss = process.memory_info().rss
    started = time.time()
    last_checkpoint = completed
    with torch.inference_mode():
        for start in range(completed, indices.shape[0], args.batch_size):
            stop = min(start + args.batch_size, indices.shape[0])
            current = indices[start:stop]
            ids = torch.from_numpy(np.asarray(tokens[current], dtype=np.int64))
            key, value = project_layer0(model, ids, args.heads)
            for name, tensor in [("key", key), ("value", value)]:
                maps[name][start:stop] = tensor.cpu().numpy().astype(np.float16)
                max_abs[name] = max(max_abs[name], float(tensor.abs().max()))
            peak_rss = max(peak_rss, process.memory_info().rss)
            if stop - last_checkpoint >= args.checkpoint_every or stop == indices.shape[0]:
                for mapping in maps.values():
                    mapping.flush()
                atomic_json(
                    state_path,
                    {
                        "completed": int(stop),
                        "shape": list(shape),
                        "layer": 0,
                        "heads": args.heads,
                        "max_abs": max_abs,
                        "elapsed_current_run_seconds": time.time() - started,
                        "peak_rss_bytes": int(peak_rss),
                    },
                )
                last_checkpoint = stop
                print("qwen-kv-cache {}/{} elapsed={:.1f}s".format(stop, indices.shape[0], time.time() - started), flush=True)

    # Formula-fidelity check: the derived layer-0 cache must match an ordinary
    # model forward pass before any result can be finalized.
    verification_source_index = int(indices[0])
    verify_ids = torch.from_numpy(
        np.asarray(tokens[verification_source_index : verification_source_index + 1], dtype=np.int64)
    )
    with torch.inference_mode():
        derived_key, derived_value = project_layer0(model, verify_ids, args.heads)
        forward = model(input_ids=verify_ids, use_cache=True, return_dict=True)
        reference_key, reference_value = forward.past_key_values[0]
        selected = torch.tensor(args.heads, dtype=torch.long)
        reference_key = reference_key.index_select(1, selected)
        reference_value = reference_value.index_select(1, selected)
        key_error = float((derived_key - reference_key).abs().max())
        value_error = float((derived_value - reference_value).abs().max())
    fidelity_tolerance = 2e-5
    if key_error > fidelity_tolerance or value_error > fidelity_tolerance:
        raise RuntimeError(
            "layer-0 derivation failed the forward-pass check: K={} V={}".format(key_error, value_error)
        )

    layer = model.model.layers[0]
    manifest = {
        "status": "complete",
        "model": str(Path(args.model).resolve()),
        "model_type": config.model_type,
        "architecture": "Qwen2-family grouped-query attention",
        "source_tokens": str(tokens_path),
        "source_tokens_sha256": sha256(tokens_path),
        "source_indices": str(Path(args.indices).resolve()) if args.indices else None,
        "source_indices_sha256": sha256(Path(args.indices).resolve()) if args.indices else None,
        "layer": 0,
        "heads": args.heads,
        "hidden_dimension": hidden_dim,
        "attention_heads": attention_heads,
        "key_value_heads": kv_heads,
        "head_dim": head_dim,
        "sequence_length": sequence,
        "shape": list(shape),
        "storage_dtype": "float16",
        "files": {key: str(value.resolve()) for key, value in paths.items()},
        "sha256": {key: sha256(value) for key, value in paths.items()},
        "projection_sha256": {
            "k_proj_weight": tensor_sha256(layer.self_attn.k_proj.weight),
            "v_proj_weight": tensor_sha256(layer.self_attn.v_proj.weight),
            "input_layernorm_weight": tensor_sha256(layer.input_layernorm.weight),
        },
        "forward_pass_fidelity": {
            "source_index": verification_source_index,
            "max_absolute_key_error": key_error,
            "max_absolute_value_error": value_error,
            "tolerance": fidelity_tolerance,
            "pass": True,
        },
        "max_abs": max_abs,
        "elapsed_current_run_seconds": time.time() - started,
        "peak_rss_bytes": int(peak_rss),
        "torch_version": torch.__version__,
    }
    atomic_json(output / (args.tag + "_manifest.json"), manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

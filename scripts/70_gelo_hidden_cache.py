#!/usr/bin/env python3
"""Build resumable FP16 hidden-state memmaps for the MS MARCO GELO study."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import psutil
import torch
from transformers import AutoModelForCausalLM


def atomic_json(path: Path, value: Dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(str(temporary), str(path))


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", required=True)
    parser.add_argument("--indices")
    parser.add_argument("--model", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--layers", type=int, nargs="+", default=[4, 8, 12])
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--threads", type=int, default=20)
    parser.add_argument("--checkpoint-every", type=int, default=1024)
    args = parser.parse_args()

    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    tokens = np.load(args.tokens, mmap_mode="r")
    if tokens.ndim != 2:
        raise ValueError("tokens must have shape [example, sequence]")
    if args.indices:
        indices = np.load(args.indices, allow_pickle=False).astype(np.int64)
    else:
        indices = np.arange(tokens.shape[0], dtype=np.int64)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    state_path = output / (args.tag + "_state.json")
    completed = 0
    if state_path.exists():
        state = json.loads(state_path.read_text(encoding="utf-8"))
        completed = int(state.get("completed", 0))
        if state.get("indices_count") != int(indices.shape[0]) or state.get("layers") != args.layers:
            raise RuntimeError("resume metadata does not match requested cache")

    print("loading model {}".format(args.model), flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, local_files_only=True, torch_dtype=torch.float32, low_cpu_mem_usage=True
    ).eval().to("cpu")
    if max(args.layers) > int(model.config.n_layer):
        raise ValueError("requested layer exceeds model depth")
    hidden_dim = int(model.config.n_embd)
    shape = (int(indices.shape[0]), int(tokens.shape[1]), hidden_dim)
    maps: Dict[int, np.memmap] = {}
    paths: Dict[int, Path] = {}
    for layer in args.layers:
        path = output / ("{}_layer{}_n{}_s{}_d{}.f16".format(args.tag, layer, *shape))
        paths[layer] = path
        mode = "r+" if path.exists() else "w+"
        maps[layer] = np.memmap(str(path), mode=mode, dtype=np.float16, shape=shape)

    process = psutil.Process()
    peak_rss = process.memory_info().rss
    started = time.time()
    last_checkpoint = completed
    with torch.inference_mode():
        for start in range(completed, indices.shape[0], args.batch_size):
            stop = min(start + args.batch_size, indices.shape[0])
            ids = torch.from_numpy(np.asarray(tokens[indices[start:stop]], dtype=np.int64))
            result = model(
                input_ids=ids,
                attention_mask=torch.ones_like(ids),
                output_hidden_states=True,
                return_dict=True,
                use_cache=False,
            )
            for layer in args.layers:
                maps[layer][start:stop] = (
                    result.hidden_states[layer].detach().cpu().numpy().astype(np.float16)
                )
            peak_rss = max(peak_rss, process.memory_info().rss)
            if stop - last_checkpoint >= args.checkpoint_every or stop == indices.shape[0]:
                for mapping in maps.values():
                    mapping.flush()
                atomic_json(
                    state_path,
                    {
                        "tag": args.tag,
                        "model": args.model,
                        "layers": args.layers,
                        "shape": list(shape),
                        "indices": args.indices,
                        "indices_count": int(indices.shape[0]),
                        "completed": int(stop),
                        "elapsed_current_run_seconds": time.time() - started,
                        "peak_rss_bytes": int(peak_rss),
                    },
                )
                last_checkpoint = stop
                print(
                    "cache {}/{} elapsed={:.1f}s rss={:.2f}GiB".format(
                        stop, indices.shape[0], time.time() - started, peak_rss / 2**30
                    ),
                    flush=True,
                )
    del model
    checksums: Dict[str, str] = {}
    for layer, path in paths.items():
        checksums[str(layer)] = sha256_file(path)
    atomic_json(
        output / (args.tag + "_manifest.json"),
        {
            "tag": args.tag,
            "model": args.model,
            "layers": args.layers,
            "shape": list(shape),
            "storage_dtype": "float16",
            "tokens": str(Path(args.tokens).resolve()),
            "indices": str(Path(args.indices).resolve()) if args.indices else None,
            "indices_count": int(indices.shape[0]),
            "files": {str(layer): str(path.resolve()) for layer, path in paths.items()},
            "sha256": checksums,
            "elapsed_current_run_seconds": time.time() - started,
            "peak_rss_bytes": int(peak_rss),
            "torch_version": torch.__version__,
        },
    )
    print("completed cache {}".format(args.tag), flush=True)


if __name__ == "__main__":
    main()

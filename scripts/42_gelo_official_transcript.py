#!/usr/bin/env python3
"""Run the row-space presence attack on the official GELO wrapper transcript.

The script imports GeloObfuscatedLinear from a pinned checkout, captures the
exact tensor passed to its remote linear client, and ranks public candidate
hidden states by projection residual.  It does not reimplement the mixing path.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


def load_helpers() -> object:
    path = Path(__file__).with_name("41_gelo_rowspace_presence.py")
    spec = importlib.util.spec_from_file_location("gelo_rowspace_helpers", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load helpers from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class CaptureRemote:
    def __init__(self, output_dim: int):
        self.output_dim = output_dim
        self.last: torch.Tensor | None = None

    def run_linear(self, name: str, x: torch.Tensor) -> torch.Tensor:
        del name
        self.last = x.detach().float().cpu().clone()
        return torch.zeros(
            (x.shape[0], self.output_dim),
            device=x.device,
            dtype=x.dtype,
        )


def official_transcript(
    data_rows: np.ndarray,
    shield_fraction: float,
    shield_scale: float,
    precision: str,
    wrapper_class: type,
    orthogonal_fn: object,
) -> np.ndarray:
    rows, dim = data_rows.shape
    dtype = torch.float32 if precision == "float32" else torch.bfloat16
    hidden = torch.from_numpy(data_rows.astype("float32")).reshape(1, rows, dim)
    hidden = hidden.to(dtype)
    shield_count = int(np.ceil(rows * shield_fraction))
    effective_rows = rows + shield_count

    wrapper = wrapper_class(nn.Linear(dim, dim, bias=False))
    remote = CaptureRemote(dim)
    wrapper._remote_client = remote
    wrapper._remote_name = "capture"
    wrapper._gelo_gauss_scale = shield_scale
    wrapper.set_context_A(
        orthogonal_fn(
            b=1,
            s=effective_rows,
            device=hidden.device,
            dtype=hidden.dtype,
        )
    )
    with torch.no_grad():
        wrapper(hidden)
    if remote.last is None:
        raise RuntimeError("official GELO wrapper did not call the remote client")
    return remote.last.numpy().astype("float64")


def one_trial(
    dictionary: np.ndarray,
    source_count: int,
    shield_fraction: float,
    shield_scale: float,
    precision: str,
    rng: np.random.Generator,
    helpers: object,
    wrapper_class: type,
    orthogonal_fn: object,
) -> dict:
    candidate_count, seq_len, dim = dictionary.shape
    source_ids = np.sort(
        rng.choice(candidate_count, size=source_count, replace=False)
    )
    data_rows = dictionary[source_ids].reshape(source_count * seq_len, dim)
    observed = official_transcript(
        data_rows,
        shield_fraction,
        shield_scale,
        precision,
        wrapper_class,
        orthogonal_fn,
    )
    basis, rank, observed_condition = helpers.observed_row_basis(observed)
    score = helpers.residuals(dictionary, basis)
    predicted = np.argsort(score)[:source_count]
    true_score = score[source_ids]
    false_mask = np.ones(candidate_count, dtype=bool)
    false_mask[source_ids] = False
    false_score = score[false_mask]
    exact = set(predicted.tolist()) == set(source_ids.tolist())
    return {
        "exact_set": exact,
        "recall": len(set(predicted.tolist()) & set(source_ids.tolist()))
        / source_count,
        "max_true_residual": float(np.max(true_score)),
        "min_false_residual": float(np.min(false_score)),
        "gap": float(np.min(false_score) - np.max(true_score)),
        "observed_rank": rank,
        "observed_condition": observed_condition,
        "observed_rows": int(observed.shape[0]),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--official-repo", required=True)
    parser.add_argument("--model", default="gpt2")
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--candidates", type=int, default=128)
    parser.add_argument("--seq-len", type=int, default=16)
    parser.add_argument("--source-count", type=int, default=4)
    parser.add_argument("--trials", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260802)
    parser.add_argument("--layers", type=int, nargs="+", default=[2, 4, 8, 12])
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()

    official_repo = Path(args.official_repo).resolve()
    sys.path.insert(0, str(official_repo))
    from gelo.gelo_attention import (  # type: ignore
        GeloObfuscatedLinear,
        _batched_random_orthogonal,
    )

    helpers = load_helpers()
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)
    _, texts, encoded = helpers.load_texts(
        args.model, args.cache_dir, args.candidates, args.seq_len
    )
    dictionaries, model_revision = helpers.hidden_dictionary(
        args.model,
        args.cache_dir,
        encoded,
        args.layers,
        args.batch_size,
    )
    commit = subprocess.run(
        ["git", "-C", str(official_repo), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    wrapper_path = official_repo / "gelo" / "gelo_attention.py"
    wrapper_sha256 = hashlib.sha256(wrapper_path.read_bytes()).hexdigest()

    conditions = [
        (0.0, 1.0, "float32"),
        (0.12, 25.0, "float32"),
        (0.12, 25.0, "bfloat16"),
    ]
    output = {
        "official_repository": "https://github.com/noskill/gelo",
        "official_commit": commit,
        "wrapper_file": "gelo/gelo_attention.py",
        "wrapper_sha256": wrapper_sha256,
        "capture_point": "GeloObfuscatedLinear._project remote client input",
        "model": args.model,
        "model_revision": model_revision,
        "dataset": "Salesforce/wikitext:wikitext-2-raw-v1:validation",
        "candidate_count": args.candidates,
        "seq_len": args.seq_len,
        "source_count": args.source_count,
        "trials": args.trials,
        "seed": args.seed,
        "layers": args.layers,
        "results": [],
    }
    for layer, dictionary in dictionaries.items():
        for fraction, scale, precision in conditions:
            started = time.time()
            trials = [
                one_trial(
                    dictionary,
                    args.source_count,
                    fraction,
                    scale,
                    precision,
                    rng,
                    helpers,
                    GeloObfuscatedLinear,
                    _batched_random_orthogonal,
                )
                for _ in range(args.trials)
            ]
            record = {
                "layer": layer,
                "mixing": "official orthogonal QR",
                "shield_fraction": fraction,
                "shield_scale": scale,
                "precision": precision,
                "exact_sets": int(sum(t["exact_set"] for t in trials)),
                "mean_recall": float(np.mean([t["recall"] for t in trials])),
                "min_gap": float(min(t["gap"] for t in trials)),
                "max_true_residual": float(
                    max(t["max_true_residual"] for t in trials)
                ),
                "min_false_residual": float(
                    min(t["min_false_residual"] for t in trials)
                ),
                "rank_min": int(min(t["observed_rank"] for t in trials)),
                "rank_max": int(max(t["observed_rank"] for t in trials)),
                "elapsed_seconds": time.time() - started,
            }
            output["results"].append(record)
            print(json.dumps(record, sort_keys=True), flush=True)
    output["candidate_texts"] = texts
    Path(args.output).write_text(
        json.dumps(output, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
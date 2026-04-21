"""
Zero-query weight recovery in STIP/Centaur-like threat model.

In STIP (Yang et al., 2024) and Centaur (Liu et al., 2025), the inference
server holds permuted weight matrices W_sigma = W[sigma, :] in plaintext.
Unlike SAFHIRE, no chosen-input queries are needed: the server directly
observes each column and sorts it.

This script:
  1. Loads a pretrained model.
  2. Applies a random row permutation sigma to each Conv2d weight matrix,
     simulating the server-side view in STIP/Centaur.
  3. Recovers sorted spectra by sorting each column of W_sigma -- zero queries.
  4. Computes max_error = max_j ||sort(W_sigma[:,j]) - sort(W[:,j]+b)||_inf.
     This is always 0: sorting undoes the row permutation exactly.

Reports per-layer: (d, max_error, queries_used=0).
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch

from lib.models import ResNet20, ResNet56
from lib.attack import get_conv_layers


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher-path", required=True)
    ap.add_argument("--architecture", default="resnet20", choices=["resnet20", "resnet56"])
    ap.add_argument("--output-path", required=True)
    ap.add_argument("--precision", type=int, default=256)
    ap.add_argument("--seed", type=int, default=0)
    return ap.parse_args()


def load_model(arch, path):
    model = ResNet20() if arch == "resnet20" else ResNet56()
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
    else:
        model.load_state_dict(ckpt)
    model.eval()
    return model


def main():
    args = parse_args()
    np.random.seed(args.seed)
    t0 = time.time()

    model = load_model(args.architecture, args.teacher_path)
    print(f"[model] {args.architecture} loaded from {args.teacher_path}", flush=True)

    p = args.precision
    layers = get_conv_layers(model)
    print(f"[layers] {len(layers)} Conv2d layers", flush=True)
    print(f"[threat model] server-side plaintext permuted weights (STIP/Centaur)",
          flush=True)
    print(f"\n{'Layer':<40} {'d':>6} {'MaxErr':>12} {'Queries':>9}", flush=True)

    results = []
    for (lname, W_mat, b_vec, _, C_out, d) in layers:
        W_q = np.round(W_mat * p) / p
        b_q = np.round(b_vec * p) / p

        perm = np.random.permutation(C_out)
        W_perm = W_q[perm, :]

        max_err = 0.0
        for i in range(d):
            true_sorted = np.sort(W_q[:, i] + b_q)
            recovered = np.sort(W_perm[:, i])
            err = np.max(np.abs(recovered - true_sorted))
            max_err = max(max_err, err)

        print(f"{lname:<40} {d:>6} {max_err:>12.2e} {'0':>9}", flush=True)
        results.append({
            "name": lname,
            "C_out": int(C_out),
            "d": int(d),
            "max_error": float(max_err),
            "queries_used": 0,
        })

    wall = time.time() - t0
    out = {
        "architecture": args.architecture,
        "precision": p,
        "threat_model": "server_plaintext_permuted (STIP/Centaur)",
        "seed": args.seed,
        "wall_seconds": wall,
        "layers": results,
        "total_queries": 0,
        "global_max_error": float(max(r["max_error"] for r in results)),
    }
    with open(args.output_path, "w") as f:
        json.dump(out, f, indent=2)

    print(f"\n[summary] global_max_error={out['global_max_error']:.2e}  "
          f"total_queries=0  layers={len(results)}", flush=True)
    print(f"[done] wall={wall:.1f}s  wrote {args.output_path}", flush=True)


if __name__ == "__main__":
    main()

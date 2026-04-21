"""
Property inference via sorted spectra: quantization bitwidth and sparsity.

Unique-value counts on sorted column spectra distinguish precision levels
(p=4 vs p=256 gives 7 vs 217 unique values). Sorted column norms expose
50% magnitude pruning directly, since zeroed weights collapse the norm
distribution.
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch
import torch.nn.utils.prune as prune

from lib.models import ResNet20, ResNet56
from lib.attack import get_conv_layers


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher-path", required=True)
    ap.add_argument("--architecture", default="resnet20", choices=["resnet20", "resnet56"])
    ap.add_argument("--output-path", required=True)
    ap.add_argument("--precisions", type=str, default="4,16,64,256")
    ap.add_argument("--prune-amount", type=float, default=0.5)
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
    t0 = time.time()

    model = load_model(args.architecture, args.teacher_path)
    print(f"[model] {args.architecture} loaded", flush=True)

    precisions = [int(x) for x in args.precisions.split(",")]
    layers = get_conv_layers(model)

    # ── (A) Quantization fingerprinting ──────────────────────────────────────
    print(f"\n=== (A) Quantization bitwidth fingerprinting ===", flush=True)
    print(f"{'Layer':<35} " + "".join(f"  p={p:>4}" for p in precisions), flush=True)

    quant_results = []
    for (lname, W_mat, b_vec, _, C_out, d) in layers:
        true_cols = W_mat + b_vec[:, None]
        row = f"{lname:<35}"
        layer_quant = {"name": lname, "precisions": []}
        for p in precisions:
            gamma = 1.0 / p
            W_q = np.round(true_cols / gamma) * gamma
            unique_vals = len(np.unique(W_q))
            val_range = float(W_q.max() - W_q.min())
            row += f"  {unique_vals:>6}"
            layer_quant["precisions"].append({
                "p": p, "unique_values": int(unique_vals),
                "value_range": val_range,
            })
        print(row, flush=True)
        quant_results.append(layer_quant)

    # ── (B) Sparsity / pruning detection ─────────────────────────────────────
    print(f"\n=== (B) Sparsity detection (prune_amount={args.prune_amount}) ===",
          flush=True)

    # Dense model column norms (first Conv2d layer as reference)
    dense_layers = get_conv_layers(model)

    # Apply unstructured magnitude pruning to a copy
    model_pruned = load_model(args.architecture, args.teacher_path)
    import torch.nn as nn
    conv_modules = [m for m in model_pruned.modules() if isinstance(m, nn.Conv2d)]
    for m in conv_modules:
        prune.l1_unstructured(m, name="weight", amount=args.prune_amount)
        prune.remove(m, "weight")
    pruned_layers = get_conv_layers(model_pruned)

    print(f"{'Layer':<35} {'Dense norm':>12} {'Pruned norm':>12} {'Zero frac':>10}",
          flush=True)

    sparsity_results = []
    for (dense_row, pruned_row) in zip(dense_layers, pruned_layers):
        lname = dense_row[0]
        W_dense  = dense_row[1] + dense_row[2][:, None]
        W_pruned = pruned_row[1] + pruned_row[2][:, None]

        dense_norms  = np.linalg.norm(W_dense,  axis=0)
        pruned_norms = np.linalg.norm(W_pruned, axis=0)
        zero_frac = float(np.mean(W_pruned == 0))

        dense_mean  = float(np.mean(dense_norms))
        pruned_mean = float(np.mean(pruned_norms))

        print(f"{lname:<35} {dense_mean:>12.3f} {pruned_mean:>12.3f} {zero_frac:>9.1%}",
              flush=True)
        sparsity_results.append({
            "name": lname,
            "dense_norm_mean": dense_mean,
            "dense_norm_std": float(np.std(dense_norms)),
            "pruned_norm_mean": pruned_mean,
            "pruned_norm_std": float(np.std(pruned_norms)),
            "zero_fraction": zero_frac,
        })

    # Summary for paper
    dense_all  = float(np.mean([r["dense_norm_mean"]  for r in sparsity_results]))
    pruned_all = float(np.mean([r["pruned_norm_mean"] for r in sparsity_results]))
    print(f"\n[summary] all-layer mean norm: dense={dense_all:.1f}  "
          f"pruned={pruned_all:.1f}", flush=True)

    wall = time.time() - t0
    out = {
        "architecture": args.architecture,
        "prune_amount": args.prune_amount,
        "wall_seconds": wall,
        "quantization": quant_results,
        "sparsity": sparsity_results,
        "summary": {
            "dense_norm_all_layers": dense_all,
            "pruned_norm_all_layers": pruned_all,
        },
    }
    with open(args.output_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[done] wall={wall:.1f}s  wrote {args.output_path}", flush=True)


if __name__ == "__main__":
    main()

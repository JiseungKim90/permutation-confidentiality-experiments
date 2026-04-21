"""
Empirical validation of Proposition 5 (fresh-permutation query complexity).

For each Conv2d layer, simulates the coupon-collector attack under independently
resampled permutations and measures T_emp (queries to recover all d* distinct
sorted spectra). Reports T_emp / T_theory per layer.

T_theory = d * (ln(d*) + c)  where c = 3 (confidence constant).
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
from lib.attack import get_conv_layers, round_then_sort


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher-path", required=True)
    ap.add_argument("--architecture", default="resnet20", choices=["resnet20", "resnet56"])
    ap.add_argument("--output-path", required=True)
    ap.add_argument("--precision", type=int, default=256)
    ap.add_argument("--n-trials", type=int, default=30)
    ap.add_argument("--confidence", type=float, default=3.0,
                    help="Constant c in T_theory = d*(ln(d*)+c)")
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


def count_queries_to_cover(W_q, b_q, gamma, d_star):
    """Simulate fresh-permutation attack; return #queries until all d* spectra seen."""
    C_out = W_q.shape[0]
    d = W_q.shape[1]
    Delta = gamma / 2
    B = Delta - 1e-12

    true_cols = W_q + b_q[:, None]
    sorted_spectra = set()
    for i in range(d):
        key = tuple(np.round(np.sort(true_cols[:, i]) / gamma).astype(int))
        sorted_spectra.add(key)

    seen = set()
    queries = 0
    while len(seen) < d_star:
        i = np.random.randint(d)
        perm = np.random.permutation(C_out)
        noise = np.random.normal(0, Delta / 3, C_out).clip(-B, B)
        observed = true_cols[perm, i] + noise
        recovered = round_then_sort(observed, gamma)
        key = tuple(np.round(recovered / gamma).astype(int))
        seen.add(key)
        queries += 1
        if queries > d_star * 200:
            break
    return queries


def main():
    args = parse_args()
    t0 = time.time()

    model = load_model(args.architecture, args.teacher_path)
    print(f"[model] {args.architecture} loaded from {args.teacher_path}", flush=True)

    p = args.precision
    gamma = 1.0 / p
    c = args.confidence
    layers = get_conv_layers(model)
    print(f"[layers] {len(layers)} Conv2d layers, n_trials={args.n_trials}", flush=True)
    print(f"{'Layer':<40} {'d':>6} {'d*':>6} {'T_emp':>10} {'T_theory':>10} {'ratio':>8}", flush=True)

    results = []
    for (lname, W_mat, b_vec, _, C_out, d) in layers:
        W_q = np.round(W_mat * p) / p
        b_q = np.round(b_vec * p) / p

        true_cols = W_q + b_q[:, None]
        spectra = set()
        for i in range(d):
            key = tuple(np.round(np.sort(true_cols[:, i]) / gamma).astype(int))
            spectra.add(key)
        d_star = len(spectra)

        T_theory = d * (np.log(d_star) + c)

        T_trials = []
        for _ in range(args.n_trials):
            T = count_queries_to_cover(W_q, b_q, gamma, d_star)
            T_trials.append(T)

        T_emp = float(np.mean(T_trials))
        ratio = T_emp / T_theory

        print(f"{lname:<40} {d:>6} {d_star:>6} {T_emp:>10.1f} {T_theory:>10.1f} {ratio:>7.2f}x", flush=True)

        results.append({
            "name": lname,
            "d": int(d),
            "d_star": int(d_star),
            "T_emp_mean": T_emp,
            "T_emp_std": float(np.std(T_trials)),
            "T_theory": float(T_theory),
            "ratio": float(ratio),
            "T_trials": [int(t) for t in T_trials],
        })

    wall = time.time() - t0
    out = {
        "architecture": args.architecture,
        "precision": p,
        "n_trials": args.n_trials,
        "confidence_c": c,
        "wall_seconds": wall,
        "layers": results,
    }
    with open(args.output_path, "w") as f:
        json.dump(out, f, indent=2)

    ratios = [r["ratio"] for r in results]
    print(f"\n[summary] ratio range [{min(ratios):.2f}, {max(ratios):.2f}] across {len(results)} layers", flush=True)
    print(f"[done] wall={wall:.1f}s  wrote {args.output_path}", flush=True)


if __name__ == "__main__":
    main()

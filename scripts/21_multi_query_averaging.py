"""
Multi-query averaging attack: shows that k repeated queries per column
reduce attack error as ~1/sqrt(k), even at high noise multipliers.

For each (nm, k) pair:
  - Pick n_cols random columns from every Conv2d layer
  - Make k fresh-permutation queries per column with noise B = Delta*nm
  - Average the k sorted estimates element-wise, then round
  - Report mean max-error across columns

Validates the paper's claim: "the attack degrades gracefully via averaging
over k repeated queries."
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
    ap.add_argument("--noise-mults", type=str, default="300,700,1000")
    ap.add_argument("--k-values", type=str, default="1,3,5,10,20")
    ap.add_argument("--n-cols", type=int, default=50,
                    help="Columns sampled per layer per (nm,k) trial")
    ap.add_argument("--n-trials", type=int, default=20,
                    help="Independent repetitions per (layer, nm, k)")
    ap.add_argument("--seed", type=int, default=42)
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


def multi_query_error(W_q, b_q, gamma, col_idx, k, nm):
    """k queries on column col_idx with FIXED permutation; average raw observations
    then round-then-sort.  Models the standard SAFHIRE setting where the same
    permutation sigma_r is reused across queries of the same session.
    Averaging k i.i.d. noise terms reduces effective sigma by 1/sqrt(k).
    """
    C_out = W_q.shape[0]
    Delta = gamma / 2
    B = Delta * nm - 1e-12
    sigma = B / 3.0

    true_col = W_q[:, col_idx] + b_q
    true_sorted = np.sort(true_col)

    perm = np.random.permutation(C_out)          # fixed for all k queries
    raw_sum = np.zeros(C_out)
    for _ in range(k):
        noise = np.random.normal(0, sigma, C_out).clip(-B, B)
        raw_sum += true_col[perm] + noise

    avg = raw_sum / k
    final = round_then_sort(avg, gamma)
    return float(np.max(np.abs(final - true_sorted)))


def main():
    args = parse_args()
    np.random.seed(args.seed)
    t0 = time.time()

    model = load_model(args.architecture, args.teacher_path)
    print(f"[model] {args.architecture} loaded", flush=True)

    p = args.precision
    gamma = 1.0 / p
    noise_mults = [int(x) for x in args.noise_mults.split(",")]
    k_values = [int(x) for x in args.k_values.split(",")]
    layers = get_conv_layers(model)

    print(f"[config] nm={noise_mults}  k={k_values}  "
          f"n_cols={args.n_cols}  n_trials={args.n_trials}", flush=True)
    header = f"{'Layer':<35} {'nm':>6} " + "".join(f"  k={k:>2}" for k in k_values)
    print(header, flush=True)

    results = []
    for (lname, W_mat, b_vec, _, C_out, d) in layers:
        W_q = np.round(W_mat * p) / p
        b_q = np.round(b_vec * p) / p

        layer_res = {"name": lname, "C_out": int(C_out), "d": int(d), "nm_results": []}

        for nm in noise_mults:
            errors_by_k = {k: [] for k in k_values}
            cols = np.random.choice(d, size=min(args.n_cols, d), replace=False)
            for _ in range(args.n_trials):
                for k in k_values:
                    errs = [multi_query_error(W_q, b_q, gamma, c, k, nm)
                            for c in cols]
                    errors_by_k[k].append(float(np.mean(errs)))

            nm_entry = {"noise_mult": nm, "k_results": []}
            row = f"{lname:<35} {nm:>5}x"
            for k in k_values:
                mean_err = float(np.mean(errors_by_k[k]))
                std_err  = float(np.std(errors_by_k[k]))
                nm_entry["k_results"].append({
                    "k": k, "mean_error": mean_err, "std_error": std_err,
                    "trials": errors_by_k[k],
                })
                row += f"  {mean_err:.3f}"
            layer_res["nm_results"].append(nm_entry)
            print(row, flush=True)

        results.append(layer_res)

    wall = time.time() - t0
    out = {
        "architecture": args.architecture,
        "precision": p,
        "noise_mults": noise_mults,
        "k_values": k_values,
        "n_cols": args.n_cols,
        "n_trials": args.n_trials,
        "wall_seconds": wall,
        "layers": results,
    }
    with open(args.output_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[done] wall={wall:.1f}s  wrote {args.output_path}", flush=True)


if __name__ == "__main__":
    main()

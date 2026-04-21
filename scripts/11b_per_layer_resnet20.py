"""
Per-layer noise-injection tradeoff for ResNet-20 on CIFAR-10 (JOTA variant).

Ports scripts/11b_tradeoff_per_layer.py to:
  - Air-gapped CIFAR test set via lib.cifar_subset.CIFARSubset
  - GPU execution with forward hooks registered on every Conv2d
  - JSON output and --n-trials argument

Per noise multiplier nm:
  B       = Delta * nm - 1e-12   (Delta = 1/(2p), p=256)
  For each Conv2d, a forward hook
      1. quantizes output to the lattice gamma = 1/p
      2. adds clipped Gaussian N(0, B/3) truncated to [-B, B]
  Run the full test loop; compare predictions to a noise-free baseline.
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as T

from lib.models import ResNet20
from lib.attack import get_conv_layers, round_then_sort
from lib.cifar_subset import CIFARSubset


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher-path", required=True)
    ap.add_argument("--data-root", required=True,
                    help="Directory containing cifar_subset.npz.xz")
    ap.add_argument("--output-path", required=True)
    ap.add_argument("--precision", type=int, default=256)
    ap.add_argument("--n-trials", type=int, default=200)
    ap.add_argument("--noise-mults", type=str, default="1,10,100,1000,5000")
    ap.add_argument("--batch-size", type=int, default=500)
    return ap.parse_args()


def load_resnet20(path, device):
    model = ResNet20()
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
    else:
        model.load_state_dict(ckpt)
    model.eval()
    return model.to(device)


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[init] device={device}", flush=True)

    subset_path = os.path.join(args.data_root, "cifar_subset.npz.xz")
    if not os.path.isfile(subset_path):
        print(f"[fatal] missing {subset_path}", flush=True)
        sys.exit(2)

    model = load_resnet20(args.teacher_path, device)
    print(f"[model] loaded ResNet-20 from {args.teacher_path}", flush=True)

    transform = T.Compose([
        T.ToTensor(),
        T.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])
    testset = CIFARSubset(subset_path, train=False, transform=transform)
    loader = torch.utils.data.DataLoader(
        testset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    images_all, labels_all = [], []
    for imgs, labs in loader:
        images_all.append(imgs)
        labels_all.append(labs)
    images_all = torch.cat(images_all).to(device)
    labels_all = torch.cat(labels_all).to(device)
    print(f"[data] test tensor shape={tuple(images_all.shape)} batch={args.batch_size}", flush=True)

    p = args.precision
    gamma = 1.0 / p
    Delta = 1.0 / (2 * p)

    conv_modules = [m for m in model.modules() if isinstance(m, nn.Conv2d)]
    print(f"[hooks] {len(conv_modules)} Conv2d layers targeted", flush=True)

    def make_hook(noise_state):
        def hook(module, inputs, output):
            B = noise_state["B"]
            out = torch.round(output * p) / p
            noise = torch.randn_like(out) * (B / 3.0)
            noise = noise.clamp(-B, B)
            return out + noise
        return hook

    noise_state = {"B": 0.0}
    handles = [m.register_forward_hook(make_hook(noise_state)) for m in conv_modules]

    layers = get_conv_layers(model)
    conv1_W, conv1_b = None, None
    for lname, wm, bv, _, _, _ in layers:
        if lname == "conv1":
            conv1_W, conv1_b = wm, bv
            break

    def measure_attack_error(W_mat, b_vec, noise_mult):
        C_out, d = W_mat.shape
        W_q = np.round(W_mat * p) / p
        b_q = np.round(b_vec * p) / p
        errors = []
        for i in range(d):
            true_col = W_q[:, i] + b_q
            true_sorted = np.sort(true_col)
            B = Delta * noise_mult - 1e-12
            noise = np.random.normal(0, Delta * noise_mult / 3, C_out).clip(-B, B)
            perm = np.random.permutation(C_out)
            observed = true_col[perm] + noise
            recovered = round_then_sort(observed, gamma)
            errors.append(np.max(np.abs(recovered - true_sorted)))
        return np.mean(errors)

    with torch.no_grad():
        noise_state["B"] = 0.0
        clean_preds = model(images_all).argmax(1)
    clean_acc = (clean_preds == labels_all).float().mean().item()
    print(f"[baseline] clean test accuracy: {clean_acc*100:.2f}%", flush=True)

    noise_mults = [int(x) for x in args.noise_mults.split(",")]
    n_trials = args.n_trials

    print(f"[sweep] noise_mults={noise_mults} n_trials={n_trials}", flush=True)
    print(f"{'Mult':>8}  {'AtkErr':>12}  {'PredAgr':>16}  {'TestAcc':>16}", flush=True)

    results = []
    for nm in noise_mults:
        np.random.seed(42)
        atk_err = measure_attack_error(conv1_W, conv1_b, nm)
        noise_state["B"] = Delta * nm - 1e-12

        agr_list, acc_list = [], []
        for trial in range(n_trials):
            torch.manual_seed(trial)
            with torch.no_grad():
                noisy_preds = model(images_all).argmax(1)
            agr_list.append((noisy_preds == clean_preds).float().mean().item())
            acc_list.append((noisy_preds == labels_all).float().mean().item())

        agr_mean = np.mean(agr_list) * 100
        agr_std  = np.std(agr_list)  * 100
        acc_mean = np.mean(acc_list) * 100
        acc_std  = np.std(acc_list)  * 100
        print(f"{nm:>7}x  {atk_err:12.2e}  {agr_mean:8.2f}±{agr_std:.2f}%  "
              f"{acc_mean:8.2f}±{acc_std:.2f}%", flush=True)
        results.append({
            "noise_mult": nm,
            "attack_error": float(atk_err),
            "pred_agr_mean": float(agr_mean),
            "pred_agr_std":  float(agr_std),
            "test_acc_mean": float(acc_mean),
            "test_acc_std":  float(acc_std),
        })

    for h in handles:
        h.remove()

    out = {
        "architecture": "ResNet-20",
        "n_trials": n_trials,
        "precision": p,
        "clean_acc": float(clean_acc * 100),
        "results": results,
    }
    with open(args.output_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[done] wrote {args.output_path}", flush=True)


if __name__ == "__main__":
    main()

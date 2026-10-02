"""QAT (and float-reference) training for items Q1 and Q2 of PLAN_qat.md.

One process trains one model.  Training is time-boxed: the cosine learning-rate
schedule is driven by *elapsed wall-clock time* rather than by a step count, so
a run that is stopped by its box has still annealed its learning rate and the
epochs reached are reported rather than assumed.

Usage
  qat_train.py --arch resnet20|resnet56|resnet18|resnet34 --mode qat|float
               --seed S --minutes M --out results/qat_train_<tag>.json
               --ckpt data/qat_<tag>.pt
"""
import argparse
import json
import math
import os
import time

import numpy as np
import torch
import torch.nn as nn

from qat_common import (DATA, RESULTS, QResNetCIFAR, QResNetTV, load_cifar_float,
                        load_tv_resnet, q_layers, accumulator_penalty,
                        export_model, occupancy_stats, posthoc_quantise,
                        env_info, budget_absmax, load_verified_checkpoint)
import qat_data


CIFAR10_RESNET20_SHA256 = (
    "4118986f0df73003d572b0e397f0ac7b3f60af1f31aff3d2da164536e36f6ec8")


def evaluate(model, X, y, bs, max_n=None, threads=1):
    model.eval()
    n = X.shape[0] if max_n is None else min(max_n, X.shape[0])
    correct = 0
    with torch.no_grad():
        for s in range(0, n, bs):
            xb = torch.from_numpy(qat_data.plain(X[s:s + bs]))
            out = model(xb)
            correct += int((out.argmax(1).numpy() == y[s:s + bs]).sum())
    model.train()
    return 100.0 * correct / n


def build(args):
    """Returns (model, data, info).  `model` is a QAT model when mode == qat and
    a QAT model with 32-bit quantisers (i.e. effectively float) when float."""
    w_bits = 32 if args.mode == "float" else args.w_bits
    a_bits = 32 if args.mode == "float" else args.a_bits
    info = {}
    if args.arch in ("resnet20", "resnet56"):
        n = 3 if args.arch == "resnet20" else 9
        path = os.path.join(DATA, "cifar10_resnet20.pt" if args.arch == "resnet20"
                            else "resnet56_seed0.pt")
        checkpoint_sha256 = (args.float_checkpoint_sha256 or
                             (CIFAR10_RESNET20_SHA256
                              if args.arch == "resnet20" else ""))
        if not checkpoint_sha256:
            raise ValueError("--float-checkpoint-sha256 is required for %s" %
                             args.arch)
        fm, meta = load_cifar_float(path, checkpoint_sha256, n)
        info["float_checkpoint"] = path
        info["float_checkpoint_sha256"] = checkpoint_sha256
        info["float_checkpoint_meta"] = meta
        model = QResNetCIFAR(fm, n, w_bits, a_bits, n_classes=10)
        Xtr, ytr, Xte, yte = qat_data.load_cifar10()
        data = (Xtr, ytr, Xte, yte)
        info["dataset"] = "CIFAR-10"
        info["pad"] = 4
    else:
        layers = [2, 2, 2, 2] if args.arch == "resnet18" else [3, 4, 6, 3]
        fn = ("resnet18-f37072fd.pth" if args.arch == "resnet18"
              else "resnet34-b627a593.pth")
        path = os.path.join(DATA, fn)
        if not args.float_checkpoint_sha256:
            raise ValueError("--float-checkpoint-sha256 is required for %s" %
                             args.arch)
        fm, meta = load_tv_resnet(
            path, args.float_checkpoint_sha256, layers, 1000)
        info["float_checkpoint"] = path
        info["float_checkpoint_sha256"] = args.float_checkpoint_sha256
        info["float_checkpoint_meta"] = meta
        Xtr, ytr, Xte, yte = qat_data.load_tin()
        if args.bn_recalib > 0:
            from qat_common import reestimate_bn, IMAGENET_MEAN, IMAGENET_STD
            rng0 = np.random.default_rng([777, args.seed])
            n = Xtr.shape[0]

            def gen():
                for _ in range(args.bn_recalib):
                    idx = np.sort(rng0.integers(0, n, size=args.bs))
                    yield torch.from_numpy(qat_data.plain(Xtr[idx]))

            info["bn_reestimation"] = reestimate_bn(fm, gen(), IMAGENET_MEAN,
                                                   IMAGENET_STD)
        model = QResNetTV(fm, w_bits, a_bits, n_classes=200)
        data = (Xtr, ytr, Xte, yte)
        info["dataset"] = "Tiny-ImageNet-200 (64x64, ImageNet architecture kept, "
        info["dataset"] += "only the classifier replaced by 512->200)"
        info["pad"] = 8
    if args.init_from:
        if not args.init_from_sha256:
            raise ValueError("--init-from-sha256 is required with --init-from")
        sd = load_verified_checkpoint(args.init_from, args.init_from_sha256)
        model.load_state_dict(sd["model"] if "model" in sd else sd, strict=True)
        info["initialised_from"] = args.init_from
    return model, data, info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", required=True)
    ap.add_argument("--mode", default="qat", choices=["qat", "float"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--minutes", type=float, default=90.0)
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--wd", type=float, default=5e-4)
    ap.add_argument("--bs", type=int, default=128)
    ap.add_argument("--w-bits", type=int, default=4)
    ap.add_argument("--a-bits", type=int, default=4)
    ap.add_argument("--acc-bits", type=int, default=14)
    ap.add_argument("--acc-lambda", type=float, default=0.1)
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--eval-n", type=int, default=2000)
    ap.add_argument("--final-eval-n", type=int, default=10000)
    ap.add_argument("--bn-recalib", type=int, default=0,
                    help="forward passes of BatchNorm re-estimation on the "
                         "target data before QAT (0 = keep the checkpoint's "
                         "statistics)")
    ap.add_argument("--init-from", default="")
    ap.add_argument("--init-from-sha256", default="")
    ap.add_argument("--float-checkpoint-sha256", default="")
    ap.add_argument("--tag", required=True)
    args = ap.parse_args()

    os.environ["OMP_NUM_THREADS"] = str(args.threads)
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng([20260912, args.seed, len(args.arch)])
    t0 = time.time()
    budget = args.minutes * 60.0

    model, (Xtr, ytr, Xte, yte), info = build(args)
    model.train()

    step_params, other = [], []
    for nme, p in model.named_parameters():
        (step_params if nme.endswith(".step") else other).append(p)
    opt = torch.optim.SGD(
        [{"params": other, "weight_decay": args.wd},
         {"params": step_params, "weight_decay": 0.0}],
        lr=args.lr, momentum=0.9, nesterov=True)
    crit = nn.CrossEntropyLoss()
    acc_limit = budget_absmax(args.acc_bits)

    ntr = Xtr.shape[0]
    hist = []
    step = 0
    seen = 0
    t_train0 = time.time()
    loss_ema = None
    pen_ema = 0.0
    stop = "time box reached"
    while True:
        el = time.time() - t_train0
        if el >= budget:
            break
        frac = min(1.0, el / budget)
        lr = args.lr * 0.5 * (1.0 + math.cos(math.pi * frac))
        for g in opt.param_groups:
            g["lr"] = lr
        idx = rng.integers(0, ntr, size=args.bs)
        idx.sort()
        xb = torch.from_numpy(qat_data.augment(Xtr[idx], info["pad"], rng))
        yb = torch.from_numpy(ytr[idx])
        out = model(xb)
        loss = crit(out, yb)
        pen = accumulator_penalty(model, acc_limit)
        total = loss + args.acc_lambda * pen
        opt.zero_grad(set_to_none=True)
        total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()
        step += 1
        seen += args.bs
        lv = float(loss.detach())
        loss_ema = lv if loss_ema is None else 0.98 * loss_ema + 0.02 * lv
        pen_ema = 0.98 * pen_ema + 0.02 * float(pen.detach())
        if step % 100 == 0:
            print("  step %6d  epoch %.3f  lr %.5f  loss %.4f  accpen %.4g  %.0fs"
                  % (step, seen / ntr, lr, loss_ema, pen_ema,
                     time.time() - t_train0), flush=True)
        if step % 1000 == 0:
            a = evaluate(model, Xte, yte, 200, max_n=args.eval_n)
            hist.append({"step": step, "epoch": round(seen / ntr, 4),
                         "elapsed_sec": round(time.time() - t_train0, 1),
                         "subset_top1": a, "loss_ema": round(loss_ema, 4)})
            print("    eval@%d subset top1 %.2f" % (step, a), flush=True)

    train_sec = time.time() - t_train0
    top1 = evaluate(model, Xte, yte, 200, max_n=args.final_eval_n)

    ckpt = os.path.join(DATA, "qat_%s.pt" % args.tag)
    torch.save({"model": model.state_dict(), "args": vars(args)}, ckpt)

    layers = export_model(model)
    occ = []
    ref_qmax = 2 ** (args.w_bits - 1) - 1
    for ly in layers:
        qmax = ly["qmax"] if args.mode == "qat" else ref_qmax
        o = occupancy_stats(ly["W_int"], qmax) if args.mode == "qat" else None
        ph_int, ph_s = posthoc_quantise(ly["W_float_folded"], args.w_bits)
        o_ph = occupancy_stats(ph_int, ref_qmax)
        occ.append({
            "layer": ly["name"], "kind": ly["kind"], "shape": ly["shape"],
            "m": int(ly["W_int"].shape[0]), "d": int(ly["W_int"].shape[1]),
            "s_w_learned": ly["s_w"] if args.mode == "qat" else None,
            "s_a_in_learned": ly["s_a_in"] if args.mode == "qat" else None,
            "s_w_posthoc_maxabs": ph_s,
            "step_ratio_posthoc_over_learned": (
                ph_s / ly["s_w"] if (args.mode == "qat" and ly["s_w"] > 0)
                else None),
            "b_int_absmax": (int(np.abs(ly["b_int"]).max())
                             if args.mode == "qat" else None),
            "accumulator_int_absmax_last_batch":
                ly["accumulator_int_absmax_last_batch"],
            "qat": o, "posthoc_same_weights": o_ph,
        })

    res = {
        "tag": args.tag, "arch": args.arch, "mode": args.mode, "seed": args.seed,
        "args": vars(args), "info": info,
        "quantiser": ("LSQ, learned per-layer step, symmetric weights on "
                      "{-%d..%d}, unsigned activations on {0..%d}, BatchNorm "
                      "folded into the convolution inside the forward pass with "
                      "frozen statistics"
                      % (2 ** (args.w_bits - 1) - 1, 2 ** (args.w_bits - 1) - 1,
                         2 ** args.a_bits - 1)),
        "accumulator_constraint": ("hinge penalty lambda*relu(max|u_int|/%d - 1) "
                                   "per layer per batch, %d signed bits"
                                   % (acc_limit, args.acc_bits)),
        "steps": step, "epochs": round(seen / ntr, 4), "images_seen": seen,
        "train_seconds": round(train_sec, 1), "time_box_seconds": budget,
        "stopped_reason": stop,
        "final_top1": top1, "final_eval_images": min(args.final_eval_n,
                                                     Xte.shape[0]),
        "history": hist, "checkpoint": ckpt,
        "layer_occupancy": occ,
        "environment": env_info(),
        "total_wall_time_sec": round(time.time() - t0, 1),
    }
    out = os.path.join(RESULTS, "qat_train_%s.json" % args.tag)
    with open(out, "w") as fh:
        json.dump(res, fh, indent=1, default=str)
    print("WROTE %s  steps=%d epochs=%.3f top1=%.2f  %.0fs"
          % (out, step, seen / ntr, top1, time.time() - t0), flush=True)


if __name__ == "__main__":
    main()

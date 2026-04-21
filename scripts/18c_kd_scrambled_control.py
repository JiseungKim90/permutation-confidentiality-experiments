"""
KD scrambled-column-norm control for Table 4 causal attribution.

Question: does the colnorm / full improvement come from the leaked spectrum,
or from the column-norm regularizer acting as a generic capacity control?

Design: run three modes with the same reg_lambda = 5:
  - baseline: no regularizer, no spectral init
  - colnorm:  regularizer + init using TEACHER's sorted column norms
  - scrambled: regularizer + init using sorted column norms from an
    UNRELATED trained teacher (different seed). The multiset is a
    plausible model prior but carries no information about the real
    teacher, so if the gain survives, it was never about the leaked
    spectrum.

Matched hyperparameters and seeds with 18_kd_spectral_priors_cifar.py.
"""
import argparse
import hashlib
import importlib.util
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader, Subset


_HERE = os.path.dirname(os.path.abspath(__file__))
_KD_PATH = os.path.join(_HERE, "18_kd_spectral_priors_cifar.py")
_spec = importlib.util.spec_from_file_location("kd_18", _KD_PATH)
kd18 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(kd18)


MODEL_DIR = os.path.join(_HERE, "..", "models")
OUTPUT_DIR = os.path.join(_HERE, "..", "outputs")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--teacher-path", required=True)
    p.add_argument("--scramble-teacher-path", required=True)
    p.add_argument("--teacher-logits-source", choices=["float", "quantized"],
                   default="quantized")
    p.add_argument("--data-root", default=os.path.join(_HERE, "..", "data"))
    p.add_argument("--query-budget", type=int, default=5000)
    p.add_argument("--subset-mode", choices=["prefix", "random"], default="prefix")
    p.add_argument("--subset-seed", type=int, default=0)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--student-epochs", type=int, default=10)
    p.add_argument("--student-lr", type=float, default=0.05)
    p.add_argument("--momentum", type=float, default=0.9)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--temperature", type=float, default=4.0)
    p.add_argument("--reg-lambda", type=float, default=5.0)
    p.add_argument("--distill-seeds", type=int, default=10)
    p.add_argument("--precision", type=int, default=256)
    p.add_argument("--architecture", choices=list(kd18.ARCHITECTURES.keys()),
                   default="resnet20")
    p.add_argument("--output-path", default=None)
    return p.parse_args()


def main():
    args = parse_args()
    arch_cls = kd18.ARCHITECTURES[args.architecture]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=== KD scrambled-column-norm control ===")
    print(f"device={device}")
    print(f"teacher={os.path.basename(args.teacher_path)}")
    print(f"scramble_teacher={os.path.basename(args.scramble_teacher_path)}")
    print(f"architecture={args.architecture} "
          f"query_budget={args.query_budget} seeds={args.distill_seeds}")

    transform = T.Compose([
        T.ToTensor(),
        T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    trainset = torchvision.datasets.CIFAR10(
        root=args.data_root, train=True, download=True, transform=transform)
    testset = torchvision.datasets.CIFAR10(
        root=args.data_root, train=False, download=True, transform=transform)
    test_loader = DataLoader(testset, batch_size=args.batch_size,
                             shuffle=False, num_workers=0)

    teacher, ckpt_t = kd18.build_teacher(
        args.teacher_path, device, args.teacher_logits_source,
        args.precision, arch_cls)
    teacher_used_acc = kd18.evaluate(teacher, test_loader, device)
    print(f"[teacher] test_acc={teacher_used_acc:.2f}%")

    scramble_teacher, _ = kd18.build_teacher(
        args.scramble_teacher_path, device, args.teacher_logits_source,
        args.precision, arch_cls)
    scramble_acc = kd18.evaluate(scramble_teacher, test_loader, device)
    print(f"[scramble teacher] test_acc={scramble_acc:.2f}%")

    query_budget = min(args.query_budget, len(trainset))
    qidx = kd18.select_query_indices(len(trainset), query_budget,
                                     args.subset_mode, args.subset_seed)
    qloader = DataLoader(Subset(trainset, qidx), batch_size=args.batch_size,
                         shuffle=False, num_workers=0)
    distill_dataset = kd18.cache_teacher_logits(teacher, qloader, device)
    distill_loader = DataLoader(distill_dataset, batch_size=args.batch_size,
                                shuffle=True, num_workers=0)
    print(f"[queries] cached teacher logits on {query_budget} images")

    layer_names = kd18.spectral_layer_names(teacher, "all")
    targets_teacher = kd18.recover_conv_spectral_targets(
        teacher, args.precision, layer_names)
    targets_scramble = kd18.recover_conv_spectral_targets(
        scramble_teacher, args.precision, layer_names)

    # Swap the sorted_col_norms in teacher targets with the scrambled source's,
    # leaving layer names and shapes teacher-sourced. We only need col_norms,
    # and only apply_spectral_init(use_col_norms) + spectral_regularizer read them.
    targets_scrambled_for_student = {}
    for name in layer_names:
        t = dict(targets_teacher[name])
        s = targets_scramble[name]
        if t["shape"] != s["shape"]:
            raise RuntimeError(
                f"shape mismatch on {name}: {t['shape']} vs {s['shape']}")
        t["sorted_col_norms"] = s["sorted_col_norms"]
        targets_scrambled_for_student[name] = t

    # Sanity check: the two col_norm vectors must differ.
    any_diff = False
    for name in layer_names:
        a = targets_teacher[name]["sorted_col_norms"]
        b = targets_scrambled_for_student[name]["sorted_col_norms"]
        if not np.allclose(a, b):
            any_diff = True
            break
    if not any_diff:
        raise RuntimeError("scrambled col_norms match teacher — source "
                           "checkpoints are identical")
    print("[check] scrambled col_norms differ from teacher ✓")

    all_results = {"config": {
        "teacher_path": os.path.basename(args.teacher_path),
        "scramble_teacher_path": os.path.basename(args.scramble_teacher_path),
        "teacher_test_acc": teacher_used_acc,
        "scramble_teacher_test_acc": scramble_acc,
        "architecture": args.architecture,
        "query_budget": int(query_budget),
        "distill_seeds": args.distill_seeds,
        "student_epochs": args.student_epochs,
        "reg_lambda": args.reg_lambda,
        "precision": args.precision,
    }}

    # baseline: (False, False, False, "all")
    # colnorm:  (False, True,  True,  "all")  — teacher col_norms
    # scrambled:(False, True,  True,  "all")  — other model's col_norms
    modes = [
        ("baseline",  targets_teacher,                  False, False, False),
        ("colnorm",   targets_teacher,                  False, True,  True),
        ("scrambled", targets_scrambled_for_student,    False, True,  True),
    ]

    finals = {}
    for mode, targets_for_mode, iw, in_, ur in modes:
        hist_all = []
        fin = []
        for seed in range(args.distill_seeds):
            history = kd18.train_student(
                mode=mode,
                spectral_targets_for_mode=targets_for_mode,
                init_use_weights=iw,
                init_use_norms=in_,
                use_reg=ur,
                distill_loader=distill_loader,
                test_loader=test_loader,
                device=device,
                epochs=args.student_epochs,
                lr=args.student_lr,
                momentum=args.momentum,
                weight_decay=args.weight_decay,
                temperature=args.temperature,
                reg_lambda=args.reg_lambda,
                seed=seed,
                student_arch_cls=arch_cls,
            )
            hist_all.append(history)
            fin.append(history[-1]["test_acc"])
            print(f"  [{mode}] seed={seed} final={fin[-1]:.2f}%", flush=True)
        finals[mode] = fin
        all_results[mode] = {"histories": hist_all, "final_accs": fin}

    print("\n=== Summary ===")
    base_mean = float(np.mean(finals["baseline"]))
    for mode in ("baseline", "colnorm", "scrambled"):
        m, s = kd18.mean_and_std(finals[mode])
        gain = float(np.mean(finals[mode])) - base_mean
        print(f"{mode:10s} {m:.2f}% ± {s:.2f}  gain={gain:+.2f} pts")

    all_results["summary"] = {
        mode: {"mean": kd18.mean_and_std(finals[mode])[0],
               "std":  kd18.mean_and_std(finals[mode])[1]}
        for mode in finals
    }
    all_results["summary"]["colnorm_minus_baseline"] = (
        float(np.mean(finals["colnorm"])) - base_mean)
    all_results["summary"]["scrambled_minus_baseline"] = (
        float(np.mean(finals["scrambled"])) - base_mean)
    all_results["summary"]["colnorm_minus_scrambled"] = (
        float(np.mean(finals["colnorm"])) - float(np.mean(finals["scrambled"])))

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = args.output_path or os.path.join(
        OUTPUT_DIR,
        f"cifar_kd_scrambled_control_{args.architecture}_q{query_budget}.json")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()

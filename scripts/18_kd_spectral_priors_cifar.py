"""
Proof-of-concept model stealing on CIFAR-10 / ResNet-20:
logit-only distillation vs. distillation with sorted-spectrum priors.

Baseline:
  Standard knowledge distillation using exact teacher logits only.

Ours:
  Same logit-only distillation, plus permutation-invariant priors extracted
  from the sorted-spectrum attack on intermediate convolutional layers.
  We use:
    (1) layerwise weight-distribution matching for initialization; and
    (2) a sorted column-norm regularizer during distillation.
"""
import argparse
import hashlib
import json
import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader, Subset, TensorDataset

from lib.attack import get_conv_layers
from lib.models import ResNet20, ResNet56


ARCHITECTURES = {
    "resnet20": ResNet20,
    "resnet56": ResNet56,
}


MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs")
DEFAULT_TEACHER = os.path.join(MODEL_DIR, "resnet20_seed0.pt")
FRONTIER_LAYERS = ("conv1", "layer2.0.conv1", "layer3.0.conv1")
FIRST_LAYER = ("conv1",)

# Each ablation mode is a (init_use_weights, init_use_norms, use_reg, scope) tuple.
ABLATION_MODES = {
    "baseline":  (False, False, False, "all"),
    "first":     (True,  True,  True,  "first"),
    "colnorm":   (False, True,  True,  "all"),
    "full":      (True,  True,  True,  "all"),
}


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def file_sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sanitize_path_for_output(path):
    if os.path.isabs(path):
        return os.path.basename(path)
    return path


def load_model(path, device, arch_cls):
    model = arch_cls().to(device)
    ckpt = torch.load(path, map_location=device, weights_only=False)
    if "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
    else:
        model.load_state_dict(ckpt)
    model.eval()
    return model, ckpt


def quantize_state_dict(state_dict, p):
    q_state = {}
    for key, value in state_dict.items():
        if torch.is_tensor(value) and torch.is_floating_point(value):
            q_state[key] = torch.round(value * p) / p
        else:
            q_state[key] = value
    return q_state


def build_teacher(path, device, logits_source, p, arch_cls):
    teacher_float, ckpt = load_model(path, device, arch_cls)
    state = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt

    if logits_source == "float":
        teacher_used = teacher_float
    elif logits_source == "quantized":
        teacher_used = arch_cls().to(device)
        teacher_used.load_state_dict(quantize_state_dict(state, p))
        teacher_used.eval()
    else:
        raise ValueError(f"Unknown teacher logits source: {logits_source}")

    return teacher_used, ckpt


def evaluate(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            pred = model(images).argmax(dim=1)
            total += labels.numel()
            correct += pred.eq(labels).sum().item()
    return 100.0 * correct / total


def cache_teacher_logits(teacher, loader, device):
    teacher.eval()
    xs = []
    logits = []
    ys = []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            out = teacher(images).cpu()
            xs.append(images.cpu())
            logits.append(out)
            ys.append(labels)
    return TensorDataset(torch.cat(xs), torch.cat(logits), torch.cat(ys))


def select_query_indices(n_total, budget, mode, seed):
    if mode == "prefix":
        return list(range(budget))
    if mode == "random":
        rng = np.random.default_rng(seed)
        idx = rng.choice(n_total, size=budget, replace=False)
        return sorted(int(i) for i in idx.tolist())
    raise ValueError(f"Unknown subset mode: {mode}")


def spectral_layer_names(teacher, scope):
    all_names = [name for name, *_ in get_conv_layers(teacher)]
    if scope == "all":
        return tuple(all_names)
    if scope == "frontier3":
        return tuple(name for name in all_names if name in FRONTIER_LAYERS)
    if scope == "first":
        return tuple(name for name in all_names if name in FIRST_LAYER)
    raise ValueError(f"Unknown scope: {scope}")


def recover_conv_spectral_targets(teacher, p, layer_names):
    targets = {}
    for name, W_mat, _b_vec, _has_bias, _c_out, _d in get_conv_layers(teacher):
        if name not in layer_names:
            continue
        W_q = np.round(W_mat * p) / p
        col_norms = np.linalg.norm(W_q, axis=0).astype(np.float32)
        weight_values = np.sort(W_q.reshape(-1).astype(np.float32))
        targets[name] = {
            "shape": tuple(W_mat.shape),
            "sorted_col_norms": np.sort(col_norms).astype(np.float32),
            "weight_values_sorted": weight_values,
            "weight_std": float(weight_values.std()),
            "weight_mean": float(weight_values.mean()),
        }
    return targets


def conv_weight_matrix(weight):
    return weight.view(weight.shape[0], -1)


def apply_spectral_init(student, spectral_targets, use_weight_values=True, use_col_norms=True):
    for name, module in student.named_modules():
        if not isinstance(module, torch.nn.Conv2d) or name not in spectral_targets:
            continue

        target = spectral_targets[name]
        W = module.weight.data

        if use_weight_values:
            # Match the exact recovered layerwise weight distribution up to rank.
            flat = W.reshape(-1)
            order = torch.argsort(flat)
            target_vals = torch.from_numpy(target["weight_values_sorted"]).to(
                device=flat.device, dtype=flat.dtype
            )
            if target_vals.numel() == flat.numel():
                remapped = torch.empty_like(flat)
                remapped[order] = target_vals
                flat.copy_(remapped)
            else:
                current_std = flat.std(unbiased=False).clamp_min(1e-8)
                target_std = torch.tensor(
                    target["weight_std"], device=flat.device, dtype=flat.dtype
                )
                flat.mul_(target_std / current_std)

        if use_col_norms:
            # Match the multiset of column norms without assuming any column labels.
            W_mat = conv_weight_matrix(W)
            current_norms = W_mat.norm(dim=0).clamp_min(1e-8)
            sort_idx = torch.argsort(current_norms)
            target_norms = torch.from_numpy(target["sorted_col_norms"]).to(
                device=W.device, dtype=W.dtype
            )
            scales = torch.empty_like(current_norms)
            scales[sort_idx] = target_norms / current_norms[sort_idx]
            W_mat.mul_(scales.unsqueeze(0))


def spectral_regularizer(student, spectral_targets, device):
    total = torch.zeros((), device=device)
    count = 0
    for name, module in student.named_modules():
        if not isinstance(module, torch.nn.Conv2d) or name not in spectral_targets:
            continue
        target = spectral_targets[name]
        W_mat = conv_weight_matrix(module.weight)
        current_sorted = torch.sort(W_mat.norm(dim=0), dim=0).values
        target_sorted = torch.tensor(target["sorted_col_norms"], device=device)
        total = total + F.mse_loss(current_sorted, target_sorted)
        count += 1
    if count == 0:
        return total
    return total / count


def kd_loss(student_logits, teacher_logits, temperature):
    s = F.log_softmax(student_logits / temperature, dim=1)
    t = F.softmax(teacher_logits / temperature, dim=1)
    return F.kl_div(s, t, reduction="batchmean") * (temperature ** 2)


def train_student(
    mode,
    spectral_targets_for_mode,
    init_use_weights,
    init_use_norms,
    use_reg,
    distill_loader,
    test_loader,
    device,
    epochs,
    lr,
    momentum,
    weight_decay,
    temperature,
    reg_lambda,
    seed,
    student_arch_cls,
):
    set_seed(seed)
    student = student_arch_cls().to(device)
    if init_use_weights or init_use_norms:
        apply_spectral_init(
            student,
            spectral_targets_for_mode,
            use_weight_values=init_use_weights,
            use_col_norms=init_use_norms,
        )

    optimizer = torch.optim.SGD(
        student.parameters(),
        lr=lr,
        momentum=momentum,
        weight_decay=weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    history = []

    for epoch in range(epochs):
        lr_train = optimizer.param_groups[0]["lr"]
        student.train()
        total_kd = 0.0
        total_reg = 0.0
        n_batch = 0

        for images, teacher_logits, _labels in distill_loader:
            images = images.to(device)
            teacher_logits = teacher_logits.to(device)
            optimizer.zero_grad()

            student_logits = student(images)
            loss_kd = kd_loss(student_logits, teacher_logits, temperature)
            loss = loss_kd
            reg_value = torch.zeros((), device=device)
            if use_reg:
                reg_value = spectral_regularizer(student, spectral_targets_for_mode, device)
                loss = loss + reg_lambda * reg_value

            loss.backward()
            optimizer.step()
            total_kd += loss_kd.item()
            total_reg += reg_value.item()
            n_batch += 1

        scheduler.step()
        test_acc = evaluate(student, test_loader, device)
        entry = {
            "epoch": epoch + 1,
            "lr_train": lr_train,
            "lr_next": scheduler.get_last_lr()[0],
            "kd_loss": total_kd / max(n_batch, 1),
            "reg_loss": total_reg / max(n_batch, 1),
            "test_acc": test_acc,
        }
        history.append(entry)
        print(
            f"[{mode}] seed={seed} epoch {epoch + 1:02d}/{epochs}: "
            f"kd={entry['kd_loss']:.4f}, reg={entry['reg_loss']:.6f}, "
            f"test_acc={test_acc:.2f}%"
        )

    return history


def mean_and_std(values):
    arr = np.array(values, dtype=np.float64)
    return float(arr.mean()), float(arr.std(ddof=0))


def epoch_mean(histories, key):
    return [
        float(np.mean([hist[e][key] for hist in histories]))
        for e in range(len(histories[0]))
    ]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher-path", type=str, default=DEFAULT_TEACHER)
    parser.add_argument("--teacher-logits-source", choices=["float", "quantized"], default="float")
    parser.add_argument("--data-root", type=str, default=DATA_DIR)
    parser.add_argument("--query-budget", type=int, default=5000)
    parser.add_argument("--subset-mode", choices=["prefix", "random"], default="prefix")
    parser.add_argument("--subset-seed", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--student-epochs", type=int, default=3)
    parser.add_argument("--student-lr", type=float, default=0.05)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--temperature", type=float, default=4.0)
    parser.add_argument("--reg-lambda", type=float, default=5.0)
    parser.add_argument("--distill-seeds", type=int, default=3)
    parser.add_argument("--precision", type=int, default=256)
    parser.add_argument("--prior-scope", choices=["first", "frontier3", "all"], default="frontier3")
    parser.add_argument(
        "--ablation-modes",
        type=str,
        default="baseline,full",
        help="Comma-separated list from {baseline, first, colnorm, full}.",
    )
    parser.add_argument(
        "--architecture",
        type=str,
        choices=list(ARCHITECTURES.keys()),
        default="resnet20",
        help="Student/teacher architecture.",
    )
    parser.add_argument("--output-path", type=str, default=None)
    args = parser.parse_args()
    arch_cls = ARCHITECTURES[args.architecture]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=== CIFAR-10 logit stealing with spectral priors ===")
    print(f"device={device}")
    display_args = dict(vars(args))
    for key in ("teacher_path", "data_root", "output_path"):
        if display_args.get(key) is not None:
            display_args[key] = sanitize_path_for_output(display_args[key])
    print(json.dumps(display_args, indent=2))

    transform = T.Compose([
        T.ToTensor(),
        T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    trainset = torchvision.datasets.CIFAR10(
        root=args.data_root, train=True, download=True, transform=transform
    )
    testset = torchvision.datasets.CIFAR10(
        root=args.data_root, train=False, download=True, transform=transform
    )
    test_loader = DataLoader(
        testset, batch_size=args.batch_size, shuffle=False, num_workers=0
    )

    teacher, ckpt = build_teacher(
        args.teacher_path,
        device,
        args.teacher_logits_source,
        args.precision,
        arch_cls,
    )
    teacher_checkpoint_acc = ckpt.get("test_accuracy")
    teacher_used_acc = evaluate(teacher, test_loader, device)
    print(
        f"[teacher] loaded {sanitize_path_for_output(args.teacher_path)} "
        f"(logits={args.teacher_logits_source}): "
        f"test_acc={teacher_used_acc:.2f}%"
    )

    query_budget = min(args.query_budget, len(trainset))
    query_indices = select_query_indices(
        len(trainset),
        query_budget,
        args.subset_mode,
        args.subset_seed,
    )
    query_subset = Subset(trainset, query_indices)
    query_loader = DataLoader(
        query_subset, batch_size=args.batch_size, shuffle=False, num_workers=0
    )
    distill_dataset = cache_teacher_logits(teacher, query_loader, device)
    distill_loader = DataLoader(
        distill_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0
    )
    print(f"[queries] cached exact teacher logits for {query_budget} unlabeled images")

    requested_modes = [m.strip() for m in args.ablation_modes.split(",") if m.strip()]
    for m in requested_modes:
        if m not in ABLATION_MODES:
            raise ValueError(f"Unknown ablation mode: {m}")
    print(f"[ablation] modes={requested_modes}")

    # Precompute spectral targets per scope used by any selected mode.
    scope_to_targets = {}
    for m in requested_modes:
        _, _, _, scope = ABLATION_MODES[m]
        if scope not in scope_to_targets:
            layer_names_m = spectral_layer_names(teacher, scope)
            scope_to_targets[scope] = (
                layer_names_m,
                recover_conv_spectral_targets(teacher, args.precision, layer_names_m),
            )
            print(f"[spectra/{scope}] using {len(layer_names_m)} conv layers")
            for name in layer_names_m[:3]:
                target = scope_to_targets[scope][1][name]
                print(
                    f"  {name:20s} shape={target['shape']} "
                    f"std={target['weight_std']:.4f} "
                    f"mean_col_norm={np.mean(target['sorted_col_norms']):.4f}"
                )

    query_indices_sha256 = hashlib.sha256(
        json.dumps(query_indices).encode("utf-8")
    ).hexdigest()

    all_results = {
        "config": {
            "teacher_path": sanitize_path_for_output(args.teacher_path),
            "teacher_logits_source": args.teacher_logits_source,
            "teacher_checkpoint_acc": teacher_checkpoint_acc,
            "teacher_used_acc": teacher_used_acc,
            "teacher_checkpoint_sha256": file_sha256(args.teacher_path),
            "student_architecture": args.architecture,
            "dataset": "CIFAR-10",
            "query_budget": int(query_budget),
            "subset_mode": args.subset_mode,
            "subset_seed": args.subset_seed,
            "query_indices": query_indices,
            "query_indices_sha256": query_indices_sha256,
            "batch_size": args.batch_size,
            "student_epochs": args.student_epochs,
            "optimizer": "SGD",
            "student_lr": args.student_lr,
            "momentum": args.momentum,
            "weight_decay": args.weight_decay,
            "scheduler": "CosineAnnealingLR",
            "temperature": args.temperature,
            "reg_lambda": args.reg_lambda,
            "distill_seeds": args.distill_seeds,
            "precision": args.precision,
            "ablation_modes": requested_modes,
            "scope_layers": {
                scope: list(names) for scope, (names, _) in scope_to_targets.items()
            },
        }
    }
    final_scores = {}
    epoch_scores = {}

    for mode in requested_modes:
        init_use_weights, init_use_norms, use_reg, scope = ABLATION_MODES[mode]
        targets_for_mode = scope_to_targets[scope][1]
        histories = []
        finals = []
        for seed in range(args.distill_seeds):
            history = train_student(
                mode=mode,
                spectral_targets_for_mode=targets_for_mode,
                init_use_weights=init_use_weights,
                init_use_norms=init_use_norms,
                use_reg=use_reg,
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
            histories.append(history)
            finals.append(history[-1]["test_acc"])

        final_scores[mode] = finals
        epoch_scores[mode] = [
            [hist[epoch]["test_acc"] for hist in histories]
            for epoch in range(args.student_epochs)
        ]
        all_results[mode] = {
            "histories": histories,
            "final_accs": finals,
            "scope": scope,
            "init_use_weights": init_use_weights,
            "init_use_norms": init_use_norms,
            "use_reg": use_reg,
        }

    all_results["summary"] = {
        mode: {
            "final_mean": mean_and_std(final_scores[mode])[0],
            "final_std": mean_and_std(final_scores[mode])[1],
            "epoch_mean_test_acc": epoch_mean(all_results[mode]["histories"], "test_acc"),
            "epoch_mean_kd_loss": epoch_mean(all_results[mode]["histories"], "kd_loss"),
            "epoch_mean_reg_loss": epoch_mean(all_results[mode]["histories"], "reg_loss"),
        }
        for mode in requested_modes
    }

    print("\n=== Summary ===")
    print(f"Teacher test accuracy: {teacher_used_acc:.2f}%")
    for mode in requested_modes:
        mean_final, std_final = mean_and_std(final_scores[mode])
        print(
            f"{mode:9s} final: {mean_final:.2f}% +- {std_final:.2f} "
            f"(seeds={final_scores[mode]})"
        )
        for epoch_idx, values in enumerate(epoch_scores[mode], start=1):
            mean_epoch, std_epoch = mean_and_std(values)
            print(f"  epoch {epoch_idx}: {mean_epoch:.2f}% +- {std_epoch:.2f}")

    if "baseline" in final_scores:
        baseline_mean = float(np.mean(final_scores["baseline"]))
        for mode in requested_modes:
            if mode == "baseline":
                continue
            gain = float(np.mean(final_scores[mode])) - baseline_mean
            print(f"Gain ({mode} - baseline): {gain:+.2f} pts")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = args.output_path
    if out_path is None:
        out_path = os.path.join(OUTPUT_DIR, "cifar_kd_spectral_priors.json")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"Saved JSON summary to {out_path}")


if __name__ == "__main__":
    main()

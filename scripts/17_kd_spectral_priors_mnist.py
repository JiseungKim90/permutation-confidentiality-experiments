"""
Proof-of-concept model stealing on MNIST:
logit-only distillation vs. distillation with hidden-layer spectral priors.

Teacher:
  BiasFreeMNISTNet (784 -> 128 -> 64 -> 10), trained with labels.

Baseline:
  Standard knowledge distillation using exact teacher logits only.

Ours:
  Same logit-only distillation, plus hidden-layer priors recovered from the
  sorted-spectrum attack. Concretely, we use:
    (1) spectrum-matched initialization for fc1/fc2 based on recovered
        weight-value distribution + per-column norms; and
    (2) a column-norm regularizer during distillation.

This keeps the "function leakage" channel fixed (teacher logits) and tests
whether the "identity leakage" channel (sorted spectra) accelerates model
stealing under the same unlabeled query budget.
"""
import argparse
import json
import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader, Subset, TensorDataset

from lib.attack import get_linear_layers


MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs")
TEACHER_PATH = os.path.join(MODEL_DIR, "mnist_biasfree_teacher_seed0.pt")
TARGET_LAYERS = ("fc1", "fc2")


class BiasFreeMNISTNet(nn.Module):
    """784 -> 128 -> 64 -> 10 with bias-free hidden layers."""

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 128, bias=False)
        self.fc2 = nn.Linear(128, 64, bias=False)
        self.fc3 = nn.Linear(64, 10, bias=True)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


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


def train_teacher(train_loader, test_loader, device, epochs, lr, seed):
    set_seed(seed)
    teacher = BiasFreeMNISTNet().to(device)
    opt = torch.optim.Adam(teacher.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    history = []
    for epoch in range(epochs):
        teacher.train()
        total_loss = 0.0
        n_batch = 0
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)
            opt.zero_grad()
            loss = criterion(teacher(images), labels)
            loss.backward()
            opt.step()
            total_loss += loss.item()
            n_batch += 1
        acc = evaluate(teacher, test_loader, device)
        history.append({
            "epoch": epoch + 1,
            "loss": total_loss / max(n_batch, 1),
            "test_acc": acc,
        })
        print(
            f"[teacher] epoch {epoch + 1:02d}/{epochs}: "
            f"loss={history[-1]['loss']:.4f}, test_acc={acc:.2f}%"
        )

    os.makedirs(MODEL_DIR, exist_ok=True)
    torch.save(
        {
            "model_state_dict": teacher.state_dict(),
            "history": history,
            "seed": seed,
        },
        TEACHER_PATH,
    )
    return teacher, history


def load_or_train_teacher(train_loader, test_loader, device, epochs, lr, force_train):
    if os.path.exists(TEACHER_PATH) and not force_train:
        teacher = BiasFreeMNISTNet().to(device)
        ckpt = torch.load(TEACHER_PATH, map_location=device, weights_only=False)
        teacher.load_state_dict(ckpt["model_state_dict"])
        history = ckpt.get("history", [])
        acc = evaluate(teacher, test_loader, device)
        print(f"[teacher] loaded cached checkpoint: test_acc={acc:.2f}%")
        return teacher, history

    print("[teacher] checkpoint not found; training a fresh teacher")
    return train_teacher(train_loader, test_loader, device, epochs, lr, seed=0)


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


def recover_hidden_spectral_targets(teacher, p):
    targets = {}
    for name, W, b, _has_bias in get_linear_layers(teacher):
        if name not in TARGET_LAYERS:
            continue

        # The first two hidden layers are bias-free, so the round-then-sort
        # attack recovers the exact multiset of weights in each input column.
        W_q = np.round(W * p) / p
        recovered_cols = []
        for i in range(W_q.shape[1]):
            recovered_cols.append(np.sort(W_q[:, i]).astype(np.float32))

        col_norms = np.array(
            [np.linalg.norm(col) for col in recovered_cols], dtype=np.float32
        )
        weight_values = np.concatenate(recovered_cols).astype(np.float32)
        targets[name] = {
            "shape": W.shape,
            "col_norms": col_norms,
            "weight_values": weight_values,
            "weight_mean": float(weight_values.mean()),
            "weight_std": float(weight_values.std()),
        }
    return targets


def apply_spectral_init(student, spectral_targets, seed):
    del seed  # kept for call-site compatibility while init remains deterministic
    for name, module in student.named_modules():
        if not isinstance(module, nn.Linear) or name not in spectral_targets:
            continue
        target = spectral_targets[name]

        # Start from the standard random initialization, then inject only the
        # recoverable low-order structure: layer-scale statistics and per-column
        # norms. This is more stable than attempting to sample a full weight
        # matrix from unordered multisets.
        W = module.weight.data
        current_std = W.std(unbiased=False).clamp_min(1e-8)
        target_std = torch.tensor(target["weight_std"], dtype=W.dtype)
        W.mul_(target_std / current_std)

        target_norms = torch.from_numpy(target["col_norms"]).to(W.dtype)
        current_norms = W.norm(dim=0).clamp_min(1e-8)
        W.mul_((target_norms / current_norms).unsqueeze(0))

        # Keep the prior conservative: do not use recovered exact biases.
        if module.bias is not None:
            module.bias.data.zero_()


def spectral_regularizer(student, spectral_targets, device):
    loss = torch.zeros((), device=device)
    count = 0
    for name, module in student.named_modules():
        if not isinstance(module, nn.Linear) or name not in spectral_targets:
            continue
        target = spectral_targets[name]
        target_norms = torch.tensor(target["col_norms"], device=device)
        current_norms = module.weight.norm(dim=0)
        loss = loss + F.mse_loss(current_norms, target_norms)
        count += 1
    if count == 0:
        return loss
    return loss / count


def kd_loss(student_logits, teacher_logits, temperature):
    s = F.log_softmax(student_logits / temperature, dim=1)
    t = F.softmax(teacher_logits / temperature, dim=1)
    return F.kl_div(s, t, reduction="batchmean") * (temperature ** 2)


def train_student(
    mode,
    spectral_targets,
    distill_loader,
    test_loader,
    device,
    epochs,
    lr,
    temperature,
    reg_lambda,
    seed,
):
    set_seed(seed)
    student = BiasFreeMNISTNet().to(device)
    if mode == "ours":
        apply_spectral_init(student, spectral_targets, seed=seed + 12345)

    opt = torch.optim.Adam(student.parameters(), lr=lr)
    history = []

    for epoch in range(epochs):
        student.train()
        total_kd = 0.0
        total_reg = 0.0
        n_batch = 0
        for images, teacher_logits, _labels in distill_loader:
            images = images.to(device)
            teacher_logits = teacher_logits.to(device)
            opt.zero_grad()

            student_logits = student(images)
            loss_kd = kd_loss(student_logits, teacher_logits, temperature)
            loss = loss_kd
            reg_value = torch.zeros((), device=device)
            if mode == "ours":
                reg_value = spectral_regularizer(student, spectral_targets, device)
                loss = loss + reg_lambda * reg_value

            loss.backward()
            opt.step()

            total_kd += loss_kd.item()
            total_reg += reg_value.item()
            n_batch += 1

        test_acc = evaluate(student, test_loader, device)
        entry = {
            "epoch": epoch + 1,
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

    return student, history


def mean_and_std(values):
    arr = np.array(values, dtype=np.float64)
    return float(arr.mean()), float(arr.std(ddof=0))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher-epochs", type=int, default=8)
    parser.add_argument("--teacher-lr", type=float, default=1e-3)
    parser.add_argument("--student-epochs", type=int, default=5)
    parser.add_argument("--student-lr", type=float, default=1e-3)
    parser.add_argument("--temperature", type=float, default=4.0)
    parser.add_argument("--reg-lambda", type=float, default=0.05)
    parser.add_argument("--query-budget", type=int, default=10000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--distill-seeds", type=int, default=3)
    parser.add_argument("--precision", type=int, default=256)
    parser.add_argument("--force-train-teacher", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=== MNIST logit stealing with spectral priors ===")
    print(f"device={device}")
    print(json.dumps(vars(args), indent=2))

    transform = T.ToTensor()
    trainset = torchvision.datasets.MNIST(
        root=DATA_DIR, train=True, download=True, transform=transform
    )
    testset = torchvision.datasets.MNIST(
        root=DATA_DIR, train=False, download=True, transform=transform
    )
    train_loader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    teacher, teacher_history = load_or_train_teacher(
        train_loader,
        test_loader,
        device,
        args.teacher_epochs,
        args.teacher_lr,
        args.force_train_teacher,
    )
    teacher_acc = evaluate(teacher, test_loader, device)

    query_budget = min(args.query_budget, len(trainset))
    query_indices = list(range(query_budget))
    query_subset = Subset(trainset, query_indices)
    query_loader = DataLoader(
        query_subset, batch_size=args.batch_size, shuffle=False, num_workers=0
    )
    distill_dataset = cache_teacher_logits(teacher, query_loader, device)
    distill_loader = DataLoader(
        distill_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0
    )
    print(f"[queries] cached exact teacher logits for {query_budget} unlabeled images")

    spectral_targets = recover_hidden_spectral_targets(teacher, args.precision)
    for name in TARGET_LAYERS:
        target = spectral_targets[name]
        print(
            f"[spectra] {name}: shape={target['shape']}, "
            f"weight_std={target['weight_std']:.4f}, "
            f"mean_col_norm={target['col_norms'].mean():.4f}"
        )

    all_results = {"teacher_acc": teacher_acc, "teacher_history": teacher_history}
    final_scores = {}
    epoch_scores = {}

    for mode in ("baseline", "ours"):
        histories = []
        finals = []
        for seed in range(args.distill_seeds):
            _, history = train_student(
                mode=mode,
                spectral_targets=spectral_targets,
                distill_loader=distill_loader,
                test_loader=test_loader,
                device=device,
                epochs=args.student_epochs,
                lr=args.student_lr,
                temperature=args.temperature,
                reg_lambda=args.reg_lambda,
                seed=seed,
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
        }

    print("\n=== Summary ===")
    print(f"Teacher test accuracy: {teacher_acc:.2f}%")
    for mode in ("baseline", "ours"):
        mean_final, std_final = mean_and_std(final_scores[mode])
        print(
            f"{mode:8s} final: {mean_final:.2f}% +- {std_final:.2f} "
            f"(seeds={final_scores[mode]})"
        )
        for epoch_idx, values in enumerate(epoch_scores[mode], start=1):
            mean_epoch, std_epoch = mean_and_std(values)
            print(
                f"  epoch {epoch_idx}: {mean_epoch:.2f}% +- {std_epoch:.2f}"
            )

    improvement = (
        np.mean(final_scores["ours"]) - np.mean(final_scores["baseline"])
    )
    print(f"Improvement (ours - baseline): {improvement:.2f} points")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = os.path.join(OUTPUT_DIR, "mnist_kd_spectral_priors.json")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"Saved JSON summary to {out_path}")


if __name__ == "__main__":
    main()

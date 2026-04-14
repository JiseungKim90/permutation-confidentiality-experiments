"""
Train a ResNet-56 teacher on CIFAR-10 (single model, fast 10-epoch schedule).

Used as the teacher for the distillation ablation experiment in Section 5.2
when extending to ResNet-56.

Output: models/resnet56_seed{seed}.pt
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

from lib.models import ResNet56


SAVE_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=2)
    args = parser.parse_args()

    os.makedirs(SAVE_DIR, exist_ok=True)
    out_path = os.path.join(SAVE_DIR, f"resnet56_seed{args.seed}.pt")
    print(f"=== Training ResNet-56 teacher seed={args.seed} ===", flush=True)
    print(json.dumps(vars(args), indent=2), flush=True)

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    trainset = torchvision.datasets.CIFAR10(
        root=DATA_DIR, train=True, download=True, transform=transform_train
    )
    testset = torchvision.datasets.CIFAR10(
        root=DATA_DIR, train=False, download=True, transform=transform_test
    )
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers
    )
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=256, shuffle=False, num_workers=args.num_workers
    )

    torch.manual_seed(args.seed)
    model = ResNet56()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"params: {n_params/1e6:.3f}M", flush=True)

    optimizer = optim.SGD(model.parameters(), lr=args.lr,
                          momentum=args.momentum, weight_decay=args.weight_decay)
    milestones = [max(1, args.epochs // 2), max(2, int(args.epochs * 0.8))]
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.1)
    criterion = nn.CrossEntropyLoss()

    history = []
    t_start = time.time()
    for epoch in range(args.epochs):
        model.train()
        total_loss, n_batch = 0.0, 0
        for inputs, targets in trainloader:
            optimizer.zero_grad()
            loss = criterion(model(inputs), targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_batch += 1
        scheduler.step()
        history.append({"epoch": epoch + 1, "loss": total_loss / max(n_batch, 1),
                        "lr": scheduler.get_last_lr()[0]})
        print(f"  epoch {epoch+1:02d}: loss={total_loss/max(n_batch,1):.4f}", flush=True)

    model.eval()
    correct = total = 0
    with torch.no_grad():
        for inputs, targets in testloader:
            _, pred = model(inputs).max(1)
            total += targets.size(0)
            correct += pred.eq(targets).sum().item()
    acc = 100.0 * correct / total
    elapsed = time.time() - t_start
    print(f"  test_acc: {acc:.2f}%", flush=True)
    print(f"  time: {elapsed:.0f}s", flush=True)

    torch.save({
        "model_state_dict": model.state_dict(),
        "architecture": "ResNet56",
        "seed": args.seed,
        "epochs": args.epochs,
        "test_accuracy": acc,
        "history": history,
        "n_params": n_params,
        "milestones": milestones,
    }, out_path)
    print(f"saved: {out_path}", flush=True)


if __name__ == "__main__":
    main()

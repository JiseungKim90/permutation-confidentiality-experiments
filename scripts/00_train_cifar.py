"""
Train 20 ResNet-20 models on CIFAR-10.
Produces: models/resnet20_seed{0..19}.pt, models/train_summary.json
Paper: Appendix A (tab:cifar_spec)
"""
import sys, os, time, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from lib.models import ResNet20, CIFAR_TRAIN_CONFIG

CFG = CIFAR_TRAIN_CONFIG
SAVE_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
os.makedirs(SAVE_DIR, exist_ok=True)

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

print("Loading CIFAR-10...")
trainset = torchvision.datasets.CIFAR10(root=DATA_DIR, train=True, download=True, transform=transform_train)
testset = torchvision.datasets.CIFAR10(root=DATA_DIR, train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=256, shuffle=False, num_workers=2)

print(f"Config: {json.dumps(CFG, indent=2)}")


def train_one(seed):
    torch.manual_seed(seed)
    model = ResNet20()
    optimizer = optim.SGD(model.parameters(), lr=CFG["lr"],
                          momentum=CFG["momentum"], weight_decay=CFG["weight_decay"])
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=CFG["milestones"],
                                                gamma=CFG["lr_gamma"])
    criterion = nn.CrossEntropyLoss()
    loader = torch.utils.data.DataLoader(trainset, batch_size=CFG["batch_size"],
                                          shuffle=True, num_workers=2)
    history = []
    for epoch in range(CFG["n_epochs"]):
        model.train()
        total_loss, n_batch = 0, 0
        for inputs, targets in loader:
            optimizer.zero_grad()
            loss = criterion(model(inputs), targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_batch += 1
        scheduler.step()
        history.append({"epoch": epoch, "loss": total_loss / n_batch,
                        "lr": scheduler.get_last_lr()[0]})

    model.eval()
    correct = total = 0
    with torch.no_grad():
        for inputs, targets in testloader:
            _, pred = model(inputs).max(1)
            total += targets.size(0)
            correct += pred.eq(targets).sum().item()
    acc = 100.0 * correct / total

    torch.save({"model_state_dict": model.state_dict(), "config": CFG,
                "seed": seed, "test_accuracy": acc, "history": history,
                "n_params": sum(p.numel() for p in model.parameters())},
               os.path.join(SAVE_DIR, f"resnet20_seed{seed}.pt"))
    return acc, history


results = {}
t0 = time.time()
for seed in range(CFG["n_models"]):
    t1 = time.time()
    acc, hist = train_one(seed)
    results[seed] = {"accuracy": acc, "time": time.time() - t1}
    print(f"  Seed {seed:2d}: acc={acc:.1f}%, time={time.time()-t1:.0f}s")
    sys.stdout.flush()

total = time.time() - t0
mean_acc = sum(r["accuracy"] for r in results.values()) / len(results)
summary = {"config": CFG, "results": {str(k): v for k, v in results.items()},
           "mean_accuracy": mean_acc, "total_time": total}
with open(os.path.join(SAVE_DIR, "train_summary.json"), "w") as f:
    json.dump(summary, f, indent=2)
print(f"\nDone: {len(results)} models, mean acc={mean_acc:.1f}%, total={total:.0f}s")

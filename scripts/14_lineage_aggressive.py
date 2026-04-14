"""
Lineage detection under different fine-tuning regimes.
Produces: Section 5.3 results.
Reports accuracy against the 5 selected bases and, for the aggressive
regime, also against all 20 base models (as claimed in the paper).
"""
import sys, os, glob, copy
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as T
from lib.models import ResNet20
from lib.attack import get_conv_layers, fingerprint_from_layer

MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")

def load_model(path):
    model = ResNet20()
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    if "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
    else:
        model.load_state_dict(ckpt)
    model.eval()
    return model

model_files = sorted(glob.glob(os.path.join(MODEL_DIR, "resnet20_seed*.pt")))
model_files = [f for f in model_files if not f.endswith("seed20.pt")]

transform_train = T.Compose([T.RandomCrop(32, padding=4), T.RandomHorizontalFlip(),
                              T.ToTensor(), T.Normalize((0.4914,0.4822,0.4465),(0.2470,0.2435,0.2616))])
trainset = torchvision.datasets.CIFAR10(root=os.path.join(os.path.dirname(__file__), "..", "data"),
                                         train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True)

p = 256
fp_layers = ["conv1", "layer2.0.conv1", "layer3.0.conv1"]
n_bases = 5
SUBSET_SEED = 20260409
configs = [
    {"name": "mild",       "epochs": 2,  "lr": 0.001, "subset": 5000},
    {"name": "moderate",   "epochs": 5,  "lr": 0.005, "subset": None},
    {"name": "aggressive", "epochs": 10, "lr": 0.01,  "subset": None},
]

# Fingerprints for the 5 selected parent bases
base_fps = []
for bi in range(n_bases):
    model = load_model(model_files[bi])
    layers = get_conv_layers(model)
    fp_parts = []
    for lname, wm, bv, _, _, _ in layers:
        if lname in fp_layers:
            fp_parts.extend(fingerprint_from_layer(wm, bv, p).tolist())
    base_fps.append(np.array(fp_parts))

# Fingerprints for all 20 base models (for the all-20-bases comparison)
all_fps = []
for mf in model_files:
    model = load_model(mf)
    layers = get_conv_layers(model)
    fp_parts = []
    for lname, wm, bv, _, _, _ in layers:
        if lname in fp_layers:
            fp_parts.extend(fingerprint_from_layer(wm, bv, p).tolist())
    all_fps.append(np.array(fp_parts))

print("=== Lineage Robustness Under Different Fine-Tuning Regimes ===")
print(f"Bases: {n_bases} (selected), {len(all_fps)} (all), Layers: {fp_layers}\n")

for cfg in configs:
    n_children = 3
    child_fps = []
    child_parents = []

    if cfg["subset"]:
        rng = np.random.default_rng(SUBSET_SEED)
        subset_idx = sorted(rng.choice(len(trainset), size=cfg["subset"], replace=False).tolist())
        subset = torch.utils.data.Subset(trainset, subset_idx)
        loader = torch.utils.data.DataLoader(subset, batch_size=64, shuffle=True)
    else:
        subset_idx = None
        loader = trainloader

    for bi in range(n_bases):
        base_model = load_model(model_files[bi])
        for ci in range(n_children):
            child = copy.deepcopy(base_model)
            child.train()
            torch.manual_seed(bi * 100 + ci * 10 + 9999)
            optimizer = torch.optim.SGD(child.parameters(), lr=cfg["lr"], momentum=0.9)
            criterion = nn.CrossEntropyLoss()
            for epoch in range(cfg["epochs"]):
                for images, labels in loader:
                    optimizer.zero_grad()
                    criterion(child(images), labels).backward()
                    optimizer.step()
            child.eval()

            layers = get_conv_layers(child)
            fp_parts = []
            for lname, wm, bv, _, _, _ in layers:
                if lname in fp_layers:
                    fp_parts.extend(fingerprint_from_layer(wm, bv, p).tolist())
            child_fps.append(np.array(fp_parts))
            child_parents.append(bi)
            print(f"  {cfg['name']} child {bi*n_children+ci+1}/{n_bases*n_children} done", flush=True)

    # --- Results against 5 selected bases ---
    correct_5 = 0
    within_dists = []
    between_dists_5 = []
    for cfp, parent in zip(child_fps, child_parents):
        dists = [np.linalg.norm(cfp - bfp) for bfp in base_fps]
        predicted = int(np.argmin(dists))
        correct_5 += int(predicted == parent)
        within_dists.append(dists[parent])
        for bi in range(n_bases):
            if bi != parent:
                between_dists_5.append(dists[bi])
    within_dists = np.array(within_dists)
    between_dists_5 = np.array(between_dists_5)
    sep_5 = between_dists_5.min() / within_dists.max() if within_dists.max() > 0 else float("inf")

    subset_desc = cfg["subset"] or "full"
    if subset_idx is not None:
        subset_desc = f"{cfg['subset']} random examples (seed={SUBSET_SEED})"
    print(f"--- {cfg['name']} (epochs={cfg['epochs']}, lr={cfg['lr']}, subset={subset_desc}) ---")
    print(f"  Accuracy (vs 5 bases): {correct_5}/{len(child_fps)}")
    print(f"  Within mean: {within_dists.mean():.4f}, max: {within_dists.max():.4f}")
    print(f"  Between mean (5): {between_dists_5.mean():.4f}, min: {between_dists_5.min():.4f}")
    print(f"  Separation ratio (5 bases): {sep_5:.1f}x")

    # --- Results against all 20 bases (aggressive only) ---
    if cfg["name"] == "aggressive":
        correct_20 = 0
        between_dists_20 = []
        false_positive_pairs = 0
        for cfp, parent in zip(child_fps, child_parents):
            dists_all = [np.linalg.norm(cfp - bfp) for bfp in all_fps]
            predicted = int(np.argmin(dists_all))
            correct_20 += int(predicted == parent)
            parent_dist = dists_all[parent]
            for bi in range(len(all_fps)):
                if bi != parent:
                    between_dists_20.append(dists_all[bi])
                    false_positive_pairs += int(dists_all[bi] < parent_dist)
        between_dists_20 = np.array(between_dists_20)
        sep_20 = between_dists_20.min() / within_dists.max() if within_dists.max() > 0 else float("inf")
        n_comparisons = len(child_fps) * (len(all_fps) - 1)
        print(f"  Accuracy (vs all 20): {correct_20}/{len(child_fps)}")
        print(f"  Between min (all 20): {between_dists_20.min():.4f}")
        print(f"  Separation ratio (all 20): {sep_20:.2f}x")
        print(f"  FPR: {false_positive_pairs}/{n_comparisons}")
    print()

print("Done.")

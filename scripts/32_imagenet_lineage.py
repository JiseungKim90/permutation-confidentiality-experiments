"""
Exp 32: ImageNet-scale lineage detection on ResNet-50.

Reproduces the paper's V1 / V2 / fine-tuned-V1 lineage check:
  1. load torchvision ResNet-50 IMAGENET1K_V1 and IMAGENET1K_V2;
  2. fine-tune V1 for two epochs on a fixed 500-image Imagenette-val subset;
  3. compare sorted-spectrum fingerprints of V1, V2, and V1-ft.
"""
import copy
import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.models as models
import torchvision.transforms as T
from torch.utils.data import DataLoader, Subset

from lib.attack import get_conv_layers, fingerprint_from_layer


P = 256
FP_LAYERS = ["conv1", "layer2.0.conv1", "layer3.0.conv1", "layer4.0.conv1"]
BASE_SEED = 20260414
SUBSET_SEED = 0
PER_CLASS = 50
FINE_TUNE_EPOCHS = 2
FINE_TUNE_LR = 1e-3
BATCH_SIZE = 32

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
IMAGENETTE_DIR = os.path.join(DATA_DIR, "imagenette2-320", "val")

IMAGENETTE_WNID_TO_IMAGENET_IDX = {
    "n01440764": 0,
    "n02102040": 217,
    "n02979186": 482,
    "n03000684": 491,
    "n03028079": 497,
    "n03394916": 566,
    "n03417042": 569,
    "n03425413": 571,
    "n03445777": 574,
    "n03888257": 701,
}


class ImageNetLabelSubset(torch.utils.data.Dataset):
    def __init__(self, dataset, folder_to_imagenet):
        self.dataset = dataset
        self.folder_to_imagenet = folder_to_imagenet

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, folder_label = self.dataset[idx]
        return image, self.folder_to_imagenet[int(folder_label)]


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_imagenette_subset():
    transform = T.Compose([
        T.Resize(256),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    dataset = torchvision.datasets.ImageFolder(IMAGENETTE_DIR, transform=transform)
    missing = sorted(set(dataset.classes) - set(IMAGENETTE_WNID_TO_IMAGENET_IDX))
    if missing:
        raise RuntimeError(f"Unknown ImageNette WNID folders: {missing}")

    folder_to_imagenet = {
        folder_idx: IMAGENETTE_WNID_TO_IMAGENET_IDX[wnid]
        for folder_idx, wnid in enumerate(dataset.classes)
    }
    mapped = ImageNetLabelSubset(dataset, folder_to_imagenet)

    rng = random.Random(SUBSET_SEED)
    selected = []
    for folder_idx in range(len(dataset.classes)):
        candidates = [
            idx for idx, (_path, label) in enumerate(dataset.samples)
            if label == folder_idx
        ]
        if len(candidates) < PER_CLASS:
            raise RuntimeError(
                f"Class {dataset.classes[folder_idx]} has only {len(candidates)} images"
            )
        selected.extend(rng.sample(candidates, PER_CLASS))
    selected.sort()
    return Subset(mapped, selected), dataset.classes


def fine_tune_v1(base_model, loader, device):
    child = copy.deepcopy(base_model).to(device)
    child.train()
    optimizer = torch.optim.SGD(child.parameters(), lr=FINE_TUNE_LR, momentum=0.9)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(FINE_TUNE_EPOCHS):
        total_loss = 0.0
        n_batch = 0
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            loss = criterion(child(images), labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_batch += 1
        print(
            f"  fine-tune epoch {epoch + 1}/{FINE_TUNE_EPOCHS}: "
            f"loss={total_loss / max(n_batch, 1):.4f}"
        )

    child.eval()
    return child.cpu()


def extract_fingerprint(model, precision):
    model.eval()
    layers = get_conv_layers(model)
    fp_parts = []
    for lname, wm, bv, _has_bias, _c_out, _d in layers:
        if lname in FP_LAYERS:
            fp_parts.extend(fingerprint_from_layer(wm, bv, precision).tolist())
    return np.array(fp_parts)


def print_distances(v1_fp, v2_fp, ft_fp, precision_name):
    d_v1_v2 = float(np.linalg.norm(v1_fp - v2_fp))
    d_v1_ft = float(np.linalg.norm(v1_fp - ft_fp))
    d_v2_ft = float(np.linalg.norm(v2_fp - ft_fp))
    non_lineage = min(d_v1_v2, d_v2_ft)
    ratio = non_lineage / d_v1_ft if d_v1_ft > 0 else float("inf")
    print(f"\n[{precision_name}]")
    print(f"  ||phi(V1) - phi(V2)||_2    = {d_v1_v2:.6f}")
    print(f"  ||phi(V1) - phi(V1_ft)||_2 = {d_v1_ft:.6f}")
    print(f"  ||phi(V2) - phi(V1_ft)||_2 = {d_v2_ft:.6f}")
    print(f"  non-lineage / lineage ratio = {ratio:.1f}x")
    print(f"  nearest parent for V1_ft     = {'V1' if d_v1_ft < d_v2_ft else 'V2'}")


def main():
    set_seed(BASE_SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 70)
    print("Exp 32: ImageNet-scale ResNet-50 lineage detection")
    print(f"Device: {device}")
    print(f"Fingerprint layers: {FP_LAYERS}")
    print(f"Subset: {PER_CLASS} images/class, seed={SUBSET_SEED}")
    print("=" * 70)

    subset, classes = build_imagenette_subset()
    generator = torch.Generator().manual_seed(BASE_SEED)
    loader = DataLoader(
        subset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        generator=generator,
    )
    print(f"Loaded Imagenette subset: {len(subset)} images, classes={classes}")

    print("\nLoading ResNet-50 V1/V2...")
    v1 = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1).eval()
    v2 = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2).eval()

    print("Fine-tuning V1...")
    v1_ft = fine_tune_v1(v1, loader, device)
    v1 = v1.cpu()
    v2 = v2.cpu()

    print("\nExtracting fingerprints...")
    for p_val, p_name in [(16, "4-bit"), (256, "8-bit"), (4096, "12-bit")]:
        v1_fp = extract_fingerprint(v1, p_val)
        v2_fp = extract_fingerprint(v2, p_val)
        ft_fp = extract_fingerprint(v1_ft, p_val)
        print_distances(v1_fp, v2_fp, ft_fp, p_name)

    print("\n" + "=" * 70)
    print("ALL IMAGENET LINEAGE EXPERIMENTS COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()

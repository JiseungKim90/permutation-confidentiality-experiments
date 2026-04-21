"""
Per-layer variant of 33_imagenet_tradeoff.py: clipped-Gaussian noise at
every Conv2d output via forward hooks. CPU-heavy; keep N_TRIALS small.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                ".."))

import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as T
import torchvision.models as models
from torch.utils.data import DataLoader
from lib.attack import get_conv_layers, round_then_sort

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
IMAGENETTE_DIR = os.path.join(DATA_DIR, "imagenette2-320", "val")
N_TRIALS = 3  # ResNet-50 on Imagenette is expensive on CPU
BATCH = 64

# True ImageNet-1K indices for the 10 Imagenette classes, in the order
# PyTorch ImageFolder sorts them (by WNID, alphabetic).
IMAGENETTE_WNIDS = [
    'n01440764', 'n02102040', 'n02979186', 'n03000684', 'n03028079',
    'n03394916', 'n03417042', 'n03425413', 'n03445777', 'n03888257',
]
IMAGENETTE_TO_IMAGENET = [0, 217, 482, 491, 497, 566, 569, 571, 574, 701]

print("=" * 70)
print("Exp 33b: Per-Layer Privacy-Utility Tradeoff (ImageNet ResNet-50)")
print("Dataset: Imagenette validation")
print("Device: %s" % DEVICE)
print("=" * 70)

model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
model.eval().to(DEVICE)

transform = T.Compose([
    T.Resize(256),
    T.CenterCrop(224),
    T.ToTensor(),
    T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])
valset = torchvision.datasets.ImageFolder(IMAGENETTE_DIR, transform=transform)
valloader = DataLoader(valset, batch_size=BATCH, shuffle=False, num_workers=0)
print("Validation set: %d images, %d classes" % (len(valset), len(valset.classes)))

conv_modules = [m for m in model.modules() if isinstance(m, nn.Conv2d)]
print("Found %d Conv2d layers for per-layer injection" % len(conv_modules))

p = 256
gamma = 1.0 / p
Delta = 1.0 / (2 * p)


def make_hook(state):
    def hook(module, inputs, output):
        B = state["B"]
        out = torch.round(output * p) / p
        noise = torch.randn_like(out) * (B / 3.0)
        noise = noise.clamp(-B, B)
        return out + noise
    return hook


def run_inference():
    preds = []
    labels = []
    with torch.no_grad():
        for images, lbls in valloader:
            images = images.to(DEVICE)
            out = model(images).cpu()
            preds.append(out.argmax(1))
            labels.append(lbls)
    return torch.cat(preds), torch.cat(labels)


# Clean predictions (no hooks active)
print("Computing clean predictions...")
clean_preds, all_labels_folder = run_inference()

# Map Imagenette folder labels to real ImageNet-1K class indices via the
# fixed WNID->ImageNet mapping. ImageFolder sorts folders alphabetically,
# matching IMAGENETTE_WNIDS.
assert valset.classes == IMAGENETTE_WNIDS, (
    "Imagenette folder order mismatch; got " + str(valset.classes))
folder_to_imagenet = {
    i: IMAGENETTE_TO_IMAGENET[i] for i in range(len(valset.classes))
}
all_labels_imagenet = torch.tensor(
    [folder_to_imagenet[l.item()] for l in all_labels_folder])
clean_acc = (clean_preds == all_labels_imagenet).float().mean().item()
print("Clean top-1 accuracy (real Imagenette->ImageNet labels): %.1f%%"
      % (100 * clean_acc))

# Attack error reference on ResNet-50 conv1
layers = get_conv_layers(model)
conv1_W, conv1_b = None, None
for lname, wm, bv, _, _, _ in layers:
    if lname == "conv1":
        conv1_W, conv1_b = wm, bv
        break


def measure_attack_error(W_mat, b_vec, noise_mult, n_cols=None):
    C_out, d = W_mat.shape
    W_q = np.round(W_mat * p) / p
    b_q = np.round(b_vec * p) / p
    if n_cols is None:
        n_cols = d
    n_cols = min(n_cols, d)
    errors = []
    for i in range(n_cols):
        true_col = W_q[:, i] + b_q
        true_sorted = np.sort(true_col)
        B = Delta * noise_mult - 1e-12
        noise = np.random.normal(
            0, Delta * noise_mult / 3, C_out).clip(-B, B)
        perm = np.random.permutation(C_out)
        observed = true_col[perm] + noise
        recovered = round_then_sort(observed, gamma)
        errors.append(np.max(np.abs(recovered - true_sorted)))
    return np.mean(errors)


noise_mults = [1, 10, 100, 1000, 5000]

print("\n" + "=" * 70)
print("Per-Layer Noise Tradeoff (ResNet-50, Imagenette, 8-bit)")
print("N_TRIALS = %d, BATCH = %d" % (N_TRIALS, BATCH))
print("=" * 70)
print("%8s  %12s  %16s  %16s" %
      ("Mult", "Atk err", "Pred. agr.", "Top-1 acc."))

state = {"B": 0.0}
handles = [m.register_forward_hook(make_hook(state)) for m in conv_modules]

for nm in noise_mults:
    np.random.seed(42)
    atk_err = measure_attack_error(conv1_W, conv1_b, nm, n_cols=64)

    state["B"] = Delta * nm - 1e-12
    agr_list, acc_list = [], []
    for trial in range(N_TRIALS):
        torch.manual_seed(trial)
        noisy_preds, _ = run_inference()
        agr_list.append((noisy_preds == clean_preds).float().mean().item())
        acc_list.append(
            (noisy_preds == all_labels_imagenet).float().mean().item())

    agr = np.mean(agr_list)
    acc = np.mean(acc_list)
    agr_std = np.std(agr_list)
    acc_std = np.std(acc_list)
    print("%7dx  %12.2e  %10.1f±%4.2f%%  %10.1f±%4.2f%%" %
          (nm, atk_err, 100 * agr, 100 * agr_std,
           100 * acc, 100 * acc_std))

for h in handles:
    h.remove()

print("\n" + "=" * 70)
print("PER-LAYER TRADEOFF EXPERIMENT COMPLETE")
print("=" * 70)

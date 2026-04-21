"""
Exp 33: Imagenette ResNet-50 proxy tradeoff (final-output noise).
Seeds: np=42 (attack error), torch=trial (noise trials).
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import numpy as np
import torch
import torchvision
import torchvision.transforms as T
import torchvision.models as models
from torch.utils.data import DataLoader
from lib.attack import get_conv_layers, round_then_sort

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data')
IMAGENETTE_DIR = os.path.join(DATA_DIR, 'imagenette2-320', 'val')

# ImageNette class folders -> ImageNet-1K class indices.
# PyTorch ImageFolder sorts class folders alphabetically, so valset.classes[i]
# corresponds to IMAGENETTE_WNIDS[i] / IMAGENETTE_TO_IMAGENET[i].
IMAGENETTE_WNIDS = [
    'n01440764',  # tench
    'n02102040',  # English springer
    'n02979186',  # cassette player
    'n03000684',  # chain saw
    'n03028079',  # church
    'n03394916',  # French horn
    'n03417042',  # garbage truck
    'n03425413',  # gas pump
    'n03445777',  # golf ball
    'n03888257',  # parachute
]
IMAGENETTE_TO_IMAGENET = [0, 217, 482, 491, 497, 566, 569, 571, 574, 701]

N_TRIALS = 100

print("=" * 70)
print("Exp 33: Privacy-Utility Tradeoff (ImageNet-scale ResNet-50)")
print("Dataset: ImageNette (10-class ImageNet subset)")
print("Device: %s" % DEVICE)
print("=" * 70)

# Load model
model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
model.eval().to(DEVICE)

# Load ImageNette validation set
transform = T.Compose([
    T.Resize(256),
    T.CenterCrop(224),
    T.ToTensor(),
    T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])

valset = torchvision.datasets.ImageFolder(IMAGENETTE_DIR, transform=transform)
valloader = DataLoader(valset, batch_size=64, shuffle=False, num_workers=0)

print("Validation set: %d images, %d classes" % (len(valset), len(valset.classes)))

# Compute clean outputs
print("Computing clean outputs...")
with torch.no_grad():
    all_outs = []
    all_labels = []
    for images, labels in valloader:
        images = images.to(DEVICE)
        out = model(images).cpu()
        all_outs.append(out)
        all_labels.append(labels)

all_outs = torch.cat(all_outs)
all_labels_folder = torch.cat(all_labels)

# Map folder indices to the true ImageNet-1K class indices using the fixed
# WNID->ImageNet mapping. PyTorch ImageFolder sorts class folders by WNID,
# matching the order of IMAGENETTE_WNIDS / IMAGENETTE_TO_IMAGENET above.
assert valset.classes == IMAGENETTE_WNIDS, (
    "Imagenette folder order does not match the expected WNID list; "
    f"got {valset.classes}")
folder_to_imagenet = {
    i: IMAGENETTE_TO_IMAGENET[i] for i in range(len(valset.classes))
}
all_labels_imagenet = torch.tensor(
    [folder_to_imagenet[l.item()] for l in all_labels_folder])

clean_preds_1000 = all_outs.argmax(1)
clean_acc = (clean_preds_1000 == all_labels_imagenet).float().mean().item()
print("Clean top-1 accuracy (real Imagenette->ImageNet labels): %.1f%%"
      % (100 * clean_acc))

# Attack error measurement on representative layers
layers = get_conv_layers(model)
test_layer_names = ["conv1", "layer2.0.conv1", "layer3.0.conv1", "layer4.0.conv1"]
test_layers = {}
for lname, wm, bv, _, C_out, d in layers:
    if lname in test_layer_names:
        test_layers[lname] = (wm, bv, C_out, d)

p = 256
gamma = 1.0 / p
Delta = 1.0 / (2 * p)

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
        noise = np.random.normal(0, Delta * noise_mult / 3, C_out).clip(-B, B)
        perm = np.random.permutation(C_out)
        observed = true_col[perm] + noise
        recovered = round_then_sort(observed, gamma)
        errors.append(np.max(np.abs(recovered - true_sorted)))
    return np.mean(errors)

noise_mults = [1, 10, 100, 1000, 5000]

print("\n" + "=" * 70)
print("Attack Error by Layer and Noise Multiplier (8-bit)")
print("=" * 70)
header = "%-30s" % "Layer"
for nm in noise_mults:
    header += "  %8s" % ("%dx" % nm)
print(header)
print("-" * 80)

for lname in test_layer_names:
    if lname not in test_layers:
        continue
    W_mat, b_vec, C_out, d = test_layers[lname]
    row = "%s (%dx%d)" % (lname, C_out, d)
    vals = []
    for nm in noise_mults:
        np.random.seed(42)
        err = measure_attack_error(W_mat, b_vec, nm, n_cols=min(d, 50))
        vals.append("0" if err == 0 else "%.2e" % err)
    print("%-30s  %s" % (row, "  ".join("%8s" % v for v in vals)))

print("\n" + "=" * 70)
print("Prediction Agreement and Top-1 Accuracy Under Output Noise")
print("N_TRIALS = %d, Dataset = ImageNette validation (%d images)" % (N_TRIALS, len(valset)))
print("=" * 70)
print("%8s  %15s  %12s  %12s" % ("Mult", "Atk err (conv1)", "Pred. agr.", "Top-1 acc."))
print("-" * 55)

conv1_W, conv1_b, _, _ = test_layers["conv1"]

for nm in noise_mults:
    np.random.seed(42)
    atk_err = measure_attack_error(conv1_W, conv1_b, nm)

    B = Delta * nm
    agr_list, acc_list = [], []
    for trial in range(N_TRIALS):
        torch.manual_seed(trial)
        noise = torch.randn_like(all_outs) * (B / 3)
        noise = noise.clamp(-B, B)
        noisy_preds = (all_outs + noise).argmax(1)
        agr_list.append((noisy_preds == clean_preds_1000).float().mean().item())
        acc_list.append((noisy_preds == all_labels_imagenet).float().mean().item())

    agr = np.mean(agr_list)
    acc = np.mean(acc_list)
    err_str = "0" if atk_err == 0 else "%.2e" % atk_err
    print("%7dx  %15s  %11.1f%%  %11.1f%%" % (nm, err_str, 100*agr, 100*acc))

print("\n" + "=" * 70)
print("TRADEOFF EXPERIMENT COMPLETE")
print("=" * 70)

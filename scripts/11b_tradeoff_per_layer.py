"""
Per-layer variant of 11_tradeoff_cifar.py: clipped-Gaussian noise at every
Conv2d output via forward hooks, matching SAFHIRE Remark 4.
"""
import sys, os, glob
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as T
from lib.models import ResNet20
from lib.attack import get_conv_layers, round_then_sort

MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
N_TRIALS = 20  # rerun with larger N; conv hooks still affordable on CPU.


def load_model(path):
    model = ResNet20()
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    if "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
    else:
        model.load_state_dict(ckpt)
    model.eval()
    return model


# Load first trained model from the canonical 20-model set
model_files = sorted(
    f for f in glob.glob(os.path.join(MODEL_DIR, "resnet20_seed*.pt"))
    if not f.endswith("seed20.pt")
)
model = load_model(model_files[0])
print("Loaded %s" % os.path.basename(model_files[0]))

# CIFAR-10 test set
transform = T.Compose([T.ToTensor(),
                       T.Normalize((0.4914, 0.4822, 0.4465),
                                   (0.2470, 0.2435, 0.2616))])
testset = torchvision.datasets.CIFAR10(
    root=os.path.join(os.path.dirname(__file__), "..", "data"),
    train=False, download=True, transform=transform)
# Preload all test images into a single tensor, then process in small
# batches inside each forward loop to keep per-batch memory small
# (21 hooks allocating randn_like blow up peak RAM at batch=10k).
loader = torch.utils.data.DataLoader(testset, batch_size=10000, shuffle=False)
images_all, labels_all = next(iter(loader))
BATCH = 500
print("Preloaded test tensor shape", tuple(images_all.shape), "batch", BATCH)


def infer_all(noise_state_setter=None):
    preds = []
    with torch.no_grad():
        for start in range(0, images_all.shape[0], BATCH):
            chunk = images_all[start:start + BATCH]
            out = model(chunk)
            preds.append(out.argmax(1))
    return torch.cat(preds)


# Clean predictions (no hooks active yet)
clean_preds = infer_all()
all_labels = labels_all
clean_acc = (clean_preds == all_labels).float().mean().item()
print("Clean test accuracy: %.1f%%" % (100 * clean_acc))

p = 256
gamma = 1.0 / p
Delta = 1.0 / (2 * p)

conv_modules = [m for m in model.modules() if isinstance(m, nn.Conv2d)]
print("Found %d Conv2d layers for per-layer injection" % len(conv_modules))


def make_hook(noise_state):
    """Forward hook that quantizes + adds clipped-Gaussian noise in-place."""

    def hook(module, inputs, output):
        B = noise_state["B"]
        out = torch.round(output * p) / p
        noise = torch.randn_like(out) * (B / 3.0)
        noise = noise.clamp(-B, B)
        return out + noise

    return hook


# Attack error reference on conv1 (same as 11_tradeoff_cifar.py)
layers = get_conv_layers(model)
conv1_W, conv1_b = None, None
for lname, wm, bv, _, _, _ in layers:
    if lname == "conv1":
        conv1_W, conv1_b = wm, bv
        break


def measure_attack_error(W_mat, b_vec, noise_mult):
    C_out, d = W_mat.shape
    W_q = np.round(W_mat * p) / p
    b_q = np.round(b_vec * p) / p
    errors = []
    for i in range(d):
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

print("\n=== CIFAR-10 Per-Layer Noise Tradeoff"
      " (8-bit, N_TRIALS=%d) ===" % N_TRIALS)
print("%8s  %12s  %12s  %12s" %
      ("Mult", "Atk err", "Pred. agr.", "Test acc."))

noise_state = {"B": 0.0}
handles = [m.register_forward_hook(make_hook(noise_state))
           for m in conv_modules]

for nm in noise_mults:
    np.random.seed(42)
    atk_err = measure_attack_error(conv1_W, conv1_b, nm)

    B = Delta * nm - 1e-12
    noise_state["B"] = B

    agr_list, acc_list = [], []
    for trial in range(N_TRIALS):
        torch.manual_seed(trial)
        noisy_preds = infer_all()
        agr_list.append((noisy_preds == clean_preds).float().mean().item())
        acc_list.append((noisy_preds == all_labels).float().mean().item())

    agr = np.mean(agr_list)
    acc = np.mean(acc_list)
    agr_std = np.std(agr_list)
    acc_std = np.std(acc_list)
    print("%7dx  %12.2e  %8.1f±%4.2f%%  %8.1f±%4.2f%%" %
          (nm, atk_err, 100 * agr, 100 * agr_std,
           100 * acc, 100 * acc_std))

for h in handles:
    h.remove()

"""
Privacy-utility tradeoff on trained CIFAR-10 / ResNet-20.
Produces: numbers for Table (tab:tradeoff).
Pred. agr. and Test acc. reported as mean over N_TRIALS=100 independent
noise draws (clipped Gaussian, std=B/3, clipped to [-B,B]).
Attack error uses np.random.seed(42).
"""
import sys, os, glob
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch
import torchvision
import torchvision.transforms as T
from lib.models import ResNet20
from lib.attack import get_conv_layers, round_then_sort

MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
N_TRIALS = 100

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
transform = T.Compose([T.ToTensor(), T.Normalize((0.4914,0.4822,0.4465),(0.2470,0.2435,0.2616))])
testset = torchvision.datasets.CIFAR10(
    root=os.path.join(os.path.dirname(__file__), "..", "data"),
    train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=1000, shuffle=False)

# Precompute all outputs
with torch.no_grad():
    all_preds, all_labels, all_outs = [], [], []
    for images, labels in testloader:
        out = model(images)
        all_preds.append(out.argmax(1))
        all_labels.append(labels)
        all_outs.append(out)
clean_preds = torch.cat(all_preds)
all_labels = torch.cat(all_labels)
all_outs = torch.cat(all_outs)
clean_acc = (clean_preds == all_labels).float().mean().item()
print("Clean test accuracy: %.1f%%" % (100 * clean_acc))

# Attack error: use conv1 layer
layers = get_conv_layers(model)
conv1_W, conv1_b = None, None
for lname, wm, bv, _, _, _ in layers:
    if lname == "conv1":
        conv1_W, conv1_b = wm, bv
        break

p = 256
gamma = 1.0 / p
Delta = 1.0 / (2 * p)

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

print("\n=== CIFAR-10 Privacy-Utility Tradeoff (8-bit, N_TRIALS=%d) ===" % N_TRIALS)
print("%8s  %12s  %10s  %10s" % ("Mult", "Atk err", "Pred. agr.", "Test acc."))

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
        agr_list.append((noisy_preds == clean_preds).float().mean().item())
        acc_list.append((noisy_preds == all_labels).float().mean().item())

    agr = np.mean(agr_list)
    acc = np.mean(acc_list)
    print("%7dx  %12.2e  %9.1f%%  %9.1f%%" % (nm, atk_err, 100*agr, 100*acc))

print("\nDone.")

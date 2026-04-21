"""
Exp 16: Fingerprinting accuracy under noise.
Key metric: pairwise separation of 20 models at varying noise multipliers.
Noise is added to the simulated responses before fingerprint extraction.
"""
import sys, os, glob
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import numpy as np
import torch
from lib.models import ResNet20
from lib.attack import get_conv_layers

MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'models')
BASE_SEED = 20260409

def load_model(path):
    model = ResNet20()
    ckpt = torch.load(path, map_location='cpu', weights_only=False)
    if 'model_state_dict' in ckpt:
        model.load_state_dict(ckpt['model_state_dict'])
    else:
        model.load_state_dict(ckpt)
    model.eval()
    return model

def extract_fp_with_noise(W, b, p, noise_mult, n_trials=10, seed=0):
    """
    Simulate attack under noise: add uniform noise at noise_mult * Delta,
    round-then-sort each column, compute column norms of recovered spectra.
    Returns mean fingerprint over trials.
    """
    Delta = 1.0 / (2 * p)
    noise_scale = noise_mult * Delta
    Wq = np.round(W * p) / p
    bq = np.round(b * p) / p if b is not None else np.zeros(W.shape[0])
    
    fps = []
    rng = np.random.default_rng(seed)
    for _ in range(n_trials):
        col_norms = []
        for i in range(W.shape[1]):
            y = Wq[:, i] + bq
            eta = rng.uniform(-noise_scale, noise_scale, size=y.shape)
            y_noisy = y + eta
            y_rec = np.sort(np.round(y_noisy * p) / p)
            col_norms.append(np.linalg.norm(y_rec))
        fps.append(col_norms)
    return np.mean(fps, axis=0)

model_files = sorted(
    f for f in glob.glob(os.path.join(MODEL_DIR, 'resnet20_seed*.pt'))
    if not f.endswith('seed20.pt')
)
print("=== Fingerprinting Separability Under Noise ===")
print("Models: %d, Layer: conv1" % len(model_files))
print()

p = 256
fp_layer = 'conv1'
noise_mults = [1, 10, 100, 1000, 5000]
n_trials = 5

# Load weights once
layer_weights = []
for mf in model_files:
    model = load_model(mf)
    layers = get_conv_layers(model)
    for lname, wm, bv, _, _, _ in layers:
        if lname == fp_layer:
            layer_weights.append((wm, bv))
            break

print("%-10s  %-12s  %-12s  %-12s  %-8s  %-10s" % (
    "Noise", "Sep pairs", "Within max", "Between min", "Sep ratio", "FP acc"))
print("-" * 75)

for mult in noise_mults:
    # Compute fingerprints for all models at this noise level
    fps = np.array([
        extract_fp_with_noise(
            W, b, p, mult, n_trials,
            seed=BASE_SEED + mult * 1000 + idx
        )
        for idx, (W, b) in enumerate(layer_weights)
    ])
    
    # Within-model variation: compare two sets of noisy fps
    fps2 = np.array([
        extract_fp_with_noise(
            W, b, p, mult, n_trials,
            seed=BASE_SEED + 500000 + mult * 1000 + idx
        )
        for idx, (W, b) in enumerate(layer_weights)
    ])
    within_dists = np.array([np.linalg.norm(fps[i] - fps2[i]) for i in range(len(fps))])
    within_max = within_dists.max()
    
    # Between-model distances
    between_dists = []
    sep_pairs = 0
    n = len(fps)
    n_pairs = n * (n - 1) // 2
    for i in range(n):
        for j in range(i+1, n):
            d = np.linalg.norm(fps[i] - fps[j])
            between_dists.append(d)
            if d > within_max:
                sep_pairs += 1
    between_min = min(between_dists)
    sep_ratio = between_min / within_max if within_max > 0 else float('inf')
    
    # Fingerprinting accuracy: nearest-neighbor classification
    correct = 0
    for i in range(n):
        # Query fresh fp for model i
        fp_test = extract_fp_with_noise(
            layer_weights[i][0], layer_weights[i][1], p, mult, n_trials,
            seed=BASE_SEED + 900000 + mult * 1000 + i
        )
        dists = [np.linalg.norm(fp_test - fps[j]) for j in range(n)]
        if np.argmin(dists) == i:
            correct += 1
    fp_acc = correct / n

    sr_str = ("%.1fx" % sep_ratio) if sep_ratio < 1e6 else "inf"
    print("%-10s  %-12s  %-12.4f  %-12.4f  %-8s  %-10s" % (
        "%dx" % mult, "%d/%d" % (sep_pairs, n_pairs), within_max, between_min,
        sr_str, "%d/%d" % (correct, n)))

print()
print("Done.")

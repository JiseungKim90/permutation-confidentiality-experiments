"""
Exp 32: Lineage detection on ImageNet-scale models.
Tests whether sorted-spectrum fingerprints can identify a model's parent
after weight perturbation (simulating fine-tuning drift).

Strategy: start from 5 base models (V1, V2, and 3 perturbations of V1),
create children by adding controlled noise at 3 severity levels
(simulating mild/moderate/aggressive fine-tuning drift).
All operations are weight-only -- no training data needed, no GPU needed.

Reproducibility: all random seeds are fixed and recorded.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import numpy as np
import torch
import torchvision.models as models
from lib.attack import get_conv_layers, fingerprint_from_layer

# ================================================================
# Configuration
# ================================================================
P = 256  # 8-bit precision
FP_LAYERS = ["conv1", "layer2.0.conv1", "layer3.0.conv1", "layer4.0.conv1"]
N_CHILDREN = 3
BASE_SEED = 20260414

# Perturbation configs simulate fine-tuning drift magnitudes.
# Empirically, mild fine-tuning moves weights by ~0.1-0.5% of layer std,
# moderate by ~1-3%, aggressive by ~5-10%.
CONFIGS = [
    {"name": "mild",       "noise_scale": 0.001},
    {"name": "moderate",   "noise_scale": 0.005},
    {"name": "aggressive", "noise_scale": 0.02},
]

# ================================================================
# Base model construction
# ================================================================
def load_bases():
    bases = []
    # Base 0: V1 weights
    m0 = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    m0.eval()
    bases.append(("V1", m0))

    # Base 1: V2 weights (different training recipe)
    m1 = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    m1.eval()
    bases.append(("V2", m1))

    # Bases 2-4: V1 with distinct perturbations (simulating independent training)
    for i, scale in enumerate([2e-3, 5e-3, 1e-2]):
        m = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        torch.manual_seed(BASE_SEED + 1000 + i)
        with torch.no_grad():
            for param in m.parameters():
                param.add_(torch.randn_like(param) * scale * param.std().clamp(min=1e-8))
        m.eval()
        bases.append((f"V1+{scale}", m))

    return bases

# ================================================================
# Fingerprint extraction
# ================================================================
def extract_fingerprint(model):
    layers = get_conv_layers(model)
    fp_parts = []
    for lname, wm, bv, _, _, _ in layers:
        if lname in FP_LAYERS:
            fp_parts.extend(fingerprint_from_layer(wm, bv, P).tolist())
    return np.array(fp_parts)

# ================================================================
# Child creation via perturbation
# ================================================================
def create_child(base_model, noise_scale, seed):
    """Create a child model by perturbing base weights."""
    import copy
    child = copy.deepcopy(base_model)
    torch.manual_seed(seed)
    with torch.no_grad():
        for param in child.parameters():
            param.add_(torch.randn_like(param) * noise_scale * param.std().clamp(min=1e-8))
    child.eval()
    return child

# ================================================================
# Main
# ================================================================
print("=" * 70)
print("Exp 32: Lineage Detection on ImageNet-Scale Models")
print(f"Precision: {P} (8-bit), Layers: {FP_LAYERS}")
print(f"Random seed: {BASE_SEED}")
print("=" * 70)

bases = load_bases()
n_bases = len(bases)
print(f"\nLoaded {n_bases} base models:")
for name, _ in bases:
    print(f"  {name}")

# Base fingerprints
base_fps = []
for name, model in bases:
    fp = extract_fingerprint(model)
    base_fps.append(fp)
print(f"Fingerprint dimension: {len(base_fps[0])}")

# Base separation
print(f"\nBase model pairwise L2 distances:")
for i in range(n_bases):
    for j in range(i+1, n_bases):
        d = np.linalg.norm(base_fps[i] - base_fps[j])
        print(f"  {bases[i][0]:>10s} vs {bases[j][0]:<10s}: {d:.4f}")

# Lineage detection for each regime
for cfg in CONFIGS:
    print(f"\n{'='*60}")
    print(f"Regime: {cfg['name']} (noise_scale={cfg['noise_scale']})")
    print("=" * 60)

    child_fps = []
    child_parents = []

    for bi in range(n_bases):
        for ci in range(N_CHILDREN):
            seed = BASE_SEED + bi * 100 + ci * 10
            child = create_child(bases[bi][1], cfg["noise_scale"], seed)
            fp = extract_fingerprint(child)
            child_fps.append(fp)
            child_parents.append(bi)
            del child

    # Evaluate
    correct = 0
    within_dists = []
    between_dists = []

    for idx, (cfp, parent) in enumerate(zip(child_fps, child_parents)):
        dists = [np.linalg.norm(cfp - bfp) for bfp in base_fps]
        predicted = int(np.argmin(dists))
        correct += int(predicted == parent)
        within_dists.append(dists[parent])
        for bi in range(n_bases):
            if bi != parent:
                between_dists.append(dists[bi])

    within_dists = np.array(within_dists)
    between_dists = np.array(between_dists)
    sep_ratio = between_dists.min() / within_dists.max() if within_dists.max() > 0 else float("inf")

    print(f"  Children: {len(child_fps)} ({N_CHILDREN} per base x {n_bases} bases)")
    print(f"  Accuracy: {correct}/{len(child_fps)} ({100*correct/len(child_fps):.0f}%)")
    print(f"  Within  dist -- mean: {within_dists.mean():.4f}, max: {within_dists.max():.4f}")
    print(f"  Between dist -- mean: {between_dists.mean():.4f}, min: {between_dists.min():.4f}")
    sr_str = f"{sep_ratio:.1f}x" if sep_ratio < 1e6 else "inf"
    print(f"  Separation ratio: {sr_str}")
    print(f"  Separable: {between_dists.min() > within_dists.max()}")

# ================================================================
# Cross-precision verification
# ================================================================
print(f"\n{'='*60}")
print("Cross-Precision Lineage (aggressive regime)")
print("=" * 60)

for p_val, p_name in [(16, "4-bit"), (256, "8-bit"), (4096, "12-bit")]:
    # Recompute base fps at this precision
    bp_fps = []
    for name, model in bases:
        layers = get_conv_layers(model)
        fp_parts = []
        for lname, wm, bv, _, _, _ in layers:
            if lname in FP_LAYERS:
                fp_parts.extend(fingerprint_from_layer(wm, bv, p_val).tolist())
        bp_fps.append(np.array(fp_parts))

    # Children at aggressive
    correct = 0
    total = 0
    for bi in range(n_bases):
        for ci in range(N_CHILDREN):
            seed = BASE_SEED + bi * 100 + ci * 10
            child = create_child(bases[bi][1], 0.02, seed)
            layers = get_conv_layers(child)
            fp_parts = []
            for lname, wm, bv, _, _, _ in layers:
                if lname in FP_LAYERS:
                    fp_parts.extend(fingerprint_from_layer(wm, bv, p_val).tolist())
            cfp = np.array(fp_parts)
            dists = [np.linalg.norm(cfp - bfp) for bfp in bp_fps]
            if np.argmin(dists) == bi:
                correct += 1
            total += 1
            del child

    print(f"  {p_name}: {correct}/{total} ({100*correct/total:.0f}%)")

print("\n" + "=" * 70)
print("ALL LINEAGE EXPERIMENTS COMPLETE")
print("=" * 70)

"""
Exp 30: Round-then-sort attack on ImageNet-scale pretrained models.
Verifies exact sorted-spectrum recovery and fingerprinting separation
on ResNet-50, ResNet-101, ResNet-152 (torchvision pretrained weights).

No GPU required — all operations are on extracted weight matrices.
Runtime: ~2-5 minutes on CPU.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import numpy as np
import torch
import torchvision.models as models
from lib.attack import attack_layer, get_conv_layers, fingerprint_from_layer, noise_fns_bounded

# ================================================================
# Load pretrained models
# ================================================================
MODEL_CONFIGS = [
    ("ResNet-50",  lambda: models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)),
    ("ResNet-101", lambda: models.resnet101(weights=models.ResNet101_Weights.IMAGENET1K_V1)),
    ("ResNet-152", lambda: models.resnet152(weights=models.ResNet152_Weights.IMAGENET1K_V1)),
]

precisions = [(16, "4-bit"), (256, "8-bit"), (4096, "12-bit")]

# ================================================================
# Part 1: Exact Recovery on all conv layers
# ================================================================
print("=" * 70)
print("Exp 30: Round-then-sort attack on ImageNet-scale pretrained models")
print("=" * 70)

all_models = {}
for model_name, loader in MODEL_CONFIGS:
    print(f"\nLoading {model_name}...")
    model = loader()
    model.eval()
    layers = get_conv_layers(model)
    all_models[model_name] = (model, layers)
    print(f"  Conv layers: {len(layers)}")
    total_params = sum(l[4] * l[5] for l in layers)
    print(f"  Total conv params: {total_params:,}")

    # Show layer summary
    print(f"  {'Layer':<40s} {'Shape':>12s}")
    print(f"  {'-'*40} {'-'*12}")
    for lname, W_mat, b_vec, has_bias, C_out, d in layers:
        print(f"  {lname:<40s} {C_out:>4d} x {d:<5d}")

print("\n" + "=" * 70)
print("Part 1: Exact Recovery Verification")
print("=" * 70)

for model_name, (model, layers) in all_models.items():
    for p, pname in precisions:
        print(f"\n--- {model_name}, {pname} (p={p}) ---")
        np.random.seed(42)

        n_exact = 0
        n_layers = len(layers)
        max_err_all = 0.0

        for lname, W_mat, b_vec, has_bias, C_out, d in layers:
            max_err, _ = attack_layer(W_mat, b_vec, p)
            if max_err == 0:
                n_exact += 1
            else:
                max_err_all = max(max_err_all, max_err)

        print(f"  Exact recovery: {n_exact}/{n_layers} layers")
        if max_err_all > 0:
            print(f"  Max error (non-exact layers): {max_err_all:.2e}")
        else:
            print(f"  All layers: EXACT (zero error)")

# ================================================================
# Part 2: Cross-architecture fingerprinting
# ================================================================
print("\n" + "=" * 70)
print("Part 2: Cross-Architecture Fingerprinting Separation")
print("=" * 70)

p = 256  # 8-bit
# Use first conv layer of each model
print(f"\nUsing first conv layer (conv1 / 7x7, stride 2)")
fps = {}
for model_name, (model, layers) in all_models.items():
    lname, W_mat, b_vec, _, _, _ = layers[0]
    fp = fingerprint_from_layer(W_mat, b_vec, p)
    fps[model_name] = fp
    print(f"  {model_name} [{lname}]: fingerprint dim = {len(fp)}")

# conv1 is identical shape across ResNet-50/101/152 (64 x 147)
# so fingerprints are directly comparable
model_names = list(fps.keys())
print(f"\nPairwise L2 distances (conv1, 8-bit):")
for i in range(len(model_names)):
    for j in range(i + 1, len(model_names)):
        d = np.linalg.norm(fps[model_names[i]] - fps[model_names[j]])
        print(f"  {model_names[i]} vs {model_names[j]}: {d:.6f}")

# ================================================================
# Part 3: Same-architecture fingerprinting via fine-tuning variants
# ================================================================
print("\n" + "=" * 70)
print("Part 3: Same-Architecture Fingerprinting (V1 vs V2 weights)")
print("=" * 70)

# ResNet-50 has V1 and V2 pretrained weights — use as "two checkpoints"
variant_models = [
    ("ResNet-50-V1", lambda: models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)),
    ("ResNet-50-V2", lambda: models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)),
]

p = 256
variant_fps = {}
# Collect fingerprints from multiple layers
fp_layer_indices = [0, 5, 10, 20, -1]  # sample of layers

for vname, loader in variant_models:
    print(f"\nLoading {vname}...")
    m = loader()
    m.eval()
    layers = get_conv_layers(m)
    layer_fps = []
    for idx in fp_layer_indices:
        if abs(idx) >= len(layers):
            continue
        lname, W_mat, b_vec, _, _, _ = layers[idx]
        fp = fingerprint_from_layer(W_mat, b_vec, p)
        layer_fps.append((lname, fp))
    variant_fps[vname] = layer_fps

print(f"\nPer-layer fingerprint distances (V1 vs V2):")
for (lname1, fp1), (lname2, fp2) in zip(variant_fps["ResNet-50-V1"], variant_fps["ResNet-50-V2"]):
    assert lname1 == lname2
    d = np.linalg.norm(fp1 - fp2)
    print(f"  {lname1:<40s}: {d:.6f}")

# ================================================================
# Part 4: Noise distribution robustness (ImageNet-scale layers)
# ================================================================
print("\n" + "=" * 70)
print("Part 4: Noise Robustness on ImageNet-Scale Layers")
print("=" * 70)

# Test on ResNet-50 conv1 and a deep layer
model, layers = all_models["ResNet-50"]
test_layers = [layers[0], layers[len(layers)//2], layers[-1]]

for p, pname in precisions:
    Delta = 1.0 / (2 * p)
    fns = noise_fns_bounded(Delta)
    print(f"\n--- {pname} (p={p}) ---")

    for lname, W_mat, b_vec, _, C_out, d in test_layers:
        print(f"  Layer: {lname} ({C_out}x{d})")
        for noise_name, noise_fn in fns.items():
            np.random.seed(12345)
            max_err, _ = attack_layer(W_mat, b_vec, p, noise_fn=noise_fn)
            status = "EXACT" if max_err == 0 else f"err={max_err:.2e}"
            print(f"    {noise_name:<12s}: {status}")

# ================================================================
# Part 5: Layer statistics summary
# ================================================================
print("\n" + "=" * 70)
print("Part 5: Layer-by-Layer Recovery Summary Table")
print("=" * 70)

p = 256  # 8-bit representative
print(f"\n{'Model':<12s} {'Layer':<40s} {'C_out':>5s} {'d':>6s} {'Exact':>6s} {'MaxErr':>10s}")
print("-" * 82)

for model_name, (model, layers) in all_models.items():
    np.random.seed(42)
    for lname, W_mat, b_vec, _, C_out, d in layers:
        max_err, _ = attack_layer(W_mat, b_vec, p)
        exact = "YES" if max_err == 0 else "NO"
        err_str = "0" if max_err == 0 else f"{max_err:.2e}"
        print(f"{model_name:<12s} {lname:<40s} {C_out:>5d} {d:>6d} {exact:>6s} {err_str:>10s}")
    print()

print("=" * 70)
print("ALL IMAGENET-SCALE EXPERIMENTS COMPLETE")
print("=" * 70)

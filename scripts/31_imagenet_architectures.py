"""
Exp 31: Round-then-sort attack on diverse ImageNet architectures.
Shows the attack is architecture-agnostic, not ResNet-specific.
Tests: VGG-16, DenseNet-121, MobileNet-V2, EfficientNet-B0, ConvNeXt-Tiny.
All pretrained on ImageNet. No GPU needed.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import numpy as np
import torch
import torchvision.models as models
from lib.attack import attack_layer, get_conv_layers, fingerprint_from_layer, noise_fns_bounded

ARCH_CONFIGS = [
    ("VGG-16",        lambda: models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)),
    ("DenseNet-121",  lambda: models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1)),
    ("MobileNet-V2",  lambda: models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)),
    ("EfficientNet-B0", lambda: models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)),
    ("ConvNeXt-Tiny", lambda: models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1)),
]

precisions = [(16, "4-bit"), (256, "8-bit"), (4096, "12-bit")]

print("=" * 70)
print("Exp 31: Architecture-Agnostic Attack Verification")
print("=" * 70)

summary_rows = []

for arch_name, loader in ARCH_CONFIGS:
    print(f"\n{'='*50}")
    print(f"Loading {arch_name}...")
    model = loader()
    model.eval()
    layers = get_conv_layers(model)
    total_params = sum(l[4] * l[5] for l in layers)
    print(f"  Conv layers: {len(layers)}, Total conv params: {total_params:,}")

    # Show largest layers
    sorted_by_size = sorted(layers, key=lambda l: l[4]*l[5], reverse=True)
    print(f"  Top-3 largest layers:")
    for lname, _, _, _, C_out, d in sorted_by_size[:3]:
        print(f"    {lname:<40s} {C_out:>4d} x {d:<5d} ({C_out*d:>10,} params)")

    for p, pname in precisions:
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

        status = "ALL EXACT" if n_exact == n_layers else f"{n_exact}/{n_layers}"
        print(f"  {pname}: {status}")
        summary_rows.append((arch_name, pname, n_exact, n_layers, total_params))

    # Noise robustness on largest layer
    biggest = sorted_by_size[0]
    lname, W_mat, b_vec, _, C_out, d = biggest
    p = 256
    Delta = 1.0 / (2 * p)
    fns = noise_fns_bounded(Delta)
    print(f"\n  Noise robustness (8-bit) on largest layer {lname} ({C_out}x{d}):")
    for noise_name, noise_fn in fns.items():
        np.random.seed(42)
        max_err, _ = attack_layer(W_mat, b_vec, p, noise_fn=noise_fn)
        status = "EXACT" if max_err == 0 else f"err={max_err:.2e}"
        print(f"    {noise_name:<12s}: {status}")

# Summary table
print("\n" + "=" * 70)
print("SUMMARY TABLE")
print("=" * 70)
print(f"{'Architecture':<18s} {'Conv Params':>12s} {'4-bit':>10s} {'8-bit':>10s} {'12-bit':>10s}")
print("-" * 65)

arch_data = {}
for arch_name, pname, n_exact, n_layers, total_params in summary_rows:
    if arch_name not in arch_data:
        arch_data[arch_name] = {"params": total_params, "layers": n_layers, "results": {}}
    arch_data[arch_name]["results"][pname] = (n_exact, n_layers)

for arch_name, data in arch_data.items():
    cols = []
    for pname in ["4-bit", "8-bit", "12-bit"]:
        n_ex, n_tot = data["results"][pname]
        cols.append(f"{n_ex}/{n_tot}" if n_ex == n_tot else f"{n_ex}/{n_tot}!")
    print(f"{arch_name:<18s} {data['params']:>10,}   {'  '.join(f'{c:>8s}' for c in cols)}")

print("\n" + "=" * 70)
print("ALL ARCHITECTURE EXPERIMENTS COMPLETE")
print("=" * 70)

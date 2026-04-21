"""
Experiments on trained CIFAR-10 / ResNet-20 models.
Produces: Appendix A numbers (tab:cifar_spec, fingerprinting, noise robustness)
Paper: Appendix A
Requires: trained models from 00_train_cifar.py
"""
import sys, os, glob
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch
from lib.models import ResNet20
from lib.attack import attack_layer, get_conv_layers, fingerprint_from_layer, noise_fns_bounded

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

# Load models
model_files = sorted(
    f for f in glob.glob(os.path.join(MODEL_DIR, "resnet20_seed*.pt"))
    if not f.endswith("seed20.pt")
)
n_models = len(model_files)
print(f"Found {n_models} trained models")
if n_models == 0:
    print("ERROR: Run 00_train_cifar.py first.")
    sys.exit(1)

precisions = [(16, "4-bit"), (256, "8-bit"), (4096, "12-bit")]

# ================================================================
# A1: Exact recovery verification
# ================================================================
print("\n" + "=" * 60)
print("Appendix A - Exact Recovery (tab:cifar_spec)")
print("=" * 60)

for p, pname in precisions:
    print(f"\n--- {pname} (p={p}) ---")
    n_all_exact = 0
    layer_info = None

    for mf in model_files:
        model = load_model(mf)
        layers = get_conv_layers(model)
        seed = os.path.basename(mf).replace("resnet20_seed", "").replace(".pt", "")
        np.random.seed(int(seed) * 1000)

        model_exact = True
        if layer_info is None:
            layer_info = [(l[0], l[4], l[5], l[3]) for l in layers]

        for lname, W_mat, b_vec, _, C_out, d in layers:
            max_err, _ = attack_layer(W_mat, b_vec, p)
            if max_err > 0:
                model_exact = False

        if model_exact:
            n_all_exact += 1
        print(f"  seed={seed}: all_exact={model_exact} ({len(layers)} layers)")

    print(f"\n  Summary: {n_all_exact}/{n_models} models all-exact at {pname}")
    if layer_info:
        print(f"  Layers: {len(layer_info)}")
        for lname, Co, d, has_bias in layer_info:
            print(f"    {lname:25s} {Co:3d}x{d:<4d} bias={has_bias}")

# ================================================================
# A2: Same-architecture fingerprinting
# ================================================================
print("\n" + "=" * 60)
print("Appendix A - Same-Architecture Fingerprinting")
print("=" * 60)

p = 256
n_noise_trials = 10
fp_layers = ["conv1", "layer2.0.conv1", "layer3.0.conv1"]

for fp_layer in fp_layers:
    print(f"\n--- Layer: {fp_layer} ---")
    all_fps = []  # [model_idx][trial_idx] -> norm vector

    for mi, mf in enumerate(model_files):
        model = load_model(mf)
        layers = get_conv_layers(model)
        W_mat = b_vec = None
        for lname, wm, bv, _, _, _ in layers:
            if lname == fp_layer:
                W_mat, b_vec = wm, bv
                break
        if W_mat is None:
            continue

        trial_fps = []
        for t in range(n_noise_trials):
            np.random.seed(mi * 10000 + t)
            fp = fingerprint_from_layer(W_mat, b_vec, p)
            trial_fps.append(fp)
        all_fps.append(trial_fps)

    # Within-model
    within = []
    for m_fps in all_fps:
        for a in range(n_noise_trials):
            for b_ in range(a + 1, n_noise_trials):
                within.append(np.linalg.norm(m_fps[a] - m_fps[b_]))
    within = np.array(within) if within else np.array([0.0])

    # Between-model
    between = []
    for i in range(len(all_fps)):
        for j in range(i + 1, len(all_fps)):
            between.append(np.linalg.norm(all_fps[i][0] - all_fps[j][0]))
    between = np.array(between)

    print(f"  Models: {len(all_fps)}")
    print(f"  Within max:  {within.max():.6f}")
    print(f"  Between min: {between.min():.6f}")
    print(f"  Separable:   {between.min() > within.max()}")
    if within.max() > 0:
        print(f"  Ratio:       {between.min()/within.max():.1f}x")
    else:
        print(f"  Ratio:       inf")

# ================================================================
# A3: Noise distribution robustness
# ================================================================
print("\n" + "=" * 60)
print("Appendix A - Noise Distribution Robustness")
print("=" * 60)

for p, pname in precisions:
    Delta = 1.0 / (2 * p)
    fns = noise_fns_bounded(Delta)
    print(f"\n--- {pname} (p={p}) ---")

    for noise_name, noise_fn in fns.items():
        exact_count = 0
        n_test = min(n_models, 10)
        for mi in range(n_test):
            np.random.seed(mi * 77 + abs(hash(noise_name)) % 10000)
            model = load_model(model_files[mi])
            layers = get_conv_layers(model)
            # Test conv1
            for lname, W_mat, b_vec, _, _, _ in layers:
                if lname == "conv1":
                    max_err, _ = attack_layer(W_mat, b_vec, p, noise_fn=noise_fn)
                    if max_err == 0:
                        exact_count += 1
                    break
        print(f"  {noise_name:<12s}: {exact_count}/{n_test} exact")

print("\nAll Appendix A experiments complete.")

#!/bin/bash
# GPU-oriented reproduction entrypoint.
# Usage:
#   cd experiments
#   PYTHON=/path/to/gpu/python bash run_gpu_submission.sh
#
# Covers GPU-heavy experiments not in run_all.sh:
#   32_imagenet_lineage.py          Sec 5.1     ImageNet R50 V1/V2 lineage
#   33_imagenet_tradeoff.py         Sec 5.3     Imagenette R50 final-output tradeoff
#   33b_imagenet_tradeoff_per_layer.py Sec 5.4  R50 Imagenette per-layer intermediate sweep
#   run_kd_submission.sh            Sec 5       KD ablation table (R20 + R56 student)
#   R56 variants of 19/20/22b       Sec 5 / Prop 5  Fresh-perm/STIP/property-inference R56
#     (requires models/resnet56_seed0.pt; see below)

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

PYTHON_BIN="${PYTHON:-python3}"
LOG_DIR="outputs/logs"
mkdir -p "$LOG_DIR" models data

"$PYTHON_BIN" - <<'PY'
import sys
import torch

print(f"python={sys.executable}")
print(f"torch={torch.__version__}")
print(f"cuda_available={torch.cuda.is_available()}")
if not torch.cuda.is_available():
    raise SystemExit(
        "CUDA is not available. Set PYTHON to a GPU-enabled environment "
        "or run the CPU entrypoints instead."
    )
print(f"cuda_device={torch.cuda.get_device_name(0)}")
PY

run_py() {
    local script="$1"
    local name
    name=$(basename "$script" .py)
    echo "===== Running GPU-stage $name ====="
    "$PYTHON_BIN" -u "scripts/$script" 2>&1 | tee "$LOG_DIR/${name}.log"
    echo "===== Done: $name ====="
    echo ""
}

# ImageNet-scale experiments with real labels / real fine-tuning.
run_py 32_imagenet_lineage.py
run_py 33_imagenet_tradeoff.py

# R50 Imagenette per-layer intermediate sweep (Sec 5.4 phase-transition).
# Uses torchvision pretrained weights; no checkpoint needed.
echo "===== Running 33b_imagenet_tradeoff_per_layer (R50 intermediate mults) ====="
"$PYTHON_BIN" -u scripts/33b_imagenet_tradeoff_per_layer.py \
    2>&1 | tee "$LOG_DIR/r50_imagenette_per_layer_diag_intermediate_mults.log"
echo "===== Done: 33b ====="
echo ""

# Long KD table reproduction. Requires resnet20_seed0.pt.
bash run_kd_submission.sh

# R56 variants of fresh-perm, STIP/Centaur, and property-inference experiments.
# These require a trained ResNet-56 checkpoint at models/resnet56_seed0.pt.
R56_CKPT="models/resnet56_seed0.pt"
if [ -f "$R56_CKPT" ]; then
    echo "===== Running 19_fresh_perm_empirical (R56) ====="
    "$PYTHON_BIN" -u scripts/19_fresh_perm_empirical.py \
        --teacher-path "$R56_CKPT" --architecture resnet56 \
        --output-path outputs/fresh_perm_r56_n30.json \
        2>&1 | tee "$LOG_DIR/19_fresh_perm_r56_n30.log"

    echo "===== Running 20_stip_centaur_trivial (R56) ====="
    "$PYTHON_BIN" -u scripts/20_stip_centaur_trivial.py \
        --teacher-path "$R56_CKPT" --architecture resnet56 \
        --output-path outputs/stip_centaur_r56.json \
        2>&1 | tee -a "$LOG_DIR/20_stip_centaur.log"

    echo "===== Running 22b_property_inference (R56) ====="
    "$PYTHON_BIN" -u scripts/22b_property_inference.py \
        --teacher-path "$R56_CKPT" --architecture resnet56 \
        --output-path outputs/property_inference_r56.json \
        2>&1 | tee -a "$LOG_DIR/22b_property_inference.log"
else
    echo "SKIPPED: R56 variants (19/20/22b) — $R56_CKPT not found."
    echo "  Train a ResNet-56 with scripts/00_train_cifar.py --architecture resnet56"
    echo "  and place the seed-0 checkpoint at $R56_CKPT."
fi

echo ""
echo "============================================"
echo "GPU SUBMISSION EXPERIMENTS COMPLETE"
echo "Logs: outputs/logs/"
echo "============================================"

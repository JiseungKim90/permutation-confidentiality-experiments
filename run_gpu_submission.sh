#!/bin/bash
# GPU-oriented reproduction entrypoint.
# Usage:
#   cd experiments
#   PYTHON=/path/to/gpu/python bash run_gpu_submission.sh
#
# This script intentionally separates the long GPU-friendly experiments from
# the core CPU theorem checks in run_all.sh.

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

# Long KD table reproduction. Requires both resnet20_seed0.pt and
# resnet56_seed0.pt in models/.
bash run_kd_submission.sh

echo ""
echo "============================================"
echo "GPU SUBMISSION EXPERIMENTS COMPLETE"
echo "Logs: outputs/logs/"
echo "============================================"

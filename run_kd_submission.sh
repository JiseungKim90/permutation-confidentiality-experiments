#!/bin/bash
# Reproduce the CIFAR-10 / ResNet-20 KD table used in the submission.
# Usage: cd experiments && bash run_kd_submission.sh
#
# Preconditions:
#   - models/resnet20_seed0.pt must exist.
#     Generate it via scripts/00_train_cifar.py (or bash run_all.sh).
#
# Outputs:
#   - outputs/cifar_kd_q2000_e10_s10_quantized_random.json
#   - outputs/cifar_kd_q5000_e10_s10_quantized_random.json
#   - outputs/logs/cifar_kd_q2000_e10_s10_quantized_random.log
#   - outputs/logs/cifar_kd_q5000_e10_s10_quantized_random.log

set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

mkdir -p outputs/logs models data

if [ ! -f "models/resnet20_seed0.pt" ]; then
    echo "ERROR: models/resnet20_seed0.pt not found."
    echo "Run bash run_all.sh (or python3 -u scripts/00_train_cifar.py) first."
    exit 1
fi

run_kd() {
    local budget="$1"
    local stem="cifar_kd_q${budget}_e10_s10_quantized_random"
    echo "===== Running ${stem} ====="
    python3 -u scripts/18_kd_spectral_priors_cifar.py \
        --teacher-path models/resnet20_seed0.pt \
        --teacher-logits-source quantized \
        --data-root data \
        --query-budget "${budget}" \
        --subset-mode random \
        --subset-seed 20260409 \
        --student-epochs 10 \
        --distill-seeds 10 \
        --batch-size 128 \
        --student-lr 0.05 \
        --momentum 0.9 \
        --reg-lambda 5.0 \
        --prior-scope all \
        --output-path "outputs/${stem}.json" \
        2>&1 | tee "outputs/logs/${stem}.log"
    echo "===== Done: ${stem} ====="
    echo ""
}

run_kd 2000
run_kd 5000

echo "KD submission sweep complete."

#!/bin/bash
# Reproduce the full CIFAR-10 KD ablation table used in the submission.
# Usage: cd experiments && bash run_kd_submission.sh
#
# Preconditions:
#   - models/resnet20_seed0.pt must exist.
#     Generate it via scripts/00_train_cifar.py (or bash run_all.sh).
#
# Outputs (ResNet-20, 4 ablation modes, 10 seeds):
#   - outputs/cifar_kd_q{2000,5000}_e10_s10_quantized_random.json
#   - outputs/logs/cifar_kd_q{2000,5000}_e10_s10_quantized_random.log
# Outputs (ResNet-56, 4 ablation modes, 5 seeds):
#   - outputs/cifar_kd_r56_q{2000,5000}_e10_s5_quantized_random.json
#   - outputs/logs/cifar_kd_r56_q{2000,5000}_e10_s5_quantized_random.log

set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

mkdir -p outputs/logs models data

if [ ! -f "models/resnet20_seed0.pt" ]; then
    echo "ERROR: models/resnet20_seed0.pt not found."
    echo "Run bash run_all.sh (or python3 -u scripts/00_train_cifar.py) first."
    exit 1
fi

COMMON_ARGS="--teacher-logits-source quantized \
    --data-root data \
    --subset-mode random \
    --subset-seed 0 \
    --student-epochs 10 \
    --batch-size 128 \
    --student-lr 0.05 \
    --momentum 0.9 \
    --reg-lambda 5.0 \
    --prior-scope all \
    --ablation-modes baseline,first,colnorm,full"

run_kd() {
    local arch="$1"
    local budget="$2"
    local seeds="$3"
    local stem="cifar_kd_q${budget}_e10_s${seeds}_quantized_random"
    if [ "$arch" = "resnet56" ]; then
        stem="cifar_kd_r56_q${budget}_e10_s${seeds}_quantized_random"
    fi
    echo "===== Running ${stem} ====="
    python3 -u scripts/18_kd_spectral_priors_cifar.py \
        --teacher-path models/resnet20_seed0.pt \
        --architecture "${arch}" \
        --query-budget "${budget}" \
        --distill-seeds "${seeds}" \
        ${COMMON_ARGS} \
        --output-path "outputs/${stem}.json" \
        2>&1 | tee "outputs/logs/${stem}.log"
    echo "===== Done: ${stem} ====="
    echo ""
}

# ResNet-20: 4 ablations x 2 budgets x 10 seeds
run_kd resnet20 2000 10
run_kd resnet20 5000 10

# ResNet-56: 4 ablations x 2 budgets x 5 seeds
run_kd resnet56 2000 5
run_kd resnet56 5000 5

echo "KD submission sweep complete (all ablation modes)."

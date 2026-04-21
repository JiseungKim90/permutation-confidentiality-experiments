#!/bin/bash
# Reproduce the full CIFAR-10 KD ablation table used in the submission.
# Usage: cd experiments && bash run_kd_submission.sh
#
# Preconditions:
#   - models/resnet20_seed0.pt must exist for ResNet-20.
#   - models/resnet56_seed0.pt must exist for ResNet-56.
#     Generate/provide those checkpoints before running the full table.
#
# Outputs (ResNet-20, 4 ablation modes, 10 seeds):
#   - outputs/cifar_kd_q{2000,5000}_e100_s10_quantized_random.json
#   - outputs/logs/cifar_kd_q{2000,5000}_e100_s10_quantized_random.log
# Outputs (ResNet-56, 4 ablation modes, 5 seeds):
#   - outputs/cifar_kd_r56_q{2000,5000}_e100_s5_quantized_random.json
#   - outputs/logs/cifar_kd_r56_q{2000,5000}_e100_s5_quantized_random.log

set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

mkdir -p outputs/logs models data
PYTHON_BIN="${PYTHON:-python3}"

for ckpt in models/resnet20_seed0.pt models/resnet56_seed0.pt; do
    if [ ! -f "$ckpt" ]; then
        echo "ERROR: $ckpt not found."
        echo "Provide the checkpoint before reproducing the full KD table."
        exit 1
    fi
done

COMMON_ARGS="--teacher-logits-source quantized \
    --data-root data \
    --subset-mode random \
    --subset-seed 20260409 \
    --student-epochs 100 \
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
    local teacher_path="models/${arch}_seed0.pt"
    local stem="cifar_kd_${arch}_q${budget}_e100_s${seeds}_quantized_random"
    if [ "$arch" = "resnet20" ]; then
        stem="cifar_kd_q${budget}_e100_s${seeds}_quantized_random"
    fi
    echo "===== Running ${stem} ====="
    "$PYTHON_BIN" -u scripts/18_kd_spectral_priors_cifar.py \
        --teacher-path "${teacher_path}" \
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

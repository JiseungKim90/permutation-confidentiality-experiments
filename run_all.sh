#!/bin/bash
# Reproduce all experiments.
# Usage: cd experiments && bash run_all.sh
#
# Script-to-paper mapping:
#   00_train_cifar.py        Appendix A            Train 20 ResNet-20 / CIFAR-10
#   02_dp_analysis.py        Sec 3.3, Tab 1        Local DP parameter eps_0
#   08_cifar_trained.py      Sec 4, Tab 2, Tab A1  Exact recovery (all precisions)
#   11_tradeoff_cifar.py     Sec 5.4, Tab 4        Privacy-utility tradeoff
#   14_lineage_aggressive.py Sec 5.3               Lineage (3 regimes)
#   15_bias_recovery.py      Appendix A            Bias-having MLP attack
#   16_noise_fingerprinting.py Sec 5.4, Tab 4      Fingerprinting under noise
#   18_kd_spectral_priors_cifar.py Sec 5           Logit distillation + spectral priors
#   19_fresh_perm_empirical.py   Prop 5            Fresh-perm T_emp/T_theory validation
#   20_stip_centaur_trivial.py   Sec 4             STIP/Centaur zero-query demo
#   21_multi_query_averaging.py  Sec 5             Multi-query averaging degradation
#   22_tfhe_validate.py      Appendix A            CKKS check + TFHE proxy/simulation
#   22b_property_inference.py    Sec 5             Quantization/sparsity inference
#   24_tfhe_real.py          Appendix A            Small real TFHE check + larger simulation
#   25_tfhe_resnet_transcript.py Appendix A        Trained-layer TFHE transcript check
#   30_imagenet_pretrained.py    Sec 4, Tab 3       ImageNet-scale exact recovery
#   31_imagenet_architectures.py Sec 4, Tab 3       Multi-architecture verification
#   32_imagenet_lineage.py       Sec 5.1            ImageNet-scale lineage detection
#   33_imagenet_tradeoff.py      Sec 5.3, Tab 5     ImageNet-scale tradeoff
#
# Runtime: ~7 hours total (dominated by 00 and 14)
# ImageNet-scale scripts add ~10-15 minutes (pretrained weights, no training)
# The submission KD sweep is intentionally split into
# run_kd_submission.sh because it uses 10 distillation epochs,
# 10 seeds, and two query budgets.

set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

LOG_DIR="outputs/logs"
mkdir -p "$LOG_DIR" models data

run() {
    local script="$1"
    local name=$(basename "$script" .py)
    echo "===== Running $name ====="
    python3 -u "scripts/$script" 2>&1 | tee "$LOG_DIR/${name}.log"
    echo "===== Done: $name ======"
    echo ""
}

# ---- Step 0: DP analysis (no models needed) ----
run 02_dp_analysis.py

# ---- Step 1: Train models (skip if already present) ----
if [ ! -f "models/resnet20_seed19.pt" ]; then
    echo "Training 20 ResNet-20 models (~5 hours on CPU)..."
    run 00_train_cifar.py
else
    echo "Models already present, skipping 00_train_cifar.py"
fi

# ---- Step 2: Exact recovery ----
run 08_cifar_trained.py

# ---- Step 2: Privacy-utility tradeoff ----
run 11_tradeoff_cifar.py

# ---- Step 3: Lineage detection ----
run 14_lineage_aggressive.py

# ---- Step 4: Bias-having MLP ----
run 15_bias_recovery.py

# ---- Step 5: Fingerprinting under noise ----
run 16_noise_fingerprinting.py

# ---- Step 6: Fresh permutation / supplementary experiments ----
# These need the trained R20 checkpoint (models/resnet20_seed0.pt).
TEACHER="models/resnet20_seed0.pt"

echo "===== Running 19_fresh_perm_empirical (R20) ====="
python3 -u scripts/19_fresh_perm_empirical.py \
    --teacher-path "$TEACHER" --architecture resnet20 \
    --output-path outputs/fresh_perm_r20_n30.json \
    2>&1 | tee "$LOG_DIR/19_fresh_perm_r20_n30.log"

echo "===== Running 20_stip_centaur_trivial (R20) ====="
python3 -u scripts/20_stip_centaur_trivial.py \
    --teacher-path "$TEACHER" --architecture resnet20 \
    --output-path outputs/stip_centaur_r20.json \
    2>&1 | tee "$LOG_DIR/20_stip_centaur.log"

echo "===== Running 21_multi_query_averaging (R20) ====="
python3 -u scripts/21_multi_query_averaging.py \
    --teacher-path "$TEACHER" --architecture resnet20 \
    --output-path outputs/multi_query_avg_r20.json \
    2>&1 | tee "$LOG_DIR/21_multi_query_avg_r20.log"

echo "===== Running 22b_property_inference (R20) ====="
python3 -u scripts/22b_property_inference.py \
    --teacher-path "$TEACHER" --architecture resnet20 \
    --output-path outputs/property_inference_r20.json \
    2>&1 | tee "$LOG_DIR/22b_property_inference.log"
echo ""

# ---- Step 7: FHE validation ----
run 22_tfhe_validate.py

if python3 -c "import concrete" 2>/dev/null; then
    run 24_tfhe_real.py
    run 25_tfhe_resnet_transcript.py
else
    echo "SKIPPED: 24_tfhe_real.py and 25_tfhe_resnet_transcript.py (concrete-python not installed)"
fi

# ---- ImageNet-scale experiments (pretrained weights, no training needed) ----
echo ""
echo "===== ImageNet-scale experiments ====="
run 30_imagenet_pretrained.py
run 31_imagenet_architectures.py
run 32_imagenet_lineage.py
run 33_imagenet_tradeoff.py

echo ""
echo "============================================"
echo "ALL EXPERIMENTS COMPLETE"
echo "Logs: outputs/logs/"
echo "For the submission KD table, run: bash run_kd_submission.sh"
echo "============================================"

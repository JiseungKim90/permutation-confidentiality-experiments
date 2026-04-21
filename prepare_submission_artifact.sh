#!/bin/bash
# Stage a clean canonical submission artifact directory.
# Usage: cd experiments && bash prepare_submission_artifact.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

OUT_DIR="submission_artifact"
rm -rf "$OUT_DIR"
mkdir -p "$OUT_DIR"
mkdir -p "$OUT_DIR/lib" "$OUT_DIR/scripts" "$OUT_DIR/outputs/logs" "$OUT_DIR/outputs/figures" "$OUT_DIR/models" "$OUT_DIR/data"

copy_file() {
    local src="$1"
    local dst="$2"
    mkdir -p "$(dirname "$dst")"
    cp "$src" "$dst"
}

# Top-level docs / entrypoints
copy_file "README_SUBMISSION.md" "$OUT_DIR/README.md"
copy_file "ARTIFACT_MANIFEST.md" "$OUT_DIR/ARTIFACT_MANIFEST.md"
copy_file "requirements.txt" "$OUT_DIR/requirements.txt"
copy_file "run_all.sh" "$OUT_DIR/run_all.sh"
copy_file "run_kd_submission.sh" "$OUT_DIR/run_kd_submission.sh"
copy_file "run_submission_artifact.sh" "$OUT_DIR/run_submission_artifact.sh"
copy_file "validate_submission_artifact.sh" "$OUT_DIR/validate_submission_artifact.sh"

# Libraries
copy_file "lib/__init__.py" "$OUT_DIR/lib/__init__.py"
copy_file "lib/attack.py" "$OUT_DIR/lib/attack.py"
copy_file "lib/models.py" "$OUT_DIR/lib/models.py"

# Canonical scripts
for script in \
    00_train_cifar.py \
    02_dp_analysis.py \
    08_cifar_trained.py \
    11_tradeoff_cifar.py \
    14_lineage_aggressive.py \
    15_bias_recovery.py \
    16_noise_fingerprinting.py \
    18_kd_spectral_priors_cifar.py \
    22_tfhe_validate.py \
    24_tfhe_real.py \
    25_tfhe_resnet_transcript.py \
    30_imagenet_pretrained.py \
    31_imagenet_architectures.py \
    32_imagenet_lineage.py \
    33_imagenet_tradeoff.py
do
    copy_file "scripts/$script" "$OUT_DIR/scripts/$script"
done

# Canonical outputs
copy_file "outputs/figures/fig_dp_vacuousness.png" \
          "$OUT_DIR/outputs/figures/fig_dp_vacuousness.png"

for log in \
    00_train_cifar.log \
    02_dp_analysis.log \
    08_cifar_trained.log \
    11_tradeoff_cifar.log \
    14_lineage_aggressive.log \
    15_bias_recovery.log \
    16_noise_fingerprinting.log \
    22_tfhe_validate.log \
    24_tfhe_real.log \
    25_tfhe_resnet_transcript.log \
    25_tfhe_resnet_transcript.json \
    25_tfhe_resnet_transcript_doublecheck.log \
    25_tfhe_resnet_transcript_doublecheck.json \
    30_imagenet_pretrained.log \
    31_imagenet_architectures.log \
    32_imagenet_lineage.log \
    33_imagenet_tradeoff.log \
    cifar_kd_q2000_e10_s10_quantized_random.log \
    cifar_kd_q5000_e10_s10_quantized_random.log
do
    copy_file "outputs/logs/$log" "$OUT_DIR/outputs/logs/$log"
done

for json_file in \
    cifar_kd_q2000_e10_s10_quantized_random.json \
    cifar_kd_q5000_e10_s10_quantized_random.json
do
    copy_file "outputs/$json_file" "$OUT_DIR/outputs/$json_file"
done

# Placeholder directories expected by the scripts.
touch "$OUT_DIR/models/.gitkeep"
touch "$OUT_DIR/data/.gitkeep"

echo "Prepared clean submission artifact in: $OUT_DIR/"

#!/bin/bash
# Validate the clean canonical submission artifact.
# Usage: cd submission_artifact && bash validate_submission_artifact.sh

set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

required_files=(
  "README.md"
  "ARTIFACT_MANIFEST.md"
  "requirements.txt"
  "run_all.sh"
  "run_kd_submission.sh"
  "run_submission_artifact.sh"
  "lib/attack.py"
  "lib/models.py"
  "scripts/00_train_cifar.py"
  "scripts/02_dp_analysis.py"
  "scripts/08_cifar_trained.py"
  "scripts/11_tradeoff_cifar.py"
  "scripts/14_lineage_aggressive.py"
  "scripts/15_bias_recovery.py"
  "scripts/16_noise_fingerprinting.py"
  "scripts/18_kd_spectral_priors_cifar.py"
  "scripts/22_tfhe_validate.py"
  "scripts/24_tfhe_real.py"
  "scripts/25_tfhe_resnet_transcript.py"
  "scripts/30_imagenet_pretrained.py"
  "scripts/31_imagenet_architectures.py"
  "scripts/32_imagenet_lineage.py"
  "scripts/33_imagenet_tradeoff.py"
  "outputs/cifar_kd_q2000_e10_s10_quantized_random.json"
  "outputs/cifar_kd_q5000_e10_s10_quantized_random.json"
  "outputs/logs/00_train_cifar.log"
  "outputs/logs/cifar_kd_q2000_e10_s10_quantized_random.log"
  "outputs/logs/cifar_kd_q5000_e10_s10_quantized_random.log"
  "outputs/logs/25_tfhe_resnet_transcript.log"
  "outputs/logs/25_tfhe_resnet_transcript.json"
  "outputs/logs/25_tfhe_resnet_transcript_doublecheck.log"
  "outputs/logs/25_tfhe_resnet_transcript_doublecheck.json"
  "outputs/logs/30_imagenet_pretrained.log"
  "outputs/logs/31_imagenet_architectures.log"
  "outputs/logs/32_imagenet_lineage.log"
  "outputs/logs/33_imagenet_tradeoff.log"
)

for path in "${required_files[@]}"; do
  if [ ! -f "$path" ]; then
    echo "MISSING: $path"
    exit 1
  fi
done

if rg -n \
  -g '!validate_submission_artifact.sh' \
  -g '!package_submission_artifact.sh' \
  "/home/user|/Users/|Dropbox|codex_lab614|210\\.117" .; then
  echo "ERROR: found machine-local absolute paths in submission artifact"
  exit 1
fi

echo "Submission artifact validation passed."

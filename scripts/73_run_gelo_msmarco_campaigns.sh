#!/usr/bin/env bash
set -euo pipefail

campaign="${1:?usage: $0 core|robust}"
threads="${2:-14}"
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="/home/ubuntu/research-vault/venvs/p050-ndss-gelo-v1/bin/python"
run_root="$root/outputs/gelo_msmarco_100k_dedup_20260802"
official="/home/ubuntu/research-vault/projects/P050/ndss-2027/third_party/gelo"

export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS="$threads"
export MKL_NUM_THREADS="$threads"

common=(
  "$python_bin" "$root/scripts/71_gelo_msmarco_evaluate.py"
  --trials "$run_root/data/trials.json"
  --public-cache "$run_root/cache/public_manifest.json"
  --private-cache "$run_root/cache/private_full_manifest.json"
  --official-repo "$official"
  --source-models public private
  --bank-sizes 10000 50000 100000
  --sampled-rows 16
  --calibration-trials 20
  --evaluation-trials 50
  --chunk-size 128
  --score-dtype float64
  --score-backend torch
  --control-trials 5
  --threads "$threads"
  --seed 20260805
)

case "$campaign" in
  core)
    exec "${common[@]}" \
      --output-dir "$run_root/evaluation_core_torch" \
      --layers 4 8 12 \
      --conditions ideal gelo_nonorth manifold_stress
    ;;
  robustness)
    exec "${common[@]}" \
      --output-dir "$run_root/evaluation_robustness_torch" \
      --layers 8 \
      --conditions gelo_gaussian quantized_gaussian quantized_nonorth manifold
    ;;
  *)
    echo "unknown campaign: $campaign" >&2
    exit 2
    ;;
esac


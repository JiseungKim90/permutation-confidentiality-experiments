#!/usr/bin/env bash
set -euo pipefail

repo_root="/home/ubuntu/research-vault/projects/P050/ndss-2027/artifact"
run_root="$repo_root/outputs/gelo_msmarco_100k_dedup_20260802"
campaign_root="$run_root/fullbank_validation"
python_bin="/home/ubuntu/research-vault/venvs/p050-ndss-gelo-v1/bin/python"
official="/home/ubuntu/research-vault/projects/P050/ndss-2027/third_party/gelo"
tokens="$run_root/data/tokens.npy"
public_cache="$run_root/cache/public_manifest.json"
threads="${1:-20}"

mkdir -p "$campaign_root/logs" "$campaign_root/data" "$campaign_root/evaluation"

# Keep all heavy experiments sequential: the drift sweep already waits for the
# core/robustness chain, and this campaign waits for the drift sweep.
while tmux has-session -t '=p050_gelo_followup' 2>/dev/null; do
  sleep 30
done

export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS="$threads"
export MKL_NUM_THREADS="$threads"

"$python_bin" "$repo_root/scripts/80_prepare_gelo_fullbank_trials.py" \
  --tokens "$tokens" \
  --output-dir "$campaign_root/data" \
  --trials 300 \
  --source-count 4 \
  --seed 20260810 \
  > "$campaign_root/logs/prepare_trials.log" 2>&1

"$python_bin" "$repo_root/scripts/71_gelo_msmarco_evaluate.py" \
  --trials "$campaign_root/data/trials_fullbank.json" \
  --public-cache "$public_cache" \
  --official-repo "$official" \
  --output-dir "$campaign_root/evaluation" \
  --layers 8 \
  --source-models public \
  --conditions ideal gelo_nonorth manifold_stress \
  --bank-sizes 100000 \
  --sampled-rows 16 \
  --calibration-trials 100 \
  --calibration-alpha 0.01 \
  --closed-trials 100 \
  --partial-trials 50 \
  --open-trials 300 \
  --chunk-size 128 \
  --score-dtype float64 \
  --score-backend torch \
  --control-trials 10 \
  --threads "$threads" \
  --seed 20260811 \
  > "$campaign_root/logs/evaluate.log" 2>&1

"$python_bin" "$repo_root/scripts/81_finalize_gelo_fullbank_validation.py" \
  --repo-root "$repo_root" \
  --campaign-root "$campaign_root" \
  > "$campaign_root/logs/finalize.log" 2>&1

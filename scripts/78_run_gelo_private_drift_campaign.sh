#!/usr/bin/env bash
set -euo pipefail

repo_root="/home/ubuntu/research-vault/projects/P050/ndss-2027/artifact"
run_root="$repo_root/outputs/gelo_msmarco_100k_dedup_20260802"
drift_root="$run_root/private_drift_followup"
python_bin="/home/ubuntu/research-vault/venvs/p050-ndss-gelo-v1/bin/python"
base_model="/home/ubuntu/research-vault/cache/p050-gelo/hub/models--openai-community--gpt2/snapshots/607a30d783dfa663caf39e06633721c8d4cfcd7e"
official="/home/ubuntu/research-vault/projects/P050/ndss-2027/third_party/gelo"
tokens="$run_root/data/tokens.npy"
public_cache="$run_root/cache/public_manifest.json"
threads="${1:-20}"

mkdir -p "$drift_root/logs" "$drift_root/data" "$drift_root/cache" "$drift_root/evaluation"

while tmux has-session -t p050_gelo_full 2>/dev/null; do
  sleep 30
done

export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS="$threads"
export MKL_NUM_THREADS="$threads"

"$python_bin" "$repo_root/scripts/76_prepare_gelo_drift_trials.py" \
  --repo-root "$repo_root" \
  --tokens "$tokens" \
  --output-dir "$drift_root/data" \
  --trials 300 \
  --source-count 4 \
  --seed 20260807 \
  > "$drift_root/logs/prepare_trials.log" 2>&1

"$python_bin" "$repo_root/scripts/77_gelo_private_drift_sweep.py" \
  --tokens "$tokens" \
  --model "$base_model" \
  --output-dir "$drift_root/checkpoints" \
  --checkpoints 0 100 400 1600 3200 \
  --prefix-depth 8 \
  --probe-layer 8 \
  --probe-sequences 128 \
  --utility-sequences 320 \
  --batch-size 4 \
  --eval-batch-size 8 \
  --learning-rate 1e-5 \
  --threads "$threads" \
  --seed 20260808 \
  > "$drift_root/logs/train_checkpoints.log" 2>&1

for step in 0 100 400 1600 3200; do
  tag=$(printf "step%04d" "$step")
  if [[ "$step" -eq 0 ]]; then
    model="$base_model"
    private_cache="$public_cache"
  else
    model="$drift_root/checkpoints/$tag"
    cache_dir="$drift_root/cache/$tag"
    cache_tag="private_${tag}_sources"
    "$python_bin" "$repo_root/scripts/70_gelo_hidden_cache.py" \
      --tokens "$tokens" \
      --indices "$drift_root/data/source_indices_drift.npy" \
      --model "$model" \
      --output-dir "$cache_dir" \
      --tag "$cache_tag" \
      --layers 8 \
      --batch-size 64 \
      --threads "$threads" \
      > "$drift_root/logs/cache_${tag}.log" 2>&1
    private_cache="$cache_dir/${cache_tag}_manifest.json"
  fi

  "$python_bin" "$repo_root/scripts/71_gelo_msmarco_evaluate.py" \
    --trials "$drift_root/data/trials_drift.json" \
    --public-cache "$public_cache" \
    --private-cache "$private_cache" \
    --official-repo "$official" \
    --output-dir "$drift_root/evaluation/$tag" \
    --layers 8 \
    --source-models private \
    --conditions ideal \
    --bank-sizes 10000 50000 100000 \
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
    --seed 20260809 \
    > "$drift_root/logs/evaluate_${tag}.log" 2>&1
done

"$python_bin" "$repo_root/scripts/79_finalize_gelo_private_drift.py" \
  --repo-root "$repo_root" \
  --drift-root "$drift_root" \
  --checkpoints 0 100 400 1600 3200 \
  > "$drift_root/logs/finalize.log" 2>&1

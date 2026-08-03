#!/usr/bin/env bash
set -euo pipefail

repo_root="/home/ubuntu/research-vault/projects/P050/ndss-2027/artifact"
project_root="/home/ubuntu/research-vault/projects/P050/ndss-2027"
python_bin="/home/ubuntu/research-vault/venvs/p050-ndss-gelo-v1/bin/python"
official="$project_root/third_party/kvcloak"
gelo_root="$repo_root/outputs/gelo_msmarco_100k_dedup_20260802"
run_root="$repo_root/outputs/kvcloak_msmarco_100k_20260803"
model="/home/ubuntu/research-vault/cache/p050-gelo/hub/models--openai-community--gpt2/snapshots/607a30d783dfa663caf39e06633721c8d4cfcd7e"
threads="${1:-20}"
pinned="6b40f36edb2f337557543e7e60b10022308883d4"

mkdir -p "$project_root/third_party" "$run_root/logs" "$run_root/data" "$run_root/cache"
if [[ ! -d "$official/.git" ]]; then
  git clone https://github.com/SiO-2/kvcloak.git "$official"
  git -C "$official" checkout --detach "$pinned"
fi
actual="$(git -C "$official" rev-parse HEAD)"
if [[ "$actual" != "$pinned" ]]; then
  echo "KV-Cloak official checkout mismatch: $actual" >&2
  exit 1
fi

# Heavy CPU work stays sequential with the already-running GELO campaign.
while tmux has-session -t '=p050_gelo_validation' 2>/dev/null; do
  sleep 60
done

export OPENBLAS_NUM_THREADS="$threads"
export OMP_NUM_THREADS="$threads"
export MKL_NUM_THREADS="$threads"

tokens32="$gelo_root/data/tokens.npy"
hidden32="$gelo_root/cache/public_manifest.json"

"$python_bin" "$repo_root/scripts/85_kvcloak_prepare_trials.py" \
  --tokens "$tokens32" --output-dir "$run_root/data/b16_full" \
  --block-size 16 --bank-sizes 10000 50000 100000 \
  --calibration-trials 100 --closed-trials 100 --open-trials 300 --seed 20260813 \
  > "$run_root/logs/prepare_b16_full.log" 2>&1

"$python_bin" "$repo_root/scripts/85_kvcloak_prepare_trials.py" \
  --tokens "$tokens32" --output-dir "$run_root/data/b16_10k" \
  --block-size 16 --bank-sizes 10000 \
  --calibration-trials 50 --closed-trials 50 --open-trials 100 --seed 20260815 \
  > "$run_root/logs/prepare_b16_10k.log" 2>&1

for block_size in 8 32; do
  "$python_bin" "$repo_root/scripts/85_kvcloak_prepare_trials.py" \
    --tokens "$tokens32" --output-dir "$run_root/data/b${block_size}_10k" \
    --block-size "$block_size" --bank-sizes 10000 \
    --calibration-trials 50 --closed-trials 50 --open-trials 100 \
    --seed "$((20260815 + block_size))" \
    > "$run_root/logs/prepare_b${block_size}_10k.log" 2>&1
done

for layer in 4 8; do
  "$python_bin" "$repo_root/scripts/84_kvcloak_gpt2_kv_cache.py" \
    --hidden-manifest "$hidden32" --model "$model" \
    --output-dir "$run_root/cache" --tag "gpt2_layer${layer}_heads0_5_11" \
    --layer "$layer" --heads 0 5 11 --batch-size 256 --threads "$threads" \
    > "$run_root/logs/cache_layer${layer}.log" 2>&1
done

cache8="$run_root/cache/gpt2_layer8_heads0_5_11_manifest.json"
cache4="$run_root/cache/gpt2_layer4_heads0_5_11_manifest.json"

"$python_bin" "$repo_root/scripts/86_kvcloak_codebook_evaluate.py" \
  --cache-manifest "$cache8" --trials "$run_root/data/b16_full/trials.json" \
  --official-repo "$official" --repo-root "$repo_root" \
  --output-dir "$run_root/evaluation_main" --head-position 0 \
  --kv-types key value --conditions official_reuse --precision bfloat16 \
  --threads "$threads" --seed 20260814 \
  > "$run_root/logs/evaluate_main.log" 2>&1

additional=()

# Layer/head robustness at the official b=16 and bfloat16 setting.
for layer in 4 8; do
  cache_var="$run_root/cache/gpt2_layer${layer}_heads0_5_11_manifest.json"
  for head_position in 0 1 2; do
    if [[ "$layer" == "8" && "$head_position" == "0" ]]; then
      continue
    fi
    out="$run_root/robustness/layer${layer}_headpos${head_position}_b16_bf16"
    mkdir -p "$out"
    "$python_bin" "$repo_root/scripts/86_kvcloak_codebook_evaluate.py" \
      --cache-manifest "$cache_var" --trials "$run_root/data/b16_10k/trials.json" \
      --official-repo "$official" --repo-root "$repo_root" --output-dir "$out" \
      --head-position "$head_position" --kv-types key value --conditions official_reuse \
      --precision bfloat16 --threads "$threads" --seed "$((20260820 + layer * 10 + head_position))" \
      > "$run_root/logs/layer${layer}_headpos${head_position}.log" 2>&1
    additional+=("$out")
  done
done

# Block-size and precision robustness on a fixed real layer/head.
for block_size in 8 32; do
  for precision in float32 float16 bfloat16; do
    out="$run_root/robustness/layer8_head0_b${block_size}_${precision}"
    mkdir -p "$out"
    "$python_bin" "$repo_root/scripts/86_kvcloak_codebook_evaluate.py" \
      --cache-manifest "$cache8" --trials "$run_root/data/b${block_size}_10k/trials.json" \
      --official-repo "$official" --repo-root "$repo_root" --output-dir "$out" \
      --head-position 0 --kv-types key value --conditions official_reuse \
      --precision "$precision" --threads "$threads" --seed "$((20260900 + block_size))" \
      > "$run_root/logs/b${block_size}_${precision}.log" 2>&1
    additional+=("$out")
  done
done

# Exact key-lifetime boundary. These conditions deliberately retain failures.
boundary="$run_root/key_refresh_boundary"
"$python_bin" "$repo_root/scripts/86_kvcloak_codebook_evaluate.py" \
  --cache-manifest "$cache8" --trials "$run_root/data/b16_10k/trials.json" \
  --official-repo "$official" --repo-root "$repo_root" --output-dir "$boundary" \
  --head-position 0 --kv-types key value \
  --conditions official_reuse refresh_left refresh_a_row refresh_a_vector refresh_m fresh_all \
  --precision bfloat16 --threads "$threads" --seed 20260916 \
  > "$run_root/logs/key_refresh_boundary.log" 2>&1
additional+=("$boundary")

# Real b=64 result from the same deterministic MS MARCO source archive.
collection="$($python_bin -c 'import json; print(json.load(open("'"$gelo_root/data/manifest.json"'"))["collection"])')"
"$python_bin" "$repo_root/scripts/69_gelo_msmarco_prepare.py" \
  --collection "$collection" --tokenizer "$model" --output-dir "$run_root/data/msmarco64" \
  --count 120000 --reservoir-size 600000 --seq-len 64 --seed 20260802 \
  --trials 1 --source-count 1 \
  > "$run_root/logs/prepare_msmarco64.log" 2>&1
"$python_bin" "$repo_root/scripts/85_kvcloak_prepare_trials.py" \
  --tokens "$run_root/data/msmarco64/tokens.npy" --output-dir "$run_root/data/b64_10k" \
  --block-size 64 --bank-sizes 10000 --calibration-trials 50 --closed-trials 50 \
  --open-trials 100 --seed 20260964 \
  > "$run_root/logs/prepare_b64_10k.log" 2>&1
"$python_bin" "$repo_root/scripts/84_kvcloak_gpt2_kv_cache.py" \
  --tokens "$run_root/data/msmarco64/tokens.npy" --model "$model" \
  --output-dir "$run_root/cache" --tag "gpt2_layer0_heads0_5_11_s64" \
  --layer 0 --heads 0 5 11 --batch-size 256 --threads "$threads" \
  > "$run_root/logs/cache_layer0_s64.log" 2>&1
out64="$run_root/robustness/layer0_head0_b64_bfloat16"
"$python_bin" "$repo_root/scripts/86_kvcloak_codebook_evaluate.py" \
  --cache-manifest "$run_root/cache/gpt2_layer0_heads0_5_11_s64_manifest.json" \
  --trials "$run_root/data/b64_10k/trials.json" --official-repo "$official" \
  --repo-root "$repo_root" --output-dir "$out64" --head-position 0 \
  --kv-types key value --conditions official_reuse --precision bfloat16 \
  --threads "$threads" --seed 20261064 \
  > "$run_root/logs/b64_bfloat16.log" 2>&1
additional+=("$out64")

"$python_bin" "$repo_root/scripts/87_finalize_kvcloak_campaign.py" \
  --repo-root "$repo_root" --campaign-root "$run_root" \
  --smoke "$run_root/smoke/campaign_complete.json" \
  --trials "$run_root/data/b16_full/trials.json" \
  --main-evaluation "$run_root/evaluation_main" \
  --additional-evaluations "${additional[@]}" \
  > "$run_root/logs/finalize.log" 2>&1

#!/usr/bin/env bash
set -euo pipefail

repo_root="/home/ubuntu/research-vault/projects/P050/ndss-2027/artifact"
project_root="/home/ubuntu/research-vault/projects/P050/ndss-2027"
python_bin="/home/ubuntu/research-vault/venvs/p050-ndss-gelo-v1/bin/python"
official="$project_root/third_party/kvcloak"
gelo_root="$repo_root/outputs/gelo_msmarco_100k_dedup_20260802"
baseline_root="$repo_root/outputs/kvcloak_msmarco_100k_20260803"
run_root="$repo_root/outputs/kvcloak_qwen_crossmodel_20260804"
model="/home/ubuntu/research-vault/cache/p050-kvcloak-qwen/Qwen2.5-1.5B"
threads="${1:-20}"
pinned="6b40f36edb2f337557543e7e60b10022308883d4"

mkdir -p "$run_root/logs" "$run_root/data" "$run_root/cache"
while tmux has-session -t '=p050_kvcloak_qwen_download' 2>/dev/null; do
  sleep 30
done
if [[ ! -s "$model/config.json" ]] || [[ ! -s "$model/model.safetensors" ]]; then
  echo "Qwen2.5-1.5B download is incomplete" >&2
  exit 1
fi
if [[ "$(git -C "$official" rev-parse HEAD)" != "$pinned" ]]; then
  echo "KV-Cloak official checkout mismatch" >&2
  exit 1
fi

export OPENBLAS_NUM_THREADS="$threads"
export OMP_NUM_THREADS="$threads"
export MKL_NUM_THREADS="$threads"
collection="$($python_bin -c 'import json; print(json.load(open("'"$gelo_root/data/manifest.json"'"))["collection"])')"

"$python_bin" "$repo_root/scripts/69_gelo_msmarco_prepare.py" \
  --collection "$collection" --tokenizer "$model" \
  --output-dir "$run_root/data/msmarco128" --count 120000 \
  --reservoir-size 600000 --seq-len 128 --seed 20260804 \
  --trials 1 --source-count 1 \
  > "$run_root/logs/prepare_msmarco128.log" 2>&1
tokens="$run_root/data/msmarco128/tokens.npy"

"$python_bin" "$repo_root/scripts/85_kvcloak_prepare_trials.py" \
  --tokens "$tokens" --output-dir "$run_root/data/b64_full" \
  --block-size 64 --bank-sizes 10000 50000 100000 \
  --calibration-trials 100 --closed-trials 100 --open-trials 300 \
  --seed 20261064 > "$run_root/logs/prepare_b64_full.log" 2>&1

"$python_bin" "$repo_root/scripts/85_kvcloak_prepare_trials.py" \
  --tokens "$tokens" --output-dir "$run_root/data/b64_head1" \
  --block-size 64 --bank-sizes 10000 \
  --calibration-trials 50 --closed-trials 50 --open-trials 100 \
  --seed 20262064 > "$run_root/logs/prepare_b64_head1.log" 2>&1

"$python_bin" "$repo_root/scripts/85_kvcloak_prepare_trials.py" \
  --tokens "$tokens" --output-dir "$run_root/data/b128_boundary" \
  --block-size 128 --bank-sizes 10000 \
  --calibration-trials 100 --closed-trials 100 --open-trials 300 \
  --seed 20261128 > "$run_root/logs/prepare_b128_boundary.log" 2>&1

"$python_bin" "$repo_root/scripts/90_prepare_kvcloak_cache_indices.py" \
  --trials "$run_root/data/b64_full/trials.json" \
  "$run_root/data/b64_head1/trials.json" \
  "$run_root/data/b128_boundary/trials.json" \
  --output "$run_root/data/cache_indices.npy" \
  > "$run_root/logs/prepare_cache_indices.log" 2>&1

"$python_bin" "$repo_root/scripts/89_kvcloak_qwen_kv_cache.py" \
  --tokens "$tokens" --model "$model" --indices "$run_root/data/cache_indices.npy" \
  --output-dir "$run_root/cache" --tag qwen2_5_1_5b_layer0_kvheads0_1_s128 \
  --layer 0 --heads 0 1 --batch-size 128 --threads "$threads" \
  > "$run_root/logs/cache_qwen_layer0.log" 2>&1
cache="$run_root/cache/qwen2_5_1_5b_layer0_kvheads0_1_s128_manifest.json"

"$python_bin" "$repo_root/scripts/86_kvcloak_codebook_evaluate.py" \
  --cache-manifest "$cache" --trials "$run_root/data/b64_full/trials.json" \
  --official-repo "$official" --repo-root "$repo_root" \
  --output-dir "$run_root/evaluation_b64_head0" --head-position 0 \
  --kv-types key value --conditions official_reuse --precision bfloat16 \
  --threads "$threads" --seed 20261264 \
  > "$run_root/logs/evaluate_b64_head0.log" 2>&1

"$python_bin" "$repo_root/scripts/86_kvcloak_codebook_evaluate.py" \
  --cache-manifest "$cache" --trials "$run_root/data/b64_head1/trials.json" \
  --official-repo "$official" --repo-root "$repo_root" \
  --output-dir "$run_root/evaluation_b64_head1" --head-position 1 \
  --kv-types key value --conditions official_reuse --precision bfloat16 \
  --threads "$threads" --seed 20262264 \
  > "$run_root/logs/evaluate_b64_head1.log" 2>&1

"$python_bin" "$repo_root/scripts/86_kvcloak_codebook_evaluate.py" \
  --cache-manifest "$cache" --trials "$run_root/data/b128_boundary/trials.json" \
  --official-repo "$official" --repo-root "$repo_root" \
  --output-dir "$run_root/evaluation_b128_boundary" --head-position 0 \
  --kv-types key value --conditions official_reuse --precision bfloat16 \
  --threads "$threads" --seed 20261328 \
  > "$run_root/logs/evaluate_b128_boundary.log" 2>&1

"$python_bin" "$repo_root/scripts/91_finalize_kvcloak_qwen_crossmodel.py" \
  --repo-root "$repo_root" --campaign-root "$run_root" \
  --smoke "$baseline_root/smoke/campaign_complete.json" \
  --cache-manifest "$cache" \
  --indices-manifest "$run_root/data/cache_indices.manifest.json" \
  --trials-b64 "$run_root/data/b64_full/trials.json" \
  --main-b64 "$run_root/evaluation_b64_head0" \
  --trials-head1 "$run_root/data/b64_head1/trials.json" \
  --head1-b64 "$run_root/evaluation_b64_head1" \
  --trials-b128 "$run_root/data/b128_boundary/trials.json" \
  --boundary-b128 "$run_root/evaluation_b128_boundary" \
  > "$run_root/logs/finalize.log" 2>&1

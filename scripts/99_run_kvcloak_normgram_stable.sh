#!/usr/bin/env bash
set -euo pipefail

repo=/home/ubuntu/research-vault/projects/P050/ndss-2027/artifact
project=/home/ubuntu/research-vault/projects/P050/ndss-2027
python=/home/ubuntu/research-vault/venvs/p050-ndss-gelo-v1/bin/python
official="$project/third_party/kvcloak"
qwen=outputs/kvcloak_qwen_crossmodel_20260804
gpt2=outputs/kvcloak_msmarco_100k_20260803
root=outputs/kvcloak_normgram_stable_20260804
threads="${1:-20}"

cd "$repo"
mkdir -p "$root/logs"

"$python" scripts/98_kvcloak_normgram_stable.py \
  --repo-root "$repo" --official-repo "$official" \
  --cache-manifest "$qwen/cache/qwen2_5_1_5b_layer0_kvheads0_1_s128_manifest.json" \
  --trials "$qwen/data/b128_boundary/trials.json" --output-dir "$root/qwen_b128" \
  --head-position 0 --precision bfloat16 --evaluation-seed 20261328 \
  --conditions official_reuse refresh_left --pair-count 512 --threads "$threads" \
  > "$root/logs/qwen_b128.log" 2>&1

"$python" scripts/98_kvcloak_normgram_stable.py \
  --repo-root "$repo" --official-repo "$official" \
  --cache-manifest "$gpt2/cache/gpt2_layer0_heads0_5_11_s64_manifest.json" \
  --trials "$gpt2/data/b64_10k/trials.json" --output-dir "$root/gpt2_b64" \
  --head-position 0 --precision bfloat16 --evaluation-seed 20261064 \
  --conditions official_reuse refresh_left --pair-count 512 --threads "$threads" \
  > "$root/logs/gpt2_b64.log" 2>&1

"$python" scripts/98_kvcloak_normgram_stable.py \
  --repo-root "$repo" --official-repo "$official" \
  --cache-manifest "$qwen/cache/qwen2_5_1_5b_layer0_kvheads0_1_s128_manifest.json" \
  --trials "$qwen/data/b64_head1/trials.json" --output-dir "$root/qwen_b64" \
  --head-position 0 --precision bfloat16 --evaluation-seed 20261264 \
  --conditions official_reuse refresh_left --pair-count 512 --threads "$threads" \
  > "$root/logs/qwen_b64.log" 2>&1


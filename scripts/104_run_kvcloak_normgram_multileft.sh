#!/usr/bin/env bash
set -euo pipefail

repo=/home/ubuntu/research-vault/projects/P050/ndss-2027/artifact
project=/home/ubuntu/research-vault/projects/P050/ndss-2027
python=/home/ubuntu/research-vault/venvs/p050-ndss-gelo-v1/bin/python
official="$project/third_party/kvcloak"
qwen=outputs/kvcloak_qwen_crossmodel_20260804
gpt2=outputs/kvcloak_msmarco_100k_20260803
holdout=outputs/kvcloak_normgram_confirmatory_20260804
root=outputs/kvcloak_normgram_multileft_20260804
threads="${1:-20}"
metric_sha="$2"
left_seeds=(20263001 20263002 20263003 20263004 20263005)

cd "$repo"
mkdir -p "$root/logs"

"$python" scripts/103_kvcloak_normgram_multileft.py \
  --repo-root "$repo" --official-repo "$official" \
  --cache-manifest "$qwen/cache/qwen2_5_1_5b_layer0_kvheads0_1_s128_manifest.json" \
  --trials "$holdout/data/qwen_b128/trials.json" --output-dir "$root/qwen_b128" \
  --head-position 0 --precision bfloat16 --evaluation-seed 20261328 \
  --left-seeds "${left_seeds[@]}" --pair-count 512 --threads "$threads" \
  --frozen-metric-code "$repo/scripts/98_kvcloak_normgram_stable.py" \
  --frozen-metric-sha256 "$metric_sha" \
  > "$root/logs/qwen_b128.log" 2>&1

"$python" scripts/103_kvcloak_normgram_multileft.py \
  --repo-root "$repo" --official-repo "$official" \
  --cache-manifest "$gpt2/cache/gpt2_layer0_heads0_5_11_s64_manifest.json" \
  --trials "$holdout/data/gpt2_b64/trials.json" --output-dir "$root/gpt2_b64" \
  --head-position 0 --precision bfloat16 --evaluation-seed 20261064 \
  --left-seeds "${left_seeds[@]}" --pair-count 512 --threads "$threads" \
  --frozen-metric-code "$repo/scripts/98_kvcloak_normgram_stable.py" \
  --frozen-metric-sha256 "$metric_sha" \
  > "$root/logs/gpt2_b64.log" 2>&1


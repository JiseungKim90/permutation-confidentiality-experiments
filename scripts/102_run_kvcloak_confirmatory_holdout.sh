#!/usr/bin/env bash
set -euo pipefail

repo=/home/ubuntu/research-vault/projects/P050/ndss-2027/artifact
project=/home/ubuntu/research-vault/projects/P050/ndss-2027
python=/home/ubuntu/research-vault/venvs/p050-ndss-gelo-v1/bin/python
official="$project/third_party/kvcloak"
qwen=outputs/kvcloak_qwen_crossmodel_20260804
gpt2=outputs/kvcloak_msmarco_100k_20260803
root=outputs/kvcloak_normgram_confirmatory_20260804
threads="${1:-20}"
expected_metric_sha="$2"

cd "$repo"
actual_metric_sha="$(sha256sum scripts/98_kvcloak_normgram_stable.py | awk '{print $1}')"
test "$actual_metric_sha" = "$expected_metric_sha"
mkdir -p "$root/logs"

qwen_tokens="$("$python" -c 'import json,sys; print(json.load(open(sys.argv[1]))["tokens"])' "$qwen/data/b128_boundary/trials.json")"
gpt2_tokens="$("$python" -c 'import json,sys; print(json.load(open(sys.argv[1]))["tokens"])' "$gpt2/data/b64_10k/trials.json")"

"$python" scripts/101_prepare_kvcloak_confirmatory_trials.py \
  --tokens "$qwen_tokens" \
  --exclude-trials "$qwen/data/b128_boundary/trials.json" "$qwen/data/b64_head1/trials.json" \
  --allowed-indices "$qwen/data/cache_indices.npy" \
  --output-dir "$root/data/qwen_b128" --block-size 128 --seed 20261428 \
  > "$root/logs/qwen_trials.log" 2>&1

"$python" scripts/101_prepare_kvcloak_confirmatory_trials.py \
  --tokens "$gpt2_tokens" --exclude-trials "$gpt2/data/b64_10k/trials.json" \
  --output-dir "$root/data/gpt2_b64" --block-size 64 --seed 20261164 \
  > "$root/logs/gpt2_trials.log" 2>&1

"$python" scripts/98_kvcloak_normgram_stable.py \
  --repo-root "$repo" --official-repo "$official" \
  --cache-manifest "$qwen/cache/qwen2_5_1_5b_layer0_kvheads0_1_s128_manifest.json" \
  --trials "$root/data/qwen_b128/trials.json" --output-dir "$root/qwen_b128" \
  --head-position 0 --precision bfloat16 --evaluation-seed 20261328 \
  --conditions official_reuse refresh_left --pair-count 512 --threads "$threads" \
  > "$root/logs/qwen_b128.log" 2>&1

"$python" scripts/98_kvcloak_normgram_stable.py \
  --repo-root "$repo" --official-repo "$official" \
  --cache-manifest "$gpt2/cache/gpt2_layer0_heads0_5_11_s64_manifest.json" \
  --trials "$root/data/gpt2_b64/trials.json" --output-dir "$root/gpt2_b64" \
  --head-position 0 --precision bfloat16 --evaluation-seed 20261064 \
  --conditions official_reuse refresh_left --pair-count 512 --threads "$threads" \
  > "$root/logs/gpt2_b64.log" 2>&1

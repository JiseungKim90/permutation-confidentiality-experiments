#!/usr/bin/env bash
set -euo pipefail

repo_root="/home/ubuntu/research-vault/projects/P050/ndss-2027/artifact"
run_root="$repo_root/outputs/gelo_msmarco_100k_dedup_20260802"
python_bin="/home/ubuntu/research-vault/venvs/p050-ndss-gelo-v1/bin/python"

while tmux has-session -t p050_gelo_full 2>/dev/null; do
  sleep 30
done

exec "$python_bin" "$repo_root/scripts/74_finalize_gelo_msmarco_campaign.py" \
  --run-root "$run_root" \
  --repo-root "$repo_root"

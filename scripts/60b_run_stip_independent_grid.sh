#!/usr/bin/env bash
set -euo pipefail

ROOT="${P050_ROOT:-/home/user/p050_journal_targets}"
SEEDS="${SEEDS:-20260803 20260804}"
FRACTIONS="${FRACTIONS:-0.25}"
STEPS="${STEPS:-200}"
SELECTION="${SELECTION:-frequency}"

SCIFACT="/home/user/.cache/huggingface/datasets/BeIR___scifact/corpus/0.0.0/b3b5335604bf5ee3c4447671af975ea25143d4f5/scifact-corpus.arrow"
NFCORPUS="/home/user/.cache/huggingface/datasets/BeIR___nfcorpus/corpus/0.0.0/b5026a0e96e8a7ac4f95f482a596389289d46269/nfcorpus-corpus.arrow"
mkdir -p "$ROOT/results" "$ROOT/checkpoints" "$ROOT/logs"

for seed in $SEEDS; do
  for fraction in $FRACTIONS; do
    fraction_tag="${fraction/./}"
    tag="stip_independent_embedding_${SELECTION}_f${fraction_tag}_seed${seed}"
    result="$ROOT/results/${tag}.json"
    checkpoint="$ROOT/checkpoints/${tag}.pt"
    log="$ROOT/logs/${tag}.log"
    if [[ -s "$result" && -s "$checkpoint" ]]; then
      echo "SKIP $tag"
      continue
    fi
    echo "START $tag $(date -Iseconds)"
    /usr/bin/time -v python3 "$ROOT/experiments/60_stip_independent_embedding_training.py" \
      --model gpt2 \
      --cache-dir "$ROOT/models" \
      --arrow "$SCIFACT" \
      --arrow "$NFCORPUS" \
      --output "$result" \
      --checkpoint "$checkpoint" \
      --selection "$SELECTION" \
      --train-fraction "$fraction" \
      --steps "$STEPS" \
      --batch-size 2 \
      --sequence-length 64 \
      --learning-rate 0.0005 \
      --eval-batches 10 \
      --max-tokens 500000 \
      --orbit-batch-size 512 \
      --threads 16 \
      --log-every 20 \
      --seed "$seed" \
      --device cpu > "$log" 2>&1
    python3 - "$result" <<'PY'
import json
import sys
record = json.load(open(sys.argv[1], encoding="utf-8"))
selected = record["orbit_recovery"]["top1"]["selected"]["rate"]
if record["changed_unselected_rows"] != 0:
    raise SystemExit("unselected row changed")
print(
    "DONE",
    sys.argv[1],
    "selected_top1=", selected,
    "certificate=", record["orbit_recovery"]["half_margin_certificate"]["selected"]["rate"],
    "loss=", record["validation_loss_before"], "->", record["validation_loss_after"],
)
PY
  done
done

echo "GRID_COMPLETE $(date -Iseconds)"
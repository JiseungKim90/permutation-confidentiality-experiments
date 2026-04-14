#!/bin/bash
# One-command entrypoint for the full submission artifact.
# Usage: cd experiments && bash run_submission_artifact.sh

set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

bash run_all.sh
bash run_kd_submission.sh

echo ""
echo "============================================"
echo "FULL SUBMISSION ARTIFACT COMPLETE"
echo "Core logs: outputs/logs/"
echo "Core summaries: outputs/*.json"
echo "============================================"

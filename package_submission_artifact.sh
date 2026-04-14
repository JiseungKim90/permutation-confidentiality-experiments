#!/bin/bash
# Build a tar.gz archive for the clean canonical submission artifact.
# Usage: cd experiments && bash package_submission_artifact.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

bash prepare_submission_artifact.sh

cd submission_artifact
bash validate_submission_artifact.sh
cd ..

ARCHIVE_NAME="submission_artifact.tar.gz"
rm -f "$ARCHIVE_NAME"
tar czf "$ARCHIVE_NAME" submission_artifact

echo "Created $ARCHIVE_NAME"

#!/usr/bin/env bash
# Full pipeline: timing gate -> tuning (seed 100) -> final (seeds 0-4) -> compressed (5/15 epochs).
set -euo pipefail
cd "$(dirname "$0")"
for mode in timing tune final compressed; do
  echo "=== mode: $mode ==="
  .venv/bin/python method.py "$mode"
done
echo "=== analyze ==="
.venv/bin/python analyze.py
echo "PIPELINE_DONE"

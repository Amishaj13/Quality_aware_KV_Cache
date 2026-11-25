#!/usr/bin/env bash
# KIVI (4-bit, 2-bit) and SnapKV (1024, 2048 prompt tokens kept) on custom + LongCodeU +
# position, then the comparison report. Thin wrapper around scripts/run_methods.sh.
#
#   bash scripts/run_kivi_snapkv.sh                 # everything (~3-4 h on an L4 / A10G)
#   METHODS="kivi4 snapkv:1024" bash scripts/run_kivi_snapkv.sh
set -euo pipefail
cd "$(dirname "$0")/.."
METHODS=${METHODS:-"kivi4 kivi2 snapkv:1024 snapkv:2048"} LOG=results/raw_logs/kivi_snapkv.log \
  bash scripts/run_methods.sh

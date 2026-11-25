#!/usr/bin/env bash
# CPU offloading only: self-test, then the 30 short custom tasks, then a check that the
# answers equal the full cache's; only if that passes, LongCodeU and the position test.
# About 1.5 h on one A10G. Safe to restart: finished samples are skipped.
#
#   bash scripts/run_offload.sh
set -euo pipefail
cd "$(dirname "$0")/.."
LOG=results/raw_logs/offload.log
mkdir -p results/raw_logs

echo "=== self-test (GPU offloading tests included) ===" | tee -a "$LOG"
python -m pytest -q tests/test_cache_methods.py 2>&1 | tee -a "$LOG"

echo "=== offload / custom ($(date '+%H:%M')) ===" | tee -a "$LOG"
python main.py run --cache-method offload --resume 2>&1 | tee -a "$LOG"
python scripts/check_same_answers.py baseline_001 baseline_001_offload 2>&1 | tee -a "$LOG"

WORKLOADS="longcodeu position" METHODS="offload" LOG="$LOG" bash scripts/run_methods.sh
python scripts/check_same_answers.py longcodeu_8k_full longcodeu_8k_offload 2>&1 | tee -a "$LOG" || true

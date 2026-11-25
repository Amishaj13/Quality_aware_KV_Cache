#!/usr/bin/env bash
# StreamingLLM (1024 and 2048 tokens kept, same budgets as SnapKV) and CPU offloading of
# the full cache, on custom + LongCodeU + position; then the report with every method.
# About 3-4 h on one A10G (ml.g5.xlarge). Safe to restart: finished samples are skipped.
#
#   bash scripts/run_streaming_offload.sh
set -euo pipefail
cd "$(dirname "$0")/.."

echo "=== quick self-test of the cache methods (includes the GPU offloading test) ==="
python -m pytest -q tests/test_cache_methods.py

METHODS=${METHODS:-"streamingllm:1024 streamingllm:2048 offload"} \
LOG=results/raw_logs/streaming_offload.log \
  bash scripts/run_methods.sh

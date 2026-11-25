#!/usr/bin/env bash
# Run KV-cache methods on the same three workloads as the full-cache baseline, then
# rebuild the comparison report (all methods found in results/, old ones included).
#
#   METHODS="streamingllm:1024 streamingllm:2048 offload" bash scripts/run_methods.sh
#   METHODS="kivi4 snapkv:1024" WORKLOADS="longcodeu" bash scripts/run_methods.sh
#   DEPTHS="0 0.5 1" bash scripts/run_methods.sh        # shorter position sweep
#
# A method spec is a cache method, optionally with a budget after a colon:
#   kivi4 | kivi2 | snapkv:<prompt tokens kept> | streamingllm:<tokens kept> | offload | full
# Every run uses --resume, so after an interruption just start the script again.
# Needs the full-cache runs (baseline_001, longcodeu_8k_full, position_full_d*) in results/.
set -euo pipefail
cd "$(dirname "$0")/.."

METHODS=${METHODS:?set METHODS, e.g. METHODS=\"streamingllm:1024 offload\"}
WORKLOADS=${WORKLOADS:-"custom longcodeu position"}
DEPTHS=${DEPTHS:-"0 0.25 0.5 0.75 1"}
LOG=${LOG:-results/raw_logs/run_methods.log}
mkdir -p results/raw_logs

for spec in $METHODS; do
  method=${spec%%:*}
  args=(--cache-method "$method")
  if [[ "$spec" == *:* ]]; then
    case $method in
      snapkv)       args+=(--snapkv-budget "${spec#*:}") ;;
      streamingllm) args+=(--streaming-budget "${spec#*:}") ;;
      *) echo "method $method takes no budget ($spec)"; exit 2 ;;
    esac
  fi
  for wl in $WORKLOADS; do
    echo "=== $spec / $wl ($(date '+%H:%M')) ==="
    case $wl in
      custom)    python main.py run "${args[@]}" --resume ;;
      longcodeu) python main.py run --config configs/experiment_longcodeu.yaml "${args[@]}" --resume ;;
      position)  python main.py position "${args[@]}" --depths $DEPTHS --resume ;;
      *) echo "unknown workload $wl"; exit 2 ;;
    esac
  done
done 2>&1 | tee -a "$LOG"

python main.py compare
echo "Report: results/comparison/methods_comparison.md"

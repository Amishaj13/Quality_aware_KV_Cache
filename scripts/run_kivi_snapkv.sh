#!/usr/bin/env bash
# Run KIVI (4-bit, 2-bit) and SnapKV (1024, 2048 prompt tokens kept) on the same three
# workloads as the full-cache baseline, then build the comparison report.
#
#   bash scripts/run_kivi_snapkv.sh                 # everything (~3-4 h on an L4 / A10G)
#   METHODS="kivi4 snapkv" bash scripts/run_kivi_snapkv.sh
#   DEPTHS="0 0.5 1" bash scripts/run_kivi_snapkv.sh  # shorter position sweep
#   WORKLOADS="longcodeu position" bash scripts/run_kivi_snapkv.sh
#
# Every run uses --resume, so after an interruption just start the script again.
# Needs the full-cache runs (baseline_001, longcodeu_8k_full, position_full_d*) in results/.
set -euo pipefail
cd "$(dirname "$0")/.."

METHODS=${METHODS:-"kivi4 kivi2 snapkv:1024 snapkv:2048"}
WORKLOADS=${WORKLOADS:-"custom longcodeu position"}
DEPTHS=${DEPTHS:-"0 0.25 0.5 0.75 1"}
mkdir -p results/raw_logs

for spec in $METHODS; do
  method=${spec%%:*}
  args=(--cache-method "$method")
  if [[ "$spec" == *:* ]]; then args+=(--snapkv-budget "${spec#*:}"); fi
  for wl in $WORKLOADS; do
    echo "=== $spec / $wl ==="
    case $wl in
      custom)    python main.py run "${args[@]}" --resume ;;
      longcodeu) python main.py run --config configs/experiment_longcodeu.yaml "${args[@]}" --resume ;;
      position)  python main.py position "${args[@]}" --depths $DEPTHS --resume ;;
    esac
  done
done 2>&1 | tee -a results/raw_logs/kivi_snapkv.log

python main.py compare
echo "Report: results/comparison/methods_comparison.md"

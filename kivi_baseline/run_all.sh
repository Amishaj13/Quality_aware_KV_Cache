#!/usr/bin/env bash
# Runs FP16, KIVI 4-bit and KIVI 2-bit back to back, then builds the report.
# Usage:  source <WORKDIR>/env.sh && bash run_all.sh
# Quick smoke test first (~5 min):  QUICK=1 bash run_all.sh
set -uo pipefail
cd "$(dirname "$0")"
OUT=results/results.jsonl
mkdir -p results logs

if [ "${QUICK:-0}" = "1" ]; then
  COMMON="--contexts 4096 --trials 1 --gen-tokens 32 --out results/smoke.jsonl"
  OUT=results/smoke.jsonl
else
  COMMON="--contexts 8192 16384 32768 --batch-sizes ${BATCH_SIZES:-1} --gen-tokens 128 --trials 3 --out $OUT"
fi

python bench_kivi.py --method fp16                       $COMMON 2>&1 | tee logs/fp16.log
python bench_kivi.py --method kivi --k-bits 4 --v-bits 4 $COMMON 2>&1 | tee logs/kivi_k4v4.log
python bench_kivi.py --method kivi --k-bits 2 --v-bits 2 $COMMON 2>&1 | tee logs/kivi_k2v2.log

python make_report.py --inp "$OUT" --outdir results

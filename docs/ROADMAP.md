# Roadmap after the first review (Dec 2026 – Mar 2027)

KIVI, SnapKV, StreamingLLM and CPU offloading are implemented (`cache_methods/kivi.py`, `snapkv.py`, `streamingllm.py`, `offload.py`); the rest is not yet. The code is arranged so each item plugs in without
changing the runner.

## 1. Compression methods (`cache_methods/`)
Each method subclasses `CacheMethod` (`prepare`, `new_cache`, `generate_kwargs`, `describe`)
and is added to `IMPLEMENTED` in `cache_methods/__init__.py`. Until then the config
validator rejects the name, so no run can silently fall back to the full cache.

| Method | Idea | Integration note |
|---|---|---|
| StreamingLLM | attention sinks + recent window | **done**: own cache layer; exact prefill, then sinks + sliding recent window (budget 1024/2048 like SnapKV) |
| CPU offloading | full cache in CPU memory, layer-wise prefetch | **done**: transformers `DynamicCache(offloading=True)`; reports KV bytes on GPU vs CPU |
| H2O | keep heavy-hitter tokens by accumulated attention | needs attention scores: eager attention or a patched sdpa |
| SnapKV | select prompt KV using an observation window | **done**: forward pre-hook recomputes window queries; cache evicts once after prefill |
| PyramidKV | per-layer budgets, larger in lower layers | builds on SnapKV selection |
| KIVI | 2/4-bit asymmetric K/V quantization | **done**: own cache layer (packed uint8 codes, per-channel K / per-token V, 128-token fp16 residual); no custom kernel |
| EvolKV | evolutionary search of layer budgets | runs on top of a budgeted eviction method |

Formulas (`evaluation/comparison.py`, tested):
QR = Q_compressed / Q_full x 100, MR = (M_full - M_compressed) / M_full x 100, CR = M_full / M_compressed.
Q = pass@1 / functional-correctness rate on the same samples; M = measured KV-cache bytes
(peak GPU memory reported separately, since weights dominate it at batch 1).

Base paper: Yuan et al., *KV Cache Compression, But What Must We Give in Return? A Comprehensive
Benchmark of Long Context Capable Approaches* (EMNLP 2024 Findings). It compares KIVI, FlexGen,
StreamingLLM, H2O, InfLLM, LLMLingua-2 and linear-time models on LongBench + Needle-in-a-Haystack.
SnapKV, PyramidKV and EvolKV are not in its main comparison, so they extend it; LongCodeU is a
separate coding benchmark. Prefer original implementations or validated integrations over
re-implementations.

Every result row already carries `cache_method`, `kv_cache_bytes` (measured) and
`kv_cache_bytes_theoretical_full`, so KV reduction = 1 - measured / full, computed
only once a compressed run exists. Quality retention = compressed quality / full-cache
quality x 100 on the same samples.

## 2. Benchmarks (check version, licence and terms before use)
Done: LongCodeU 0~8K subset (`datasets/prepare_longcodeu.py`, Apache-2.0). Next: its 8~16K bucket (fits 24 GB
with the full cache) and 16~32K (borderline; the natural target for compression), then RepoBench, CrossCodeEval, CodeRepoQA, SWE-bench Lite, BugsInPy, TestGenEval,
EvalPlus. Each becomes a loader that emits the same sample schema as
`datasets/sample_tasks.json` (task, context, hidden evaluation fields, group_id for splits).

## 3. Evaluations
- Position sensitivity: the full-cache sweep at 0 / 25 / 50 / 75 / 100 % already exists
  (`scripts/run_position_sweep.py`); rerun it per method and compare the curves.
- Longer contexts: 16K / 32K (Qwen2.5-Coder supports 32K natively; 24 GB GPUs may OOM at 32K in bf16 — record OOM rows).
- Retrieval precision/recall/F1 and cross-file dependency preservation (NetworkX import graph of repo tasks).
- Long-generation stability at 256 / 512 / 1024 new tokens.
- Quality-memory Pareto fronts per workflow.

## 4. Recommendation engine
Rule-based first (lookup over measured Pareto fronts by workflow, context length,
free GPU memory, minimum quality, latency limit). Train a model only once there are
enough reliable measurements.

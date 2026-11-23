# Roadmap after the first review (Dec 2026 – Mar 2027)

Nothing below is implemented yet. The code is arranged so each item plugs in without
changing the runner.

## 1. Compression methods (`cache_methods/`)
Each method subclasses `CacheMethod` (`prepare`, `new_cache`, `generate_kwargs`, `describe`)
and is added to `IMPLEMENTED` in `cache_methods/__init__.py`. Until then the config
validator rejects the name, so no run can silently fall back to the full cache.

| Method | Idea | Integration note |
|---|---|---|
| StreamingLLM | attention sinks + recent window | simplest first: custom `Cache` subclass that evicts the middle |
| H2O | keep heavy-hitter tokens by accumulated attention | needs attention scores: eager attention or a patched sdpa |
| SnapKV | select prompt KV using an observation window | acts once after prefill |
| PyramidKV | per-layer budgets, larger in lower layers | builds on SnapKV selection |
| KIVI | 2/4-bit asymmetric K/V quantization | `kivi_baseline/` already measures it on Llama-3.1-8B (pinned to transformers 4.43). Port to Qwen2 or use HF's quantized cache as a stand-in; record which |
| EvolKV | evolutionary search of layer budgets | runs on top of a budgeted eviction method |

Every result row already carries `cache_method`, `kv_cache_bytes` (measured) and
`kv_cache_bytes_theoretical_full`, so KV reduction = 1 - measured / full, computed
only once a compressed run exists. Quality retention = compressed quality / full-cache
quality x 100 on the same samples.

## 2. Benchmarks (check version, licence and terms before use)
LongCodeU, RepoBench, CrossCodeEval, CodeRepoQA, SWE-bench Lite, BugsInPy, TestGenEval,
EvalPlus. Each becomes a loader that emits the same sample schema as
`datasets/sample_tasks.json` (task, context, hidden evaluation fields, group_id for splits).

## 3. Evaluations
- Position sensitivity: `context_fill.depth` at 0.1 / 0.3 / 0.5 / 0.7 / 0.9 (already supported).
- Longer contexts: 16K / 32K (Qwen2.5-Coder supports 32K natively; 24 GB GPUs may OOM at 32K in bf16 — record OOM rows).
- Retrieval precision/recall/F1 and cross-file dependency preservation (NetworkX import graph of repo tasks).
- Long-generation stability at 256 / 512 / 1024 new tokens.
- Quality-memory Pareto fronts per workflow.

## 4. Recommendation engine
Rule-based first (lookup over measured Pareto fronts by workflow, context length,
free GPU memory, minimum quality, latency limit). Train a model only once there are
enough reliable measurements.

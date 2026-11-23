# Baseline: FP16 KV cache vs KIVI on Llama 3.1 8B (SageMaker)

Goal for this step: show, on the same GPU and the same prompts, how much KV-cache
memory KIVI (2-bit and 4-bit) saves against the standard FP16 cache, what it costs
in speed, and whether the model still recalls information at the beginning, middle
and end of a long coding context.

| File | What it does |
|---|---|
| `setup_env.sh` | One-time setup: conda env, torch 2.4.1, transformers 4.43.1, flash-attn wheel, KIVI clone + CUDA kernel build |
| `bench_kivi.py` | Runs one method (FP16 or KIVI) at 8K / 16K / 32K: KV bytes, peak GPU memory, prefill time, decode speed, and needle recall |
| `run_all.sh` | Runs FP16, KIVI-4bit, KIVI-2bit back to back, then the report |
| `make_report.py` | Writes `results/summary.md` (tables) and PNG charts |

Total time once the environment is ready: about 45 to 75 minutes on one GPU.

---

## 0. Do this first (it can take a while to be approved)

1. **Llama 3.1 licence.** Log in to Hugging Face, open
   <https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct> and accept the licence.
   Approval is usually quick but can take hours.
2. **HF token.** <https://huggingface.co/settings/tokens>, create a *Read* token.
3. **SageMaker GPU quota.** AWS console → Service Quotas → Amazon SageMaker →
   search for the instance below with "for Studio JupyterLab Apps" (or "for notebook
   instance usage" if you use classic notebook instances). If the value is 0,
   request 1. If you already have quota for any instance in the table, use that one.

## 1. Pick the instance

KIVI and flash-attention need an Ampere-or-newer GPU. T4 (`ml.g4dn.*`) and V100
(`ml.p3.*`) will **not** work.

| Instance | GPU | Fits | Recommendation |
|---|---|---|---|
| `ml.g6e.xlarge` | L40S 48 GB | everything, incl. FP16 at 32K and batch 4 | **Best** if you have quota |
| `ml.g5.2xlarge` | A10G 24 GB | FP16 to 16K; FP16 at 32K is borderline (may OOM), KIVI at 32K fits | Good default, widely available |
| `ml.g6.xlarge` / `ml.g6.2xlarge` | L4 24 GB | same as g5, a bit slower | Fine; the benchmark loads weights straight to the GPU, so the 16 GB of RAM on `xlarge` is enough |

Why memory is tight on 24 GB: the FP16 weights alone are about 15 GiB, and the FP16
KV cache of Llama 3.1 8B is 128 KiB per token (32 layers × 8 KV heads × 128 dims ×
K and V × 2 bytes), so 4 GiB at 32K, plus about 3 GiB of prefill activations.
If FP16 runs out of memory at 32K and KIVI does not, that is a result worth showing.

## 2. Create the workspace

**SageMaker Studio (recommended):** Studio → JupyterLab → *Create JupyterLab space* →
instance from the table above, **storage 100 GB** (default 5 GB is too small; if the
max is lower, ask your admin or use a notebook instance) → Run space → Open → Terminal.

**Studio Code Editor space** (use this if your GPU quota is "Studio CodeEditor Apps running on ...", e.g. `ml.g6.xlarge`): Studio → Code Editor → *Create Code Editor space* → pick that instance, storage 100 GB → Run space → Open → Terminal (menu ☰ → Terminal → New Terminal). Upload the zip by dragging it into the Explorer pane. Everything else below is the same.

**Classic notebook instance:** SageMaker → Notebook instances → Create → instance from
the table, volume size **100 GB** → Open JupyterLab → Terminal.

## 3. Set up the environment (~15 to 20 min)

Upload this folder (drag and drop into JupyterLab), then in the terminal:

```bash
cd kivi-baseline
bash setup_env.sh
```

It prints `KIVI import OK` at the end. Then:

```bash
source ~/kivi-work/env.sh            # or ~/SageMaker/kivi-work/env.sh on a notebook instance
huggingface-cli login                # paste the HF token
```

## 4. Run

Smoke test first (about 5 minutes, 4K context, catches setup problems early):

```bash
QUICK=1 bash run_all.sh
```

Full run (8K / 16K / 32K, batch 1):

```bash
bash run_all.sh
```

On a 48 GB GPU you can also show the batching effect, where KIVI's savings matter most:

```bash
BATCH_SIZES="1 4" bash run_all.sh
```

Keep the terminal open, or run it with `nohup bash run_all.sh > run.log 2>&1 &`
and follow it with `tail -f run.log`.

Outputs in `results/`:
`summary.md`, `kv_cache_mib.png`, `peak_gpu_gib.png`, `decode_tok_per_s.png`,
`ttft_s.png`, `needle_recall.png`, and the raw `results.jsonl`.
To re-make the charts for batch 4: `python make_report.py --batch 4`.

## 5. What is measured

| Metric | How |
|---|---|
| KV cache size | Bytes of every tensor in the cache after prefill + generation (KIVI: packed 2/4-bit codes + scales + zero points + the 128-token FP16 window) |
| Compression | Theoretical FP16 KV size ÷ measured cache bytes |
| Peak GPU memory | `torch.cuda.max_memory_allocated()` over prefill + decode (weights + cache + activations) |
| Prefill time (TTFT) | Time to process the prompt and produce the first token |
| Decode speed | ms per token and tokens/s over 128 greedy tokens |
| Positional recall | A constant `DEPLOY_ACCESS_CODE = "NNNNN"` hidden at 10% / 50% / 90% of a synthetic Python file; the model must return the number. 3 trials per position and length |

Fairness: both methods use flash-attention for the prompt and the same hand-written
greedy loop, and only the last position goes through the output layer (full-prompt
logits would take ~16 GB at 32K and swamp the comparison).

KIVI settings (paper defaults): group size 32, residual length 128.

## 6. Expected numbers (sanity check)

Per-token cache cost for Llama 3.1 8B, independent of the GPU:

| Context | FP16 | KIVI 4-bit (≈3.2× smaller) | KIVI 2-bit (≈5.3× smaller) |
|---|---|---|---|
| 8K | 1.00 GiB | ≈0.33 GiB | ≈0.20 GiB |
| 16K | 2.00 GiB | ≈0.64 GiB | ≈0.39 GiB |
| 32K | 4.00 GiB | ≈1.27 GiB | ≈0.77 GiB |

If your measured KV sizes are close to these, the setup is correct.

What to tell your guide about the other metrics:

- **Peak memory** falls by the KV saving only. At batch 1 the 15 GiB of weights
  dominate, so the peak drops by ~3 GiB at 32K, not 5×. The saving grows with batch
  size and context, which is why KIVI reports its throughput gains at large batch.
- **Speed at batch 1** may be the same or slower with KIVI: each step dequantizes the
  cache in a custom kernel, and this research implementation re-lays out the cache
  every step. Its throughput benefit comes from fitting more sequences on the GPU.
- **Recall**: 4-bit should match FP16; 2-bit is where any loss at the middle of the
  context would show up. That is the question the project goes on to study.

## 7. Compatibility notes and workarounds (already handled in the scripts)

- KIVI pins `transformers==4.43.1` and `torch==2.4.1`; its Llama 3 / GQA support was
  added for 4.43. Newer transformers versions break its model class, so the env pins them.
- KIVI only runs with flash-attention (`config.use_flash = True` is required). The
  setup installs the prebuilt flash-attn 2.6.3 wheel; `pip install flash-attn` without
  a wheel would compile for hours.
- Without `config._flash_attn_2_enabled = True`, KIVI's model allocates a dense causal
  mask (2 GiB at 32K) that flash-attention never uses. The benchmark sets it so
  KIVI's peak memory is not overstated.
- KIVI's CUDA kernel (`kivi_gemv`) needs `nvcc`. The setup installs CUDA 12.1 into the
  conda env (matching torch's cu121) and builds only for your GPU's architecture.
- KIVI breaks if the prompt is shorter than the residual window (128 tokens) with GQA
  models like Llama 3.1. All prompts here are far longer, and the warm-up uses 1024 tokens.
- The KIVI repo's `requirements.txt` is an old, conflicting pin list (torch 2.1,
  transformers 4.36). Do not install it; `setup_env.sh` installs what is needed.

## Troubleshooting

| Symptom | Fix |
|---|---|
| `401` / `GatedRepoError` when loading the model | Licence not yet approved, or `huggingface-cli login` not run in this terminal |
| `nvcc not found` / `cuda_runtime.h` missing during kernel build | Re-run `bash setup_env.sh` (it skips finished steps) and check the conda `cuda-toolkit` install succeeded |
| `No space left on device` | Increase space storage to 100 GB; the model is 16 GB and the env ~12 GB |
| `OOM` rows for FP16 at 32K on a 24 GB GPU | Expected; report it, or use `ml.g6e.xlarge` |
| Killed while loading the model | Instance has too little RAM; use a 2xlarge |

## Next steps for the project

The same harness extends naturally: add HF's built-in quantized cache (`quanto` int4)
and FP8 KV via vLLM as further methods, add token-eviction baselines (StreamingLLM,
H2O, SnapKV), and replace the needle test with real coding tasks (e.g. RepoBench or
LongCodeArena) for task-success scores.

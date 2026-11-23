"""FP16 KV cache vs KIVI (2/4-bit) on Llama 3.1 8B: memory, speed and positional recall.

One process benchmarks one method. Results are appended as JSON lines so that
make_report.py can build the comparison table and charts.

Examples:
  python bench_kivi.py --method fp16
  python bench_kivi.py --method kivi --k-bits 2 --v-bits 2
  python bench_kivi.py --method kivi --k-bits 4 --v-bits 4 --contexts 8192 16384

Both methods use flash-attention for prefill and the same manual greedy loop
(prefill once, then one token at a time), so the only difference is the cache.
Only the last position goes through lm_head: computing fp32 logits for a 32K
prompt would by itself need ~16 GB and hide the KV cache difference.
"""
import argparse
import gc
import json
import os
import random
import re
import sys
import time

import torch

MODEL_DEFAULT = "meta-llama/Llama-3.1-8B-Instruct"


# --------------------------------------------------------------------------- utils
def sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def gib(x):
    return round(x / 1024**3, 3)


def mib(x):
    return round(x / 1024**2, 1)


def cache_bytes(obj):
    """Bytes held by every tensor in a cache (HF DynamicCache or KIVI's nested tuples)."""
    if obj is None:
        return 0
    if torch.is_tensor(obj):
        return obj.numel() * obj.element_size()
    if hasattr(obj, "key_cache") and hasattr(obj, "value_cache"):
        return cache_bytes(obj.key_cache) + cache_bytes(obj.value_cache)
    if isinstance(obj, (list, tuple)):
        return sum(cache_bytes(o) for o in obj)
    return 0


def fp16_kv_bytes(cfg, tokens, batch):
    head_dim = cfg.hidden_size // cfg.num_attention_heads
    return 2 * cfg.num_hidden_layers * cfg.num_key_value_heads * head_dim * 2 * tokens * batch


# --------------------------------------------------------------------------- model
def load_model(args):
    from transformers import AutoConfig, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model)
    # Load shards straight onto the GPU so a 16 GB-RAM instance (e.g. ml.g6.xlarge)
    # never holds the whole 16 GB model in CPU memory.
    device_map = args.device if args.device != "cpu" else None
    if args.method == "fp16":
        from transformers import LlamaForCausalLM

        model = LlamaForCausalLM.from_pretrained(
            args.model,
            torch_dtype=torch.float16,
            attn_implementation=args.attn,
            low_cpu_mem_usage=True,
            device_map=device_map,
        )
    else:
        sys.path.insert(0, args.kivi_dir)
        from models.llama_kivi import LlamaForCausalLM_KIVI

        cfg = AutoConfig.from_pretrained(args.model)
        cfg.k_bits = args.k_bits
        cfg.v_bits = args.v_bits
        cfg.group_size = args.group_size
        cfg.residual_length = args.residual_length
        cfg.use_flash = True
        # Without this flag KIVI builds a dense (T x T) causal mask even though prefill
        # uses flash-attention: 2 GiB at 32K, which would unfairly inflate its peak memory.
        cfg._flash_attn_2_enabled = True
        model = LlamaForCausalLM_KIVI.from_pretrained(
            args.model, config=cfg, torch_dtype=torch.float16, low_cpu_mem_usage=True,
            device_map=device_map,
        )
    if device_map is None:
        model = model.to(args.device)
    model.eval()
    return model, tok


@torch.no_grad()
def run_greedy(model, method, input_ids, new_tokens, stop_ids=()):
    """Prefill + greedy decode. Returns generated ids and timing / memory stats."""
    from transformers import DynamicCache

    pkv = DynamicCache() if method == "fp16" else None
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    sync()
    t0 = time.perf_counter()
    out = model.model(input_ids=input_ids, past_key_values=pkv, use_cache=True)
    pkv = out.past_key_values
    nxt = model.lm_head(out.last_hidden_state[:, -1:, :]).argmax(-1)
    sync()
    t1 = time.perf_counter()
    kv_after_prefill = cache_bytes(pkv)
    peak_prefill = torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0

    generated = [nxt]
    for _ in range(new_tokens - 1):
        if stop_ids and nxt[0, 0].item() in stop_ids:
            break
        out = model.model(input_ids=nxt, past_key_values=pkv, use_cache=True)
        pkv = out.past_key_values
        nxt = model.lm_head(out.last_hidden_state[:, -1:, :]).argmax(-1)
        generated.append(nxt)
    sync()
    t2 = time.perf_counter()

    stats = dict(
        prefill_s=t1 - t0,
        decode_s=t2 - t1,
        decode_steps=len(generated) - 1,
        kv_bytes_after_prefill=kv_after_prefill,
        kv_bytes_final=cache_bytes(pkv),
        peak_prefill_bytes=peak_prefill,
        peak_total_bytes=torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0,
    )
    del out, pkv
    return torch.cat(generated, dim=1), stats


def free():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# --------------------------------------------------------------------------- haystack
VERBS = ["load", "parse", "validate", "compute", "merge", "fetch", "render", "update", "resolve", "encode"]
NOUNS = ["config", "user", "order", "invoice", "session", "token", "record", "payload", "cache", "report"]


def code_chunk(rng, i):
    v, n = rng.choice(VERBS), rng.choice(NOUNS)
    a, b = rng.randint(2, 97), rng.randint(2, 97)
    return (
        f"def {v}_{n}_{i}(items, limit={a}):\n"
        f'    """{v.capitalize()} {n} entries and return the filtered result."""\n'
        f"    result = []\n"
        f"    for idx, item in enumerate(items):\n"
        f"        if idx >= limit:\n"
        f"            break\n"
        f"        value = item.get('{n}_id', 0) * {b} % {a + b}\n"
        f"        if value > {a}:\n"
        f"            result.append({{'id': idx, '{n}': value}})\n"
        f"    return result\n\n\n"
    )


def build_code_tokens(tok, n_tokens, seed=0):
    """Synthetic Python file of roughly n_tokens tokens, as a list of chunks."""
    rng = random.Random(seed)
    chunks, total, i = [], 0, 0
    while total < n_tokens:
        c = code_chunk(rng, i)
        chunks.append(c)
        total += len(tok(c, add_special_tokens=False).input_ids)
        i += 1
    return chunks


def needle_prompt(tok, ctx_len, depth, secret, seed):
    """Coding-flavoured needle-in-a-haystack: a constant hidden at `depth` of the file."""
    question = (
        "In the Python file above, what is the exact value assigned to DEPLOY_ACCESS_CODE? "
        "Reply with the number only."
    )
    needle = (
        "# Deployment settings (do not edit)\n"
        f'DEPLOY_ACCESS_CODE = "{secret}"\n\n\n'
    )
    # leave room for the chat template, needle and question
    chunks = build_code_tokens(tok, ctx_len - 200, seed=seed)
    pos = int(len(chunks) * depth)
    chunks.insert(pos, needle)
    code = "".join(chunks)
    msgs = [{"role": "user", "content": f"```python\n{code}```\n\n{question}"}]
    ids = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt")
    return ids


# --------------------------------------------------------------------------- benchmarks
def label_of(args):
    return "FP16" if args.method == "fp16" else f"KIVI-K{args.k_bits}V{args.v_bits}"


def emit(args, row):
    row.update(
        method=label_of(args),
        model=args.model,
        gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        group_size=args.group_size if args.method == "kivi" else None,
        residual_length=args.residual_length if args.method == "kivi" else None,
    )
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "a") as f:
        f.write(json.dumps(row) + "\n")
    print(json.dumps(row), flush=True)


def perf_bench(model, tok, args, weights_bytes):
    chunks = build_code_tokens(tok, max(args.contexts) + 64, seed=1234)
    all_ids = tok("".join(chunks), return_tensors="pt").input_ids
    assert all_ids.shape[1] >= max(args.contexts), "haystack too short"
    for ctx in args.contexts:
        for bs in args.batch_sizes:
            ids = all_ids[:, :ctx].repeat(bs, 1).to(args.device)
            row = dict(kind="perf", context=ctx, batch=bs, gen_tokens=args.gen_tokens)
            try:
                best = None
                for _ in range(args.repeats):
                    free()
                    _, st = run_greedy(model, args.method, ids, args.gen_tokens)
                    if best is None or st["prefill_s"] + st["decode_s"] < best["prefill_s"] + best["decode_s"]:
                        best = st
                st = best
                theo = fp16_kv_bytes(model.config, ctx + args.gen_tokens - 1, bs)
                row.update(
                    status="ok",
                    kv_cache_mib=mib(st["kv_bytes_final"]),
                    kv_cache_prefill_mib=mib(st["kv_bytes_after_prefill"]),
                    fp16_kv_theoretical_mib=mib(theo),
                    kv_compression_x=round(theo / max(st["kv_bytes_final"], 1), 2),
                    weights_gib=gib(weights_bytes),
                    peak_gpu_gib=gib(st["peak_total_bytes"]),
                    peak_prefill_gib=gib(st["peak_prefill_bytes"]),
                    ttft_s=round(st["prefill_s"], 3),
                    decode_ms_per_token=round(1000 * st["decode_s"] / max(st["decode_steps"], 1), 2),
                    decode_tok_per_s=round(bs * st["decode_steps"] / max(st["decode_s"], 1e-9), 2),
                    e2e_tok_per_s=round(bs * (st["decode_steps"] + 1) / (st["prefill_s"] + st["decode_s"]), 2),
                )
            except torch.cuda.OutOfMemoryError:
                row.update(status="OOM")
            except RuntimeError as e:  # some kernels raise plain RuntimeError on OOM
                if "out of memory" not in str(e).lower():
                    raise
                row.update(status="OOM")
            emit(args, row)
            del ids
            free()


def quality_bench(model, tok, args):
    stop_ids = {tok.eos_token_id, *tok.convert_tokens_to_ids(["<|eot_id|>", "<|eom_id|>"])}
    for ctx in args.contexts:
        for depth in args.depths:
            for trial in range(args.trials):
                rng = random.Random(ctx * 1000 + int(depth * 100) * 10 + trial)
                secret = str(rng.randint(10000, 99999))
                ids = needle_prompt(tok, ctx, depth, secret, seed=trial).to(args.device)
                row = dict(kind="needle", context=ctx, depth=depth, trial=trial,
                           prompt_tokens=ids.shape[1], expected=secret)
                try:
                    free()
                    gen, _ = run_greedy(model, args.method, ids, 12, stop_ids=stop_ids)
                    answer = tok.decode(gen[0], skip_special_tokens=True).strip()
                    row.update(status="ok", answer=answer,
                               correct=secret in re.sub(r"\D", "", answer))
                except torch.cuda.OutOfMemoryError:
                    row.update(status="OOM", correct=None)
                emit(args, row)
                del ids
                free()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--method", choices=["fp16", "kivi"], required=True)
    p.add_argument("--model", default=MODEL_DEFAULT)
    default_kivi = os.environ.get("KIVI_DIR") or next(
        (d for d in (os.path.expanduser("~/kivi-work/KIVI"), os.path.expanduser("~/SageMaker/kivi-work/KIVI"))
         if os.path.isdir(d)), "./KIVI")
    p.add_argument("--kivi-dir", default=default_kivi)
    p.add_argument("--k-bits", type=int, default=2, choices=[2, 4])
    p.add_argument("--v-bits", type=int, default=2, choices=[2, 4])
    p.add_argument("--group-size", type=int, default=32)
    p.add_argument("--residual-length", type=int, default=128, help="recent tokens kept in FP16")
    p.add_argument("--contexts", type=int, nargs="+", default=[8192, 16384, 32768])
    p.add_argument("--batch-sizes", type=int, nargs="+", default=[1])
    p.add_argument("--gen-tokens", type=int, default=128)
    p.add_argument("--repeats", type=int, default=1)
    p.add_argument("--depths", type=float, nargs="+", default=[0.1, 0.5, 0.9],
                   help="needle position as a fraction of the context (begin / middle / end)")
    p.add_argument("--trials", type=int, default=3)
    p.add_argument("--skip-perf", action="store_true")
    p.add_argument("--skip-quality", action="store_true")
    p.add_argument("--out", default="results/results.jsonl")
    p.add_argument("--attn", default="flash_attention_2", help=argparse.SUPPRESS)
    p.add_argument("--device", default="cuda", help=argparse.SUPPRESS)
    args = p.parse_args()

    torch.manual_seed(0)
    print(f">> loading {args.model} as {label_of(args)}", flush=True)
    model, tok = load_model(args)
    weights_bytes = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
    print(f">> weights on GPU: {gib(weights_bytes)} GiB", flush=True)

    # warm-up: triton / CUDA kernels compile on first use; must exceed residual_length
    chunks = build_code_tokens(tok, 1024, seed=7)
    warm = tok("".join(chunks), return_tensors="pt").input_ids[:, :1024].to(args.device)
    run_greedy(model, args.method, warm, 8)
    free()

    if not args.skip_perf:
        perf_bench(model, tok, args, weights_bytes)
    if not args.skip_quality:
        quality_bench(model, tok, args)
    print(">> done", flush=True)


if __name__ == "__main__":
    main()

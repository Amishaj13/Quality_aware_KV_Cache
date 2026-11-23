"""GPU check + feasibility estimate for the target model at the configured context.

Estimate = weights + full KV cache for (context_length) tokens + an activation margin.
It is a planning number, not a measurement; the real check is running the model.
  python scripts/check_gpu.py [--model ...] [--context 8192]
"""
import argparse
import json

import _bootstrap  # noqa: F401
import yaml

from utils.paths import resolve


def kv_bytes_per_token(cfg, bytes_per_elem=2):
    head_dim = getattr(cfg, "head_dim", None) or cfg.hidden_size // cfg.num_attention_heads
    return 2 * cfg.num_hidden_layers * (cfg.num_key_value_heads or cfg.num_attention_heads) * head_dim * bytes_per_elem


def estimate(model_name, context, n_params=None):
    from transformers import AutoConfig
    import os

    cfg = AutoConfig.from_pretrained(model_name, token=os.environ.get("HF_TOKEN") or None)
    if n_params is None:
        # rough parameter count from the architecture (embeddings + layers + lm_head)
        h, i, L, v = cfg.hidden_size, cfg.intermediate_size, cfg.num_hidden_layers, cfg.vocab_size
        kvh = cfg.num_key_value_heads
        hd = getattr(cfg, "head_dim", None) or h // cfg.num_attention_heads
        attn = h * h * 2 + h * kvh * hd * 2
        mlp = 3 * h * i
        tied = getattr(cfg, "tie_word_embeddings", False)
        n_params = v * h * (1 if tied else 2) + L * (attn + mlp)
    per_tok = kv_bytes_per_token(cfg)
    gib = 1024**3
    out = {
        "model": model_name,
        "max_position_embeddings": cfg.max_position_embeddings,
        "layers": cfg.num_hidden_layers, "kv_heads": cfg.num_key_value_heads,
        "approx_params_billion": round(n_params / 1e9, 2),
        "weights_bf16_gib": round(n_params * 2 / gib, 2),
        "weights_4bit_gib_approx": round(n_params * 0.55 / gib, 2),
        "kv_cache_per_token_kib_bf16": round(per_tok / 1024, 1),
        "kv_cache_at_context_gib_bf16": round(per_tok * context / gib, 3),
        "context": context,
    }
    # activations during an 8K prefill with sdpa: a few hundred MB to ~1-2 GB; use 2 GiB margin
    out["estimated_peak_bf16_gib"] = round(out["weights_bf16_gib"] + out["kv_cache_at_context_gib_bf16"] + 2.0, 2)
    return out


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen2.5-Coder-7B-Instruct")
    p.add_argument("--context", type=int, default=8192)
    a = p.parse_args(argv)
    import torch

    rules = yaml.safe_load(open(resolve("configs/hardware_config.yaml")))["vram_rules"]
    print("PyTorch:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())
    total_gb = None
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            pr = torch.cuda.get_device_properties(i)
            free, total = torch.cuda.mem_get_info(i)
            print("GPU:", pr.name, f"(compute capability {pr.major}.{pr.minor})")
            print("Total VRAM:", round(total / 1024**3, 2), "GiB")
            print("Free VRAM:", round(free / 1024**3, 2), "GiB")
            print("bf16 supported:", torch.cuda.is_bf16_supported())
            total_gb = total / 1024**3 if total_gb is None else total_gb
    else:
        print("No CUDA GPU. Inference with the 7B model is not feasible here.")

    try:
        est = estimate(a.model, a.context)
        print(json.dumps(est, indent=2))
    except Exception as e:
        est = None
        print(f"could not read model config ({type(e).__name__}: {e}); check internet / HF token")

    if total_gb:
        rule = next(r for r in rules if total_gb * 1.0737 <= r["max_gb"])
        print(f"\nPlanning rule for {total_gb * 1.0737:.0f} GB: {rule['advice']}")
        if est:
            fits = est["estimated_peak_bf16_gib"] < total_gb * 0.93
            print(f"Estimated bf16 peak at {a.context} tokens: {est['estimated_peak_bf16_gib']} GiB -> "
                  f"{'expected to fit' if fits else 'may NOT fit; consider 4-bit weights or a shorter context'} "
                  "(estimate only; verify by running)")


if __name__ == "__main__":
    main()

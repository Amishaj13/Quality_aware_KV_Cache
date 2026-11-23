"""Load the tokenizer and model once. Inference lives in inference/, not here."""
import os
import time

import torch


class ModelLoadError(RuntimeError):
    """Download, authentication, or loading failure, with a readable reason."""


def hf_token():
    """HF token from the environment only (HF_TOKEN or HUGGING_FACE_HUB_TOKEN). Never logged."""
    return os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN") or None


def pick_dtype(name="auto", device="cuda"):
    if name == "auto":
        if device.startswith("cuda") and torch.cuda.is_available():
            return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        return torch.float32
    return {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[name]


def load_tokenizer(model_name, revision="main", trust_remote_code=False):
    from transformers import AutoTokenizer

    try:
        tok = AutoTokenizer.from_pretrained(model_name, revision=revision, token=hf_token(),
                                            trust_remote_code=trust_remote_code)
    except Exception as e:  # network, 401/404, corrupt files
        raise ModelLoadError(f"tokenizer load failed for {model_name}: {type(e).__name__}: {e}") from e
    if not getattr(tok, "chat_template", None):
        raise ModelLoadError(f"{model_name} has no chat template; use an *-Instruct model")
    return tok


def model_context_info(model_or_config):
    """Context-length facts from the model config (advertised, not guaranteed to fit in memory)."""
    cfg = getattr(model_or_config, "config", model_or_config)
    rope = getattr(cfg, "rope_scaling", None)
    return {
        "max_position_embeddings": getattr(cfg, "max_position_embeddings", None),
        "sliding_window": getattr(cfg, "sliding_window", None) if getattr(cfg, "use_sliding_window", False) else None,
        "rope_scaling": rope if isinstance(rope, (dict, type(None))) else str(rope),
        "num_hidden_layers": getattr(cfg, "num_hidden_layers", None),
        "num_attention_heads": getattr(cfg, "num_attention_heads", None),
        "num_key_value_heads": getattr(cfg, "num_key_value_heads", None),
        "head_dim": getattr(cfg, "head_dim", None) or (cfg.hidden_size // cfg.num_attention_heads),
    }


def load_model(cfg):
    """Load tokenizer + model from the merged config. Returns (model, tokenizer, info dict).

    The model is loaded straight onto the GPU (device_map) so a 16 GB-RAM instance such
    as ml.g6.xlarge never holds the whole 15 GB of bf16 weights in CPU memory.
    """
    from transformers import AutoModelForCausalLM

    m = cfg["model"]
    name, revision = m["name"], m.get("revision", "main")
    device_map = m.get("device_map", "cuda")
    if device_map in ("cuda", "auto") and not torch.cuda.is_available():
        raise ModelLoadError("no CUDA GPU detected. Run on a GPU instance, or set model.device_map: cpu "
                             "with a small model for pipeline debugging only.")
    dtype = pick_dtype(m.get("dtype", "auto"), "cuda" if device_map != "cpu" else "cpu")

    tok = load_tokenizer(name, revision, m.get("trust_remote_code", False))

    kwargs = dict(revision=revision, token=hf_token(), trust_remote_code=m.get("trust_remote_code", False),
                  low_cpu_mem_usage=True, attn_implementation=m.get("attn_implementation", "sdpa"))
    # transformers renamed torch_dtype -> dtype in 4.56; both are accepted by recent versions
    kwargs["dtype" if _transformers_at_least("4.56") else "torch_dtype"] = dtype
    if device_map != "cpu":
        kwargs["device_map"] = device_map
    if m.get("quantization", "none") == "4bit":
        try:
            from transformers import BitsAndBytesConfig
            import bitsandbytes  # noqa: F401
        except ImportError as e:
            raise ModelLoadError("model.quantization=4bit needs bitsandbytes: pip install bitsandbytes") from e
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=dtype)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        before = torch.cuda.memory_allocated()
    t0 = time.perf_counter()
    try:
        model = AutoModelForCausalLM.from_pretrained(name, **kwargs)
    except torch.cuda.OutOfMemoryError as e:
        raise ModelLoadError(f"CUDA out of memory while loading {name} in {dtype}. "
                             "Try model.quantization: 4bit or the fallback model.") from e
    except Exception as e:
        raise ModelLoadError(f"model load failed for {name}: {type(e).__name__}: {e}") from e
    load_s = time.perf_counter() - t0
    model.eval()

    n_params = sum(p.numel() for p in model.parameters())
    weight_bytes = sum(p.numel() * p.element_size() for p in model.parameters()) + \
        sum(b.numel() * b.element_size() for b in model.buffers())
    info = {
        "model": name,
        "revision": revision,
        "dtype": str(dtype).replace("torch.", ""),
        "quantization": m.get("quantization", "none"),
        "attn_implementation": m.get("attn_implementation", "sdpa"),
        "device": str(next(model.parameters()).device),
        "num_parameters": n_params,
        "parameter_bytes": weight_bytes,
        "load_seconds": round(load_s, 2),
        "is_primary_model": name == "Qwen/Qwen2.5-Coder-7B-Instruct" and m.get("quantization", "none") == "none",
        **model_context_info(model),
    }
    if torch.cuda.is_available() and device_map != "cpu":
        info["gpu_model_memory_bytes"] = torch.cuda.memory_allocated() - before
        info["gpu_peak_during_load_bytes"] = torch.cuda.max_memory_allocated()
    else:
        info["gpu_model_memory_bytes"] = None
        info["gpu_peak_during_load_bytes"] = None
    return model, tok, info


def _transformers_at_least(version):
    import transformers
    from packaging.version import Version

    return Version(transformers.__version__.split("+")[0]) >= Version(version)

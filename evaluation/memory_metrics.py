"""GPU memory and KV-cache size measurements.

kv_cache_bytes is the size of the tensors actually stored in the cache object that
generate() returns, not an estimate. If the cache cannot be inspected, it is None.
The theoretical full-precision size is reported in a separate, clearly named field.
"""
import torch


def cuda_ok():
    return torch.cuda.is_available()


def _tensor_bytes(t):
    return t.numel() * t.element_size()


def cache_bytes(obj, _seen=None):
    """Bytes of every tensor held by a KV cache.

    Handles transformers DynamicCache (old .key_cache/.value_cache lists and the
    newer .layers[i].keys/.values layout), legacy tuple-of-tuples caches, and KIVI's
    nested tuples. Returns None if nothing recognisable is found.
    """
    if obj is None:
        return None
    _seen = _seen if _seen is not None else set()
    if id(obj) in _seen:
        return 0
    _seen.add(id(obj))
    if torch.is_tensor(obj):
        return _tensor_bytes(obj)
    if isinstance(obj, (list, tuple)):
        vals = [cache_bytes(o, _seen) for o in obj]
        vals = [v for v in vals if v is not None]
        return sum(vals) if vals else None
    total, found = 0, False
    if hasattr(obj, "layers") and isinstance(getattr(obj, "layers"), (list, tuple)):
        for layer in obj.layers:
            for attr in ("keys", "values"):
                t = getattr(layer, attr, None)
                if torch.is_tensor(t):
                    total += _tensor_bytes(t)
                    found = True
    if not found:
        for attr in ("key_cache", "value_cache"):
            v = getattr(obj, attr, None)
            if v is not None:
                b = cache_bytes(v, _seen)
                if b is not None:
                    total += b
                    found = True
    return total if found else None


def theoretical_kv_bytes(model_config, tokens, batch=1, bytes_per_elem=2):
    """Full-precision KV size from the architecture: 2 (K,V) x layers x kv_heads x head_dim x tokens."""
    c = model_config
    head_dim = getattr(c, "head_dim", None) or c.hidden_size // c.num_attention_heads
    kv_heads = getattr(c, "num_key_value_heads", None) or c.num_attention_heads
    return 2 * c.num_hidden_layers * kv_heads * head_dim * tokens * batch * bytes_per_elem


def reset_peak():
    if cuda_ok():
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()


def snapshot():
    """Current/peak allocator stats and free/total device memory. None values on CPU."""
    if not cuda_ok():
        return {k: None for k in ("allocated_bytes", "reserved_bytes", "peak_allocated_bytes",
                                  "peak_reserved_bytes", "free_bytes", "total_bytes")}
    torch.cuda.synchronize()
    free, total = torch.cuda.mem_get_info()
    return {
        "allocated_bytes": torch.cuda.memory_allocated(),
        "reserved_bytes": torch.cuda.memory_reserved(),
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
        "free_bytes": free,
        "total_bytes": total,
    }


def gib(x):
    return None if x is None else round(x / 1024**3, 3)

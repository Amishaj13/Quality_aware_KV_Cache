"""CPU offloading of the full KV cache (transformers' DynamicCache(offloading=True)).

Nothing is evicted or quantized: every token's K/V is kept in 16-bit, so the model's
answers must be the same as with the full cache (up to tiny GPU timing-dependent float
differences, which greedy decoding almost never turns into a different token).
What changes is WHERE the cache lives:

  * after a layer has used its K/V, they are copied to CPU memory;
  * while layer i runs, layer i+1's K/V are copied back to the GPU on a side CUDA stream
    (prefetch), so the copy overlaps with computation as much as possible.

So at any moment only about two layers' K/V (of 28 for Qwen2.5-7B) sit on the GPU. The
price is that every decoding step moves the whole cache over PCIe twice (to the GPU and
back), which makes each new token slower, more so the longer the context.

Measured per request (in addition to the usual fields):
  kv_cache_gpu_bytes  K/V bytes on the GPU when generation ends (about 1 layer)
  kv_cache_cpu_bytes  K/V bytes in CPU memory when generation ends
kv_cache_bytes is still the total size of the cache (unchanged vs full). The comparison
report uses kv_cache_gpu_bytes for "KV memory saved" (for every other method the two are
the same, as their whole cache is on the GPU), and peak GPU memory shows the real effect.

Needs a CUDA GPU: transformers' offloading uses CUDA streams.
"""
import torch

from cache_methods.base import CacheMethod


def bytes_by_device(cache):
    """(gpu_bytes, cpu_bytes) of all K/V tensors in a transformers Cache."""
    gpu = cpu = 0
    for layer in getattr(cache, "layers", None) or []:
        for t in (getattr(layer, "keys", None), getattr(layer, "values", None)):
            if torch.is_tensor(t):
                n = t.numel() * t.element_size()
                if t.device.type == "cpu":
                    cpu += n
                else:
                    gpu += n
    return gpu, cpu


class OffloadCache(CacheMethod):
    name = "offload"
    compresses = False

    def new_cache(self, model):
        from transformers.cache_utils import DynamicCache
        if not torch.cuda.is_available():
            raise RuntimeError("cache_method 'offload' needs a CUDA GPU (transformers offloads with CUDA streams)")
        return DynamicCache(config=model.config, offloading=True)

    def cache_stats(self, cache):
        if cache is None:
            return {}
        if torch.cuda.is_available():
            torch.cuda.synchronize()          # finish pending non-blocking copies before reading devices
        gpu, cpu = bytes_by_device(cache)
        return {"kv_cache_gpu_bytes": gpu, "kv_cache_cpu_bytes": cpu}

    def describe(self):
        return {"cache_method": self.name, "kv_offload": "cpu"}

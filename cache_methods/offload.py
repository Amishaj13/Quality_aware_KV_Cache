"""CPU offloading of the full KV cache.

Nothing is evicted or quantized: every token's K/V is kept in 16-bit, so the model's
answers must be the same as with the full cache. What changes is WHERE the cache lives:

  * after a layer has used its K/V, they are copied to CPU memory;
  * while layer i runs, layer i+1's K/V are copied back to the GPU on a side CUDA stream
    (prefetch), so the copy overlaps with computation as much as possible.

So at any moment only about two layers' K/V (of 28 for Qwen2.5-7B) sit on the GPU. The
price is that every decoding step moves the whole cache over PCIe twice (to the GPU and
back), which makes each new token slower, more so the longer the context.

Why not transformers' own DynamicCache(offloading=True): with it, every offload answer
on the 7B model was wrong (pass@1 0) while the tiny-model test passed. The likely cause is
that its prefetched tensors are allocated on the side stream but used and freed on the main
stream without `record_stream`, so the GPU allocator can hand a still-in-use block to the
next prefetch, which then overwrites K/V before attention reads them (this only shows when
the GPU runs behind the CPU, as with a 7B model). This class does the same thing with the
two synchronisations made explicit:
  1. the main stream waits for the prefetch and marks the prefetched tensors as used by it
     (`record_stream`), so their memory is not reused until the main stream is done;
  2. a prefetch of layer i waits for the event recorded after layer i was copied to the
     CPU, so it never reads a CPU buffer that is still being written.

Measured per request (in addition to the usual fields):
  kv_cache_gpu_bytes  K/V bytes on the GPU when generation ends (about 1 layer)
  kv_cache_cpu_bytes  K/V bytes in CPU memory when generation ends
kv_cache_bytes is still the total size of the cache (unchanged vs full). The comparison
report uses kv_cache_gpu_bytes for "KV memory saved", and peak GPU memory shows the real effect.

Needs a CUDA GPU.
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


def _make_cache_class():
    from transformers.cache_utils import Cache, DynamicLayer

    class OffloadedKVCache(Cache):
        def __init__(self, num_layers):
            super().__init__(layers=[DynamicLayer() for _ in range(num_layers)])
            self.n = num_layers
            self.side = torch.cuda.Stream()
            self.offloaded = [None] * num_layers      # CUDA event: layer i's copy to the CPU is done

        def _to_gpu(self, i, device):
            """Start copying layer i back to the GPU on the side stream (if it is on the CPU)."""
            layer = self.layers[i]
            if not layer.is_initialized or layer.keys.device.type != "cpu":
                return
            with torch.cuda.stream(self.side):
                if self.offloaded[i] is not None:
                    self.side.wait_event(self.offloaded[i])
                layer.keys = layer.keys.to(device, non_blocking=True)
                layer.values = layer.values.to(device, non_blocking=True)

        def update(self, key_states, value_states, layer_idx, *args, **kwargs):
            device = key_states.device
            main = torch.cuda.current_stream(device)
            layer = self.layers[layer_idx]
            self._to_gpu(layer_idx, device)              # no-op if it was prefetched already
            main.wait_stream(self.side)                  # prefetch of this layer has finished
            if layer.is_initialized and layer.keys.numel():
                layer.keys.record_stream(main)           # main stream uses memory allocated on `side`
                layer.values.record_stream(main)
            self._to_gpu((layer_idx + 1) % self.n, device)   # overlap the next layer's copy
            keys, values = layer.update(key_states, value_states, *args, **kwargs)
            layer.keys = keys.to("cpu", non_blocking=True)
            layer.values = values.to("cpu", non_blocking=True)
            ev = torch.cuda.Event()
            ev.record(main)
            self.offloaded[layer_idx] = ev
            return keys, values

    return OffloadedKVCache


class OffloadCache(CacheMethod):
    name = "offload"
    compresses = False

    def __init__(self):
        self._cls = None

    def new_cache(self, model):
        if not torch.cuda.is_available():
            raise RuntimeError("cache_method 'offload' needs a CUDA GPU")
        if self._cls is None:
            self._cls = _make_cache_class()
        return self._cls(model.config.get_text_config(decoder=True).num_hidden_layers)

    def cache_stats(self, cache):
        if cache is None:
            return {}
        if torch.cuda.is_available():
            torch.cuda.synchronize()          # finish pending non-blocking copies before reading devices
        gpu, cpu = bytes_by_device(cache)
        return {"kv_cache_gpu_bytes": gpu, "kv_cache_cpu_bytes": cpu}

    def describe(self):
        return {"cache_method": self.name, "kv_offload": "cpu"}

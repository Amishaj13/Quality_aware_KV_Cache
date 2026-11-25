"""StreamingLLM: keep the first few tokens ("attention sinks") plus a sliding window of the
most recent tokens, and drop everything in between (Xiao et al., ICLR 2024).

Per layer and KV head the cache never holds more than `budget_tokens` tokens:
  sinks   the first `sink_tokens` tokens of the prompt (default 4). Models put a lot of
          attention on the very first tokens whatever they contain; keeping them stops
          the attention from falling apart once the middle is evicted.
  window  the most recent (budget_tokens - sink_tokens) tokens.

Unlike SnapKV, the choice does not look at what the tokens contain: it is fixed by
position. Code in the middle of a long prompt is therefore always dropped once the
prompt is longer than the budget, which is exactly what this baseline is meant to show.

How it runs here (the usual "StreamingLLM as a KV-compression baseline" setup, the same
one the SnapKV and PyramidKV papers use, so the comparison with snapkv<N> is fair):
  * prefill is exact: the prompt attends to all of itself, then the stored K/V are cut
    to sinks + window;
  * during decoding the window slides: each new token attends to sinks + window + itself,
    and the oldest window token is dropped. The cache size stays at budget_tokens.
  * Rotary positions are not re-numbered after eviction: kept tokens keep their original
    positions and new tokens continue from the true length (as in SnapKV here). The paper
    re-numbers positions inside the cache so it can run past the model's trained length;
    at 8K tokens Qwen2.5 (32K trained) never gets there, so this changes nothing that
    matters for these tests and keeps the two eviction methods directly comparable.
"""
import torch

from cache_methods.base import CacheMethod


def keep_slice(t, sinks, budget):
    """t: [B, H, T, D]. Sinks + most recent (budget - sinks) tokens along dim 2."""
    T = t.shape[-2]
    if T <= budget:
        return t
    return torch.cat([t[..., :sinks, :], t[..., T - (budget - sinks):, :]], dim=-2).contiguous()


def _make_layer_class():
    from transformers.cache_utils import DynamicLayer

    class StreamingLayer(DynamicLayer):
        def __init__(self, budget_tokens, sink_tokens):
            super().__init__()
            self.budget, self.sinks = budget_tokens, sink_tokens
            self.cumulative_length = 0
            self.prompt_tokens = self.kept_prompt_tokens = None
            self.evicted_tokens = 0

        def update(self, key_states, value_states, *args, **kwargs):
            if not self.is_initialized:
                self.lazy_initialization(key_states, value_states)
            first = self.cumulative_length == 0
            self.cumulative_length += key_states.shape[-2]
            if first:
                # the prompt attends to all of itself; only what is stored gets cut
                self.prompt_tokens = key_states.shape[-2]
                self.keys = keep_slice(key_states, self.sinks, self.budget)
                self.values = keep_slice(value_states, self.sinks, self.budget)
                self.kept_prompt_tokens = self.keys.shape[-2]
                self.evicted_tokens = self.prompt_tokens - self.kept_prompt_tokens
                return key_states, value_states
            k = torch.cat([self.keys, key_states], dim=-2)
            v = torch.cat([self.values, value_states], dim=-2)
            # new tokens see sinks + window + themselves, then the window slides
            self.keys = keep_slice(k, self.sinks, self.budget)
            self.values = keep_slice(v, self.sinks, self.budget)
            self.evicted_tokens += k.shape[-2] - self.keys.shape[-2]
            return k, v

        def get_seq_length(self):
            # logical length: generate() uses it for positions and to count consumed tokens
            return self.cumulative_length

        def get_mask_sizes(self, *args, **kwargs):
            q = args[-1] if args else kwargs.get("query_length", kwargs.get("cache_position"))
            q = q.shape[0] if torch.is_tensor(q) else int(q)
            stored = self.keys.shape[-2] if self.is_initialized and self.keys.numel() else 0
            return stored + q, 0

        def stored_tensors(self):
            return [t for t in (self.keys, self.values) if torch.is_tensor(t)]

        def crop(self, max_length):
            raise NotImplementedError("StreamingLLM cache does not support cropping")

    return StreamingLayer


class StreamingLLMCache(CacheMethod):
    compresses = True

    def __init__(self, budget_tokens=1024, sink_tokens=4):
        if sink_tokens >= budget_tokens:
            raise ValueError("streamingllm.sink_tokens must be smaller than streamingllm.budget_tokens")
        self.budget, self.sinks = budget_tokens, sink_tokens
        self.name = f"streaming{budget_tokens}"
        self._layer_cls = None

    @classmethod
    def from_config(cls, cfg, budget=None):
        s = cfg.get("streamingllm") or {}
        return cls(budget_tokens=int(budget or s.get("budget_tokens", 1024)),
                   sink_tokens=int(s.get("sink_tokens", 4)))

    def new_cache(self, model):
        from transformers.cache_utils import Cache
        if self._layer_cls is None:
            self._layer_cls = _make_layer_class()
        n = model.config.get_text_config(decoder=True).num_hidden_layers
        return Cache(layers=[self._layer_cls(self.budget, self.sinks) for _ in range(n)])

    def cache_stats(self, cache):
        layers = getattr(cache, "layers", None) or []
        if not layers or layers[0].prompt_tokens is None:
            return {}
        l0 = layers[0]
        return {"streaming_prompt_tokens": l0.prompt_tokens,
                "streaming_kept_prompt_tokens": l0.kept_prompt_tokens,
                "streaming_evicted_tokens": l0.evicted_tokens,
                "streaming_compressed": l0.evicted_tokens > 0}

    def describe(self):
        return {"cache_method": self.name, "streaming_budget_tokens": self.budget,
                "streaming_sink_tokens": self.sinks}

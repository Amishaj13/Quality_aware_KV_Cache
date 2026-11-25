"""SnapKV: keep only the prompt tokens the end of the prompt pays attention to
(Li et al., NeurIPS 2024).

After the prompt has been processed (prefill), for every layer and KV head:
  1. take the queries of the last `window` prompt tokens (the "observation window",
     which normally contains the question/instruction);
  2. compute their attention over all earlier prompt tokens and sum it over the window;
  3. smooth the scores with a 1-D max-pool of size `kernel_size`, so that neighbours
     of an important token are kept too (code needs whole identifiers and lines);
  4. keep the top (budget_tokens - window) earlier tokens plus the window itself,
     and drop the rest of the prompt's K/V.
Generated tokens are then appended normally (not compressed). Prompts that already fit
in budget_tokens are not compressed at all.

GQA models (Qwen2.5 has 28 query heads, 4 KV heads) share each KV head between several
query heads; their scores are averaged before choosing, so every KV head keeps one set
of tokens.

Implementation: a forward pre-hook on every attention module recomputes the window
queries (q_proj + RoPE, with the module's own weights) during prefill and hands them to
the cache. The model's code is not modified, and prefill attention itself is exact.
Rotary positions are not shifted after eviction: kept tokens keep their original
positions, and new tokens continue from the true prompt length, as in SnapKV.
"""
import torch
import torch.nn.functional as F

from cache_methods.base import CacheMethod


def _rotate_half(x):
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


def select_indices(q_win, keys, window, keep, kernel_size, pooling="max"):
    """Return sorted indices [B, Hkv, keep] of the earlier prompt tokens to keep.

    q_win: [B, Hq, W, D] window queries (post-RoPE); keys: [B, Hkv, L, D] (post-RoPE).
    """
    B, Hq, W, D = q_win.shape
    Hkv, L = keys.shape[1], keys.shape[2]
    groups = Hq // Hkv
    k = keys.repeat_interleave(groups, dim=1) if groups > 1 else keys
    att = torch.matmul(q_win.float(), k.float().transpose(-1, -2)) / (D ** 0.5)     # [B,Hq,W,L]
    # causal mask inside the window (window query i cannot see window tokens after it)
    causal = torch.ones(W, W, dtype=torch.bool, device=att.device).triu(1)
    att[..., L - W:].masked_fill_(causal, float("-inf"))
    att = att.softmax(dim=-1)
    score = att[..., : L - W].sum(dim=-2)                                            # [B,Hq,L-W]
    score = score.view(B, Hkv, groups, L - W).mean(dim=2)                           # [B,Hkv,L-W]
    if kernel_size and kernel_size > 1:
        pool = F.max_pool1d if pooling == "max" else F.avg_pool1d
        score = pool(score, kernel_size=kernel_size, padding=kernel_size // 2, stride=1)[..., : L - W]
    idx = score.topk(keep, dim=-1).indices
    return idx.sort(dim=-1).values


def _make_layer_class():
    from transformers.cache_utils import DynamicLayer

    class SnapKVLayer(DynamicLayer):
        def __init__(self, budget_tokens, window, kernel_size, pooling):
            super().__init__()
            self.budget, self.window = budget_tokens, window
            self.kernel_size, self.pooling = kernel_size, pooling
            self.window_queries = None     # set by the attention pre-hook during prefill
            self.cumulative_length = 0
            self.prompt_tokens = self.kept_prompt_tokens = None

        def update(self, key_states, value_states, *args, **kwargs):
            if not self.is_initialized:
                self.lazy_initialization(key_states, value_states)
            first = self.cumulative_length == 0
            self.cumulative_length += key_states.shape[-2]
            L = key_states.shape[-2]
            if first:
                self.prompt_tokens = L
                q, self.window_queries = self.window_queries, None
                if q is not None and L > self.budget:
                    idx = select_indices(q, key_states, self.window, self.budget - self.window,
                                         self.kernel_size, self.pooling)
                    gather = idx.unsqueeze(-1).expand(-1, -1, -1, key_states.shape[-1])
                    self.keys = torch.cat([key_states[..., : L - self.window, :].gather(2, gather),
                                           key_states[..., L - self.window:, :]], dim=-2).contiguous()
                    self.values = torch.cat([value_states[..., : L - self.window, :].gather(2, gather),
                                             value_states[..., L - self.window:, :]], dim=-2).contiguous()
                else:
                    self.keys, self.values = key_states, value_states
                self.kept_prompt_tokens = self.keys.shape[-2]
                return key_states, value_states      # the prompt itself attends to everything
            self.keys = torch.cat([self.keys, key_states], dim=-2)
            self.values = torch.cat([self.values, value_states], dim=-2)
            return self.keys, self.values

        def get_seq_length(self):
            # logical length: generate() uses it to know how many tokens were consumed
            return self.cumulative_length

        def get_mask_sizes(self, query_length):
            stored = self.keys.shape[-2] if self.is_initialized and self.keys.numel() else 0
            return stored + query_length, 0

        def stored_tensors(self):
            return [t for t in (self.keys, self.values) if torch.is_tensor(t)]

        def crop(self, max_length):
            raise NotImplementedError("SnapKV cache does not support cropping")

    return SnapKVLayer


def _attention_modules(model):
    """(layer_idx, module) for every decoder self-attention module."""
    out = []
    for m in model.modules():
        if hasattr(m, "q_proj") and hasattr(m, "layer_idx") and hasattr(m, "head_dim"):
            out.append((m.layer_idx, m))
    return out


def _make_pre_hook(window):
    def hook(module, args, kwargs):
        cache = kwargs.get("past_key_values")
        layers = getattr(cache, "layers", None)
        if not getattr(cache, "is_snapkv", False) or not layers:
            return None
        hidden = kwargs.get("hidden_states", args[0] if args else None)
        if hidden is None or hidden.shape[1] <= window:
            return None
        layer = layers[module.layer_idx]
        if layer.cumulative_length:           # only the prefill pass
            return None
        pe = kwargs.get("position_embeddings")
        if pe is None and len(args) > 1:
            pe = args[1]
        cos, sin = pe
        h = hidden[:, -window:]
        q = module.q_proj(h).view(h.shape[0], window, -1, module.head_dim)
        if hasattr(module, "q_norm"):          # Qwen3-style per-head norm before RoPE
            q = module.q_norm(q)
        q = q.transpose(1, 2)
        c, s = cos[:, -window:].unsqueeze(1), sin[:, -window:].unsqueeze(1)
        layer.window_queries = q * c + _rotate_half(q) * s
        return None
    return hook


class SnapKVCache(CacheMethod):
    compresses = True

    def __init__(self, budget_tokens=1024, window=32, kernel_size=7, pooling="max"):
        if window >= budget_tokens:
            raise ValueError("snapkv.window must be smaller than snapkv.budget_tokens")
        self.budget, self.window, self.kernel_size, self.pooling = budget_tokens, window, kernel_size, pooling
        self.name = f"snapkv{budget_tokens}"
        self._layer_cls = None

    @classmethod
    def from_config(cls, cfg, budget=None):
        s = cfg.get("snapkv") or {}
        return cls(budget_tokens=int(budget or s.get("budget_tokens", 1024)), window=int(s.get("window", 32)),
                   kernel_size=int(s.get("kernel_size", 7)), pooling=s.get("pooling", "max"))

    def prepare(self, model, config):
        mods = _attention_modules(model)
        if not mods:
            raise RuntimeError("SnapKV: no attention modules with q_proj/layer_idx found in this model")
        handles = getattr(model, "_snapkv_hooks", None)
        if handles:                            # re-used model (position sweep): replace old hooks
            for h in handles:
                h.remove()
        model._snapkv_hooks = [m.register_forward_pre_hook(_make_pre_hook(self.window), with_kwargs=True)
                               for _, m in mods]
        return model

    def new_cache(self, model):
        from transformers.cache_utils import Cache
        if self._layer_cls is None:
            self._layer_cls = _make_layer_class()
        n = model.config.get_text_config(decoder=True).num_hidden_layers
        cache = Cache(layers=[self._layer_cls(self.budget, self.window, self.kernel_size, self.pooling)
                              for _ in range(n)])
        cache.is_snapkv = True
        return cache

    def cache_stats(self, cache):
        layers = getattr(cache, "layers", None) or []
        if not layers or layers[0].prompt_tokens is None:
            return {}
        return {"snapkv_prompt_tokens": layers[0].prompt_tokens,
                "snapkv_kept_prompt_tokens": layers[0].kept_prompt_tokens,
                "snapkv_compressed": layers[0].kept_prompt_tokens < layers[0].prompt_tokens}

    def describe(self):
        return {"cache_method": self.name, "snapkv_budget_tokens": self.budget, "snapkv_window": self.window,
                "snapkv_kernel_size": self.kernel_size, "snapkv_pooling": self.pooling}

"""KV-cache method interface.

Implemented: the full cache (baseline), KIVI (2/4-bit quantization) and SnapKV (prompt
token eviction). The remaining planned methods are registered by name so configs fail
loudly instead of silently falling back to the full cache. Each method implements
CacheMethod (see base.py) and is selected with experiment.cache_method; its settings
live in the top-level `kivi:` / `snapkv:` config blocks.
"""
from cache_methods.base import CacheMethod
from cache_methods.full import FullCache
from cache_methods.kivi import KIVICache
from cache_methods.snapkv import SnapKVCache

# name -> factory(cfg). "kivi" takes its bit width from kivi.bits; kivi4/kivi2 fix it.
IMPLEMENTED = {
    "full": lambda cfg: FullCache(),
    "kivi": lambda cfg: KIVICache.from_config(cfg),
    "kivi4": lambda cfg: KIVICache.from_config(cfg, bits=4),
    "kivi2": lambda cfg: KIVICache.from_config(cfg, bits=2),
    "snapkv": lambda cfg: SnapKVCache.from_config(cfg),
}

# Planned (see docs/ROADMAP.md). Not implemented.
PLANNED = {
    "h2o": "Heavy-Hitter Oracle token eviction",
    "streamingllm": "attention sinks + recent window",
    "pyramidkv": "layer-wise pyramidal KV budget",
    "evolkv": "evolutionary layer-wise budget search",
}


def get_cache_method(name: str, cfg=None) -> CacheMethod:
    if name in PLANNED:
        raise NotImplementedError(f"cache method '{name}' ({PLANNED[name]}) is not implemented yet")
    if name not in IMPLEMENTED:
        raise KeyError(f"unknown cache method '{name}'")
    return IMPLEMENTED[name](cfg or {})


def method_tag(cfg) -> str:
    """Short label used in experiment ids and charts: full, kivi4, kivi2, snapkv1024."""
    return get_cache_method(cfg["experiment"].get("cache_method", "full"), cfg).name

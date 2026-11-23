"""KV-cache method interface.

Only the standard full cache is implemented in this phase. The planned methods are
registered by name so configs fail loudly instead of silently falling back to the
full cache. Each future method implements CacheMethod (see base.py).
"""
from cache_methods.base import CacheMethod
from cache_methods.full import FullCache

IMPLEMENTED = {"full": FullCache}

# Planned for the next three months (see docs/ROADMAP.md). Not implemented.
PLANNED = {
    "h2o": "Heavy-Hitter Oracle token eviction",
    "streamingllm": "attention sinks + recent window",
    "kivi": "2/4-bit asymmetric KV quantization (see kivi_baseline/ for the Llama-3.1 harness)",
    "snapkv": "observation-window based prompt KV selection",
    "pyramidkv": "layer-wise pyramidal KV budget",
    "evolkv": "evolutionary layer-wise budget search",
}


def get_cache_method(name: str) -> CacheMethod:
    if name in PLANNED:
        raise NotImplementedError(f"cache method '{name}' ({PLANNED[name]}) is not implemented yet")
    if name not in IMPLEMENTED:
        raise KeyError(f"unknown cache method '{name}'")
    return IMPLEMENTED[name]()

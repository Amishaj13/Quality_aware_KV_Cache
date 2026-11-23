"""Standard full KV cache: transformers' default DynamicCache, nothing evicted or quantized."""
from cache_methods.base import CacheMethod


class FullCache(CacheMethod):
    name = "full"
    compresses = False

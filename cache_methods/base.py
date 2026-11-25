"""Interface every KV-cache method implements, so the runner never changes per method."""


class CacheMethod:
    #: short name used in configs and result rows
    name = "base"
    #: True if this method changes what is stored in the cache (compression/eviction)
    compresses = False

    def prepare(self, model, config):
        """One-time setup after the model is loaded (e.g. patch attention). Default: none."""
        return model

    def new_cache(self, model):
        """Return a fresh past_key_values object for one request, or None to let
        transformers create its default DynamicCache."""
        return None

    def generate_kwargs(self):
        """Extra keyword arguments for model.generate()."""
        return {"use_cache": True}

    def cache_stats(self, cache):
        """Extra per-request facts read from the finished cache (e.g. tokens kept). Default: none."""
        return {}

    def describe(self):
        """Settings that must be stored with every result row."""
        return {"cache_method": self.name}

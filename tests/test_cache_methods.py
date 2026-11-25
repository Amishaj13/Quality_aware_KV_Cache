"""KIVI and SnapKV: numerical checks on a small random Qwen2-architecture model (CPU).

These check that each method does what it claims (quantization error, memory, which
tokens are kept, exactness when nothing is compressed). They say nothing about quality
on real tasks: that is measured by the GPU runs.
"""
import pytest
import torch

from cache_methods import get_cache_method, method_tag
from cache_methods.kivi import _pack, _unpack, dequantize, quantize
from cache_methods.snapkv import select_indices
from evaluation.memory_metrics import cache_bytes, theoretical_kv_bytes
from utils.config import ConfigError, load_config


@pytest.fixture(scope="module")
def small_model():
    from transformers import Qwen2Config, Qwen2ForCausalLM
    torch.manual_seed(0)
    cfg = Qwen2Config(vocab_size=500, hidden_size=128, intermediate_size=256, num_hidden_layers=2,
                      num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=4096)
    m = Qwen2ForCausalLM(cfg).to(torch.bfloat16).eval()
    m.config._attn_implementation = "eager"
    return m


def generate(model, method, ids, n=20):
    method.prepare(model, {})
    cache = method.new_cache(model)
    kw = {"past_key_values": cache} if cache is not None else {}
    out = model.generate(input_ids=ids, attention_mask=torch.ones_like(ids), max_new_tokens=n, min_new_tokens=n,
                         do_sample=False, return_dict_in_generate=True, **kw)
    return out.sequences[0, ids.shape[1]:].tolist(), out.past_key_values


# --------------------------------------------------------------------------- KIVI
@pytest.mark.parametrize("bits", [2, 4])
def test_pack_roundtrip(bits):
    codes = torch.randint(0, 2 ** bits, (2, 3, 5, 64), dtype=torch.uint8)
    packed = _pack(codes, bits)
    assert packed.shape[-1] == 64 * bits // 8
    assert torch.equal(_unpack(packed, bits), codes)


@pytest.mark.parametrize("bits,max_err", [(4, 0.5 / 15), (2, 0.5 / 3)])
@pytest.mark.parametrize("axis", ["channel", "token"])
def test_quantization_error_is_within_half_a_step(bits, max_err, axis):
    x = torch.rand(1, 2, 64, 64)            # values in [0, 1): one step = range / levels
    y = dequantize(quantize(x, bits, 32, axis), bits, 32, axis, x.dtype)
    assert (x - y).abs().max() <= max_err + 1e-4


def test_kivi_keys_use_per_channel_stats():
    x = torch.randn(1, 1, 32, 64)
    x[..., 5] *= 100                         # one outlier channel
    q = quantize(x, 2, 32, "channel")
    y = dequantize(q, 2, 32, "channel", x.dtype)
    other = [c for c in range(64) if c != 5]
    # the outlier channel does not destroy the precision of the other channels
    assert (x[..., other] - y[..., other]).abs().mean() < 0.5


@pytest.mark.parametrize("bits", [4, 2])
def test_kivi_cache_memory_and_length(small_model, bits):
    ids = torch.randint(0, 500, (1, 300))
    method = get_cache_method(f"kivi{bits}", {})
    toks, cache = generate(small_model, method, ids, n=20)
    assert len(toks) == 20
    L = 300 + 20 - 1
    assert cache.get_seq_length() == L
    layer = cache.layers[0]
    assert layer.n_quantized == 256 and layer.keys.shape[-2] == L - 256    # residual < 128
    full = theoretical_kv_bytes(small_model.config, L, bytes_per_elem=2)
    measured = cache_bytes(cache)
    # quantized part: bits/16 of bf16 + min/scale overhead; residual stays bf16
    hd, kvh, nl = 32, 2, 2
    expect = nl * 2 * kvh * (256 * hd * bits // 8 + (L - 256) * hd * 2)
    expect += nl * kvh * 2 * 2 * (256 // 32) * hd       # key stats: per channel per 32-token group
    expect += nl * kvh * 2 * 2 * 256 * (hd // 32)       # value stats: per token per 32-channel group
    assert measured == expect
    assert measured < full


def test_kivi4_mostly_matches_full_cache(small_model):
    ids = torch.randint(0, 500, (1, 300))
    full, _ = generate(small_model, get_cache_method("full", {}), ids)
    k4, _ = generate(small_model, get_cache_method("kivi4", {}), ids)
    assert k4[0] == full[0]                  # first token comes from exact prefill attention
    assert sum(a == b for a, b in zip(k4, full)) >= 10


# --------------------------------------------------------------------------- SnapKV
def test_snapkv_selects_what_the_window_attends_to():
    torch.manual_seed(0)
    B, Hkv, groups, L, W, D = 1, 2, 2, 64, 8, 16
    keys = torch.randn(B, Hkv, L, D) * 0.1
    q = torch.randn(B, Hkv * groups, W, D) * 0.1
    # make tokens 10 and 40 strongly attended for KV head 0 and 1 respectively
    keys[0, 0, 10] = q[0, 0:2].mean(dim=(0, 1)) * 50
    keys[0, 1, 40] = q[0, 2:4].mean(dim=(0, 1)) * 50
    idx = select_indices(q, keys, W, keep=4, kernel_size=1)
    assert 10 in idx[0, 0].tolist() and 40 in idx[0, 1].tolist()
    assert torch.equal(idx, idx.sort(dim=-1).values)
    # pooling keeps the neighbours of an important token
    idx = select_indices(q, keys, W, keep=3, kernel_size=3)
    assert set(idx[0, 0].tolist()) == {9, 10, 11}


def test_snapkv_matches_reference_attention(small_model):
    """Indices chosen from the hook's recomputed queries equal those from the model's own
    attention weights (output_attentions with eager attention)."""
    from cache_methods.snapkv import _make_layer_class
    ids = torch.randint(0, 500, (1, 200))
    W, budget, ks = 16, 64, 1
    method = get_cache_method("snapkv", {"snapkv": {"budget_tokens": budget, "window": W, "kernel_size": ks}})
    method.prepare(small_model, {})
    cache = method.new_cache(small_model)
    captured = {}
    cls = type(cache.layers[0])
    orig = cls.update

    def spy(self, k, v, *a, **kw):
        if self.cumulative_length == 0:
            captured[len(captured)] = (self.window_queries.clone(), k.clone())
        return orig(self, k, v, *a, **kw)
    cls.update = spy
    try:
        with torch.no_grad():
            out = small_model(input_ids=ids, past_key_values=cache, use_cache=True, output_attentions=True)
    finally:
        cls.update = orig
    assert cache.layers[0].keys.shape[-2] == budget and cache.get_seq_length() == 200
    for layer_idx, att in enumerate(out.attentions):
        q, k = captured[layer_idx]
        mine = select_indices(q, k, W, budget - W, ks)
        ref = att[:, :, -W:, : 200 - W].float().sum(dim=-2)
        ref = ref.view(1, 2, 2, -1).mean(dim=2).topk(budget - W, dim=-1).indices.sort(dim=-1).values
        overlap = (mine.unsqueeze(-1) == ref.unsqueeze(-2)).any(-1).float().mean()
        assert overlap > 0.95       # bf16 rounding can swap near-tied tokens


def test_snapkv_is_exact_when_prompt_fits_budget(small_model):
    ids = torch.randint(0, 500, (1, 150))
    full, _ = generate(small_model, get_cache_method("full", {}), ids)
    snap, cache = generate(small_model, get_cache_method("snapkv", {"snapkv": {"budget_tokens": 256}}), ids)
    assert snap == full
    assert cache.layers[0].kept_prompt_tokens == 150


def test_snapkv_compresses_long_prompt(small_model):
    ids = torch.randint(0, 500, (1, 400))
    method = get_cache_method("snapkv", {"snapkv": {"budget_tokens": 128}})
    toks, cache = generate(small_model, method, ids, n=20)
    assert len(toks) == 20
    assert cache.get_seq_length() == 419                       # logical length (positions)
    assert cache.layers[0].keys.shape[-2] == 128 + 19          # stored: budget + generated
    assert cache_bytes(cache) == theoretical_kv_bytes(small_model.config, 147, bytes_per_elem=2)
    stats = method.cache_stats(cache)
    assert stats == {"snapkv_prompt_tokens": 400, "snapkv_kept_prompt_tokens": 128, "snapkv_compressed": True}


def test_snapkv_prepare_is_idempotent(small_model):
    method = get_cache_method("snapkv", {})
    method.prepare(small_model, {})
    method.prepare(small_model, {})
    assert len(small_model._snapkv_hooks) == 2


# --------------------------------------------------------------------------- config
def test_method_tags_and_config():
    assert method_tag(load_config(overrides={"experiment": {"cache_method": "kivi2"}})) == "kivi2"
    assert method_tag(load_config(overrides={"experiment": {"cache_method": "kivi"}, "kivi": {"bits": 2}})) == "kivi2"
    assert method_tag(load_config(overrides={"experiment": {"cache_method": "snapkv"},
                                             "snapkv": {"budget_tokens": 2048}})) == "snapkv2048"
    with pytest.raises(ConfigError, match="kivi.bits"):
        load_config(overrides={"kivi": {"bits": 3}})
    with pytest.raises(ConfigError, match="window"):
        load_config(overrides={"snapkv": {"budget_tokens": 16, "window": 32}})


# --------------------------------------------------------------------------- StreamingLLM
def test_streaming_keep_slice():
    from cache_methods.streamingllm import keep_slice
    t = torch.arange(20).view(1, 1, 20, 1)
    assert keep_slice(t, 4, 10).flatten().tolist() == [0, 1, 2, 3, 14, 15, 16, 17, 18, 19]
    assert keep_slice(t, 4, 30).shape[-2] == 20          # shorter than the budget: untouched


def test_streaming_cache_stays_within_budget(small_model):
    ids = torch.randint(0, 500, (1, 300))
    method = get_cache_method("streamingllm", {"streamingllm": {"budget_tokens": 64, "sink_tokens": 4}})
    assert method.name == "streaming64"
    toks, cache = generate(small_model, method, ids, n=20)
    assert len(toks) == 20
    assert cache.get_seq_length() == 300 + 20 - 1         # logical length keeps counting
    for layer in cache.layers:
        assert layer.keys.shape[-2] == 64 and layer.values.shape[-2] == 64
    stats = method.cache_stats(cache)
    assert stats["streaming_prompt_tokens"] == 300 and stats["streaming_kept_prompt_tokens"] == 64
    assert stats["streaming_evicted_tokens"] == 300 + 19 - 64 and stats["streaming_compressed"]
    assert cache_bytes(cache) == theoretical_kv_bytes(small_model.config, 64, bytes_per_elem=2)


def test_streaming_keeps_sinks_and_latest_tokens(small_model):
    """The stored keys are exactly the sinks and the newest tokens of the full cache."""
    ids = torch.randint(0, 500, (1, 200))
    _, full = generate(small_model, get_cache_method("full", {}), ids, n=1)
    method = get_cache_method("streamingllm", {"streamingllm": {"budget_tokens": 50, "sink_tokens": 4}})
    _, st = generate(small_model, method, ids, n=1)
    fk, sk = full.layers[0].keys, st.layers[0].keys
    assert torch.equal(sk[..., :4, :], fk[..., :4, :])
    assert torch.equal(sk[..., 4:, :], fk[..., -46:, :])


def test_streaming_is_exact_when_everything_fits(small_model):
    ids = torch.randint(0, 500, (1, 150))
    full, _ = generate(small_model, get_cache_method("full", {}), ids)
    st, cache = generate(small_model, get_cache_method("streamingllm", {"streamingllm": {"budget_tokens": 256}}), ids)
    assert st == full
    assert not get_cache_method("streamingllm", {"streamingllm": {"budget_tokens": 256}}).cache_stats(cache)[
        "streaming_compressed"]


def test_streaming_config_validation(tmp_path):
    with pytest.raises(ConfigError):
        load_config(overrides={"experiment": {"cache_method": "streamingllm"},
                               "streamingllm": {"budget_tokens": 4, "sink_tokens": 4}})
    cfg = load_config(overrides={"experiment": {"cache_method": "streamingllm"},
                                 "streamingllm": {"budget_tokens": 2048}})
    assert method_tag(cfg) == "streaming2048"


# --------------------------------------------------------------------------- CPU offloading
def test_offload_registered_and_tagged():
    cfg = load_config(overrides={"experiment": {"cache_method": "offload"}})
    assert method_tag(cfg) == "offload"
    assert not get_cache_method("offload").compresses


def test_offload_bytes_by_device():
    from transformers.cache_utils import DynamicCache
    from cache_methods.offload import bytes_by_device
    c = DynamicCache()
    c.update(torch.zeros(1, 2, 10, 8, dtype=torch.bfloat16), torch.zeros(1, 2, 10, 8, dtype=torch.bfloat16), 0)
    assert bytes_by_device(c) == (0, 2 * 2 * 10 * 8 * 2)   # on a CPU-only machine everything is "cpu"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="offloading needs a CUDA GPU")
def test_offload_matches_full_cache_and_moves_kv_to_cpu():
    from transformers import Qwen2Config, Qwen2ForCausalLM
    torch.manual_seed(0)
    # 4 layers: when generation ends, offloading keeps one layer (the prefetched layer 0) on the GPU
    model = Qwen2ForCausalLM(Qwen2Config(vocab_size=500, hidden_size=128, intermediate_size=256, num_hidden_layers=4,
                                         num_attention_heads=4, num_key_value_heads=2)).to("cuda", torch.bfloat16).eval()
    model.config._attn_implementation = "eager"
    ids = torch.randint(0, 500, (1, 300), device="cuda")
    full, fc = generate(model, get_cache_method("full", {}), ids)
    method = get_cache_method("offload", {})
    off, oc = generate(model, method, ids)
    assert off == full                                        # nothing is changed, only moved
    total = cache_bytes(oc)
    assert total == cache_bytes(fc)                           # same cache, different place
    stats = method.cache_stats(oc)
    assert stats["kv_cache_gpu_bytes"] + stats["kv_cache_cpu_bytes"] == total
    assert stats["kv_cache_gpu_bytes"] <= total // 4          # at most one of the 4 layers left on the GPU
    assert stats["kv_cache_cpu_bytes"] >= 3 * total // 4


@pytest.mark.skipif(not torch.cuda.is_available(), reason="offloading needs a CUDA GPU")
def test_offload_matches_full_cache_when_gpu_is_busy():
    """A wider, deeper model and a long prompt, so the GPU runs behind the CPU like with the
    7B model. Stream/allocator races in offloading only show up in that situation."""
    from transformers import Qwen2Config, Qwen2ForCausalLM
    torch.manual_seed(0)
    model = Qwen2ForCausalLM(Qwen2Config(vocab_size=2000, hidden_size=1024, intermediate_size=4096,
                                         num_hidden_layers=12, num_attention_heads=16, num_key_value_heads=4,
                                         max_position_embeddings=8192)).to("cuda", torch.bfloat16).eval()
    ids = torch.randint(0, 2000, (1, 3000), device="cuda")
    full, _ = generate(model, get_cache_method("full", {}), ids, n=40)
    off, _ = generate(model, get_cache_method("offload", {}), ids, n=40)
    assert off == full


@pytest.mark.parametrize("impl", ["eager", "sdpa"])
def test_streaming_decode_equals_masked_full_attention(impl):
    """One decoding step with the StreamingLLM cache gives the same logits as the full cache
    with every evicted token masked out (original positions kept)."""
    from transformers import DynamicCache, Qwen2Config, Qwen2ForCausalLM
    torch.manual_seed(0)
    m = Qwen2ForCausalLM(Qwen2Config(vocab_size=500, hidden_size=128, intermediate_size=256, num_hidden_layers=2,
                                     num_attention_heads=4, num_key_value_heads=2)).eval()
    m.config._attn_implementation = impl
    ids = torch.randint(0, 500, (1, 300))
    nxt = torch.tensor([[7]])
    with torch.no_grad():
        c = get_cache_method("streamingllm", {"streamingllm": {"budget_tokens": 64}}).new_cache(m)
        m(input_ids=ids, past_key_values=c, use_cache=True)
        mine = m(input_ids=nxt, past_key_values=c, use_cache=True).logits
        dc = DynamicCache()
        m(input_ids=ids, past_key_values=dc, use_cache=True)
        am = torch.zeros(1, 301, dtype=torch.long)
        am[:, :4] = 1
        am[:, 300 - 60:] = 1
        ref = m(input_ids=nxt, past_key_values=dc, attention_mask=am, position_ids=torch.tensor([[300]]),
                use_cache=True).logits
    assert (mine - ref).abs().max() < 1e-4

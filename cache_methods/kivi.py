"""KIVI: tuning-free asymmetric 2/4-bit KV-cache quantization (Liu et al., ICML 2024).

What is stored, per layer and KV head:
  keys    quantized PER CHANNEL: each channel gets its own min/scale over a group of
          `group_size` consecutive tokens (keys have a few channels with very large
          values, so sharing a scale across channels would crush the others).
  values  quantized PER TOKEN: each token gets its own min/scale over groups of
          `group_size` channels.
  residual  the most recent tokens (fewer than `residual_length`) stay in full
          precision. Whenever the residual reaches `residual_length` tokens it is
          quantized as one block and appended to the quantized store.

The quantized codes are really packed into uint8 (two 4-bit or four 2-bit codes per
byte), and min/scale are kept in the model dtype, so kv_cache_bytes measures the true
storage. Attention still runs in 16-bit: the quantized part is dequantized on the fly
for every decoding step, so the output quality is exactly what a fused KIVI kernel
would produce, but decoding is somewhat slower than the official CUDA kernels.

Prefill attention uses the unquantized prompt K/V (as in the official implementation);
quantization affects every generated token.

Differences from the paper's reference code, kept small on purpose:
  * values are flushed in blocks of `residual_length`, like keys (the reference
    quantizes values one token at a time as they leave the window). Both keep
    between 0 and residual_length-1 recent tokens in full precision.
"""
import torch

from cache_methods.base import CacheMethod


# ----------------------------------------------------------------------------- packing
def _pack(codes, bits):
    """codes: uint8 tensor of values < 2**bits, last dim divisible by 8//bits."""
    per = 8 // bits
    c = codes.view(*codes.shape[:-1], codes.shape[-1] // per, per)
    out = c[..., 0].clone()
    for i in range(1, per):
        out |= c[..., i] << (bits * i)
    return out


def _unpack(packed, bits):
    per = 8 // bits
    mask = (1 << bits) - 1
    parts = [(packed >> (bits * i)) & mask for i in range(per)]
    return torch.stack(parts, dim=-1).view(*packed.shape[:-1], packed.shape[-1] * per)


def quantize(x, bits, group_size, axis):
    """Asymmetric min-max quantization.

    x: [B, H, T, D]. axis="channel": groups of `group_size` tokens share stats per channel
    (T must be a multiple of group_size). axis="token": groups of `group_size` channels
    share stats per token (D must be a multiple of group_size).
    Returns dict(codes=packed uint8, mn, scale, shape, axis).
    """
    B, H, T, D = x.shape
    levels = (1 << bits) - 1
    dtype, x = x.dtype, x.float()          # compute codes in fp32, store stats in the model dtype
    if axis == "channel":
        g = x.view(B, H, T // group_size, group_size, D)
        mn = g.amin(dim=3, keepdim=True)
        mx = g.amax(dim=3, keepdim=True)
    else:
        g = x.view(B, H, T, D // group_size, group_size)
        mn = g.amin(dim=4, keepdim=True)
        mx = g.amax(dim=4, keepdim=True)
    mn, scale = mn.to(dtype), ((mx - mn) / levels).clamp(min=1e-5).to(dtype)
    codes = ((g - mn.float()) / scale.float()).round_().clamp_(0, levels).to(torch.uint8)
    return {"codes": _pack(codes.view(B, H, T, D), bits), "mn": mn, "scale": scale}


def dequantize(q, bits, group_size, axis, dtype):
    codes = _unpack(q["codes"], bits)
    B, H, T, D = codes.shape
    if axis == "channel":
        g = codes.view(B, H, T // group_size, group_size, D)
    else:
        g = codes.view(B, H, T, D // group_size, group_size)
    return (g.to(dtype) * q["scale"] + q["mn"]).view(B, H, T, D)


def _cat_q(a, b):
    if a is None:
        return b
    # channel-axis stats are [B,H,T/G,1,D]; token-axis stats [B,H,T,D/G,1]: both concat on dim 2
    return {k: torch.cat([a[k], b[k]], dim=2) for k in a}


# ----------------------------------------------------------------------------- cache
def _make_layer_class():
    from transformers.cache_utils import DynamicLayer

    class KIVILayer(DynamicLayer):
        """One layer of the KIVI cache. `keys`/`values` hold the full-precision residual."""

        def __init__(self, k_bits=4, v_bits=4, group_size=32, residual_length=128):
            super().__init__()
            self.k_bits, self.v_bits = k_bits, v_bits
            self.group_size, self.residual_length = group_size, residual_length
            self.qk = self.qv = None          # quantized blocks
            self.n_quantized = 0
            self.cumulative_length = 0

        def _flush(self, k, v):
            """Quantize k/v ([B,H,T,D], T a multiple of residual_length) into the store."""
            self.qk = _cat_q(self.qk, quantize(k, self.k_bits, self.group_size, "channel"))
            self.qv = _cat_q(self.qv, quantize(v, self.v_bits, self.v_group, "token"))
            self.n_quantized += k.shape[-2]

        def _dequantized(self):
            if self.qk is None:
                return None, None
            return (dequantize(self.qk, self.k_bits, self.group_size, "channel", self.dtype),
                    dequantize(self.qv, self.v_bits, self.v_group, "token", self.dtype))

        def update(self, key_states, value_states, *args, **kwargs):
            if not self.is_initialized:
                self.lazy_initialization(key_states, value_states)
                # value groups run along channels; tiny test models can have head_dim < group_size
                self.v_group = min(self.group_size, key_states.shape[-1])
                if key_states.shape[-1] % self.v_group:
                    raise ValueError(f"head_dim {key_states.shape[-1]} is not a multiple of "
                                     f"kivi.group_size {self.group_size}")
            self.cumulative_length += key_states.shape[-2]
            is_prefill = self.n_quantized == 0 and self.keys.numel() == 0
            dk, dv = self._dequantized()
            res_k = torch.cat([self.keys, key_states], dim=-2)
            res_v = torch.cat([self.values, value_states], dim=-2)
            if dk is None:
                k_out, v_out = res_k, res_v
            else:
                k_out, v_out = torch.cat([dk, res_k], dim=-2), torch.cat([dv, res_v], dim=-2)
            del dk, dv
            # move whole residual blocks into the quantized store
            n_flush = (res_k.shape[-2] // self.residual_length) * self.residual_length
            if n_flush:
                self._flush(res_k[..., :n_flush, :].contiguous(), res_v[..., :n_flush, :].contiguous())
                res_k, res_v = res_k[..., n_flush:, :].contiguous(), res_v[..., n_flush:, :].contiguous()
            self.keys, self.values = res_k, res_v
            # prefill attends to the exact prompt K/V; later steps see the quantized history
            return (key_states, value_states) if is_prefill else (k_out, v_out)

        def get_seq_length(self):
            return self.cumulative_length

        def stored_tensors(self):
            """Every tensor this layer keeps between steps (for kv_cache_bytes)."""
            out = [t for t in (self.keys, self.values) if torch.is_tensor(t)]
            for q in (self.qk, self.qv):
                if q is not None:
                    out += list(q.values())
            return out

        def crop(self, max_length):
            raise NotImplementedError("KIVI cache does not support cropping")

    return KIVILayer


class KIVICache(CacheMethod):
    compresses = True

    def __init__(self, bits=4, group_size=32, residual_length=128):
        if bits not in (2, 4):
            raise ValueError("kivi bits must be 2 or 4")
        if residual_length % group_size:
            raise ValueError("kivi.residual_length must be a multiple of kivi.group_size")
        self.bits, self.group_size, self.residual_length = bits, group_size, residual_length
        self.name = f"kivi{bits}"
        self._layer_cls = None

    @classmethod
    def from_config(cls, cfg, bits=None):
        k = cfg.get("kivi") or {}
        return cls(bits=bits or int(k.get("bits", 4)), group_size=int(k.get("group_size", 32)),
                   residual_length=int(k.get("residual_length", 128)))

    def new_cache(self, model):
        from transformers.cache_utils import Cache
        if self._layer_cls is None:
            self._layer_cls = _make_layer_class()
        n = model.config.get_text_config(decoder=True).num_hidden_layers
        layers = [self._layer_cls(self.bits, self.bits, self.group_size, self.residual_length) for _ in range(n)]
        return Cache(layers=layers)

    def describe(self):
        return {"cache_method": self.name, "kivi_bits": self.bits, "kivi_group_size": self.group_size,
                "kivi_residual_length": self.residual_length}

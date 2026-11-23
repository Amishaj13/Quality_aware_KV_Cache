"""Full-KV-cache inference for one prepared prompt, with timing and memory measurement.

What is timed (time.perf_counter, CUDA synchronised before and after):
  total_inference_seconds  model.generate() only; tokenisation, prompt building,
                           decoding to text and model loading are excluded.
  ttft_seconds             start of generate() -> first new token exists (TokenTimer).
  tpot_ms                  (end - first token) / (output_tokens - 1); None for 1 token.
  prefill_time_seconds     a separate prefill-only forward pass over the same prompt
                           (experiment.measure_prefill); None when disabled.
"""
import time

import torch

from cache_methods import get_cache_method
from evaluation import memory_metrics as mm
from evaluation.latency_metrics import TokenTimer, latency_from_stamps


def _sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def cache_kwarg_name(model):
    """transformers >= 5.3x renamed forward(past_key_values=...) to forward(cache=...)."""
    import inspect
    params = inspect.signature(model.forward).parameters
    return "cache" if "cache" in params else "past_key_values"


def greedy_generation_kwargs(cfg, tokenizer, max_new_tokens):
    e = cfg["experiment"]
    eos = tokenizer.eos_token_id
    # Qwen chat models end turns with <|im_end|>; make sure it stops generation
    im_end = tokenizer.convert_tokens_to_ids("<|im_end|>") if "<|im_end|>" in tokenizer.get_vocab() else None
    eos_ids = [i for i in {eos, im_end} if i is not None]
    return dict(
        max_new_tokens=max_new_tokens,
        do_sample=False,              # greedy decoding; temperature is not used
        temperature=None, top_p=None, top_k=None,  # clear sampling defaults from generation_config
        repetition_penalty=float(e.get("repetition_penalty", 1.0)),
        num_beams=1,
        eos_token_id=eos_ids or None,
        pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else eos,
    )


class InferenceRunner:
    """Holds a loaded model; call run() once per sample."""

    def __init__(self, model, tokenizer, cfg):
        self.model, self.tokenizer, self.cfg = model, tokenizer, cfg
        self.cache_method = get_cache_method(cfg["experiment"]["cache_method"])
        self.model = self.cache_method.prepare(model, cfg)
        self.device = next(model.parameters()).device
        self.cache_kw = cache_kwarg_name(model)

    @torch.no_grad()
    def measure_prefill(self, input_ids):
        """Prefill-only forward. Logits only for the last position (full-vocab logits for an
        8K prompt would need ~5 GB and distort the memory picture)."""
        cache = self.cache_method.new_cache(self.model)
        _sync()
        t0 = time.perf_counter()
        kw = {self.cache_kw: cache} if cache is not None else {}
        try:
            out = self.model(input_ids=input_ids, use_cache=True, logits_to_keep=1, **kw)
        except TypeError:  # transformers < 4.50 used num_logits_to_keep
            out = self.model(input_ids=input_ids, use_cache=True, num_logits_to_keep=1, **kw)
        _sync()
        dt = time.perf_counter() - t0
        kv = mm.cache_bytes(getattr(out, "past_key_values", None) or getattr(out, "cache", None))
        del out
        return dt, kv

    @torch.no_grad()
    def run(self, input_ids_list, max_new_tokens):
        cfg = self.cfg
        input_ids = torch.tensor([input_ids_list], dtype=torch.long, device=self.device)
        attn = torch.ones_like(input_ids)
        rec = {"prefill_time_seconds": None, "kv_cache_bytes_after_prefill": None}

        if cfg["experiment"].get("measure_prefill", True):
            rec["prefill_time_seconds"], rec["kv_cache_bytes_after_prefill"] = self.measure_prefill(input_ids)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        before = mm.snapshot()
        mm.reset_peak()
        timer = TokenTimer()
        gen_kwargs = greedy_generation_kwargs(cfg, self.tokenizer, max_new_tokens)
        gen_kwargs.update(self.cache_method.generate_kwargs())   # use_cache=True for the full cache
        cache = self.cache_method.new_cache(self.model)
        if cache is not None:
            gen_kwargs[self.cache_kw] = cache

        timer.start()
        out = self.model.generate(input_ids=input_ids, attention_mask=attn, stopping_criteria=[timer],
                                  return_dict_in_generate=True, **gen_kwargs)
        _sync()
        t_end = time.perf_counter()
        after = mm.snapshot()

        new_ids = out.sequences[0, input_ids.shape[1]:].tolist()
        # strip trailing eos/pad so the count is the real answer length
        stop = set(gen_kwargs["eos_token_id"] or []) | {gen_kwargs["pad_token_id"]}
        generated_count = len(new_ids)
        answer_ids = list(new_ids)
        while answer_ids and answer_ids[-1] in stop:
            answer_ids.pop()
        lat = latency_from_stamps(timer.t_start, timer.stamps, t_end, generated_count)

        kv_bytes = mm.cache_bytes(getattr(out, "past_key_values", None) or getattr(out, "cache", None))
        rec.update(
            output_tokens=generated_count,
            output_tokens_excluding_eos=len(answer_ids),
            hit_max_new_tokens=generated_count >= max_new_tokens,
            response=self.tokenizer.decode(answer_ids, skip_special_tokens=True),
            **lat,
            kv_cache_bytes=kv_bytes,   # measured from the returned cache (prompt + generated tokens)
            kv_cache_tokens=input_ids.shape[1] + generated_count - 1 if kv_bytes is not None else None,
            kv_cache_bytes_theoretical_full=mm.theoretical_kv_bytes(
                self.model.config, input_ids.shape[1] + generated_count - 1,
                bytes_per_elem=self.model.dtype.itemsize),
            gpu_allocated_before_bytes=before["allocated_bytes"],
            peak_allocated_gpu_bytes=after["peak_allocated_bytes"],
            peak_reserved_gpu_bytes=after["peak_reserved_bytes"],
            gpu_free_after_bytes=after["free_bytes"],
            gpu_total_bytes=after["total_bytes"],
        )
        # extra memory used by this request on top of what was allocated before it
        if after["peak_allocated_bytes"] is not None and before["allocated_bytes"] is not None:
            rec["request_peak_extra_bytes"] = after["peak_allocated_bytes"] - before["allocated_bytes"]
        else:
            rec["request_peak_extra_bytes"] = None
        del out
        return rec

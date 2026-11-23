"""TTFT / TPOT from per-token timestamps recorded during generate().

TokenTimer is passed to generate() as a StoppingCriteria. transformers calls it once
after every decoding step, i.e. right after each new token exists, so the first call
marks the first token. CUDA is synchronised before reading the clock so timestamps
reflect finished GPU work. It never stops generation.
"""
import time

import torch
from transformers import StoppingCriteria


def _sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


class TokenTimer(StoppingCriteria):
    def __init__(self):
        self.t_start = None
        self.stamps = []

    def start(self):
        _sync()
        self.t_start = time.perf_counter()
        self.stamps = []

    def __call__(self, input_ids, scores, **kwargs):
        _sync()
        self.stamps.append(time.perf_counter())
        return torch.zeros(input_ids.shape[0], dtype=torch.bool, device=input_ids.device)


def latency_from_stamps(t_start, stamps, t_end, output_tokens):
    """TTFT = first token time - start. TPOT = (end - first token) / (n - 1); None if n < 2.

    output_tokens is the number of tokens actually generated (stamps count steps; they
    agree unless generation was interrupted).
    """
    total = t_end - t_start
    if not stamps:
        return {"ttft_seconds": None, "tpot_ms": None, "total_inference_seconds": total,
                "decode_tokens_per_second": None}
    first = stamps[0]
    ttft = first - t_start
    tpot = None
    if output_tokens and output_tokens > 1:
        tpot = (t_end - first) / (output_tokens - 1) * 1000.0
    decode_tps = None
    if output_tokens and output_tokens > 1 and t_end > first:
        decode_tps = (output_tokens - 1) / (t_end - first)
    return {"ttft_seconds": ttft, "tpot_ms": tpot, "total_inference_seconds": total,
            "decode_tokens_per_second": decode_tps}

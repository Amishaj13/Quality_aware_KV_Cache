"""Test 4 (inference) and the prompt-leakage / context-length rules."""
import copy
import json

import pytest

from evaluation import memory_metrics as mm
from inference.generation import ContextOverflow, PROMPT_FIELDS, build_messages, prepare_prompt
from inference.inference_runner import InferenceRunner

HIDDEN = ("reference_answer", "expected_behavior", "tests", "reference_solution", "buggy_solutions",
          "expected_answers", "keywords")


@pytest.fixture(scope="module")
def runner(loaded, cfg):
    model, tok, _ = loaded
    return InferenceRunner(model, tok, cfg)


def test_full_cache_inference_completes(runner, cfg, samples, tmp_path):
    tok = runner.tokenizer
    ids, info = prepare_prompt(tok, samples[0], cfg)
    r = runner.run(ids, info["max_new_tokens"])
    # token counts
    assert info["input_tokens"] == len(ids)
    assert 1 <= r["output_tokens"] <= info["max_new_tokens"]
    assert r["output_tokens_excluding_eos"] <= r["output_tokens"]
    # timing
    assert r["total_inference_seconds"] > 0 and r["ttft_seconds"] > 0
    assert r["ttft_seconds"] <= r["total_inference_seconds"]
    if r["output_tokens"] > 1:
        assert r["tpot_ms"] > 0
    else:
        assert r["tpot_ms"] is None
    # full cache: measured bytes equal the architecture's full-precision size
    assert r["kv_cache_bytes"] == r["kv_cache_bytes_theoretical_full"]
    assert r["kv_cache_bytes_after_prefill"] < r["kv_cache_bytes"]
    # memory: real numbers on GPU, None (unavailable) on CPU
    if mm.cuda_ok():
        assert r["peak_allocated_gpu_bytes"] > 0 and r["peak_reserved_gpu_bytes"] >= r["peak_allocated_gpu_bytes"]
    else:
        assert r["peak_allocated_gpu_bytes"] is None and r["peak_reserved_gpu_bytes"] is None
    # response is saved
    out = tmp_path / "r.json"
    out.write_text(json.dumps(r))
    assert json.loads(out.read_text())["response"] == r["response"]


def test_prompt_never_contains_hidden_fields(samples):
    for s in samples:
        prompt = json.dumps(build_messages(s))
        for k in HIDDEN:
            v = s.get(k)
            if isinstance(v, str) and len(v) > 12 and v not in s["context"] and v not in s["task"]:
                assert v not in prompt, (s["sample_id"], k)
            if isinstance(v, list):
                for t in v:
                    if isinstance(t, str) and len(t) > 25 and t not in s["context"]:
                        assert t not in prompt, (s["sample_id"], k)
    assert set(PROMPT_FIELDS) == {"task", "context"}


def test_context_overflow_skip_and_truncate(runner, cfg, samples):
    tok = runner.tokenizer
    small = copy.deepcopy(cfg)
    small["experiment"]["context_length"] = 120
    small["experiment"]["max_new_tokens"] = 16
    with pytest.raises(ContextOverflow):
        prepare_prompt(tok, samples[-1], small)
    small["experiment"]["context_length"] = 400
    small["experiment"]["overflow_policy"] = "truncate"
    ids, info = prepare_prompt(tok, samples[-1], small)
    assert info["truncated"] is True and info["input_tokens"] <= 400 - 16
    assert info["input_tokens_before_truncation"] > info["input_tokens"]


def test_context_fill_reaches_target(runner, cfg, samples):
    tok = runner.tokenizer
    c = copy.deepcopy(cfg)
    c["context_fill"] = {"enabled": True, "target_tokens": 2000, "depth": 0.5, "seed": 1}
    ids, info = prepare_prompt(tok, samples[0], c)
    assert 1800 <= info["input_tokens"] <= 2000
    assert info["filler_tokens"] > 0 and info["context_fill_depth"] == 0.5
    text = tok.decode(ids)
    assert samples[0]["context"].strip().splitlines()[0] in text

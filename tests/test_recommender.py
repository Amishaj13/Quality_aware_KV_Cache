"""Recommender: memory model, quality rules, constraint handling, choice. No GPU needed.

The synthetic tests build a tiny results/json/ in tmp_path; the last test runs on the real
results when they are present in the repository.
"""
import json
from pathlib import Path

import numpy as np
import pytest

from recommender import profiles as P
from recommender.engine import Request, recommend

B = 57344   # bytes per cached token of Qwen2.5-Coder-7B


def _row(sid, method, prompt, gen, ok_answer, tpot, kv, gpu=None, **extra):
    return {"sample_id": sid, "status": "ok", "eval_type": "unit_tests", "input_tokens": prompt,
            "output_tokens": gen, "functional_correctness": ok_answer, "tpot_ms": tpot,
            "kv_cache_bytes": kv, "kv_cache_gpu_bytes": gpu, **extra}


def _write(tmp_path):
    """custom workload: 20 tasks at prompts 3000-5000. full solves 10; lossy kivi2 solves 8;
    snapkv1024 solves 4; offload is identical to full."""
    j = tmp_path / "json"
    j.mkdir(parents=True)
    rng = np.random.default_rng(0)
    runs = {m: [] for m in ("full", "kivi2", "snapkv1024", "offload")}
    for i in range(20):
        p, g = int(3000 + 100 * i), 100
        cached = p + g - 1
        full_ok = i % 2 == 0
        tf = 35 + 0.35 * p / 1000
        runs["full"].append(_row(f"s{i}", "full", p, g, full_ok, tf, B * cached))
        runs["kivi2"].append(_row(f"s{i}", "kivi2", p, g, full_ok and i not in (0, 2), tf * 1.2,
                                  2_000_000 + 11_000 * cached + rng.normal(0, 1000)))
        runs["snapkv1024"].append(_row(f"s{i}", "snapkv1024", p, g, full_ok and i < 8, tf * 0.97,
                                       B * (1024 + g - 1), snapkv_compressed=True))
        runs["offload"].append(_row(f"s{i}", "offload", p, g, full_ok, tf * (1.0 + 0.13 * p / 1000),
                                    B * cached, gpu=B * cached / 28))
    (j / "baseline_001.json").write_text(json.dumps(runs["full"]))
    for m in ("kivi2", "snapkv1024", "offload"):
        (j / f"baseline_001_{m}.json").write_text(json.dumps(runs[m]))
    return tmp_path


@pytest.fixture
def built(tmp_path):
    return P.build(_write(tmp_path))


def test_profiles_and_memory_model(built):
    profiles, models = built
    prof = profiles["custom"]
    assert set(prof) == {"full", "kivi2", "snapkv1024", "offload"}
    assert models.bytes_per_token == B and models.layers == 28
    assert prof["kivi2"].retention == pytest.approx(80.0)
    assert prof["snapkv1024"].retention == pytest.approx(40.0)
    assert prof["offload"].retention == prof["offload"].retention_lo == 100.0      # identical answers
    assert prof["kivi2"].retention_lo < prof["kivi2"].retention < prof["kivi2"].retention_hi
    # memory formulas
    assert P.gpu_kv_bytes("full", 4000, 100, models) == B * 4099
    assert P.gpu_kv_bytes("snapkv1024", 4000, 100, models) == B * 1123
    assert P.gpu_kv_bytes("snapkv1024", 500, 100, models) == B * 599            # short prompt: nothing evicted
    assert P.gpu_kv_bytes("streaming2048", 4000, 100, models) == B * 2048
    assert P.gpu_kv_bytes("offload", 4000, 100, models) == pytest.approx(B * 4099 / 28)
    assert P.gpu_kv_bytes("kivi2", 4000, 100, models) == pytest.approx(2_000_000 + 11_000 * 4099, rel=0.01)
    assert P.prefill_peak_bytes("snapkv1024", 4000, models) == B * 4000
    assert P.prefill_peak_bytes("snapkv1024", 900, models) == 0


def test_choice_follows_constraints(built):
    profiles, models = built
    run = lambda **kw: recommend(Request("custom", **kw), profiles, models)   # noqa: E731
    assert run()["choice"].method == "offload"                      # no speed limit: least GPU memory, exact
    assert run(max_slowdown=1.3)["choice"].method == "full"         # offload too slow, kivi2 80% < 95%
    assert run(max_slowdown=1.3, min_quality=75)["choice"].method == "kivi2"
    assert run(max_slowdown=1.0, min_quality=30)["choice"].method == "snapkv1024"   # the only one faster than full
    strict = run(max_slowdown=1.3, min_quality=75, strict=True)       # lower bound of kivi2 is below 80
    assert strict["choice"].method in ("full", "kivi2")
    kivi = next(c for c in strict["candidates"] if c.method == "kivi2")
    assert kivi.retention == kivi.retention_lo
    none = run(min_quality=95, free_gpu_gib=0.001, max_slowdown=1.01)
    assert none["choice"] is None and all(c.reasons for c in none["candidates"])


def test_short_prompt_makes_eviction_exact(built):
    profiles, models = built
    out = recommend(Request("custom", prompt_tokens=800, output_tokens=50, max_slowdown=1.0), profiles, models)
    snap = next(c for c in out["candidates"] if c.method == "snapkv1024")
    assert snap.exact and snap.retention == 100 and not snap.evicts
    assert out["choice"].method == "snapkv1024"                       # same memory as full, but faster
    assert next(c for c in out["candidates"] if c.method == "kivi2").extrapolated


def test_batch_and_prefill_peak_count_against_free_memory(built):
    profiles, models = built
    out = recommend(Request("custom", prompt_tokens=4000, output_tokens=100, batch=10, min_quality=30,
                            max_slowdown=1.0), profiles, models)
    snap = next(c for c in out["candidates"] if c.method == "snapkv1024")
    assert snap.required_gib * 2**30 == pytest.approx(10 * B * 1123 + B * 4000)
    tight = recommend(Request("custom", prompt_tokens=4000, output_tokens=100, batch=10, min_quality=30,
                              max_slowdown=1.0, free_gpu_gib=snap.required_gib * 0.99), profiles, models)
    assert tight["choice"] is None


def test_pareto_front(built):
    profiles, models = built
    cands = recommend(Request("custom"), profiles, models)["candidates"]
    front = {c.method for c in cands if c.pareto}
    assert "offload" in front and "full" in front     # full is the fastest at 100 %


def test_cli_runs(tmp_path, capsys):
    import recommend as cli
    root = _write(tmp_path)
    assert cli.main(["--workload", "custom", "--results", str(root), "--max-slowdown", "1.3", "--min-quality", "75"]) == 0
    assert "Recommendation: **kivi2**" in capsys.readouterr().out
    assert cli.main(["--workload", "nope", "--results", str(root)]) == 2


@pytest.mark.skipif(not (Path(__file__).resolve().parent.parent / "results/json/longcodeu_8k_full.json").exists(),
                    reason="measured results not present")
def test_real_results_memory_model_matches():
    from recommender.report import measured_vs_predicted
    root = Path(__file__).resolve().parent.parent / "results"
    profiles, models = P.build(root)
    assert {"custom", "longcodeu", "position"} <= set(profiles)
    for r in measured_vs_predicted(profiles, models, root):
        assert abs(r["error_pct"]) < 10, r

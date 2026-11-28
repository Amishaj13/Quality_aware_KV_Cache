"""Turn the measured runs in results/json/ into per-method profiles the recommender can use.

A *profile* answers three questions for one (workload, method) pair:

  quality   how much of the full cache's quality the method keeps (retention, %), with a
            bootstrap confidence bound, because 18-150 samples are small
  memory    how many bytes of KV cache the method leaves on the GPU, as a function of
            prompt length P and generated length G (so we can predict other lengths)
  speed     how much slower each generated token is than with the full cache (slowdown)

Nothing here needs a GPU: it only reads the JSON files the benchmark runs already wrote.

Memory model (Qwen2.5-Coder-7B: 28 layers x 4 KV heads x 128 dims x K+V x 2 bytes
= 57,344 bytes per cached token; the value is measured from the full-cache runs, not assumed):

  full           b * (P + G - 1)                       every token is kept
  offload        b * (P + G - 1) / layers              one layer at a time on the GPU
  kivi<bits>     a + s * (P + G - 1)                   fitted on all KIVI samples (packed codes
                                                       + 128-token fp16 residual + scales)
  snapkv<N>      b * (min(P, N) + G - 1)               keeps N prompt tokens, then grows
  streaming<N>   b * min(P + G - 1, N)                 fixed window: 4 sinks + N-4 recent tokens

SnapKV and StreamingLLM build the full cache during prefill and evict afterwards, so for a
moment one request needs the full-size cache. The engine adds that transient peak.
"""
import glob
import json
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

WORKLOADS = {
    # name: (full-cache experiment id, pattern for other methods, short description)
    "custom": ("baseline_001", "baseline_001_{m}", "short coding tasks (~240-token prompts)"),
    "longcodeu": ("longcodeu_8k_full", "longcodeu_8k_{m}", "long-code understanding, LongCodeU 0-8K (~4.5K tokens)"),
    "position": ("position_full_d{d}", "position_{m}_d{d}", "find the relevant code inside an ~8K-token file"),
}
ALIASES = {"short": "custom", "code": "custom", "lcu": "longcodeu", "long": "longcodeu",
           "retrieval": "position", "needle": "position"}
DEPTHS = ["000", "025", "050", "075", "100"]
CODE_EVALS = ("unit_tests", "testgen")   # pass@1 is computed over these (same as evaluation/comparison.py)
METHOD_ORDER = ["full", "kivi4", "kivi2", "snapkv1024", "snapkv2048", "streaming1024", "streaming2048", "offload"]


def canonical_workload(name):
    n = name.lower()
    n = ALIASES.get(n, n)
    if n not in WORKLOADS:
        raise ValueError(f"unknown workload {name!r}; choose from {', '.join(WORKLOADS)}")
    return n


# ---------------------------------------------------------------- loading

def _load(path):
    with open(path) as f:
        return json.load(f)


def _ok(rows):
    return {r["sample_id"]: r for r in rows if r.get("status") == "ok"}


def _runs(json_dir, workload):
    """{method: {sample_id: row}} for one workload; position depths are pooled (id@depth)."""
    full_id, pattern, _ = WORKLOADS[workload]
    json_dir = Path(json_dir)
    out = {}
    if workload == "position":
        tags = {re.match(r"position_(.+)_d\d{3}\.json$", Path(p).name).group(1)
                for p in glob.glob(str(json_dir / "position_*_d[0-9][0-9][0-9].json"))}
        for m in tags:
            rows = {}
            for d in DEPTHS:
                p = json_dir / f"position_{m}_d{d}.json"
                if p.exists():
                    for sid, r in _ok(_load(p)).items():
                        rows[f"{sid}@{d}"] = {**r, "_depth": int(d)}
            if rows:
                out[m] = rows
        return out
    fp = json_dir / f"{full_id}.json"
    if fp.exists():
        out["full"] = _ok(_load(fp))
    prefix = pattern.split("{m}")[0]
    for p in glob.glob(str(json_dir / f"{prefix}*.json")):
        m = re.match(re.escape(prefix) + r"([a-z0-9]+)\.json$", Path(p).name)
        if m and m.group(1) != "full":
            out[m.group(1)] = _ok(_load(p))
    return out


def _quality_vector(rows, ids, workload):
    """Per-sample quality on the given ids: 0/1 pass for code tasks, LongCodeU score in [0, 1]."""
    if workload == "longcodeu":
        return np.array([float(rows[i].get("longcodeu_score") or 0.0) for i in ids])
    return np.array([1.0 if rows[i].get("functional_correctness") is True else 0.0 for i in ids])


def _quality_ids(full, comp, workload):
    ids = sorted(set(full) & set(comp))
    if workload != "longcodeu":
        ids = [i for i in ids if full[i].get("eval_type") in CODE_EVALS]
    return ids


def bootstrap_retention(qf, qm, n_boot=2000, seed=0):
    """Paired bootstrap of retention = mean(qm) / mean(qf) x 100. Returns (point, 5th, 95th pct)."""
    if len(qf) == 0 or qf.mean() == 0:
        return None, None, None
    point = qm.mean() / qf.mean() * 100
    if np.array_equal(qf, qm):
        return point, point, point          # identical answers: no uncertainty about the difference
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(qf), size=(n_boot, len(qf)))
    f, m = qf[idx].mean(1), qm[idx].mean(1)
    r = np.where(f > 0, m / np.where(f > 0, f, 1) * 100, np.nan)
    lo, hi = np.nanpercentile(r, [5, 95])
    return point, float(lo), float(hi)


# ---------------------------------------------------------------- profiles

@dataclass
class Profile:
    workload: str
    method: str
    n: int                         # samples the quality numbers are based on
    quality_full: float            # full-cache quality on those samples (0-100)
    quality_method: float
    retention: float               # point estimate, %
    retention_lo: float            # 5th percentile of the paired bootstrap (one-sided 95 % lower bound)
    retention_hi: float
    prompt_tokens: tuple           # (min, mean, max) prompt length measured for this workload
    output_tokens_mean: float
    evicted_share: float = None    # SnapKV / StreamingLLM: share of prompts that were actually cut
    by_depth: dict = field(default_factory=dict)   # position only: depth -> (retention, n)


@dataclass
class Models:
    """Length -> memory / speed models fitted on all workloads together."""
    bytes_per_token: float
    layers: int
    kivi: dict                     # method -> (intercept, slope) in bytes
    slowdown: dict                 # method -> (intercept, slope per 1K prompt tokens)
    tpot_full: tuple               # (intercept ms, slope ms per 1K prompt tokens)


def budget_of(method):
    m = re.match(r"(snapkv|streaming)(\d+)$", method)
    return (m.group(1), int(m.group(2))) if m else (None, None)


def gpu_kv_bytes(method, prompt, gen, models):
    """Predicted KV bytes left on the GPU after generating `gen` tokens for a `prompt`-token prompt."""
    b, cached = models.bytes_per_token, prompt + gen - 1
    kind, n = budget_of(method)
    if method == "full":
        return b * cached
    if method == "offload":
        return b * cached / models.layers
    if method in models.kivi:
        a, s = models.kivi[method]
        return max(a + s * cached, 0.0)
    if kind == "snapkv":
        return b * (min(prompt, n) + gen - 1)
    if kind == "streaming":
        return b * min(cached, n)
    raise ValueError(f"no memory model for method {method!r}")


def prefill_peak_bytes(method, prompt, models):
    """KV bytes one request briefly needs during prefill (eviction methods build the full cache first)."""
    if budget_of(method)[0] and prompt > budget_of(method)[1]:
        return models.bytes_per_token * prompt
    return 0.0


def evicts(method, prompt):
    kind, n = budget_of(method)
    return bool(kind) and prompt > n


def build(results_dir="results"):
    """Read every finished run and return ({workload: {method: Profile}}, Models)."""
    json_dir = Path(results_dir) / "json"
    profiles, mem_pts, slow_pts, tpot_pts = {}, {}, {}, []
    bpt, layer_ratio = [], []
    for wl in WORKLOADS:
        runs = _runs(json_dir, wl)
        full = runs.get("full")
        if not full:
            continue
        profiles[wl] = {}
        prompts = np.array([r["input_tokens"] for r in full.values()])
        outs = np.array([r["output_tokens"] for r in full.values()])
        for r in full.values():
            cached = r["input_tokens"] + r["output_tokens"] - 1
            if cached > 0 and r.get("kv_cache_bytes"):
                bpt.append(r["kv_cache_bytes"] / cached)
            if r.get("tpot_ms"):
                tpot_pts.append((r["input_tokens"] / 1000, r["tpot_ms"]))
        for m in sorted(runs, key=lambda m: (METHOD_ORDER.index(m) if m in METHOD_ORDER else 99, m)):
            comp = runs[m]
            ids = _quality_ids(full, comp, wl)
            qf, qm = _quality_vector(full, ids, wl), _quality_vector(comp, ids, wl)
            point, lo, hi = bootstrap_retention(qf, qm) if m != "full" else (100.0, 100.0, 100.0)
            if point is None:
                continue
            p = Profile(wl, m, len(ids), round(100 * qf.mean(), 1), round(100 * qm.mean(), 1),
                        point, lo, hi, (int(prompts.min()), float(prompts.mean()), int(prompts.max())),
                        float(outs.mean()))
            flag = next((k for k in ("snapkv_compressed", "streaming_compressed") if k in next(iter(comp.values()))), None)
            if flag:
                p.evicted_share = 100 * np.mean([bool(r.get(flag)) for r in comp.values()])
            if wl == "position" and m != "full":
                for d in sorted({full[i]["_depth"] for i in ids}):
                    di = [i for i in ids if full[i]["_depth"] == d]
                    f_, c_ = _quality_vector(full, di, wl), _quality_vector(comp, di, wl)
                    p.by_depth[d] = (None if f_.mean() == 0 else c_.mean() / f_.mean() * 100, len(di))
            profiles[wl][m] = p
            # points for the length models (all samples, not just the scored ones)
            for sid in set(full) & set(comp):
                r, f = comp[sid], full[sid]
                cached = r["input_tokens"] + r["output_tokens"] - 1
                mem_pts.setdefault(m, []).append((cached, r.get("kv_cache_bytes")))
                if m == "offload" and r.get("kv_cache_gpu_bytes") and r.get("kv_cache_bytes"):
                    layer_ratio.append(r["kv_cache_bytes"] / r["kv_cache_gpu_bytes"])
                if m != "full" and r.get("tpot_ms") and f.get("tpot_ms"):
                    slow_pts.setdefault(m, []).append((f["input_tokens"] / 1000, r["tpot_ms"] / f["tpot_ms"]))
    if not profiles:
        raise FileNotFoundError(f"no full-cache runs found in {json_dir}")

    kivi = {}
    for m, pts in mem_pts.items():
        if m.startswith("kivi"):
            x, y = np.array([p[0] for p in pts]), np.array([p[1] for p in pts], dtype=float)
            s, a = np.polyfit(x, y, 1)
            kivi[m] = (float(a), float(s))
    slowdown = {}
    for m, pts in slow_pts.items():
        x, y = np.array(pts).T
        s, a = np.polyfit(x, y, 1) if len(set(x)) > 1 else (0.0, float(y.mean()))
        slowdown[m] = (float(a), float(s))
    tx, ty = np.array(tpot_pts).T
    ts, ta = np.polyfit(tx, ty, 1)
    models = Models(bytes_per_token=float(np.median(bpt)),
                    layers=int(round(np.median(layer_ratio))) if layer_ratio else 28,
                    kivi=kivi, slowdown=slowdown, tpot_full=(float(ta), float(ts)))
    return profiles, models


def predicted_slowdown(method, prompt, models):
    if method == "full":
        return 1.0
    a, s = models.slowdown.get(method, (1.0, 0.0))
    return max(a + s * prompt / 1000, 0.5)


def predicted_tpot_full(prompt, models):
    a, s = models.tpot_full
    return a + s * prompt / 1000

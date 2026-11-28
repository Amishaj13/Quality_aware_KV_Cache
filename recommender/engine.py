"""Rule-based, quality-aware recommender over the measured profiles (ROADMAP section 4).

Given a workload and the constraints of a deployment, it predicts every method's
quality, GPU KV memory and speed at that context length, keeps the methods that meet
all constraints, and returns the one that leaves the least KV cache on the GPU.

Rules, in plain words:

1. Quality. Full cache and CPU offloading are exact (offload gave answers identical to the
   full cache on all samples). SnapKV / StreamingLLM with a budget N are exact when the
   prompt has at most N tokens, because nothing is evicted. Otherwise a method keeps the
   retention measured on that workload. With --strict the 95 % lower confidence bound is
   used instead of the average, so a method only passes if it is *reliably* good enough.
2. Memory. KV bytes on the GPU from the fitted length models in profiles.py, times the
   batch size, plus the brief full-size prefill of one request for eviction methods.
3. Speed. Slowdown per generated token vs the full cache, fitted against prompt length.
4. Choice. Among methods meeting all constraints, the one leaving the least KV cache on the
   GPU wins (the prefill peak only decides whether a method fits, not which saves most);
   within 2 % of each other, higher quality, then lower slowdown, breaks the tie.
"""
from dataclasses import asdict, dataclass, field

from recommender import profiles as P

GIB = 2**30


@dataclass
class Request:
    workload: str
    prompt_tokens: int = None        # default: the workload's measured mean prompt length
    output_tokens: int = None        # default: the workload's measured mean output length
    free_gpu_gib: float = None       # GPU memory available for the KV cache (after weights)
    batch: int = 1                   # concurrent requests sharing that memory
    min_quality: float = 95.0        # minimum quality retention, % of the full cache
    max_slowdown: float = None       # e.g. 1.3 = at most 30 % slower per token
    max_tpot_ms: float = None        # absolute limit on time per generated token
    strict: bool = False             # use the 95 % lower bound of retention
    depth: int = None                # position workload only: depth of the relevant code (0-100)


@dataclass
class Candidate:
    method: str
    retention: float                 # used for the decision (capped at 100, or the lower bound)
    retention_measured: float
    retention_lo: float
    exact: bool                      # quality is identical to the full cache by construction
    gpu_kv_gib: float                # per request, after generation
    required_gib: float              # batch x per-request + prefill peak
    kv_reduction_pct: float
    slowdown: float
    tpot_ms: float
    evicts: bool
    extrapolated: bool
    n: int
    pareto: bool = False
    meets: bool = False
    reasons: list = field(default_factory=list)


def _quality(prof, req, prompt):
    """(retention used, measured point, lower bound, exact?) for one method at this prompt length."""
    m = prof.method
    if m == "full":
        return 100.0, 100.0, 100.0, True
    if m == "offload" and prof.retention_lo >= 100.0:
        return 100.0, prof.retention, prof.retention_lo, True
    if P.budget_of(m)[0] and not P.evicts(m, prompt):
        return 100.0, 100.0, 100.0, True
    point, lo = prof.retention, prof.retention_lo
    if req.depth is not None and prof.by_depth:
        d = min(prof.by_depth, key=lambda k: abs(k - req.depth))
        if prof.by_depth[d][0] is not None:
            point = prof.by_depth[d][0]
    used = lo if req.strict else min(point, 100.0)
    return used, point, lo, False


def evaluate(req, profiles, models):
    wl = P.canonical_workload(req.workload)
    if wl not in profiles:
        raise ValueError(f"no measured runs for workload {wl!r}")
    profs = profiles[wl]
    any_prof = profs["full"]
    prompt = int(req.prompt_tokens or round(any_prof.prompt_tokens[1]))
    gen = int(req.output_tokens or round(any_prof.output_tokens_mean))
    full_kv = P.gpu_kv_bytes("full", prompt, gen, models)
    tpot_full = P.predicted_tpot_full(prompt, models)
    lo_p, _, hi_p = any_prof.prompt_tokens

    cands = []
    for m, prof in profs.items():
        try:
            kv = P.gpu_kv_bytes(m, prompt, gen, models)
        except ValueError:
            continue
        used, point, lo, exact = _quality(prof, req, prompt)
        slow = P.predicted_slowdown(m, prompt, models)
        required = req.batch * kv + P.prefill_peak_bytes(m, prompt, models)
        c = Candidate(m, round(used, 1), round(point, 1), round(lo, 1), exact, kv / GIB, required / GIB,
                      round(100 * (1 - kv / full_kv), 1), round(slow, 2), round(tpot_full * slow, 1),
                      P.evicts(m, prompt), not exact and not (0.8 * lo_p <= prompt <= 1.1 * hi_p), prof.n)
        if c.retention < req.min_quality:
            c.reasons.append(f"quality {c.retention:.0f}% < {req.min_quality:.0f}%")
        if req.free_gpu_gib is not None and c.required_gib > req.free_gpu_gib:
            c.reasons.append(f"needs {c.required_gib:.2f} GiB > {req.free_gpu_gib:.2f} GiB free")
        if req.max_slowdown is not None and c.slowdown > req.max_slowdown:
            c.reasons.append(f"{c.slowdown:.2f}x slower > {req.max_slowdown:.2f}x limit")
        if req.max_tpot_ms is not None and c.tpot_ms > req.max_tpot_ms:
            c.reasons.append(f"{c.tpot_ms:.0f} ms/token > {req.max_tpot_ms:.0f} ms limit")
        c.meets = not c.reasons
        cands.append(c)

    for c in cands:   # Pareto front: no other method is at least as good on all three and better on one
        c.pareto = not any(
            o is not c and o.gpu_kv_gib <= c.gpu_kv_gib and o.retention >= c.retention and o.slowdown <= c.slowdown
            and (o.gpu_kv_gib < c.gpu_kv_gib or o.retention > c.retention or o.slowdown < c.slowdown)
            for o in cands)
    return {"workload": wl, "prompt_tokens": prompt, "output_tokens": gen, "candidates": cands,
            "full_kv_gib": full_kv / GIB, "measured_prompt_range": (lo_p, hi_p)}


def choose(cands):
    ok = [c for c in cands if c.meets]
    if not ok:
        return None
    best_mem = min(c.gpu_kv_gib for c in ok)
    close = [c for c in ok if c.gpu_kv_gib <= best_mem * 1.02 + 1e-12]
    return sorted(close, key=lambda c: (-c.retention, c.slowdown, c.gpu_kv_gib))[0]


def recommend(req, profiles=None, models=None, results_dir="results"):
    if profiles is None:
        profiles, models = P.build(results_dir)
    out = evaluate(req, profiles, models)
    out["choice"] = choose(out["candidates"])
    out["request"] = req
    return out


def to_dict(out):
    d = {k: v for k, v in out.items() if k not in ("candidates", "choice", "request")}
    d["request"] = asdict(out["request"])
    d["choice"] = None if out["choice"] is None else out["choice"].method
    d["candidates"] = [asdict(c) for c in out["candidates"]]
    return d

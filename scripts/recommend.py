"""Recommend a KV-cache method for a workload and a set of constraints.  No GPU needed.

  python main.py recommend --workload longcodeu --free-gpu-gib 7 --min-quality 95
  python main.py recommend --workload position --prompt-tokens 8000 --batch 16 --max-slowdown 1.3
  python main.py recommend --workload custom --strict          # use the 95 % lower confidence bound
  python main.py recommend --report                            # write results/recommender/

Inputs
  --workload        custom | longcodeu | position  (what kind of coding job)
  --prompt-tokens   context length of the prompt   (default: the workload's measured mean)
  --output-tokens   tokens to generate             (default: the workload's measured mean)
  --free-gpu-gib    GPU memory left for the KV cache after the model weights
                    (A10G 24 GB with the 7B model in bf16: about 7 GiB)
  --batch           requests served at the same time (they share that memory)
  --min-quality     minimum quality retention, % of the full cache (default 95)
  --max-slowdown    latency limit as a multiple of the full cache's time per token
  --max-tpot-ms     latency limit in milliseconds per generated token
  --depth           position workload: where the relevant code sits (0, 25, 50, 75, 100 %)

Output: the method that leaves the least KV cache on the GPU while meeting every limit,
then a table of all methods with why each one passed or failed and whether it is on the
Pareto front (no other method is better on memory, quality and speed at once).
"""
import argparse
import json
import sys

import _bootstrap  # noqa: F401
from recommender import profiles as P
from recommender.engine import Request, recommend, to_dict


def table(out):
    rows = ["| method | quality kept | GPU KV / request | needed (batch + prefill) | KV saved | slowdown | ms/token | Pareto | verdict |",
            "|:--|--:|--:|--:|--:|--:|--:|:-:|:--|"]
    order = sorted(out["candidates"], key=lambda c: c.gpu_kv_gib)
    for c in order:
        q = f"{c.retention:.0f}%" + (" (exact)" if c.exact else f" (measured {c.retention_measured:.0f}, low {c.retention_lo:.0f})")
        if c.extrapolated:
            q += " ⚠"
        verdict = ("**chosen**" if out["choice"] is c else "meets all limits") if c.meets else "; ".join(c.reasons)
        rows.append(f"| {c.method} | {q} | {mib(c.gpu_kv_gib)} | {c.required_gib:.2f} GiB | {c.kv_reduction_pct:.0f}% | "
                    f"{c.slowdown:.2f}x | {c.tpot_ms:.0f} | {'●' if c.pareto else ''} | {verdict} |")
    return "\n".join(rows)


def mib(gib):
    return f"{gib * 1024:.0f} MiB" if gib < 1 else f"{gib:.2f} GiB"


def quality_words(c):
    if c.exact:
        return "quality identical to the full cache"
    if c.retention_measured >= 100:
        return f"no quality loss measured ({c.retention_measured:.0f}% of the full cache, within noise)"
    return f"quality {c.retention_measured:.0f}% of the full cache"


def explain(out):
    r, ch = out["request"], out["choice"]
    lim = [f"quality ≥ {r.min_quality:.0f}%" + (" (95% lower bound)" if r.strict else "")]
    if r.free_gpu_gib is not None:
        lim.append(f"≤ {r.free_gpu_gib:g} GiB for the KV cache")
    if r.max_slowdown is not None:
        lim.append(f"≤ {r.max_slowdown:g}x slowdown")
    if r.max_tpot_ms is not None:
        lim.append(f"≤ {r.max_tpot_ms:g} ms/token")
    head = (f"Workload {out['workload']}, prompt {out['prompt_tokens']} tokens, {out['output_tokens']} generated, "
            f"batch {r.batch}. Limits: {', '.join(lim)}.")
    if ch is None:
        best = max(out["candidates"], key=lambda c: (c.meets, -len(c.reasons), c.retention))
        return head + "\n\nNo method meets every limit. Closest: " + f"{best.method} ({'; '.join(best.reasons)})."
    full = next(c for c in out["candidates"] if c.method == "full")
    msg = (f"{head}\n\nRecommendation: **{ch.method}**: {ch.kv_reduction_pct:.0f}% less KV cache on the GPU "
           f"({mib(ch.gpu_kv_gib)} instead of {mib(full.gpu_kv_gib)} per request), "
           f"{quality_words(ch)}, "
           f"{ch.slowdown:.2f}x the full cache's time per token.")
    if ch.method == "offload":
        msg += f" Needs about {full.gpu_kv_gib * r.batch:.2f} GiB of free CPU memory for the cache."
    if ch.extrapolated:
        lo, hi = out["measured_prompt_range"]
        msg += f"\n⚠ The prompt length is outside what was measured for this workload ({lo}-{hi} tokens); treat the quality as an estimate."
    return msg


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workload", default="longcodeu")
    ap.add_argument("--prompt-tokens", type=int)
    ap.add_argument("--output-tokens", type=int)
    ap.add_argument("--free-gpu-gib", type=float)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--min-quality", type=float, default=95.0)
    ap.add_argument("--max-slowdown", type=float)
    ap.add_argument("--max-tpot-ms", type=float)
    ap.add_argument("--depth", type=int)
    ap.add_argument("--strict", action="store_true")
    ap.add_argument("--json", action="store_true", help="print the full result as JSON")
    ap.add_argument("--results", default="results")
    ap.add_argument("--report", action="store_true", help="write Pareto fronts and example scenarios to results/recommender/")
    a = ap.parse_args(argv)

    profiles, models = P.build(a.results)
    if a.report:
        from recommender.report import write_report
        write_report(profiles, models, a.results)
        return 0
    req = Request(a.workload, a.prompt_tokens, a.output_tokens, a.free_gpu_gib, a.batch, a.min_quality,
                  a.max_slowdown, a.max_tpot_ms, a.strict, a.depth)
    try:
        out = recommend(req, profiles, models)
    except ValueError as e:
        print(e, file=sys.stderr)
        return 2
    if a.json:
        print(json.dumps(to_dict(out), indent=2, default=float))
    else:
        print(explain(out) + "\n\n" + table(out))
        print("\nquality kept: low = 95% lower confidence bound (small samples). ● = on the Pareto front. "
              "⚠ = outside the measured prompt lengths.")
    return 0 if out["choice"] else 1


if __name__ == "__main__":
    sys.exit(main())

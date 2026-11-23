"""Builds the comparison table (markdown) and charts (PNG) from bench_kivi.py output."""
import argparse
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

# fixed colour per method (colour follows the method, never its rank)
COLORS = {"FP16": "#2a78d6", "KIVI-K4V4": "#eb6834", "KIVI-K2V2": "#1baf7a"}
ORDER = ["FP16", "KIVI-K4V4", "KIVI-K2V2"]


def load(path):
    rows = [json.loads(l) for l in open(path) if l.strip()]
    df = pd.DataFrame(rows)
    methods = [m for m in ORDER if m in set(df.method)] + sorted(set(df.method) - set(ORDER))
    return df, methods


def style(ax, title, ylabel):
    ax.set_title(title, loc="left", fontsize=12, fontweight="bold", color="#0b0b0b")
    ax.set_ylabel(ylabel, color="#52514e")
    ax.grid(axis="y", color="#e6e5e0", linewidth=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color("#b5b4ae")
    ax.tick_params(colors="#52514e", length=0)


def grouped_bars(df, methods, xcol, ycol, title, ylabel, path, fmt="{:.0f}", xfmt=str):
    xs = sorted(df[xcol].unique())
    width = 0.8 / len(methods)
    fig, ax = plt.subplots(figsize=(8, 4.2), dpi=150)
    for i, m in enumerate(methods):
        sub = df[df.method == m].set_index(xcol)
        for j, x in enumerate(xs):
            pos = j + (i - (len(methods) - 1) / 2) * width
            if x not in sub.index or pd.isna(sub.loc[x, ycol]):
                ax.text(pos, 0, "OOM" if x in sub.index else "", ha="center", va="bottom",
                        fontsize=8, color="#52514e", rotation=90)
                continue
            val = sub.loc[x, ycol]
            ax.bar(pos, val, width * 0.92, color=COLORS.get(m, "#888"), label=m if j == 0 else None)
            ax.text(pos, val, fmt.format(val), ha="center", va="bottom", fontsize=7.5, color="#0b0b0b")
    ax.set_xticks(range(len(xs)))
    ax.set_xticklabels([xfmt(x) for x in xs])
    style(ax, title, ylabel)
    ax.legend(frameon=False, ncols=len(methods), loc="upper left", bbox_to_anchor=(0, -0.1))
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def ctx_label(c):
    return f"{c / 1024:g}K"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inp", default="results/results.jsonl")
    ap.add_argument("--outdir", default="results")
    ap.add_argument("--batch", type=int, default=1, help="batch size to chart")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    df, methods = load(args.inp)
    md = ["# FP16 KV cache vs KIVI on Llama 3.1 8B\n"]

    perf = df[df.kind == "perf"].drop_duplicates(["method", "context", "batch"], keep="last")
    if len(perf):
        gpu = perf.gpu.iloc[0]
        md.append(f"GPU: **{gpu}** · generated tokens per run: {int(perf.gen_tokens.iloc[0])}\n")
        cols = ["method", "context", "batch", "status", "kv_cache_mib", "kv_compression_x",
                "peak_gpu_gib", "ttft_s", "decode_ms_per_token", "decode_tok_per_s"]
        cols = [c for c in cols if c in perf.columns]
        t = perf[cols].copy()
        t["method"] = pd.Categorical(t.method, methods, ordered=True)
        t = t.sort_values(["batch", "context", "method"])
        md.append("## Memory and speed\n")
        md.append(t.to_markdown(index=False))
        md.append("\n`kv_compression_x` = theoretical FP16 KV size / measured cache bytes "
                  "(includes KIVI's scales, zero-points and the FP16 residual window).\n")

        # headline numbers vs FP16 at each context, for the charted batch size
        ok = perf[(perf.status == "ok") & (perf.batch == args.batch)]
        base = ok[ok.method == "FP16"].set_index("context")
        lines = []
        for m in methods:
            if m == "FP16":
                continue
            for _, r in ok[ok.method == m].iterrows():
                c = r.context
                kv_ref = base.kv_cache_mib.get(c, r.fp16_kv_theoretical_mib)
                s = f"- **{m} @ {ctx_label(c)}**: KV cache {r.kv_cache_mib:.0f} MiB vs {kv_ref:.0f} MiB FP16 " \
                    f"({100 * (1 - r.kv_cache_mib / kv_ref):.0f}% smaller)"
                if c in base.index:
                    s += f", peak GPU {r.peak_gpu_gib:.2f} vs {base.peak_gpu_gib[c]:.2f} GiB, " \
                         f"decode {r.decode_tok_per_s:.1f} vs {base.decode_tok_per_s[c]:.1f} tok/s"
                else:
                    s += ", FP16 ran out of memory at this length"
                lines.append(s)
        if lines:
            md.append("## Headline (batch %d)\n" % args.batch)
            md.extend(lines)
            md.append("")

        chart = perf[perf.batch == args.batch].copy()
        for col in ("kv_cache_mib", "peak_gpu_gib", "decode_tok_per_s", "ttft_s"):
            if col in chart:
                chart.loc[chart.status != "ok", col] = float("nan")
        grouped_bars(chart, methods, "context", "kv_cache_mib", "KV cache size (measured)", "MiB",
                     f"{args.outdir}/kv_cache_mib.png", xfmt=ctx_label)
        grouped_bars(chart, methods, "context", "peak_gpu_gib", "Peak GPU memory (weights + cache + activations)",
                     "GiB", f"{args.outdir}/peak_gpu_gib.png", fmt="{:.1f}", xfmt=ctx_label)
        grouped_bars(chart, methods, "context", "decode_tok_per_s", f"Decode throughput, batch {args.batch}",
                     "tokens / s", f"{args.outdir}/decode_tok_per_s.png", fmt="{:.1f}", xfmt=ctx_label)
        grouped_bars(chart, methods, "context", "ttft_s", "Prefill time (time to first token)",
                     "seconds", f"{args.outdir}/ttft_s.png", fmt="{:.2f}", xfmt=ctx_label)

    nd = df[df.kind == "needle"]
    if len(nd):
        nd = nd.drop_duplicates(["method", "context", "depth", "trial"], keep="last")
        nd = nd.assign(correct=nd.correct.astype(float))
        rec = nd.groupby(["method", "context", "depth"]).correct.mean().unstack("depth")
        rec.columns = [{0.1: "begin (10%)", 0.5: "middle (50%)", 0.9: "end (90%)"}.get(c, f"{c:.0%}")
                       for c in rec.columns]
        rec = (100 * rec).round(0).reset_index()
        rec["method"] = pd.Categorical(rec.method, methods, ordered=True)
        rec = rec.sort_values(["context", "method"])
        md.append("## Positional recall (needle in synthetic Python code, % correct)\n")
        md.append(rec.to_markdown(index=False))
        md.append("")

        overall = nd.groupby(["method", "depth"]).correct.mean().reset_index()
        overall["recall_pct"] = 100 * overall.correct
        grouped_bars(overall, methods, "depth", "recall_pct", "Needle recall by position (all context lengths)",
                     "% correct", f"{args.outdir}/needle_recall.png",
                     xfmt=lambda d: {0.1: "begin", 0.5: "middle", 0.9: "end"}.get(d, f"{d:.0%}"))

    with open(f"{args.outdir}/summary.md", "w") as f:
        f.write("\n".join(md) + "\n")
    print("\n".join(md))
    print(f"\n>> wrote {args.outdir}/summary.md and charts")


if __name__ == "__main__":
    main()

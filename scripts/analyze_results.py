"""Charts + summary for one experiment's results.

  python scripts/analyze_results.py                       # experiment_id from the config
  python scripts/analyze_results.py --experiment-id baseline_001

Writes results/charts/<id>_*.png, results/csv/<id>_summary.csv and
results/<id>_summary.md. A 20-30 sample run is a pipeline demonstration, not a
statistically conclusive benchmark; the summary says so.
"""
import argparse
import json
import sys

import _bootstrap  # noqa: F401
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from evaluation.comparison import pass_at_1  # noqa: E402
from utils.config import WORKFLOWS, load_config  # noqa: E402
from utils.paths import results_dirs  # noqa: E402

# Fixed colour + marker per workflow (colour follows the workflow, never its rank).
COLORS = dict(zip(WORKFLOWS, ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#6250d6"]))
MARKERS = dict(zip(WORKFLOWS, ["o", "s", "^", "D", "v"]))
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e6e5e0"
GIB = 1024**3


def style(ax, title, xlabel=None, ylabel=None, grid_axis="y"):
    ax.set_title(title, loc="left", fontsize=12, fontweight="bold", color=INK)
    if xlabel:
        ax.set_xlabel(xlabel, color=INK2)
    if ylabel:
        ax.set_ylabel(ylabel, color=INK2)
    ax.grid(axis=grid_axis, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#b5b4ae")
    ax.tick_params(colors=INK2, length=0)


def legend(ax, workflows):
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], marker=MARKERS[w], linestyle="", markersize=8, color=COLORS[w], label=w)
               for w in workflows]
    ax.legend(handles=handles, frameon=False, ncols=len(workflows), loc="upper left",
              bbox_to_anchor=(0, -0.14), fontsize=9)


def footnote(fig, label):
    fig.text(0.01, 0.005, label, fontsize=7.5, color=INK2, ha="left", va="bottom")


def per_sample_bars(df, col, scale, title, ylabel, path, label):
    d = df.dropna(subset=[col]).copy()
    if d.empty:
        return None
    fig, ax = plt.subplots(figsize=(max(8, 0.32 * len(d) + 2), 4.4), dpi=150)
    x = range(len(d))
    ax.bar(x, d[col] / scale, width=0.8, color=[COLORS.get(w, "#888") for w in d.workflow],
           edgecolor="white", linewidth=1)
    ax.set_xticks(list(x))
    ax.set_xticklabels(d.sample_id.str.replace("python_", ""), rotation=90, fontsize=7.5)
    style(ax, title, "sample", ylabel)
    legend(ax, [w for w in WORKFLOWS if w in set(d.workflow)])
    footnote(fig, label)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(path)
    plt.close(fig)
    return path


def scatter(df, xcol, ycol, title, xlabel, ylabel, path, label):
    d = df.dropna(subset=[xcol, ycol])
    if d.empty:
        return None
    fig, ax = plt.subplots(figsize=(7, 4.6), dpi=150)
    for w in WORKFLOWS:
        s = d[d.workflow == w]
        if len(s):
            ax.scatter(s[xcol], s[ycol], s=46, marker=MARKERS[w], color=COLORS[w], edgecolor="white",
                       linewidth=1.5, zorder=3)
    style(ax, title, xlabel, ylabel, grid_axis="both")
    legend(ax, [w for w in WORKFLOWS if w in set(d.workflow)])
    footnote(fig, label)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(path)
    plt.close(fig)
    return path


def correctness_chart(df, path, label):
    d = df[df.functional_correctness.notna()]
    if d.empty:
        return None
    g = d.groupby("workflow").functional_correctness.agg(["sum", "count"])
    g = g.reindex([w for w in WORKFLOWS if w in g.index])
    fig, ax = plt.subplots(figsize=(7, 4.2), dpi=150)
    pct = 100 * g["sum"] / g["count"]
    bars = ax.bar(range(len(g)), pct, width=0.6, color=[COLORS[w] for w in g.index])
    for i, (b, (_, r)) in enumerate(zip(bars, g.iterrows())):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 1.5, f"{int(r['sum'])}/{int(r['count'])}",
                ha="center", va="bottom", fontsize=9, color=INK)
    ax.set_xticks(range(len(g)))
    ax.set_xticklabels(g.index)
    ax.set_ylim(0, 110)
    style(ax, "Correct answers by workflow (full KV cache)", None, "% correct")
    footnote(fig, label + "\nExplanation tasks (keyword-scored) have no executable ground truth and are not counted here.")
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(path)
    plt.close(fig)
    return path


def longcodeu_chart(df, path, label, csv_path):
    """Mean LongCodeU score (F1 of recall/precision, or CodeBLEU/BLEU) per subtask."""
    if "longcodeu_score" not in df or df.longcodeu_score.isna().all():
        return None
    d = df[df.longcodeu_score.notna()].copy()
    # runs before this fix did not store subtask; recover it from ids like lcu_cu_p_12 / lcu_dra_t1_7
    missing = d.subtask.isna() & d.sample_id.str.startswith("lcu_")
    d.loc[missing, "subtask"] = d.loc[missing, "sample_id"].str[4:].str.rsplit("_", n=1).str[0].str.upper()
    g = d.groupby("subtask").agg(n=("longcodeu_score", "size"), score=("longcodeu_score", "mean"),
                                 recall=("longcodeu_recall", "mean"), precision=("longcodeu_precision", "mean"),
                                 metric=("longcodeu_metric", "first"), input_tokens=("input_tokens", "mean"))
    order = [t for t in ("CU_P", "CU_SA", "CU_DFA", "DRA_T1", "DRA_T2", "SRE_T1", "SRE_T2", "LDU") if t in g.index]
    g = g.reindex(order)
    g.round(4).to_csv(csv_path)
    fig, ax = plt.subplots(figsize=(8, 4.2), dpi=150)
    bars = ax.bar(range(len(g)), 100 * g.score, width=0.6, color=COLORS["longcode"])
    for b, (_, r) in zip(bars, g.iterrows()):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 1.5, f"{100 * r.score:.0f} (n={int(r.n)})",
                ha="center", va="bottom", fontsize=8.5, color=INK)
    ax.set_xticks(range(len(g)))
    ax.set_xticklabels(g.index)
    ax.set_ylim(0, 110)
    style(ax, "LongCodeU score by task (full KV cache)", None, "score × 100")
    footnote(fig, label + "\nScore = F1 of recall/precision (CU_P, CU_DFA, DRA, SRE) or CodeBLEU/BLEU (CU_SA, LDU); "
             "re-implemented metrics")
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(path)
    plt.close(fig)
    return path


def summarize(df):
    ok = df[df.status == "ok"]

    def pct(series):
        s = series.dropna()
        return round(100 * s.astype(bool).mean(), 1) if len(s) else None

    def r(v, n=3):
        return None if v is None or pd.isna(v) else round(float(v), n)

    rows = {
        "experiment_id": df.experiment_id.iloc[0] if len(df) else None,
        "model": df.model.iloc[0] if len(df) else None,
        "cache_method": df.cache_method.iloc[0] if len(df) else None,
        "num_samples": len(df),
        "num_successful": len(ok),
        "num_failed": len(df) - len(ok),
        "failed_by_status": json.dumps(df[df.status != "ok"].status.value_counts().to_dict()),
        "avg_input_tokens": r(ok.input_tokens.mean(), 1) if len(ok) else None,
        "avg_output_tokens": r(ok.output_tokens.mean(), 1) if len(ok) else None,
        "mean_inference_seconds": r(ok.total_inference_seconds.mean()) if len(ok) else None,
        "median_inference_seconds": r(ok.total_inference_seconds.median()) if len(ok) else None,
        "mean_ttft_seconds": r(ok.ttft_seconds.mean(), 4) if "ttft_seconds" in ok and len(ok) else None,
        "mean_tpot_ms": r(ok.tpot_ms.mean(), 2) if "tpot_ms" in ok and len(ok) else None,
        "mean_peak_allocated_gib": r(ok.peak_allocated_gpu_bytes.mean() / GIB) if ok.peak_allocated_gpu_bytes.notna().any() else None,
        "max_peak_allocated_gib": r(ok.peak_allocated_gpu_bytes.max() / GIB) if ok.peak_allocated_gpu_bytes.notna().any() else None,
        "max_peak_reserved_gib": r(ok.peak_reserved_gpu_bytes.max() / GIB) if ok.peak_reserved_gpu_bytes.notna().any() else None,
        "mean_kv_cache_mib": r(ok.kv_cache_bytes.mean() / 1024**2, 1) if ok.kv_cache_bytes.notna().any() else None,
        "syntax_valid_pct": pct(ok.syntax_valid) if "syntax_valid" in ok else None,
        "functional_correctness_pct": pct(ok.functional_correctness) if "functional_correctness" in ok else None,
        "pass_at_1_pct": None if pass_at_1(df.to_dict("records")) is None else round(100 * pass_at_1(df.to_dict("records")), 1),
        "mean_decode_tokens_per_second": r(ok.decode_tokens_per_second.mean(), 2) if "decode_tokens_per_second" in ok and ok.decode_tokens_per_second.notna().any() else None,
        "longcodeu_mean_score": r(ok.longcodeu_score.mean(), 4) if "longcodeu_score" in ok and ok.longcodeu_score.notna().any() else None,
        "functional_correctness_testable_n": int(ok.functional_correctness.notna().sum()) if len(ok) else 0,
        "unit_test_pass_rate_mean": r(ok.unit_test_pass_rate.mean()) if "unit_test_pass_rate" in ok and ok.unit_test_pass_rate.notna().any() else None,
    }
    per_wf = []
    for w, g in ok.groupby("workflow"):
        per_wf.append({"workflow": w, "n": len(g),
                       "functional_correctness_pct": pct(g.functional_correctness),
                       "syntax_valid_pct": pct(g.syntax_valid),
                       "mean_keyword_coverage": r(g.keyword_coverage.mean()) if "keyword_coverage" in g and g.keyword_coverage.notna().any() else None,
                       "mean_inference_seconds": r(g.total_inference_seconds.mean()),
                       "mean_input_tokens": r(g.input_tokens.mean(), 1)})
    return rows, pd.DataFrame(per_wf)


def analyze(experiment_id, results_directory="results"):
    dirs = results_dirs(results_directory)
    src = dirs["json"] / f"{experiment_id}.json"
    if not src.exists():
        sys.exit(f"no results at {src}; run scripts/run_experiments.py first")
    df = pd.DataFrame(json.loads(src.read_text()))
    for c in ("longcodeu_score", "longcodeu_recall", "longcodeu_precision", "longcodeu_metric", "subtask",
              "functional_correctness", "syntax_valid", "peak_allocated_gpu_bytes", "peak_reserved_gpu_bytes",
              "kv_cache_bytes", "ttft_seconds", "tpot_ms", "unit_test_pass_rate", "keyword_coverage",
              "input_tokens", "output_tokens", "total_inference_seconds"):
        if c not in df:
            df[c] = None
    df = df.sort_values("sample_id").reset_index(drop=True)
    ok = df[df.status == "ok"]
    primary = bool(ok.get("is_primary_model", pd.Series([False])).fillna(False).all()) if len(ok) else False
    model = df.model.iloc[0]
    label = f"{experiment_id} · {model} · full KV cache · n={len(ok)}" + \
            ("" if primary else " · PRELIMINARY (not the primary bf16 Qwen-7B run)")

    c = dirs["charts"]
    charts = [
        per_sample_bars(ok, "total_inference_seconds", 1, "Total inference time by sample", "seconds",
                        c / f"{experiment_id}_inference_time.png", label),
        per_sample_bars(ok, "peak_allocated_gpu_bytes", GIB, "Peak allocated GPU memory by sample", "GiB",
                        c / f"{experiment_id}_peak_gpu_memory.png", label),
        scatter(ok, "input_tokens", "total_inference_seconds", "Input tokens vs inference time", "input tokens",
                "seconds", c / f"{experiment_id}_input_tokens_vs_time.png", label),
        scatter(ok, "output_tokens", "total_inference_seconds", "Output tokens vs inference time", "output tokens",
                "seconds", c / f"{experiment_id}_output_tokens_vs_time.png", label),
        correctness_chart(ok, c / f"{experiment_id}_correctness.png", label),
    ]
    charts.append(longcodeu_chart(ok, c / f"{experiment_id}_longcodeu_by_task.png", label, dirs["csv"] / f"{experiment_id}_longcodeu_by_task.csv"))
    summary, per_wf = summarize(df)
    pd.DataFrame([summary]).to_csv(dirs["csv"] / f"{experiment_id}_summary.csv", index=False)
    per_wf.to_csv(dirs["csv"] / f"{experiment_id}_by_workflow.csv", index=False)

    md = [f"# Results summary: {experiment_id}", "",
          f"Model: `{model}`. Cache method: full KV cache (no compression).", "",
          ("" if primary else "**Preliminary pipeline-validation run** (not the primary bf16 Qwen2.5-Coder-7B model).\n"),
          f"This is a {len(df)}-sample demonstration of the pipeline. It is not statistically conclusive, "
          "and no compression method has been evaluated, so no memory savings or quality retention are reported.", "",
          "| metric | value |", "|---|---|"]
    md += [f"| {k} | {v} |" for k, v in summary.items()]
    md += ["", "## By workflow", "", per_wf.to_markdown(index=False) if len(per_wf) else "(no successful samples)",
           "", "## Charts", ""] + [f"![]({p.relative_to(dirs['root']).as_posix()})" for p in charts if p]
    (dirs["root"] / f"{experiment_id}_summary.md").write_text("\n".join(md) + "\n")
    print("\n".join(md[:len(summary) + 8]))
    print("\ncharts:", *[str(p) for p in charts if p], sep="\n  ")
    return summary, [p for p in charts if p]


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--experiment-id")
    p.add_argument("--config", default="configs/experiment_config.yaml")
    p.add_argument("--results-directory")
    a = p.parse_args(argv)
    cfg = load_config(a.config)
    analyze(a.experiment_id or cfg["experiment"]["experiment_id"],
            a.results_directory or cfg["output"].get("results_directory", "results"))


if __name__ == "__main__":
    main()

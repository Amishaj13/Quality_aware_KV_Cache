"""Compare every KV-cache method against the full cache on the same samples.

  python scripts/compare_methods.py            (or: python main.py compare)

Looks for finished runs in results/json/ for three workloads:
  custom      baseline_001 (full)        vs baseline_001_<method>
  longcodeu   longcodeu_8k_full          vs longcodeu_8k_<method>
  position    position_full_dXXX         vs position_<method>_dXXX   (all depths pooled)

For each method and workload, only samples that finished (status ok) in BOTH runs are
compared. Quality is pass@1 for the custom and position tasks (executable tasks) and the
mean LongCodeU score for LongCodeU.

    quality retention  = Q_method / Q_full x 100     (100 = no loss)
    memory reduction   = 1 - KV_method / KV_full     (KV bytes actually stored at the end)

Writes results/comparison/methods_comparison.{md,csv}, a per-depth table, and charts.
With 30 or 80 samples a difference of a few points is within noise; the report says so.
"""
import argparse
import glob
import json
import re
import sys

import _bootstrap  # noqa: F401
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from analyze_results import footnote, style  # noqa: E402
from evaluation.comparison import compression_ratio, memory_reduction, pass_at_1, quality_retention  # noqa: E402
from utils.paths import results_dirs  # noqa: E402

WORKLOADS = {
    "custom": ("baseline_001", "baseline_001_{m}"),
    "longcodeu": ("longcodeu_8k_full", "longcodeu_8k_{m}"),
}
# fixed colour per method (follows the method, never its rank)
METHOD_ORDER = ["full", "kivi4", "kivi2", "snapkv1024", "snapkv2048", "snapkv512", "snapkv4096"]
PALETTE = ["#52514e", "#2a78d6", "#eb6834", "#1baf7a", "#6250d6", "#eda100", "#b5338a"]  # gray = full-cache reference
MARKERS = ["o", "s", "D", "^", "v", "P", "X"]


def method_style(m):
    i = METHOD_ORDER.index(m) if m in METHOD_ORDER else len(METHOD_ORDER) - 1
    return PALETTE[i], MARKERS[i]


def _load(path):
    return pd.DataFrame(json.load(open(path)))


def _quality(df, workload):
    recs = df.to_dict("records")
    if workload == "longcodeu":
        s = df[df.status == "ok"].get("longcodeu_score")
        return None if s is None or s.dropna().empty else float(s.dropna().mean())
    return pass_at_1(recs)


def compare_pair(full, comp, workload):
    """Rows of both runs restricted to samples ok in both."""
    ok = set(full[full.status == "ok"].sample_id) & set(comp[comp.status == "ok"].sample_id)
    f, c = full[full.sample_id.isin(ok)], comp[comp.sample_id.isin(ok)]
    if not ok:
        return {"n_compared": 0, "n_failed_method": int((comp.status != "ok").sum())}
    qf, qc = _quality(f, workload), _quality(c, workload)
    mf, mc = f.kv_cache_bytes.mean(), c.kv_cache_bytes.mean()
    row = {
        "n_compared": len(ok),
        "n_failed_method": int((comp.status != "ok").sum()),
        "quality_full": None if qf is None else round(100 * qf, 1),
        "quality_method": None if qc is None else round(100 * qc, 1),
        "quality_retention_pct": None if quality_retention(qc, qf) is None else round(quality_retention(qc, qf), 1),
        "kv_full_mib": round(mf / 2**20, 1),
        "kv_method_mib": round(mc / 2**20, 1),
        "memory_reduction_pct": None if memory_reduction(mf, mc) is None else round(memory_reduction(mf, mc), 1),
        "compression_ratio": None if compression_ratio(mf, mc) is None else round(compression_ratio(mf, mc), 2),
        "peak_gpu_gib_full": round(f.peak_allocated_gpu_bytes.mean() / 2**30, 2)
        if f.peak_allocated_gpu_bytes.notna().any() else None,
        "peak_gpu_gib_method": round(c.peak_allocated_gpu_bytes.mean() / 2**30, 2)
        if c.peak_allocated_gpu_bytes.notna().any() else None,
        "tpot_ms_full": round(f.tpot_ms.mean(), 1) if f.tpot_ms.notna().any() else None,
        "tpot_ms_method": round(c.tpot_ms.mean(), 1) if c.tpot_ms.notna().any() else None,
        "mean_input_tokens": round(f.input_tokens.mean()),
    }
    if "snapkv_compressed" in c:
        row["snapkv_compressed_share_pct"] = round(100 * c.snapkv_compressed.fillna(False).astype(bool).mean(), 1)
    return row


def methods_for(pattern_prefix, suffix, dirs):
    """Method tags present for a workload, e.g. baseline_001_<m>.json -> m."""
    out = []
    for p in glob.glob(str(dirs["json"] / f"{pattern_prefix}*{suffix}.json")):
        m = re.match(re.escape(pattern_prefix) + r"([a-z0-9]+)" + re.escape(suffix) + r"\.json$",
                     p.rsplit("/", 1)[-1])
        if m and m.group(1) != "full":
            out.append(m.group(1))
    return sorted(set(out), key=lambda m: (METHOD_ORDER.index(m) if m in METHOD_ORDER else 99, m))


def load_position(prefix, dirs):
    frames = []
    for p in sorted(glob.glob(str(dirs["json"] / f"{prefix}_d*.json"))):
        m = re.search(r"_d(\d{3})\.json$", p)
        if m:
            d = _load(p)
            d["depth_pct"] = int(m.group(1))
            d["sample_id"] = d.sample_id + f"@{m.group(1)}"
            frames.append(d)
    return pd.concat(frames, ignore_index=True) if frames else None


def run(results_directory="results"):
    dirs = results_dirs(results_directory)
    out_dir = dirs["root"] / "comparison"
    out_dir.mkdir(exist_ok=True)
    rows, depth_rows = [], []

    for wl, (full_id, pattern) in WORKLOADS.items():
        fp = dirs["json"] / f"{full_id}.json"
        if not fp.exists():
            print(f"[{wl}] no full-cache run ({fp.name}); skipped")
            continue
        full = _load(fp)
        prefix, suffix = pattern.split("{m}")
        for m in methods_for(prefix, suffix, dirs):
            comp = _load(dirs["json"] / f"{pattern.format(m=m)}.json")
            rows.append({"workload": wl, "method": m, **compare_pair(full, comp, wl)})

    full_pos = load_position("position_full", dirs)
    if full_pos is not None:
        tags = sorted({re.match(r"position_(.+)_d\d{3}\.json$", p.rsplit("/", 1)[-1]).group(1)
                       for p in glob.glob(str(dirs["json"] / "position_*_d[0-9][0-9][0-9].json"))} - {"full"},
                      key=lambda m: (METHOD_ORDER.index(m) if m in METHOD_ORDER else 99, m))
        for m in tags:
            comp = load_position(f"position_{m}", dirs)
            depths = sorted(set(comp.depth_pct) & set(full_pos.depth_pct))
            f, c = full_pos[full_pos.depth_pct.isin(depths)], comp[comp.depth_pct.isin(depths)]
            rows.append({"workload": "position", "method": m, **compare_pair(f, c, "position")})
            for d in depths:
                r = compare_pair(f[f.depth_pct == d], c[c.depth_pct == d], "position")
                depth_rows.append({"method": m, "depth_pct": d, **r})
        for d in sorted(set(full_pos.depth_pct)):
            q = pass_at_1(full_pos[full_pos.depth_pct == d].to_dict("records"))
            depth_rows.append({"method": "full", "depth_pct": d, "quality_method": None if q is None else round(100 * q, 1)})

    if not rows:
        sys.exit("nothing to compare: run a compressed method first (see README, 'KIVI and SnapKV')")
    t = pd.DataFrame(rows)
    t.to_csv(out_dir / "methods_comparison.csv", index=False)
    dt = pd.DataFrame(depth_rows)
    if len(dt):
        dt.to_csv(out_dir / "position_by_depth.csv", index=False)

    charts = [tradeoff_chart(t, out_dir / "quality_vs_memory.png")]
    if len(dt):
        charts.append(depth_chart(dt, out_dir / "position_by_depth.png"))

    cols = ["workload", "method", "n_compared", "quality_full", "quality_method", "quality_retention_pct",
            "kv_full_mib", "kv_method_mib", "memory_reduction_pct", "compression_ratio",
            "peak_gpu_gib_full", "peak_gpu_gib_method", "tpot_ms_full", "tpot_ms_method", "mean_input_tokens"]
    if "snapkv_compressed_share_pct" in t:
        cols.append("snapkv_compressed_share_pct")
    md = ["# KV-cache methods vs the full cache", "",
          "Quality: pass@1 (% of executable tasks solved) for `custom` and `position`; mean LongCodeU score "
          "× 100 for `longcodeu`. Retention = method ÷ full × 100. Memory = KV-cache bytes actually stored at "
          "the end of generation, averaged over the compared samples.", "",
          "Small samples (30 custom tasks, 80 LongCodeU, 150 position prompts): differences of a few points "
          "are within noise.", "",
          t[cols].to_markdown(index=False), ""]
    if len(dt):
        piv = dt.pivot_table(index="method", columns="depth_pct", values="quality_method")
        md += ["## Position test: pass@1 by depth of the relevant code", "", piv.to_markdown(), ""]
    md += ["## Charts", ""] + [f"![]({p.name})" for p in charts if p]
    (out_dir / "methods_comparison.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))
    return t


def tradeoff_chart(t, path):
    d = t.dropna(subset=["quality_retention_pct", "memory_reduction_pct"])
    if d.empty:
        return None
    wls = [w for w in ("custom", "longcodeu", "position") if w in set(d.workload)]
    fig, axes = plt.subplots(1, len(wls), figsize=(4.6 * len(wls), 4.4), dpi=150, squeeze=False)
    for ax, wl in zip(axes[0], wls):
        s = d[d.workload == wl]
        ax.axhline(100, color="#b5b4ae", lw=1, ls="--", zorder=1)
        for _, r in s.iterrows():
            col, mk = method_style(r.method)
            ax.scatter(r.memory_reduction_pct, r.quality_retention_pct, s=70, marker=mk, color=col,
                       edgecolor="white", linewidth=2, zorder=3)
            ax.annotate(r.method, (r.memory_reduction_pct, r.quality_retention_pct), xytext=(6, 4),
                        textcoords="offset points", fontsize=8.5, color="#0b0b0b")
        ax.set_xlim(-5, 100)
        lo = min(50, s.quality_retention_pct.min() - 10)
        ax.set_ylim(lo, max(115, s.quality_retention_pct.max() + 10))
        style(ax, wl, "KV memory saved (%)", "quality kept (% of full cache)", grid_axis="both")
    footnote(fig, "Top right is best. Dashed line = full-cache quality. Each point is one method on the same samples.")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(path)
    plt.close(fig)
    return path


def depth_chart(dt, path):
    fig, ax = plt.subplots(figsize=(8, 4.6), dpi=150)
    methods = [m for m in METHOD_ORDER if m in set(dt.method)] + sorted(set(dt.method) - set(METHOD_ORDER))
    for m in methods:
        s = dt[dt.method == m].dropna(subset=["quality_method"]).sort_values("depth_pct")
        col, mk = method_style(m)
        ax.plot(s.depth_pct, s.quality_method, color=col, marker=mk, lw=2, ms=8, label=m,
                markeredgecolor="white", markeredgewidth=1.5)
    depths = sorted(set(dt.depth_pct))
    ax.set_xticks(depths)
    ax.set_xticklabels([f"{d}%" for d in depths])
    ax.set_ylim(-5, 105)
    style(ax, "pass@1 vs position of the relevant code (~8K-token prompts)",
          "position of the task code in the context", "pass@1 (%)")
    ax.legend(frameon=False, ncols=min(len(methods), 6), loc="upper left", bbox_to_anchor=(0, -0.16), fontsize=9)
    footnote(fig, "Same 30 tasks at every depth, padded with unrelated code.")
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(path)
    plt.close(fig)
    return path


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results-directory", default="results")
    a = p.parse_args(argv)
    run(a.results_directory)
    return 0


if __name__ == "__main__":
    sys.exit(main())

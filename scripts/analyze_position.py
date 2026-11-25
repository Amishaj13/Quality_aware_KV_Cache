"""Position-sensitivity chart and table from run_position_sweep.py output.

  python scripts/analyze_position.py [--prefix position_full]

Writes results/csv/<prefix>_by_depth.csv and results/charts/<prefix>_correctness_by_position.png.
"""
import argparse
import glob
import json
import os
import re
import sys

import _bootstrap  # noqa: F401
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from analyze_results import COLORS, MARKERS, footnote, style  # noqa: E402
from evaluation.comparison import pass_at_1  # noqa: E402
from utils.config import WORKFLOWS  # noqa: E402
from utils.paths import results_dirs  # noqa: E402


def load(prefix, dirs):
    rows = []
    for path in sorted(glob.glob(str(dirs["json"] / f"{prefix}_d*.json"))):
        m = re.search(r"_d(\d{3})\.json$", path)
        if not m:
            continue
        for r in json.load(open(path)):
            r["depth_pct"] = int(m.group(1))
            rows.append(r)
    return pd.DataFrame(rows)


def analyze(prefix="position_full", results_directory="results"):
    dirs = results_dirs(results_directory)
    df = load(prefix, dirs)
    if df.empty:
        sys.exit(f"no {prefix}_dXXX.json files in {dirs['json']}; run scripts/run_position_sweep.py first")
    for c in ("functional_correctness", "ttft_seconds", "input_tokens", "is_primary_model"):
        if c not in df:
            df[c] = None
    table = []
    for d, g in df.groupby("depth_pct"):
        ok = g[g.status == "ok"]
        testable = ok.functional_correctness.dropna()
        table.append({"depth_pct": d, "n": len(g), "ok": len(ok),
                      "correct_pct": round(100 * testable.astype(bool).mean(), 1) if len(testable) else None,
                      "pass_at_1_pct": None if pass_at_1(g.to_dict("records")) is None
                      else round(100 * pass_at_1(g.to_dict("records")), 1),
                      "mean_input_tokens": round(ok.input_tokens.mean(), 0) if len(ok) else None,
                      "mean_ttft_seconds": round(ok.ttft_seconds.mean(), 4) if len(ok) else None,
                      **{f"{w}_correct_pct": (round(100 * ok[ok.workflow == w].functional_correctness.dropna()
                                                    .astype(bool).mean(), 1)
                                              if ok[ok.workflow == w].functional_correctness.notna().any() else None)
                         for w in WORKFLOWS}})
    t = pd.DataFrame(table)
    t.to_csv(dirs["csv"] / f"{prefix}_by_depth.csv", index=False)

    primary = bool(df[df.status == "ok"].is_primary_model.fillna(False).all())
    label = f"{prefix} · {df.model.iloc[0]} · {df.cache_method.iloc[0]} KV cache · ~{int(t.mean_input_tokens.mean() or 0)} input tokens" + \
            ("" if primary else " · PRELIMINARY")
    fig, ax = plt.subplots(figsize=(8, 4.6), dpi=150)
    for w in WORKFLOWS:
        col = f"{w}_correct_pct"
        s = t.dropna(subset=[col])
        if len(s):
            ax.plot(s.depth_pct, s[col], color=COLORS[w], marker=MARKERS[w], lw=2, ms=8, label=w,
                    markeredgecolor="white", markeredgewidth=1.5)
    ax.set_xticks(sorted(t.depth_pct))
    ax.set_xticklabels([f"{d}%" for d in sorted(t.depth_pct)])
    ax.set_ylim(-5, 105)
    ax.set_xlim(min(t.depth_pct) - 5, max(t.depth_pct) + 5)
    style(ax, "Correct answers vs position of the relevant code", "position of the task code in the context",
          "% correct", grid_axis="y")
    ax.legend(frameon=False, ncols=5, loc="upper left", bbox_to_anchor=(0, -0.16), fontsize=9)
    footnote(fig, label + "\nExplanation tasks (keyword-scored) are not counted.")
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    out = dirs["charts"] / f"{prefix}_correctness_by_position.png"
    fig.savefig(out)
    plt.close(fig)
    print(t.to_string(index=False))
    print("chart:", out)
    return t, out


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--prefix", default="position_full")
    p.add_argument("--results-directory", default="results")
    a = p.parse_args(argv)
    analyze(a.prefix, a.results_directory)


if __name__ == "__main__":
    main()

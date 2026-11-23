"""Draws docs/architecture.png (research pipeline diagram for slides).  python docs/make_architecture_diagram.py"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402

INK, INK2, EDGE = "#0b0b0b", "#52514e", "#b5b4ae"
NOW, LATER = "#e8f1fc", "#f4f3ef"

boxes = {
    # name: (x, y, w, h, title, subtitle, fill)
    "env": (0.2, 4.6, 2.6, 1.0, "Environment", "server_report · check_gpu\nsetup_env · lock file", NOW),
    "data": (0.2, 3.1, 2.6, 1.0, "Coding tasks", "30 Python tasks · 5 workflows\nhidden tests · group splits", NOW),
    "cfg": (0.2, 1.6, 2.6, 1.0, "Configs (YAML)", "model · experiment · hardware\nvalidated", NOW),
    "model": (3.5, 4.6, 2.8, 1.0, "Model loader", "Qwen2.5-Coder-7B-Instruct\nbf16 · sdpa · loaded once", NOW),
    "prompt": (3.5, 3.1, 2.8, 1.0, "Prompt builder", "chat template · token count\ncontext budget · optional 8K fill", NOW),
    "cache": (3.5, 1.6, 2.8, 1.0, "KV-cache method", "full (now)\nH2O · StreamingLLM · KIVI · SnapKV ...", NOW),
    "infer": (7.0, 3.1, 2.8, 1.0, "Inference runner", "generate(use_cache=True)\nTTFT · TPOT · KV bytes · peak GPU", NOW),
    "eval": (7.0, 1.6, 2.8, 1.0, "Evaluation", "AST syntax · hidden unit tests\ntest-gen bug catching · QA match", NOW),
    "res": (10.5, 3.1, 2.6, 1.0, "Results", "JSONL (progressive)\nJSON · CSV · S3 (optional)", NOW),
    "ana": (10.5, 1.6, 2.6, 1.0, "Analysis", "charts · summary CSV/MD", NOW),
    "rec": (10.5, 0.1, 2.6, 1.0, "Recommender (later)", "workflow + memory + quality\n→ method & cache budget", LATER),
}
arrows = [("env", "model"), ("data", "prompt"), ("cfg", "cache"), ("model", "infer"), ("prompt", "infer"),
          ("cache", "infer"), ("infer", "eval"), ("infer", "res"), ("eval", "res"), ("res", "ana"), ("ana", "rec")]

fig, ax = plt.subplots(figsize=(13.5, 6.4), dpi=160)
ax.set_xlim(0, 13.3)
ax.set_ylim(-0.1, 6.2)
ax.axis("off")
for k, (x, y, w, h, t, s, fill) in boxes.items():
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08", fc=fill, ec=EDGE,
                                lw=1.2, ls="--" if fill == LATER else "-"))
    ax.text(x + 0.12, y + h - 0.2, t, fontsize=10.5, fontweight="bold", color=INK, va="top")
    ax.text(x + 0.12, y + h - 0.5, s, fontsize=8, color=INK2, va="top", linespacing=1.35)


def anchor(k, side):
    x, y, w, h = boxes[k][:4]
    return {"r": (x + w, y + h / 2), "l": (x, y + h / 2), "b": (x + w / 2, y), "t": (x + w / 2, y + h)}[side]


for a, b in arrows:
    ax_, ay, aw, ah = boxes[a][:4]
    bx, by = boxes[b][:2]
    if abs(ax_ - bx) < 0.5:  # same column: vertical
        p0, p1 = (anchor(a, "b"), anchor(b, "t")) if ay > by else (anchor(a, "t"), anchor(b, "b"))
    else:
        p0, p1 = anchor(a, "r"), anchor(b, "l")
    ax.annotate("", xy=p1, xytext=p0, arrowprops=dict(arrowstyle="-|>", color=INK2, lw=1.1,
                                                      connectionstyle="arc3,rad=0.0", shrinkA=2, shrinkB=2))
ax.text(0.2, 6.05, "Quality-aware KV-cache research pipeline: full-cache baseline (phase 1)", fontsize=13,
        fontweight="bold", color=INK)
ax.text(0.2, 0.35, "Solid: implemented now.  Dashed: planned for the next three months.", fontsize=8.5, color=INK2)
out = Path(__file__).resolve().parent / "architecture.png"
fig.savefig(out, bbox_inches="tight")
print("wrote", out)

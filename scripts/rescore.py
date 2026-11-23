"""Re-evaluate saved responses without running the model again (e.g. after a scorer fix).

  python scripts/rescore.py --experiment-id baseline_001
  python main.py rescore --experiment-id baseline_001

Reads results/json/<id>.json, re-runs evaluate_sample on each saved response, and rewrites
the JSONL/JSON/CSV. Timing, memory and the responses themselves are unchanged.
"""
import argparse
import json

import _bootstrap  # noqa: F401

from evaluation.correctness import evaluate_sample
from utils.dataset import load_samples
from utils.paths import results_dirs
from utils.results_io import ResultWriter


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--experiment-id", required=True)
    p.add_argument("--dataset", help="defaults to the path stored in the run metadata")
    p.add_argument("--results-directory", default="results")
    a = p.parse_args(argv)
    dirs = results_dirs(a.results_directory)
    rows = json.loads((dirs["json"] / f"{a.experiment_id}.json").read_text())
    ds = a.dataset
    if not ds:
        meta = dirs["json"] / f"{a.experiment_id}_run_meta.json"
        ds = json.loads(meta.read_text())["config"]["dataset"]["path"] if meta.exists() else "datasets/sample_tasks.json"
    samples = {s["sample_id"]: s for s in load_samples(ds)[0]}
    changed = 0
    for r in rows:
        s = samples.get(r["sample_id"])
        if r.get("status") != "ok" or s is None or r.get("response") is None:
            continue
        before = r.get("functional_correctness")
        r.update(evaluate_sample(s, r["response"]))
        changed += before != r.get("functional_correctness")
        print(f"{r['sample_id']} {r['workflow']:<10} correct: {before} -> {r.get('functional_correctness')}")
    w = ResultWriter(dirs, a.experiment_id)
    w.jsonl.write_text("".join(json.dumps(r, default=str) + "\n" for r in rows))
    w.rebuild()
    print(f"rescored {len(rows)} rows, {changed} changed. Now run: python main.py analyze --experiment-id {a.experiment_id}")


if __name__ == "__main__":
    main()

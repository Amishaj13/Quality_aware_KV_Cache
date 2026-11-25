"""Check that a method that must not change the answers (e.g. offload) really gives the
same responses as the full cache on the samples both runs finished.

  python scripts/check_same_answers.py baseline_001 baseline_001_offload [--min-share 0.8]

Exits with code 1 if fewer than --min-share of the responses are identical.
"""
import argparse
import json
import sys
from pathlib import Path


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("full_id")
    p.add_argument("method_id")
    p.add_argument("--min-share", type=float, default=0.8)
    p.add_argument("--results-directory", default="results")
    a = p.parse_args(argv)
    load = lambda i: {r["sample_id"]: r for r in json.load(open(Path(a.results_directory) / "json" / f"{i}.json"))
                      if r.get("status") == "ok"}
    full, meth = load(a.full_id), load(a.method_id)
    common = sorted(set(full) & set(meth))
    if not common:
        sys.exit("no samples finished in both runs")
    same = [s for s in common if full[s].get("response") == meth[s].get("response")]
    share = len(same) / len(common)
    print(f"{len(same)}/{len(common)} responses identical to the full cache ({100 * share:.0f}%)")
    if share < a.min_share:
        diff = next(s for s in common if s not in same)
        print(f"example {diff}:\n  full:   {full[diff].get('response', '')[:200]!r}\n"
              f"  method: {meth[diff].get('response', '')[:200]!r}")
        print(f"FAILED: fewer than {100 * a.min_share:.0f}% identical")
        return 1
    print("OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())

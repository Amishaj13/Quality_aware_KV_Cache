"""Print why samples failed (evaluation details + the start of each response).

  python scripts/show_failures.py --experiment-id baseline_001 --workflow testgen
"""
import argparse
import json

import _bootstrap  # noqa: F401

from utils.paths import results_dirs


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--experiment-id", required=True)
    p.add_argument("--workflow")
    p.add_argument("--all", action="store_true", help="include correct samples too")
    p.add_argument("--chars", type=int, default=600)
    a = p.parse_args(argv)
    rows = json.loads((results_dirs()["json"] / f"{a.experiment_id}.json").read_text())
    for r in rows:
        if a.workflow and r.get("workflow") != a.workflow:
            continue
        if not a.all and r.get("functional_correctness") is not False:
            continue
        print("=" * 80)
        print(r["sample_id"], r.get("workflow"), "status:", r.get("status"), "syntax_valid:", r.get("syntax_valid"),
              "out_tokens:", r.get("output_tokens"), "tests:", r.get("tests_passed"), "/", r.get("tests_total"))
        print("eval_details:", json.dumps(r.get("eval_details"), indent=1)[:1500])
        print("--- response ---")
        print((r.get("response") or "")[:a.chars])


if __name__ == "__main__":
    main()

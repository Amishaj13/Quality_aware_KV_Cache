"""Single entry point for the full-KV-cache baseline pipeline.

  python main.py report        # environment + GPU report -> results/server_report.json
  python main.py check-gpu     # GPU check and 7B feasibility estimate
  python main.py download      # download model weights to the HF cache
  python main.py first         # one coding task end to end (first inference)
  python main.py run [args]    # full baseline over the dataset (args go to run_experiments.py)
  python main.py analyze       # charts + summary for the configured experiment
  python main.py position      # position sweep 0/25/50/75/100 % at ~8K tokens (full cache)
  python main.py analyze-position
  python main.py compare       # every method vs the full cache -> results/comparison/
  python main.py recommend --workload longcodeu --free-gpu-gib 7 --max-slowdown 1.3
                               # pick the method that saves most GPU KV memory within your limits
  python main.py recommend --report   # Pareto fronts + example choices -> results/recommender/
  python main.py rescore --experiment-id ID   # re-score saved responses (no GPU needed)
  python main.py show-failures --experiment-id ID [--workflow testgen]
  python main.py prepare-longcodeu   # download LongCodeU, build the 0~8K subset (datasets/processed/)
  python main.py s3 [--dry-run]# copy results to an existing S3 bucket
  python main.py demo          # report -> run -> analyze in one go
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "datasets"))

COMMANDS = {
    "report": "server_report", "check-gpu": "check_gpu", "download": "download_model",
    "first": "baseline_inference", "run": "run_experiments", "analyze": "analyze_results",
    "position": "run_position_sweep", "prepare-longcodeu": "prepare_longcodeu",
    "rescore": "rescore", "show-failures": "show_failures", "analyze-position": "analyze_position", "compare": "compare_methods", "recommend": "recommend", "s3": "sync_to_s3",
}


def main():
    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help"):
        print(__doc__)
        return 0
    cmd, rest = sys.argv[1], sys.argv[2:]
    import importlib

    if cmd == "demo":
        importlib.import_module("server_report").main([])
        rc = importlib.import_module("run_experiments").main(rest)
        if rc:
            return rc
        analyze_args = []
        for flag in ("--experiment-id", "--config"):
            if flag in rest:
                analyze_args += [flag, rest[rest.index(flag) + 1]]
        importlib.import_module("analyze_results").main(analyze_args)
        return 0
    if cmd not in COMMANDS:
        print(f"unknown command {cmd}\n{__doc__}")
        return 2
    return importlib.import_module(COMMANDS[cmd]).main(rest) or 0


if __name__ == "__main__":
    sys.exit(main())

"""Position-sensitivity sweep (Lost-in-the-Middle) for one KV-cache method (default: full).

Each task's real code is embedded in ~target_tokens of unrelated synthetic Python at
several relative positions (default 0%, 25%, 50%, 75%, 100% of the context, as in the
project plan). The model is loaded once; each depth is saved as its own experiment
(<prefix>_d000, _d025, ...). This full-cache sweep is the reference curve that each
compression method will later be compared against.

  python scripts/run_position_sweep.py                                   # 5 depths, ~8K tokens, 30 tasks
  python scripts/run_position_sweep.py --depths 0 0.5 1 --max-samples 10 --target-tokens 4096
  python scripts/run_position_sweep.py --resume
  python scripts/run_position_sweep.py --cache-method kivi4        # saved as position_kivi4_d000 ...
  python scripts/run_position_sweep.py --cache-method snapkv --snapkv-budget 1024
  python scripts/run_position_sweep.py --cache-method streamingllm --streaming-budget 1024   # position_streaming1024_d000 ...
"""
import argparse
import copy
import sys

import _bootstrap  # noqa: F401

from cache_methods import method_tag
from models.model_loader import load_model
from run_experiments import log, run
from utils.config import load_config

DEFAULT_DEPTHS = [0.0, 0.25, 0.5, 0.75, 1.0]


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default="configs/experiment_longctx.yaml")
    p.add_argument("--model-config", default="configs/model_config.yaml")
    p.add_argument("--model")
    p.add_argument("--device-map")
    p.add_argument("--depths", type=float, nargs="+", default=DEFAULT_DEPTHS)
    p.add_argument("--target-tokens", type=int)
    p.add_argument("--max-samples", type=int)
    p.add_argument("--cache-method", help="full | kivi4 | kivi2 | kivi | snapkv | streamingllm | offload (default: from the config)")
    p.add_argument("--snapkv-budget", type=int)
    p.add_argument("--streaming-budget", type=int)
    p.add_argument("--prefix", help="experiment id prefix (default: position_<method>, e.g. position_kivi4)")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--overwrite", action="store_true")
    a = p.parse_args(argv)

    ov = {"model": {}, "dataset": {}, "context_fill": {"enabled": True}, "experiment": {}, "snapkv": {}, "streamingllm": {}}
    if a.cache_method:
        ov["experiment"]["cache_method"] = a.cache_method
    if a.snapkv_budget:
        ov["snapkv"]["budget_tokens"] = a.snapkv_budget
    if a.streaming_budget:
        ov["streamingllm"]["budget_tokens"] = a.streaming_budget
    if a.model:
        ov["model"]["name"] = a.model
    if a.device_map:
        ov["model"]["device_map"] = a.device_map
    if a.max_samples:
        ov["dataset"]["max_samples"] = a.max_samples
    if a.target_tokens:
        ov["context_fill"]["target_tokens"] = a.target_tokens
    base = load_config(a.config, a.model_config, ov)
    a.prefix = a.prefix or f"position_{method_tag(base)}"
    import torch
    if torch.cuda.is_available():
        free, total = torch.cuda.mem_get_info()
        if total - free > 2 * 1024**3:
            sys.exit(f"Another process is already using {(total - free) / 1024**3:.1f} GiB of the GPU; "
                     "wait for it to finish (check with nvidia-smi).")
    log(f"loading {base['model']['name']} once for {len(a.depths)} depths ...")
    loaded = load_model(base)
    rc = 0
    for d in a.depths:
        if not 0.0 <= d <= 1.0:
            sys.exit(f"depth {d} outside [0, 1]")
        cfg = copy.deepcopy(base)
        cfg["context_fill"]["depth"] = d
        cfg["experiment"]["experiment_id"] = f"{a.prefix}_d{int(round(d * 100)):03d}"
        cfg["experiment"]["warmup"] = cfg["experiment"].get("warmup", True) and d == a.depths[0]
        log(f"=== depth {d:.2f} -> {cfg['experiment']['experiment_id']} ===")
        rc = run(cfg, resume=a.resume, overwrite=a.overwrite, loaded=loaded) or rc
    log(f"done. Analyse with: python scripts/analyze_position.py --prefix {a.prefix}")
    return rc


if __name__ == "__main__":
    sys.exit(main())

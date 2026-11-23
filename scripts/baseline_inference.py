"""First full-cache coding inference: one sample, printed with its measurements.

  python scripts/baseline_inference.py                 # first sample of the dataset
  python scripts/baseline_inference.py --sample-id python_007
"""
import argparse
import json

import _bootstrap  # noqa: F401

from evaluation.correctness import evaluate_sample
from inference.generation import prepare_prompt
from inference.inference_runner import InferenceRunner
from models.model_loader import load_model
from utils.config import load_config
from utils.dataset import load_samples
from utils.paths import results_dirs


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/experiment_config.yaml")
    p.add_argument("--model")
    p.add_argument("--device-map")
    p.add_argument("--sample-id")
    a = p.parse_args(argv)
    ov = {"model": {}}
    if a.model:
        ov["model"]["name"] = a.model
    if a.device_map:
        ov["model"]["device_map"] = a.device_map
    cfg = load_config(a.config, overrides=ov)
    samples, _ = load_samples(cfg["dataset"]["path"])
    s = next(x for x in samples if x["sample_id"] == a.sample_id) if a.sample_id else samples[0]

    model, tok, info = load_model(cfg)
    print("model:", json.dumps(info, indent=2, default=str))
    runner = InferenceRunner(model, tok, cfg)
    # untimed warm-up so CUDA kernel/library initialisation is not counted in the first measurement
    runner.run(tok("def add(a, b):", add_special_tokens=False)["input_ids"], 8)
    ids, pinfo = prepare_prompt(tok, s, cfg)
    r = runner.run(ids, pinfo["max_new_tokens"])
    ev = evaluate_sample(s, r["response"])
    print("\n=== TASK ===\n" + s["task"])
    print("\n=== RESPONSE ===\n" + r["response"])
    show = {k: r[k] for k in ("output_tokens", "prefill_time_seconds", "ttft_seconds", "tpot_ms",
                              "total_inference_seconds", "kv_cache_bytes", "kv_cache_bytes_theoretical_full",
                              "peak_allocated_gpu_bytes", "peak_reserved_gpu_bytes")}
    print("\n=== MEASUREMENTS ===\n" + json.dumps({**pinfo, **show}, indent=2))
    print("\n=== EVALUATION ===\n" + json.dumps(ev, indent=2, default=str))
    out = results_dirs()["json"] / "first_inference.json"
    out.write_text(json.dumps({"sample_id": s["sample_id"], "model_info": info, **pinfo, **r, **ev},
                              indent=2, default=str))
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()

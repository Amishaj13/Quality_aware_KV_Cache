"""Full-KV-cache baseline runner.

Loads the model once, runs every sample, evaluates it, and appends one record per
sample to results/raw_logs/<experiment_id>.jsonl immediately (JSON and CSV copies are
rebuilt after every sample), so an interrupted notebook keeps finished work.

  python scripts/run_experiments.py                                  # configs/experiment_config.yaml
  python scripts/run_experiments.py --max-samples 3                  # quick check
  python scripts/run_experiments.py --resume                         # skip samples already done
  python scripts/run_experiments.py --model Qwen/Qwen2.5-Coder-1.5B-Instruct --experiment-id fallback_001
"""
import argparse
import datetime as dt
import gc
import json
import platform
import sys
import traceback

import _bootstrap  # noqa: F401
import torch

from evaluation.correctness import evaluate_sample
from inference.generation import ContextOverflow, prepare_prompt
from inference.inference_runner import InferenceRunner
from models.model_loader import ModelLoadError, load_model
from utils.config import load_config
from utils.dataset import load_samples
from utils.paths import results_dirs
from utils.results_io import ResultWriter, new_record


class TokenizerError(RuntimeError):
    pass


def log(msg):
    print(f"[{dt.datetime.now():%H:%M:%S}] {msg}", flush=True)


def is_oom(e):
    return isinstance(e, torch.cuda.OutOfMemoryError) or "out of memory" in str(e).lower()


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default="configs/experiment_config.yaml")
    p.add_argument("--model-config", default="configs/model_config.yaml")
    p.add_argument("--model", help="override model.name (label fallback runs with --experiment-id)")
    p.add_argument("--experiment-id")
    p.add_argument("--max-samples", type=int)
    p.add_argument("--workflow")
    p.add_argument("--split")
    p.add_argument("--quantization", choices=["none", "4bit"])
    p.add_argument("--device-map", help="cuda | auto | cpu (cpu only for tiny debugging models)")
    p.add_argument("--resume", action="store_true", help="skip samples with status ok in the existing log")
    p.add_argument("--overwrite", action="store_true", help="delete earlier results for this experiment_id")
    return p.parse_args(argv)


def build_overrides(a):
    o = {"model": {}, "experiment": {}, "dataset": {}}
    if a.model:
        o["model"]["name"] = a.model
    if a.quantization:
        o["model"]["quantization"] = a.quantization
    if a.device_map:
        o["model"]["device_map"] = a.device_map
    if a.experiment_id:
        o["experiment"]["experiment_id"] = a.experiment_id
    if a.workflow:
        o["experiment"]["workflow"] = a.workflow
    if a.max_samples:
        o["dataset"]["max_samples"] = a.max_samples
    if a.split:
        o["dataset"]["split"] = a.split
    return o


def run(cfg, resume=False, overwrite=False, loaded=None):
    """Run one experiment. `loaded` = (model, tok, info) reuses an already loaded model (position sweep)."""
    e, d = cfg["experiment"], cfg["dataset"]
    dirs = results_dirs(cfg["output"].get("results_directory", "results"))
    writer = ResultWriter(dirs, e["experiment_id"])
    if overwrite:
        writer.reset()
    prior = writer.existing()
    if prior and not (resume or overwrite):
        sys.exit(f"results for experiment_id '{e['experiment_id']}' already exist in {writer.jsonl}. "
                 "Use --resume to continue, --overwrite to start again, or a new --experiment-id.")
    done = {r["sample_id"] for r in prior if r.get("status") == "ok"} if resume else set()

    samples, invalid = load_samples(d["path"], e.get("workflow", "all"), d.get("split", "all"), d.get("max_samples"))
    log(f"{len(samples)} samples selected, {len(invalid)} invalid, {len(done)} already done")

    base = dict(experiment_id=e["experiment_id"], model=cfg["model"]["name"], cache_method=e["cache_method"],
                context_length=e["context_length"])
    run_meta = {
        "started_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "python": platform.python_version(), "torch": torch.__version__,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "config": {k: v for k, v in cfg.items() if not k.startswith("_")},
    }

    for sid, reason in invalid:
        writer.append(new_record(**base, sample_id=sid, status="invalid_sample", error=reason))

    # ---- model (once)
    if not torch.cuda.is_available() and cfg["model"].get("device_map", "cuda") != "cpu":
        sys.exit("No CUDA GPU detected. Start a GPU instance (e.g. ml.g6.xlarge). "
                 "For a CPU pipeline check use --device-map cpu with a tiny model.")
    if loaded is not None:
        model, tok, info = loaded
    else:
        log(f"loading {cfg['model']['name']} ...")
    try:
        if loaded is None:
            model, tok, info = load_model(cfg)
    except ModelLoadError as err:
        log(f"MODEL LOAD FAILED: {err}")
        for s in samples:
            writer.append(new_record(**base, sample_id=s["sample_id"], workflow=s["workflow"],
                                     status="model_load_failed", error=str(err)[:1000]))
        return 2
    log(f"loaded in {info['load_seconds']} s, dtype {info['dtype']}, "
        f"GPU weights {info['gpu_model_memory_bytes'] and round(info['gpu_model_memory_bytes'] / 1024**3, 2)} GiB, "
        f"max_position_embeddings {info['max_position_embeddings']}")
    if info["max_position_embeddings"] and e["context_length"] > info["max_position_embeddings"]:
        log(f"WARNING: context_length {e['context_length']} exceeds the model's max_position_embeddings "
            f"{info['max_position_embeddings']}")
    (dirs["json"] / f"{e['experiment_id']}_run_meta.json").write_text(
        json.dumps({**run_meta, "model_info": info}, indent=2, default=str))
    base.update(model_dtype=info["dtype"], quantization=info["quantization"],
                is_primary_model=info["is_primary_model"], model_gpu_bytes=info["gpu_model_memory_bytes"])
    if not info["is_primary_model"]:
        log("NOTE: not the primary bf16 Qwen2.5-Coder-7B model; label these results as preliminary "
            "pipeline-validation results.")

    runner = InferenceRunner(model, tok, cfg)
    torch.manual_seed(e.get("seed", 0))

    if e.get("warmup", True):
        log("warm-up generation (not recorded)")
        try:
            runner.run(tok("def add(a, b):", add_special_tokens=False)["input_ids"], 8)
        except Exception as err:  # warm-up failure is reported but not fatal
            log(f"warm-up failed: {type(err).__name__}: {err}")

    n_ok = n_fail = 0
    for i, s in enumerate(samples, 1):
        if s["sample_id"] in done:
            continue
        rec = new_record(**base, sample_id=s["sample_id"], workflow=s["workflow"], split=s.get("split"),
                         eval_type=s.get("eval_type"), max_new_tokens=None, truncated=None)
        try:
            try:
                ids, pinfo = prepare_prompt(tok, s, cfg)
            except ContextOverflow:
                raise
            except Exception as err:
                raise TokenizerError(f"{type(err).__name__}: {err}") from err
            rec.update(pinfo)
            r = runner.run(ids, pinfo["max_new_tokens"])
            rec.update(r)
            ev = evaluate_sample(s, r["response"], cfg["evaluation"].get("exec_timeout_seconds", 10))
            rec.update(ev)
            rec["status"] = "ok"
            if not cfg["output"].get("save_responses", True):
                rec.pop("response", None)
            n_ok += 1
        except ContextOverflow as err:
            rec.update(status="context_overflow", error=str(err))
            n_fail += 1
        except TokenizerError as err:
            rec.update(status="tokenizer_error", error=str(err)[:500])
            n_fail += 1
        except Exception as err:
            oom = is_oom(err)
            rec.update(status="oom" if oom else "inference_error", error=f"{type(err).__name__}: {str(err)[:500]}",
                       error_traceback=traceback.format_exc()[-3000:])
            n_fail += 1
        finally:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        writer.append(rec)
        fc = rec.get("functional_correctness")
        log(f"{i}/{len(samples)} {s['sample_id']} {s['workflow']:<10} {rec['status']:<16} "
            f"in={rec.get('input_tokens')} out={rec.get('output_tokens')} "
            f"t={rec.get('total_inference_seconds') and round(rec['total_inference_seconds'], 2)}s "
            f"peak={rec.get('peak_allocated_gpu_bytes') and round(rec['peak_allocated_gpu_bytes'] / 1024**3, 2)}GiB "
            f"correct={fc}")
    log(f"done: {n_ok} ok, {n_fail} failed. Results: {writer.json} and {writer.csv}")
    return 0


def main(argv=None):
    a = parse_args(argv)
    cfg = load_config(a.config, a.model_config, build_overrides(a))
    return run(cfg, resume=a.resume, overwrite=a.overwrite)


if __name__ == "__main__":
    sys.exit(main())

"""Load and validate the YAML configs. Invalid values raise ConfigError with a clear message."""
import copy
from pathlib import Path

import yaml

from utils.paths import resolve

WORKFLOWS = ("longcode", "completion", "repoqa", "debugging", "testgen")
SPLITS = ("all", "train", "validation", "test")
DTYPES = ("auto", "bfloat16", "float16", "float32")


class ConfigError(ValueError):
    pass


def _load_yaml(path):
    path = resolve(path)
    if not path.exists():
        raise ConfigError(f"config file not found: {path}")
    with open(path) as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ConfigError(f"{path} must contain a YAML mapping")
    return data


def _deep_merge(base, override):
    out = copy.deepcopy(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def load_config(experiment_path="configs/experiment_config.yaml",
                model_path="configs/model_config.yaml", overrides=None):
    """Merge model + experiment configs (+ optional dict overrides) and validate."""
    cfg = _deep_merge(_load_yaml(model_path), _load_yaml(experiment_path))
    cfg = _deep_merge(cfg, overrides or {})
    validate_config(cfg)
    cfg["_paths"] = {"experiment": str(resolve(experiment_path)), "model": str(resolve(model_path))}
    return cfg


def _require(cond, msg):
    if not cond:
        raise ConfigError(msg)


def validate_config(cfg):
    from cache_methods import IMPLEMENTED, PLANNED

    m = cfg.get("model") or {}
    _require(isinstance(m.get("name"), str) and m["name"], "model.name must be a non-empty string")
    _require(m.get("dtype", "auto") in DTYPES, f"model.dtype must be one of {DTYPES}")
    _require(m.get("quantization", "none") in ("none", "4bit"), "model.quantization must be none or 4bit")
    _require(m.get("attn_implementation", "sdpa") in ("sdpa", "eager", "flash_attention_2"),
             "model.attn_implementation must be sdpa, eager or flash_attention_2")
    _require(isinstance(m.get("trust_remote_code", False), bool), "model.trust_remote_code must be true/false")

    e = cfg.get("experiment") or {}
    _require(isinstance(e.get("experiment_id"), str) and e["experiment_id"].strip(),
             "experiment.experiment_id must be a non-empty string")
    _require(all(c.isalnum() or c in "-_." for c in e["experiment_id"]),
             "experiment.experiment_id may only contain letters, digits, '-', '_' and '.'")
    wf = e.get("workflow", "all")
    _require(wf == "all" or wf in WORKFLOWS, f"experiment.workflow must be 'all' or one of {WORKFLOWS}")
    cm = e.get("cache_method", "full")
    if cm in PLANNED:
        raise ConfigError(f"cache_method '{cm}' is planned for the next phase and not implemented yet. "
                          f"Implemented: {sorted(IMPLEMENTED)}")
    _require(cm in IMPLEMENTED, f"unknown cache_method '{cm}'. Implemented: {sorted(IMPLEMENTED)}")
    for k in ("context_length", "max_new_tokens", "batch_size"):
        _require(isinstance(e.get(k), int) and not isinstance(e.get(k), bool) and e[k] > 0,
                 f"experiment.{k} must be a positive integer")
    _require(e["batch_size"] == 1, "experiment.batch_size must be 1 in the baseline phase")
    _require(e["max_new_tokens"] < e["context_length"], "max_new_tokens must be smaller than context_length")
    _require(isinstance(e.get("temperature", 0.0), (int, float)) and e.get("temperature", 0.0) >= 0,
             "experiment.temperature must be >= 0")
    _require(e.get("do_sample", False) is False,
             "experiment.do_sample must be false: the baseline uses greedy decoding")
    _require(isinstance(e.get("repetition_penalty", 1.0), (int, float)) and e.get("repetition_penalty", 1.0) > 0,
             "experiment.repetition_penalty must be > 0")
    _require(e.get("overflow_policy", "skip") in ("skip", "truncate"),
             "experiment.overflow_policy must be skip or truncate")

    f = cfg.get("context_fill") or {"enabled": False}
    if f.get("enabled"):
        _require(isinstance(f.get("target_tokens"), int) and 0 < f["target_tokens"] < e["context_length"],
                 "context_fill.target_tokens must be a positive int below context_length")
        _require(0.0 <= float(f.get("depth", 0.5)) <= 1.0, "context_fill.depth must be between 0 and 1")

    d = cfg.get("dataset") or {}
    _require(isinstance(d.get("path"), str), "dataset.path is required")
    hint = " (build it first: python datasets/prepare_longcodeu.py)" if "longcodeu" in d["path"] else ""
    _require(Path(resolve(d["path"])).exists(), f"dataset file not found: {resolve(d['path'])}{hint}")
    _require(d.get("split", "all") in SPLITS, f"dataset.split must be one of {SPLITS}")
    ms = d.get("max_samples")
    _require(ms is None or (isinstance(ms, int) and ms > 0), "dataset.max_samples must be a positive int or null")

    ev = cfg.get("evaluation") or {}
    _require(isinstance(ev.get("exec_timeout_seconds", 10), (int, float)) and ev.get("exec_timeout_seconds", 10) > 0,
             "evaluation.exec_timeout_seconds must be > 0")
    o = cfg.get("output") or {}
    _require(isinstance(o.get("results_directory", "results"), str), "output.results_directory must be a string")
    return cfg

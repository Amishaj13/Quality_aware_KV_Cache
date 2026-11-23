"""Result records: one JSON line per sample, written as soon as the sample finishes,
plus JSON and CSV copies rebuilt from the JSONL after every sample."""
import json
import math
from pathlib import Path

import pandas as pd

# Required schema (fields from the project brief come first). None = not measured.
SCHEMA = {
    "experiment_id": None, "sample_id": None, "model": None, "workflow": None, "cache_method": "full",
    "context_length": None, "input_tokens": None, "output_tokens": None,
    "syntax_valid": None, "functional_correctness": None,
    "prefill_time_seconds": None, "ttft_seconds": None, "tpot_ms": None, "total_inference_seconds": None,
    "kv_cache_bytes": None, "peak_allocated_gpu_bytes": None, "peak_reserved_gpu_bytes": None,
    "status": "pending",
}

# Columns kept out of the CSV (long text); they stay in the JSON/JSONL.
TEXT_FIELDS = ("response", "prompt_preview", "eval_details", "error_traceback")


def new_record(**kw):
    rec = dict(SCHEMA)
    rec.update(kw)
    return rec


def _clean(v):
    if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
        return None
    return v


class ResultWriter:
    def __init__(self, dirs, experiment_id):
        self.jsonl = Path(dirs["raw_logs"]) / f"{experiment_id}.jsonl"
        self.json = Path(dirs["json"]) / f"{experiment_id}.json"
        self.csv = Path(dirs["csv"]) / f"{experiment_id}.csv"

    def existing(self):
        if not self.jsonl.exists():
            return []
        return [json.loads(line) for line in self.jsonl.read_text().splitlines() if line.strip()]

    def reset(self):
        for p in (self.jsonl, self.json, self.csv):
            if p.exists():
                p.unlink()

    def append(self, rec):
        rec = {k: _clean(v) for k, v in rec.items()}
        with open(self.jsonl, "a") as f:
            f.write(json.dumps(rec, default=str) + "\n")
            f.flush()
        self.rebuild()

    def rebuild(self):
        rows = self.existing()
        # keep the last record per sample (a resumed run may retry failed samples)
        by_id = {}
        for r in rows:
            by_id[r["sample_id"]] = r
        rows = list(by_id.values())
        self.json.write_text(json.dumps(rows, indent=2, default=str) + "\n")
        df = pd.DataFrame(rows)
        df = df[[c for c in df.columns if c not in TEXT_FIELDS]]
        df.to_csv(self.csv, index=False)
        return rows

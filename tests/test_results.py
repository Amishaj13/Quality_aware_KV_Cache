"""Test 7 (results): JSON valid, CSV readable, charts generated.

Uses a synthetic fixture written to a temporary directory. These numbers are made up
for the test and never leave tmp_path.
"""
import json

import pandas as pd

from utils.results_io import ResultWriter, SCHEMA, new_record
from utils.paths import results_dirs


def _fixture_records():
    wfs = ["longcode", "completion", "repoqa", "debugging", "testgen"]
    rows = []
    for i in range(10):
        rows.append(new_record(experiment_id="fixture_test", sample_id=f"python_{i + 1:03d}", model="fixture-model",
                               workflow=wfs[i % 5], context_length=8192, input_tokens=200 + 50 * i,
                               output_tokens=40 + 10 * i, syntax_valid=i % 3 != 0,
                               functional_correctness=(i % 2 == 0) if i % 5 != 0 else None,
                               total_inference_seconds=1.0 + 0.1 * i, ttft_seconds=0.1, tpot_ms=25.0,
                               kv_cache_bytes=1000 * i, peak_allocated_gpu_bytes=(15 + 0.01 * i) * 1024**3,
                               peak_reserved_gpu_bytes=16 * 1024**3, status="ok", response="x" * 10,
                               is_primary_model=False))
    rows.append(new_record(experiment_id="fixture_test", sample_id="python_011", model="fixture-model",
                           workflow="repoqa", status="oom", error="CUDA out of memory"))
    return rows


def test_writer_json_csv_and_charts(tmp_path):
    from analyze_results import analyze

    dirs = results_dirs(tmp_path)
    w = ResultWriter(dirs, "fixture_test")
    for r in _fixture_records():
        w.append(r)
    data = json.loads(w.json.read_text())
    assert len(data) == 11 and all(set(SCHEMA) <= set(r) for r in data)
    assert data[-1]["status"] == "oom" and data[-1]["total_inference_seconds"] is None
    df = pd.read_csv(w.csv)
    assert len(df) == 11 and "response" not in df.columns  # long text stays in JSON only
    summary, charts = analyze("fixture_test", str(tmp_path))
    assert summary["num_samples"] == 11 and summary["num_successful"] == 10 and summary["num_failed"] == 1
    assert len(charts) == 5 and all(p.exists() and p.stat().st_size > 5000 for p in charts)
    assert (dirs["csv"] / "fixture_test_summary.csv").exists()
    # non-primary models are labelled as preliminary
    assert "Preliminary pipeline-validation run" in (dirs["root"] / "fixture_test_summary.md").read_text()


def test_resume_keeps_last_record_per_sample(tmp_path):
    dirs = results_dirs(tmp_path)
    w = ResultWriter(dirs, "resume_test")
    w.append(new_record(experiment_id="resume_test", sample_id="python_001", status="oom"))
    w.append(new_record(experiment_id="resume_test", sample_id="python_001", status="ok"))
    rows = w.rebuild()
    assert len(rows) == 1 and rows[0]["status"] == "ok"


def test_position_sweep_end_to_end(tmp_path, model_name, device_map):
    """Runs the full-cache sweep at 2 depths on 3 tasks with the test model, then the chart."""
    import run_position_sweep
    from analyze_position import analyze as analyze_pos
    from utils.config import load_config  # noqa: F401

    import yaml
    cfg = yaml.safe_load(open("configs/experiment_longctx.yaml"))
    cfg["output"]["results_directory"] = str(tmp_path)
    cfg["experiment"]["max_new_tokens"] = 16
    cfg["experiment"]["warmup"] = False
    cfg["context_fill"]["target_tokens"] = 1024
    path = tmp_path / "sweep.yaml"
    path.write_text(yaml.safe_dump(cfg))
    rc = run_position_sweep.main(["--config", str(path), "--model", model_name, "--device-map", device_map,
                                  "--depths", "0", "1", "--max-samples", "3", "--prefix", "pos_test"])
    assert rc == 0
    t, chart = analyze_pos("pos_test", str(tmp_path))
    assert list(t.depth_pct) == [0, 100] and all(t.ok == 3) and chart.exists()
    assert 900 <= t.mean_input_tokens.min() <= 1024

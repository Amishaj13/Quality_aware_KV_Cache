"""LongCodeU: metric re-implementations and the loader -> runner path on a local fixture
that mimics the Hub layout (data/<TASK>/<id>.jsonl with id, context_length, prompt, ans).
The fixture records are invented for the test; they are not LongCodeU data."""
import json

import pytest

from evaluation.longcodeu_metrics import normalize_answers, parse_name_list, score, split_units

F1 = "def load(path):\n    with open(path) as f:\n        return f.read()\n"
F2 = "def save(path, data):\n    with open(path, 'w') as f:\n        f.write(data)\n"


def test_normalize_answers_variants():
    assert normalize_answers(["a", "b"]) == ["a", "b"]
    assert normalize_answers('["a", "b"]') == ["a", "b"]
    assert normalize_answers("['a']") == ["a"]
    assert normalize_answers("plain text") == ["plain text"]


def test_cu_p_lcs():
    perfect = score("CU_P", "```python\n['load', 'save', 'run']\n```", ["load", "save", "run"])
    assert perfect["recall"] == perfect["precision"] == perfect["score"] == 1.0
    swapped = score("CU_P", "- run\n- load\n- save", ["load", "save", "run"])   # order matters (LCS)
    assert swapped["recall"] == pytest.approx(2 / 3, abs=1e-3)
    assert parse_name_list("1. load\n2. def save(") == ["load", "save"]


def test_cu_dfa_exact_lines():
    r = score("CU_DFA", "```python\nx = 1\n  x  +=  2\nprint(x)\n```", ["x = 1", "x += 2"])
    assert r["recall"] == 1.0 and r["precision"] == pytest.approx(2 / 3, abs=1e-3)


def test_dra_unit_matching():
    assert [n for n, _ in split_units(F1 + "\n" + F2)] == ["load", "save"]
    exact = score("DRA_T1", f"```python\n{F1}\n{F2}```", [F1, F2])
    assert exact["recall"] > 0.9 and exact["precision"] > 0.9
    wrong = score("DRA_T1", "```python\ndef other():\n    pass\n```", [F1, F2])
    assert wrong["score"] == 0.0
    half = score("SRE_T1", f"```python\n{F1}```", [F1, F2])
    assert 0.4 < half["recall"] < 0.6 and half["precision"] > 0.9


def test_cu_sa_and_ldu_similarity():
    assert score("CU_SA", f"```python\n{F1}```", [F1])["score"] > 0.9
    assert score("CU_SA", "```python\ndef z():\n    return 0\n```", [F1])["score"] < 0.5
    assert score("LDU", "Reads the file at path and returns its text.",
                 ["Reads the file at path and returns its text."])["score"] > 0.9


@pytest.fixture()
def fake_hub(tmp_path):
    code = "\n\n".join(F1.replace("load", f"load_{i}") for i in range(20))
    recs = {
        "CU_P": {"ans": [f"load_{i}" for i in range(20)], "q": "List every function name in order."},
        "DRA_T1": {"ans": [F2], "q": "Return the source of the functions called by load_3."},
        "LDU": {"ans": "Saves data to path.", "q": "Give the documentation of save."},
    }
    for task, r in recs.items():
        d = tmp_path / "data" / task
        d.mkdir(parents=True)
        lines = []
        for i in range(3):
            lines.append(json.dumps({"id": i, "context_length": "0~8K" if i < 2 else "8~16K",
                                     "prompt": f"You are an expert Python programmer.\n```python\n{code}\n```\n{r['q']}",
                                     "ans": r["ans"]}))
        (d / "0.jsonl").write_text("\n".join(lines) + "\n")
    return tmp_path


def test_prepare_and_run_longcodeu(fake_hub, tmp_path, model_name, device_map):
    import yaml

    import prepare_longcodeu  # noqa: E402  (datasets/ on sys.path via conftest)
    from run_experiments import run
    from utils.config import load_config
    from utils.dataset import load_samples

    out = tmp_path / "lcu.json"
    rc = prepare_longcodeu.main(["--local-dir", str(fake_hub), "--tasks", "CU_P", "DRA_T1", "LDU",
                                 "--per-task", "5", "--tokenizer", model_name, "--out", str(out)])
    assert rc == 0
    samples, invalid = load_samples(str(out))
    assert not invalid and len(samples) == 6                       # 2 per task in the 0~8K bucket
    assert all(s["raw_prompt"] and s["eval_type"] == "longcodeu" for s in samples)
    assert all(json.dumps(s["answers"]) not in s["task"] for s in samples if s["subtask"] == "CU_P")

    cfg = yaml.safe_load(open("configs/experiment_longcodeu.yaml"))
    cfg["dataset"]["path"] = str(out)
    cfg["output"]["results_directory"] = str(tmp_path / "res")
    cfg["experiment"].update(max_new_tokens=16, warmup=False, measure_prefill=False)
    cpath = tmp_path / "c.yaml"
    cpath.write_text(yaml.safe_dump(cfg))
    c = load_config(str(cpath), overrides={"model": {"name": model_name, "device_map": device_map}})
    assert run(c, overwrite=True) == 0
    rows = json.loads((tmp_path / "res/json/longcodeu_8k_full.json").read_text())
    assert len(rows) == 6 and all(r["status"] == "ok" for r in rows)
    assert all(r["longcodeu_score"] is not None and r["functional_correctness"] is None for r in rows)
    from analyze_results import analyze
    summary, charts = analyze("longcodeu_8k_full", str(tmp_path / "res"))
    assert any("longcodeu_by_task" in str(p) for p in charts) and summary["longcodeu_mean_score"] is not None
    import pandas as pd
    by_task = pd.read_csv(tmp_path / "res/csv/longcodeu_8k_full_longcodeu_by_task.csv")
    assert set(by_task.subtask) == {"CU_P", "DRA_T1", "LDU"} and by_task.n.sum() == 6
    # results saved before subtask was recorded are recovered from the sample id
    rows = json.loads((tmp_path / "res/json/longcodeu_8k_full.json").read_text())
    for r in rows:
        r.pop("subtask", None)
    (tmp_path / "res/json/longcodeu_8k_full.json").write_text(json.dumps(rows))
    analyze("longcodeu_8k_full", str(tmp_path / "res"))
    by_task = pd.read_csv(tmp_path / "res/csv/longcodeu_8k_full_longcodeu_by_task.csv")
    assert set(by_task.subtask) == {"CU_P", "DRA_T1", "LDU"}


def test_official_output_formats():
    """Formats seen in real Qwen2.5-Coder-7B outputs on the A10G run (Nov 2026)."""
    r = score("CU_P", "```\nFunction_names = [b, c, d]\n```", ["a", "b", "c", "d", "e"])
    assert r["recall"] == pytest.approx(0.6) and r["precision"] == 1.0
    prompt = "```python\n" + F1 + "\n\n" + F2 + "\nclass K:\n    def m(self):\n        return 1\n```\nInstructions"
    resp = ("```\n###### function_1 ######\n" + F1 + "\n\n###### function_2 ######\n" + F2 +
            "\n\n###### function_3 ######\ndef extra():\n    pass\n```")
    r = score("DRA_T1", resp, ["load", "save"], prompt=prompt)    # references are bare names
    assert r["name_recall"] == 1.0 and r["name_precision"] == pytest.approx(2 / 3, abs=1e-3)
    assert r["recall"] > 0.95 and r["precision"] == pytest.approx(r["recall"] * 2 / 3, abs=1e-3)
    assert score("SRE_T1", resp, ["K", "load"], prompt=prompt)["name_recall"] == 0.5

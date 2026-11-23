"""Test 6 (evaluation), dataset integrity and config validation."""
import pytest

from evaluation.correctness import (evaluate_sample, extract_code, run_generated_tests, run_unit_tests,
                                    syntax_valid)
from evaluation.quality_metrics import answer_match, keyword_coverage, token_f1
from utils.config import ConfigError, load_config

ADD_OK = "def add(a, b):\n    return a + b\n"
ADD_BAD = "def add(a, b):\n    return a - b\n"
ADD_TESTS = ["assert add(2, 3) == 5", "assert add(-1, 1) == 0"]


def test_syntax_validation():
    assert syntax_valid(ADD_OK)
    assert not syntax_valid("def add(a, b)\n    return a + b")
    assert not syntax_valid("")


def test_correct_function_passes():
    r = run_unit_tests(ADD_OK, ADD_TESTS)
    assert r["tests_passed"] == 2 and r["pass_rate"] == 1.0


def test_incorrect_function_fails():
    r = run_unit_tests(ADD_BAD, ADD_TESTS)
    assert r["tests_passed"] == 0 and r["failures"]


def test_infinite_loop_times_out():
    r = run_unit_tests("def add(a, b):\n    while True:\n        pass\n", ["assert add(1, 1) == 2"], timeout=2)
    assert r["status"] == "timeout" and r["tests_passed"] == 0


def test_extract_code_prefers_entry_point_block():
    resp = "Here:\n```python\nprint('demo')\n```\nand\n```python\ndef add(a, b):\n    return a + b\n```\n"
    assert extract_code(resp, "add").startswith("def add")
    assert extract_code("```python\ndef f():\n    return 1") .startswith("def f")  # unterminated fence


def test_every_reference_solution_passes_its_hidden_tests(samples):
    for s in samples:
        if s["eval_type"] == "unit_tests":
            r = run_unit_tests(s["reference_solution"], s["tests"])
            assert r["pass_rate"] == 1.0, (s["sample_id"], r["failures"])


def test_every_debugging_context_fails_its_tests(samples):
    for s in samples:
        if s["workflow"] == "debugging":
            assert run_unit_tests(s["context"], s["tests"])["pass_rate"] < 1.0, s["sample_id"]


def test_testgen_scoring_detects_bugs(samples):
    s = next(x for x in samples if x["eval_type"] == "testgen" and "clamp" in x["task"])
    good_tests = ("```python\nimport pytest\nfrom solution import clamp\n\n"
                  "def test_inside():\n    assert clamp(5, 0, 10) == 5\n\n"
                  "def test_low_high():\n    assert clamp(-1, 0, 10) == 0\n    assert clamp(11, 0, 10) == 10\n\n"
                  "def test_bad_bounds():\n    with pytest.raises(ValueError):\n        clamp(1, 5, 0)\n```")
    ev = evaluate_sample(s, good_tests)
    assert ev["syntax_valid"] and ev["eval_details"]["passes_on_reference"]
    assert ev["eval_details"]["bugs_caught"] == ev["eval_details"]["bugs_total"] == 2
    assert ev["functional_correctness"] is True
    weak = "```python\nfrom solution import clamp\n\ndef test_one():\n    assert clamp(5, 0, 10) == 5\n```"
    assert evaluate_sample(s, weak)["functional_correctness"] is False


def test_testgen_wrong_expectation_fails_on_reference():
    r = run_generated_tests("from solution import f\n\ndef test_f():\n    assert f() == 2\n", "def f():\n    return 1\n")
    assert r["all_passed"] is False


def test_unit_test_sample_end_to_end(samples):
    s = next(x for x in samples if x["sample_id"] == "python_001")
    ok = evaluate_sample(s, "```python\n" + s["reference_solution"] + "```")
    assert ok["syntax_valid"] and ok["functional_correctness"] and ok["unit_test_pass_rate"] == 1.0
    bad = evaluate_sample(s, "```python\ndef binary_search(items, target):\n    return 0\n```")
    assert bad["functional_correctness"] is False and 0 < bad["unit_test_pass_rate"] < 1


def test_text_metrics():
    assert token_f1("the total price", "total price") > 0.7
    assert keyword_coverage("It applies a discount and tax", [["discount"], ["tax"], ["round"]]) == pytest.approx(2 / 3)
    contains, exact = answer_match("The answer is 102.6.", ["102.6"])
    assert contains and not exact
    assert answer_match("102.6", ["102.6"]) == (True, True)
    assert answer_match("1102.6", ["102.6"])[0] is False


def test_groups_never_cross_splits(samples):
    by_group = {}
    for s in samples:
        by_group.setdefault(s["group_id"], set()).add(s["split"])
    assert all(len(v) == 1 for v in by_group.values())
    assert len(samples) == 30 and {s["workflow"] for s in samples} == {"longcode", "completion", "repoqa",
                                                                         "debugging", "testgen"}


@pytest.mark.parametrize("override,msg", [
    ({"experiment": {"cache_method": "h2o"}}, "not implemented"),
    ({"experiment": {"batch_size": 2}}, "batch_size"),
    ({"experiment": {"do_sample": True}}, "greedy"),
    ({"experiment": {"max_new_tokens": 9000}}, "smaller than context_length"),
    ({"dataset": {"path": "datasets/missing.json"}}, "not found"),
    ({"model": {"dtype": "int3"}}, "dtype"),
])
def test_config_validation_rejects_bad_values(override, msg):
    with pytest.raises(ConfigError, match=msg):
        load_config(overrides=override)


def test_default_configs_are_valid():
    cfg = load_config()
    assert cfg["experiment"]["cache_method"] == "full" and cfg["experiment"]["context_length"] == 8192
    load_config("configs/experiment_longctx.yaml")


def test_comparison_formulas():
    from evaluation.comparison import compression_ratio, memory_reduction, pass_at_1, quality_retention
    assert quality_retention(76, 80) == pytest.approx(95.0)       # example from the project plan
    assert memory_reduction(8, 2) == pytest.approx(75.0)
    assert compression_ratio(8, 2) == pytest.approx(4.0)
    assert quality_retention(50, 0) is None and memory_reduction(None, 1) is None
    recs = [{"eval_type": "unit_tests", "status": "ok", "functional_correctness": True},
            {"eval_type": "testgen", "status": "oom", "functional_correctness": None},
            {"eval_type": "keywords", "status": "ok", "functional_correctness": None}]
    assert pass_at_1(recs) == pytest.approx(0.5)

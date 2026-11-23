"""Code correctness: syntax (AST), controlled execution of hidden unit tests, and
validity / bug-detection of model-written tests.

Hidden tests and reference solutions come from the dataset and are never placed in
the model prompt. Code runs in a separate Python process with a timeout, no stdin,
an isolated interpreter (-I), a temporary working directory and (on Linux) memory and
CPU limits. This is a safeguard against hangs and accidents, not a security sandbox:
only run it on model outputs for these benchmark tasks.
"""
import ast
import json
import os
import re
import subprocess
import sys
import tempfile
import textwrap

from evaluation.quality_metrics import answer_match, keyword_coverage, token_f1

RESULT_MARK = "__KVR_RESULT__"
_FENCE = re.compile(r"```[ \t]*([A-Za-z0-9_+-]*)[^\n]*\n(.*?)```", re.S)


def extract_code(response, entry_point=None):
    """Pick the code block to evaluate from a model response.

    Preference: a python fenced block defining entry_point > the longest python block >
    any fenced block > the raw response if it parses as Python. Returns "" if none.
    """
    if not response:
        return ""
    blocks = [(lang.lower(), body) for lang, body in _FENCE.findall(response)]
    # unterminated fence (generation hit max_new_tokens): take the rest
    if not blocks and "```" in response:
        tail = response.split("```", 1)[1]
        first_nl = tail.find("\n")
        blocks = [("python", tail[first_nl + 1:] if first_nl >= 0 else tail)]
    py = [b for lang, b in blocks if lang in ("python", "py", "python3", "")]
    if entry_point:
        pat = re.compile(rf"^\s*(async\s+)?def\s+{re.escape(entry_point)}\s*\(|^\s*class\s+{re.escape(entry_point)}\b", re.M)
        for b in py:
            if pat.search(b):
                return b.strip("\n")
    if py:
        return max(py, key=len).strip("\n")
    if blocks:
        return blocks[0][1].strip("\n")
    return response.strip() if syntax_valid(response.strip()) else ""


def syntax_valid(code):
    if not code or not code.strip():
        return False
    try:
        ast.parse(code)
        return True
    except (SyntaxError, ValueError):
        return False


def _limits():  # pragma: no cover - runs in the child process
    try:
        import resource
        resource.setrlimit(resource.RLIMIT_AS, (2 * 1024**3, 2 * 1024**3))
        resource.setrlimit(resource.RLIMIT_CPU, (30, 30))
    except Exception:
        pass


def _run_python(source, timeout):
    """Run source in a fresh isolated interpreter. Returns (parsed_result | None, stderr_tail, status)."""
    with tempfile.TemporaryDirectory(prefix="kvr_exec_") as tmp:
        path = os.path.join(tmp, "harness.py")
        with open(path, "w") as f:
            f.write(source)
        env = {"PATH": os.environ.get("PATH", ""), "PYTHONDONTWRITEBYTECODE": "1", "PYTHONHASHSEED": "0"}
        try:
            proc = subprocess.run([sys.executable, "-I", path], cwd=tmp, env=env, stdin=subprocess.DEVNULL,
                                  capture_output=True, text=True, timeout=timeout,
                                  preexec_fn=_limits if os.name == "posix" else None)
        except subprocess.TimeoutExpired:
            return None, "timeout", "timeout"
    result = None
    for line in proc.stdout.splitlines():
        if line.startswith(RESULT_MARK):
            try:
                result = json.loads(line[len(RESULT_MARK):])
            except json.JSONDecodeError:
                result = None
    status = "ok" if result is not None else "crashed"
    return result, proc.stderr[-2000:], status


_UNIT_HARNESS = '''
import json as _kvr_json
_kvr_ns = {"__name__": "solution"}
_kvr_load_error = None
try:
    exec(compile(_KVR_CODE, "solution.py", "exec"), _kvr_ns)
except BaseException as _e:
    _kvr_load_error = type(_e).__name__ + ": " + str(_e)[:300]
_kvr_res = []
for _t in _KVR_TESTS:
    if _kvr_load_error:
        _kvr_res.append([False, "load: " + _kvr_load_error]); continue
    try:
        exec(compile(_t, "test", "exec"), dict(_kvr_ns))
        _kvr_res.append([True, ""])
    except BaseException as _e:
        _kvr_res.append([False, type(_e).__name__ + ": " + str(_e)[:200]])
print(%r + _kvr_json.dumps(_kvr_res))
''' % RESULT_MARK


def run_unit_tests(code, tests, setup="", timeout=10):
    """Execute candidate code, then each hidden test (assert snippet) in its own namespace copy.

    Returns dict(tests_passed, tests_total, pass_rate, failures, status).
    """
    source = (f"_KVR_CODE = {setup + chr(10) + code!r}\n_KVR_TESTS = {list(tests)!r}\n" + _UNIT_HARNESS)
    res, err, status = _run_python(source, timeout)
    if res is None:
        return {"tests_passed": 0, "tests_total": len(tests), "pass_rate": 0.0,
                "failures": [err.strip().splitlines()[-1] if err.strip() else status], "status": status}
    passed = sum(1 for ok, _ in res if ok)
    return {"tests_passed": passed, "tests_total": len(tests), "pass_rate": passed / max(len(tests), 1),
            "failures": [m for ok, m in res if not ok][:5], "status": status}


_TESTGEN_HARNESS = '''
import json as _kvr_json, sys as _kvr_sys, types as _kvr_types
_kvr_mod = _kvr_types.ModuleType(_KVR_MODULE)
exec(compile(_KVR_IMPL, _KVR_MODULE + ".py", "exec"), _kvr_mod.__dict__)
_kvr_sys.modules[_KVR_MODULE] = _kvr_mod
_kvr_ns = {"__name__": "test_generated"}
_kvr_ns.update({k: v for k, v in _kvr_mod.__dict__.items() if not k.startswith("__")})
_kvr_out = {"collected": 0, "passed": 0, "errors": []}
try:
    exec(compile(_KVR_TESTS, "test_generated.py", "exec"), _kvr_ns)
    _kvr_top_ok = True
except BaseException as _e:
    _kvr_top_ok = False
    _kvr_out["errors"].append("module: " + type(_e).__name__ + ": " + str(_e)[:200])
_kvr_out["top_level_ok"] = _kvr_top_ok
for _name, _fn in list(_kvr_ns.items()):
    if _name.startswith("test") and callable(_fn) and not isinstance(_fn, type):
        _kvr_out["collected"] += 1
        try:
            _fn()
            _kvr_out["passed"] += 1
        except BaseException as _e:
            _kvr_out["errors"].append(_name + ": " + type(_e).__name__ + ": " + str(_e)[:200])
    elif _name.startswith("Test") and isinstance(_fn, type):
        _inst = _fn()
        for _m in dir(_inst):
            if _m.startswith("test"):
                _kvr_out["collected"] += 1
                try:
                    getattr(_inst, _m)()
                    _kvr_out["passed"] += 1
                except BaseException as _e:
                    _kvr_out["errors"].append(_m + ": " + type(_e).__name__ + ": " + str(_e)[:200])
print(%r + _kvr_json.dumps(_kvr_out))
''' % RESULT_MARK


def run_generated_tests(test_code, implementation, module_name="solution", timeout=10):
    """Run model-written tests against an implementation (importable as `module_name`).

    A test suite "passes" if its module-level code runs and every collected test_*
    function passes. Suites with no test functions count as passing only if they ran
    top-level asserts without error (some models write bare asserts).
    """
    source = (f"_KVR_IMPL = {implementation!r}\n_KVR_TESTS = {test_code!r}\n_KVR_MODULE = {module_name!r}\n"
              + _TESTGEN_HARNESS)
    res, err, status = _run_python(source, timeout)
    if res is None:
        return {"all_passed": False, "collected": 0, "passed": 0, "status": status,
                "errors": [err.strip().splitlines()[-1] if err.strip() else status]}
    has_asserts = "assert" in test_code
    all_passed = res["top_level_ok"] and res["passed"] == res["collected"] and (res["collected"] > 0 or has_asserts)
    return {"all_passed": all_passed, "collected": res["collected"], "passed": res["passed"],
            "status": status, "errors": res["errors"][:5]}


def evaluate_sample(sample, response, timeout=10):
    """Score one response. Unused metrics stay None (never guessed)."""
    out = {"syntax_valid": None, "functional_correctness": None, "tests_passed": None, "tests_total": None,
           "unit_test_pass_rate": None, "exact_match": None, "answer_contains_expected": None,
           "keyword_coverage": None, "reference_token_f1": None, "eval_type": sample.get("eval_type"),
           "eval_details": None}
    et = sample.get("eval_type")
    if sample.get("reference_answer"):
        out["reference_token_f1"] = round(token_f1(response, sample["reference_answer"]), 4)

    if et == "unit_tests":
        code = extract_code(response, sample.get("entry_point"))
        out["syntax_valid"] = syntax_valid(code)
        if out["syntax_valid"]:
            r = run_unit_tests(code, sample["tests"], sample.get("test_setup", ""), timeout)
            out.update(tests_passed=r["tests_passed"], tests_total=r["tests_total"],
                       unit_test_pass_rate=round(r["pass_rate"], 4),
                       functional_correctness=r["tests_passed"] == r["tests_total"],
                       eval_details={"exec_status": r["status"], "failures": r["failures"]})
        else:
            out.update(tests_passed=0, tests_total=len(sample["tests"]), unit_test_pass_rate=0.0,
                       functional_correctness=False, eval_details={"exec_status": "syntax_error"})
    elif et == "testgen":
        code = extract_code(response)
        out["syntax_valid"] = syntax_valid(code)
        if out["syntax_valid"]:
            mod = sample.get("module_name", "solution")
            good = run_generated_tests(code, sample["reference_solution"], mod, timeout)
            bugs = [run_generated_tests(code, b, mod, timeout) for b in sample.get("buggy_solutions", [])]
            caught = sum(1 for b in bugs if not b["all_passed"])
            # valid tests must pass on the correct implementation and catch every planted bug
            out.update(functional_correctness=bool(good["all_passed"] and bugs and caught == len(bugs)),
                       tests_passed=good["passed"], tests_total=good["collected"],
                       unit_test_pass_rate=round(good["passed"] / good["collected"], 4) if good["collected"] else None,
                       eval_details={"passes_on_reference": good["all_passed"], "reference_errors": good["errors"],
                                     "bugs_caught": caught, "bugs_total": len(bugs)})
        else:
            out.update(functional_correctness=False, eval_details={"exec_status": "syntax_error"})
    elif et == "answer_match":
        contains, exact = answer_match(response, sample.get("expected_answers"))
        out.update(answer_contains_expected=contains, exact_match=exact, functional_correctness=contains)
    elif et == "keywords":
        cov = keyword_coverage(response, sample.get("keywords"))
        out["keyword_coverage"] = None if cov is None else round(cov, 4)
        # explanations have no executable ground truth: functional_correctness stays None
    return out


def dedent(s):
    return textwrap.dedent(s).strip("\n") + "\n"

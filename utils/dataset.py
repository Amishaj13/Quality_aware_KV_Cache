"""Load and validate datasets/sample_tasks.json."""
import json

from utils.config import WORKFLOWS
from utils.paths import resolve

REQUIRED = ("sample_id", "workflow", "language", "task", "context", "reference_answer",
            "expected_behavior", "max_new_tokens")
EVAL_TYPES = ("unit_tests", "testgen", "answer_match", "keywords")


class InvalidSample(ValueError):
    pass


def validate_sample(s):
    missing = [k for k in REQUIRED if k not in s]
    if missing:
        raise InvalidSample(f"{s.get('sample_id', '?')}: missing fields {missing}")
    if s["workflow"] not in WORKFLOWS:
        raise InvalidSample(f"{s['sample_id']}: unknown workflow {s['workflow']}")
    if not isinstance(s["task"], str) or not s["task"].strip():
        raise InvalidSample(f"{s['sample_id']}: empty task")
    if not isinstance(s["max_new_tokens"], int) or s["max_new_tokens"] <= 0:
        raise InvalidSample(f"{s['sample_id']}: max_new_tokens must be a positive int")
    et = s.get("eval_type")
    if et not in EVAL_TYPES:
        raise InvalidSample(f"{s['sample_id']}: eval_type must be one of {EVAL_TYPES}")
    if et == "unit_tests" and not s.get("tests"):
        raise InvalidSample(f"{s['sample_id']}: unit_tests sample without tests")
    if et == "testgen" and not (s.get("reference_solution") and s.get("buggy_solutions")):
        raise InvalidSample(f"{s['sample_id']}: testgen sample needs reference_solution and buggy_solutions")
    if et == "answer_match" and not s.get("expected_answers"):
        raise InvalidSample(f"{s['sample_id']}: answer_match sample without expected_answers")
    return s


def load_samples(path, workflow="all", split="all", max_samples=None):
    """Return (valid_samples, invalid list of (sample_id, reason))."""
    with open(resolve(path)) as f:
        raw = json.load(f)
    valid, invalid, seen = [], [], set()
    for s in raw:
        try:
            validate_sample(s)
            if s["sample_id"] in seen:
                raise InvalidSample(f"duplicate sample_id {s['sample_id']}")
            seen.add(s["sample_id"])
        except InvalidSample as e:
            invalid.append((s.get("sample_id", "?") if isinstance(s, dict) else "?", str(e)))
            continue
        if workflow != "all" and s["workflow"] != workflow:
            continue
        if split != "all" and s.get("split") != split:
            continue
        valid.append(s)
    if max_samples:
        valid = valid[:max_samples]
    return valid, invalid

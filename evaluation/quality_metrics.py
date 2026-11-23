"""Text-answer metrics for explanation and repository-QA tasks (no code execution)."""
import re
import string
from collections import Counter

_PUNCT = str.maketrans("", "", string.punctuation.replace("_", ""))


def normalize(text):
    text = (text or "").lower().translate(_PUNCT)
    return " ".join(text.split())


def token_f1(prediction, reference):
    p, r = normalize(prediction).split(), normalize(reference).split()
    if not p or not r:
        return 0.0
    common = Counter(p) & Counter(r)
    overlap = sum(common.values())
    if overlap == 0:
        return 0.0
    precision, recall = overlap / len(p), overlap / len(r)
    return 2 * precision * recall / (precision + recall)


def keyword_coverage(response, keyword_groups):
    """Fraction of keyword groups with at least one synonym present in the response.

    keyword_groups: list of lists, e.g. [["sum", "total"], ["quantity"]].
    """
    if not keyword_groups:
        return None
    norm = normalize(response)
    hits = 0
    for group in keyword_groups:
        group = [group] if isinstance(group, str) else group
        if any(re.search(r"(?<![a-z0-9_])" + re.escape(normalize(k)) + r"(?![a-z0-9_])", norm) for k in group):
            hits += 1
    return hits / len(keyword_groups)


def answer_match(response, expected_answers):
    """(contains, exact): does the response contain / exactly equal any expected answer."""
    if not expected_answers:
        return None, None
    norm = normalize(response)
    norm_code = normalize(re.sub(r"`", "", response or ""))
    exp = [normalize(str(a)) for a in expected_answers]
    contains = any(re.search(r"(?<![a-z0-9_])" + re.escape(a) + r"(?![a-z0-9_])", norm_code) for a in exp if a)
    exact = any(norm_code == a or norm == a for a in exp)
    return contains, exact

"""LongCodeU scoring (Li et al., ACL 2025, arXiv 2503.04359), re-implemented from the paper's
metric descriptions. The official scoring code was not available to check against, so these
are close re-implementations; numbers are comparable across our own runs (full cache vs
compressed), not necessarily to the paper's tables.

Task         Paper metric        Here
CU_P         LCS-R / LCS-P       LCS over the ordered list of function names
CU_DFA       EM-R / EM-P         exact match of whitespace-normalised lines
CU_SA        CodeBLEU            CodeBLEU (pip codebleu + tree-sitter-python), else BLEU-4
DRA_T1/T2,   CB-R / CB-P         per predicted unit whose name is in the reference set:
SRE_T1/T2                        CodeBLEU vs that reference unit (0 otherwise); recall = sum/|ref|,
                                 precision = sum/|pred|
LDU          BLEU                BLEU-4 with smoothing
`score` is F1 of recall/precision, or the similarity itself for CU_SA / LDU.
"""
import ast
import json
import math
import re
from collections import Counter

TASKS = ("CU_P", "CU_SA", "CU_DFA", "DRA_T1", "DRA_T2", "SRE_T1", "SRE_T2", "LDU")
_FENCE = re.compile(r"```[^\n]*\n(.*?)```", re.S)
_DEF = re.compile(r"^[ \t]*(?:async[ \t]+)?def[ \t]+([A-Za-z_]\w*)[ \t]*\(", re.M)
_TOK = re.compile(r"[A-Za-z_]\w*|\d+|\S")

_CODEBLEU = None


def _codebleu():
    global _CODEBLEU
    if _CODEBLEU is None:
        try:
            from codebleu import calc_codebleu
            calc_codebleu(["x = 1"], ["x = 1"], lang="python")
            _CODEBLEU = calc_codebleu
        except Exception:
            _CODEBLEU = False
    return _CODEBLEU


def similarity_metric_name():
    return "codebleu" if _codebleu() else "bleu4"


def bleu(pred, ref, n=4):
    p, r = _TOK.findall(pred or ""), _TOK.findall(ref or "")
    if not p or not r:
        return 0.0
    logs = []
    for k in range(1, n + 1):
        pc = Counter(tuple(p[i:i + k]) for i in range(len(p) - k + 1))
        rc = Counter(tuple(r[i:i + k]) for i in range(len(r) - k + 1))
        overlap = sum((pc & rc).values())
        total = max(sum(pc.values()), 0)
        logs.append(math.log((overlap + 1) / (total + 1)))  # add-one smoothing
    bp = 1.0 if len(p) > len(r) else math.exp(1 - len(r) / len(p))
    return bp * math.exp(sum(logs) / n)


def code_similarity(pred, ref):
    cb = _codebleu()
    if cb and pred.strip() and ref.strip():
        try:
            return float(cb([ref], [pred], lang="python")["codebleu"])
        except Exception:
            pass
    return bleu(pred, ref)


def normalize_answers(ans):
    """Reference answers arrive as a list, a JSON/Python-literal string, or plain text."""
    if ans is None:
        return []
    if isinstance(ans, str):
        s = ans.strip()
        for parse in (json.loads, ast.literal_eval):
            try:
                v = parse(s)
                if isinstance(v, (list, tuple)):
                    return [str(x) for x in v]
            except Exception:
                pass
        return [s]
    return [str(x) for x in ans]


def strip_fences(text):
    blocks = _FENCE.findall(text or "")
    return "\n".join(blocks) if blocks else (text or "")


def parse_name_list(text):
    body = strip_fences(text).strip()
    for parse in (json.loads, ast.literal_eval):
        try:
            v = parse(body)
            if isinstance(v, (list, tuple)):
                return [str(x).strip().split("(")[0].split(".")[-1] for x in v if str(x).strip()]
        except Exception:
            pass
    names = []
    for line in body.splitlines():
        line = re.sub(r"^\s*(?:[-*]|\d+[.)])\s*", "", line).strip().strip("`'\",[]")
        m = re.match(r"(?:def\s+)?([A-Za-z_][\w.]*)", line)
        if m:
            names.append(m.group(1).split(".")[-1])
    return names


def split_units(code):
    """Split code into (name, source) per top-level-or-method def."""
    code = strip_fences(code)
    starts = [(m.start(), m.group(1)) for m in _DEF.finditer(code)]
    units = []
    for i, (pos, name) in enumerate(starts):
        end = starts[i + 1][0] if i + 1 < len(starts) else len(code)
        units.append((name, code[pos:end].strip("\n")))
    return units


def _lcs(a, b):
    dp = [0] * (len(b) + 1)
    for x in a:
        prev = 0
        for j, y in enumerate(b, 1):
            cur = dp[j]
            dp[j] = prev + 1 if x == y else max(dp[j], dp[j - 1])
            prev = cur
    return dp[-1]


def _f1(r, p):
    return 0.0 if r + p == 0 else 2 * r * p / (r + p)


def _norm_line(s):
    return " ".join(s.strip().split())


def score(task, response, ans):
    ref = normalize_answers(ans)
    out = {"recall": None, "precision": None, "score": None, "metric": None}
    if task == "CU_P":
        pred, gold = parse_name_list(response), [r.strip().split("(")[0] for r in ref]
        l = _lcs(pred, gold)
        rec, prec = (l / len(gold) if gold else 0.0), (l / len(pred) if pred else 0.0)
        out.update(recall=rec, precision=prec, score=_f1(rec, prec), metric="lcs")
    elif task == "CU_DFA":
        pred = {_norm_line(x) for x in strip_fences(response).splitlines() if x.strip()}
        gold = {_norm_line(x) for r in ref for x in r.splitlines() if x.strip()}
        hit = len(pred & gold)
        rec, prec = (hit / len(gold) if gold else 0.0), (hit / len(pred) if pred else 0.0)
        out.update(recall=rec, precision=prec, score=_f1(rec, prec), metric="exact_match_lines")
    elif task == "CU_SA":
        s = max((code_similarity(strip_fences(response), r) for r in ref), default=0.0)
        out.update(score=s, metric=similarity_metric_name())
    elif task in ("DRA_T1", "DRA_T2", "SRE_T1", "SRE_T2"):
        gold = {}
        for r in ref:
            units = split_units(r) or [(r.strip().split("(")[0], r)]
            for name, src in units[:1]:
                gold[name] = src
        pred = split_units(response)
        if not pred:  # model listed names only
            pred = [(n, "") for n in parse_name_list(response)]
        total = 0.0
        for name, src in pred:
            if name in gold:
                total += code_similarity(src, gold[name]) if src else 0.0
        rec = total / len(gold) if gold else 0.0
        prec = total / len(pred) if pred else 0.0
        out.update(recall=rec, precision=prec, score=_f1(rec, prec), metric=f"{similarity_metric_name()}_by_unit")
    elif task == "LDU":
        s = max((bleu(response, r) for r in ref), default=0.0)
        out.update(score=s, metric="bleu4")
    else:
        raise ValueError(f"unknown LongCodeU task {task}")
    return {k: (round(v, 4) if isinstance(v, float) else v) for k, v in out.items()}

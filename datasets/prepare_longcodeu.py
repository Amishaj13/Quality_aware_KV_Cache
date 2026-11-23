"""Download LongCodeU and build an 8K-context subset in this pipeline's sample format.

LongCodeU (Li et al., ACL 2025, arXiv 2503.04359): 8 long-code-understanding tasks built
from 116 real Python repositories. Hugging Face: longcodeu/longcodeu-dataset, licence
Apache-2.0 (dataset card). The code inside it comes from public GitHub repositories with
their own licences; use it for research evaluation and keep the attribution. The data is
not committed to this repository; run this script on the instance.

Layout on the Hub: data/<TASK>/<id>.jsonl with fields id, context_length ("0~8K", ...),
prompt (long code + instruction, used verbatim), ans (reference answer list).

  python datasets/prepare_longcodeu.py                       # 10 per task, 0~8K, fits 8192 - 512
  python datasets/prepare_longcodeu.py --per-task 25 --tasks CU_P CU_DFA DRA_T1
  python datasets/prepare_longcodeu.py --inspect             # print the raw format of one record per task

Writes datasets/processed/longcodeu_8k.json (+ a _report.json with counts and skips).
"""
import argparse
import hashlib
import json
import os
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from evaluation.longcodeu_metrics import TASKS, normalize_answers  # noqa: E402

REPO = "longcodeu/longcodeu-dataset"
WORKFLOW = {"CU_P": "longcode", "CU_SA": "longcode", "CU_DFA": "longcode", "LDU": "longcode",
            "DRA_T1": "repoqa", "DRA_T2": "repoqa", "SRE_T1": "repoqa", "SRE_T2": "repoqa"}


def iter_task_records(task, local_dir=None, revision="main"):
    """Yield raw records of one task, from a local copy or the Hub (download is cached)."""
    if local_dir:
        files = sorted(Path(local_dir, "data", task).glob("*.jsonl"))
    else:
        from huggingface_hub import snapshot_download
        path = snapshot_download(REPO, repo_type="dataset", revision=revision,
                                 allow_patterns=[f"data/{task}/*.jsonl"],
                                 token=os.environ.get("HF_TOKEN") or None)
        files = sorted(Path(path, "data", task).glob("*.jsonl"))
    if not files:
        raise FileNotFoundError(f"no files for task {task}")
    for f in files:
        for line in f.read_text().splitlines():
            if line.strip():
                rec = json.loads(line)
                rec.setdefault("_file", f.name)
                yield rec


def to_sample(task, rec, n_tokens):
    rid = rec.get("id", rec.get("_file"))
    answers = normalize_answers(rec.get("ans"))
    group = f"longcodeu_{task}_{rid}"
    h = int(hashlib.sha256(group.encode()).hexdigest(), 16) % 10
    return {
        "sample_id": f"lcu_{task.lower()}_{rid}",
        "group_id": group,
        "split": "train" if h < 6 else ("validation" if h < 8 else "test"),
        "benchmark": "longcodeu",
        "subtask": task,
        "workflow": WORKFLOW[task],
        "language": "python",
        "raw_prompt": True,
        "task": rec["prompt"],
        "context": "",
        "context_length_bucket": rec.get("context_length"),
        "prompt_tokens_estimate": n_tokens,
        "eval_type": "longcodeu",
        "answers": answers,
        "reference_answer": answers[0] if len(answers) == 1 else json.dumps(answers)[:2000],
        "expected_behavior": f"LongCodeU {task}",
        "max_new_tokens": 512,
    }


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tasks", nargs="+", default=list(TASKS), choices=TASKS)
    p.add_argument("--per-task", type=int, default=10)
    p.add_argument("--bucket", default="0~8K", help="context_length bucket to keep")
    p.add_argument("--tokenizer", default="Qwen/Qwen2.5-Coder-7B-Instruct")
    p.add_argument("--max-prompt-tokens", type=int, default=8192 - 512,
                   help="drop prompts longer than this (context_length - max_new_tokens)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--local-dir", help="use an already downloaded copy instead of the Hub")
    p.add_argument("--out", default="datasets/processed/longcodeu_8k.json")
    p.add_argument("--inspect", action="store_true")
    a = p.parse_args(argv)

    if a.inspect:
        for t in a.tasks:
            rec = next(iter_task_records(t, a.local_dir))
            print(f"== {t}: keys {list(rec)}")
            print({k: (str(v)[:300] + "...") if len(str(v)) > 300 else v for k, v in rec.items()})
        return 0

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.tokenizer, token=os.environ.get("HF_TOKEN") or None)

    def n_tokens(prompt):
        ids = tok.apply_chat_template([{"role": "user", "content": prompt}], add_generation_prompt=True,
                                      tokenize=True)
        if hasattr(ids, "keys"):
            ids = ids["input_ids"]
        return len(ids)

    rng = random.Random(a.seed)
    out, report = [], {"source": REPO, "licence": "Apache-2.0 (dataset card)", "bucket": a.bucket,
                       "max_prompt_tokens": a.max_prompt_tokens, "tokenizer": a.tokenizer, "tasks": {}}
    for t in a.tasks:
        recs = [r for r in iter_task_records(t, a.local_dir) if r.get("context_length") == a.bucket]
        rng.shuffle(recs)
        kept, too_long, bad = [], 0, 0
        for r in recs:
            if len(kept) >= a.per_task:
                break
            if not r.get("prompt") or r.get("ans") in (None, "", []):
                bad += 1
                continue
            n = n_tokens(r["prompt"])
            if n > a.max_prompt_tokens:
                too_long += 1
                continue
            kept.append(to_sample(t, r, n))
        report["tasks"][t] = {"in_bucket": len(recs), "kept": len(kept), "skipped_too_long": too_long,
                              "skipped_invalid": bad,
                              "mean_prompt_tokens": round(sum(s["prompt_tokens_estimate"] for s in kept) / len(kept))
                              if kept else None}
        print(f"{t:7s} bucket={len(recs):4d} kept={len(kept):3d} too_long={too_long} invalid={bad}")
        out.extend(kept)
    dest = ROOT / a.out
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(out, indent=1) + "\n")
    Path(str(dest).replace(".json", "_report.json")).write_text(json.dumps(report, indent=2) + "\n")
    print(f"wrote {len(out)} samples to {dest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

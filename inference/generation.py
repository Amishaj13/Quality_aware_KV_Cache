"""Prompt construction, tokenisation and context-length validation.

Only `task` and `context` from a sample reach the model. Reference answers, hidden
tests and buggy variants never do (see PROMPT_FIELDS and the leakage test).
"""
import random

SYSTEM_PROMPT = ("You are a careful senior Python engineer. Answer the user's question about the code. "
                 "When asked for code, return complete, runnable Python.")

PROMPT_FIELDS = ("task", "context")


class ContextOverflow(ValueError):
    pass


def build_messages(sample, context_override=None):
    context = context_override if context_override is not None else sample["context"]
    user = f"```python\n{context.rstrip()}\n```\n\n{sample['task']}"
    return [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user}]


def encode_messages(tokenizer, messages):
    """Apply the model's chat template and return a list of token ids (no truncation)."""
    out = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=True)
    # transformers 5 may return a BatchEncoding/dict; 4.x returns a list
    if hasattr(out, "keys") and "input_ids" in out:
        out = out["input_ids"]
    if hasattr(out, "tolist"):
        out = out.tolist()
    if out and isinstance(out[0], list):
        out = out[0]
    return list(out)


# ------------------------------------------------------------------ long-context fill
# Same synthetic-code generator as kivi_baseline/bench_kivi.py, so long-context results
# here and in the KIVI harness use the same kind of distractor text.
VERBS = ["load", "parse", "validate", "compute", "merge", "fetch", "render", "update", "resolve", "encode"]
NOUNS = ["config", "user", "order", "invoice", "session", "token", "record", "payload", "cache", "report"]


def code_chunk(rng, i):
    v, n = rng.choice(VERBS), rng.choice(NOUNS)
    a, b = rng.randint(2, 97), rng.randint(2, 97)
    return (
        f"def {v}_{n}_{i}(items, limit={a}):\n"
        f'    """{v.capitalize()} {n} entries and return the filtered result."""\n'
        f"    result = []\n"
        f"    for idx, item in enumerate(items):\n"
        f"        if idx >= limit:\n"
        f"            break\n"
        f"        value = item.get('{n}_id', 0) * {b} % {a + b}\n"
        f"        if value > {a}:\n"
        f"            result.append({{'id': idx, '{n}': value}})\n"
        f"    return result\n\n\n"
    )


def filled_context(tokenizer, sample, target_tokens, depth, seed):
    """Surround the sample's code with unrelated functions until the full prompt reaches
    about target_tokens, inserting the real code at `depth` of the filler."""
    base = len(encode_messages(tokenizer, build_messages(sample)))
    need = target_tokens - base
    if need <= 0:
        return sample["context"], 0
    rng = random.Random(seed)
    chunks, total, i = [], 0, 0
    while True:
        c = code_chunk(rng, i)
        n = len(tokenizer(c, add_special_tokens=False)["input_ids"])
        if total + n > need:
            break
        chunks.append(c)
        total += n
        i += 1
    pos = int(round(len(chunks) * depth))
    marker = "# ---- relevant code ----\n"
    ctx = "".join(chunks[:pos]) + marker + sample["context"] + "\n\n" + "".join(chunks[pos:])
    return ctx, total


def _truncate_middle(context, tokenizer, drop_tokens):
    ids = tokenizer(context, add_special_tokens=False)["input_ids"]
    if drop_tokens >= len(ids):
        raise ContextOverflow("context too long even after removing all of it")
    keep_head = (len(ids) - drop_tokens) // 2
    keep_tail = len(ids) - drop_tokens - keep_head
    head = tokenizer.decode(ids[:keep_head])
    tail = tokenizer.decode(ids[len(ids) - keep_tail:]) if keep_tail else ""
    return head + "\n# ... [truncated] ...\n" + tail


def prepare_prompt(tokenizer, sample, cfg):
    """Return (input_ids list, prompt_info dict). Raises ContextOverflow under policy 'skip'.

    Rule: input_tokens + max_new_tokens must fit in experiment.context_length.
    """
    e = cfg["experiment"]
    max_new = int(sample.get("max_new_tokens") or e["max_new_tokens"])
    max_new = min(max_new, e["max_new_tokens"])
    budget = e["context_length"] - max_new
    fill = cfg.get("context_fill") or {}
    context, filler_tokens = sample["context"], 0
    if fill.get("enabled"):
        target = min(fill["target_tokens"], budget)
        context, filler_tokens = filled_context(tokenizer, sample, target, float(fill.get("depth", 0.5)),
                                                int(fill.get("seed", 0)))
    ids = encode_messages(tokenizer, build_messages(sample, context))
    original_tokens = len(ids)
    truncated = False
    if len(ids) > budget:
        if e.get("overflow_policy", "skip") == "skip":
            raise ContextOverflow(f"prompt has {len(ids)} tokens; budget is {budget} "
                                  f"(context_length {e['context_length']} - max_new_tokens {max_new})")
        context = _truncate_middle(context, tokenizer, len(ids) - budget + 32)
        ids = encode_messages(tokenizer, build_messages(sample, context))
        truncated = True
        if len(ids) > budget:
            raise ContextOverflow(f"still {len(ids)} tokens after truncation")
    return ids, {
        "input_tokens": len(ids),
        "input_tokens_before_truncation": original_tokens,
        "truncated": truncated,
        "filler_tokens": filler_tokens,
        "context_fill_depth": float(fill.get("depth")) if fill.get("enabled") else None,
        "max_new_tokens": max_new,
    }

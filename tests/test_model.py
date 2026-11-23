"""Test 2 (tokenizer) and Test 3 (model)."""
from inference.generation import build_messages, encode_messages
from models.model_loader import load_tokenizer


def test_tokenizer_loads_and_has_chat_template(model_name):
    tok = load_tokenizer(model_name)
    assert tok.chat_template


def test_chat_template_and_reproducible_counts(model_name, samples):
    tok = load_tokenizer(model_name)
    msgs = build_messages(samples[0])
    a = encode_messages(tok, msgs)
    b = encode_messages(tok, msgs)
    assert a == b and len(a) > 10
    text = tok.decode(a)
    assert samples[0]["task"].split()[0] in text
    assert "assistant" in text  # generation prompt appended


def test_model_loads_and_generates(loaded):
    import torch
    model, tok, info = loaded
    assert info["num_parameters"] > 0 and info["max_position_embeddings"]
    ids = encode_messages(tok, [{"role": "user", "content": "Write a Python function that adds two numbers."}])
    x = torch.tensor([ids], device=model.device)
    out = model.generate(x, max_new_tokens=8, do_sample=False, use_cache=True)
    assert out.shape[1] > x.shape[1]

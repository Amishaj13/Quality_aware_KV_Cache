"""Builds a tiny random-weight Qwen2-architecture model + tokenizer on disk, offline.

Only for CPU tests of the plumbing (tokenise -> generate -> measure -> evaluate ->
save -> chart). Its outputs are random text: results from it say nothing about
model quality and must never be reported as experimental results.
"""
import json
from pathlib import Path

QWEN_STYLE_TEMPLATE = (
    "{% for message in messages %}"
    "{{ '<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n' }}"
    "{% endfor %}"
    "{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"
)


def build_tiny_model(out_dir, seed=0):
    import torch
    from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers
    from transformers import PreTrainedTokenizerFast, Qwen2Config, Qwen2ForCausalLM

    out = Path(out_dir)
    if (out / "config.json").exists():
        return str(out)
    out.mkdir(parents=True, exist_ok=True)
    root = Path(__file__).resolve().parent.parent
    corpus = [t["context"] + "\n" + t["task"] for t in json.loads((root / "datasets/sample_tasks.json").read_text())]
    specials = ["<|endoftext|>", "<|im_start|>", "<|im_end|>"]
    tk = Tokenizer(models.BPE(unk_token=None, byte_fallback=False))
    tk.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tk.decoder = decoders.ByteLevel()
    tk.train_from_iterator(corpus * 3, trainers.BpeTrainer(vocab_size=1024, special_tokens=specials,
                                                            initial_alphabet=pre_tokenizers.ByteLevel.alphabet()))
    tok = PreTrainedTokenizerFast(tokenizer_object=tk, eos_token="<|im_end|>", pad_token="<|endoftext|>",
                                  additional_special_tokens=["<|im_start|>"])
    tok.chat_template = QWEN_STYLE_TEMPLATE
    tok.save_pretrained(out)

    torch.manual_seed(seed)
    cfg = Qwen2Config(vocab_size=len(tok), hidden_size=64, intermediate_size=128, num_hidden_layers=2,
                      num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=16384,
                      eos_token_id=tok.eos_token_id, pad_token_id=tok.pad_token_id, tie_word_embeddings=True)
    Qwen2ForCausalLM(cfg).save_pretrained(out)
    return str(out)


if __name__ == "__main__":
    import sys
    print(build_tiny_model(sys.argv[1] if len(sys.argv) > 1 else "/tmp/kvr_tiny_model"))

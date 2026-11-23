"""Download model weights + tokenizer into the Hugging Face cache (resumable).

Uses HF_TOKEN from the environment if set; never pass the token on the command line.
Set HF_HOME to put the cache on the large volume, e.g. export HF_HOME=$HOME/hf_cache
  python scripts/download_model.py [--model Qwen/Qwen2.5-Coder-7B-Instruct] [--revision main]
"""
import argparse
import os
import shutil
import sys

import _bootstrap  # noqa: F401


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen2.5-Coder-7B-Instruct")
    p.add_argument("--revision", default="main")
    p.add_argument("--min-free-gb", type=float, default=20)
    a = p.parse_args(argv)
    from huggingface_hub import snapshot_download

    cache = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
    os.makedirs(cache, exist_ok=True)
    free = shutil.disk_usage(cache).free / 1e9
    print(f"HF cache: {cache} ({free:.1f} GB free)")
    if free < a.min_free_gb:
        sys.exit(f"Only {free:.1f} GB free; the 7B model needs ~16 GB. Increase the space's storage "
                 f"or set HF_HOME to a larger volume.")
    try:
        path = snapshot_download(a.model, revision=a.revision, token=os.environ.get("HF_TOKEN") or None,
                                 allow_patterns=["*.json", "*.safetensors", "*.model", "*.txt", "*.py", "merges.txt",
                                                 "tokenizer*"])
    except Exception as e:
        sys.exit(f"Download failed: {type(e).__name__}: {e}\n"
                 "Check internet access, HF_TOKEN, and that the model name is correct.")
    size = sum(os.path.getsize(os.path.join(r, f)) for r, _, fs in os.walk(path) for f in fs)
    print(f"Downloaded to {path} ({size / 1024**3:.2f} GiB)")


if __name__ == "__main__":
    main()

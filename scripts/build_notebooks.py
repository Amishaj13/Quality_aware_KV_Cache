"""Regenerates notebooks/*.ipynb (kept in code so they stay reviewable). Outputs are empty:
run them on SageMaker to fill them in.   python scripts/build_notebooks.py"""
from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parent.parent
NB = ROOT / "notebooks"

SETUP = '''# Run from anywhere: find the project root and make it importable
import os, sys
from pathlib import Path
ROOT = Path.cwd()
while not (ROOT / "main.py").exists() and ROOT != ROOT.parent:
    ROOT = ROOT.parent
os.chdir(ROOT); sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]
PY = sys.executable
print("project root:", ROOT)'''

HF_TOKEN = '''# Hugging Face token: kept in this kernel's environment only, never written to disk or printed.
# Qwen2.5-Coder is not gated, so the token is optional, but it avoids anonymous rate limits.
import os, getpass
if not os.environ.get("HF_TOKEN"):
    os.environ["HF_TOKEN"] = getpass.getpass("HF token (Enter to skip): ").strip()
    if not os.environ["HF_TOKEN"]:
        del os.environ["HF_TOKEN"]
print("HF_TOKEN set:", bool(os.environ.get("HF_TOKEN")))
# Put the model cache on the persistent volume of the space
os.environ.setdefault("HF_HOME", str(Path.home() / "hf_cache"))
print("HF_HOME:", os.environ["HF_HOME"])'''


def md(s):
    return nbf.v4.new_markdown_cell(s.strip())


def code(s):
    return nbf.v4.new_code_cell(s.strip())


def write(name, cells):
    nb = nbf.v4.new_notebook()
    nb["cells"] = cells
    nb["metadata"] = {"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
                      "language_info": {"name": "python"}}
    nbf.write(nb, NB / name)
    print("wrote", NB / name)


write("01_environment_setup.ipynb", [
    md("""# 01 · Environment inspection and setup
Inspect the SageMaker environment **before** installing anything, then install only what is missing.
Nothing here creates AWS resources. Expected hardware: `ml.g6.xlarge` (NVIDIA L4, 24 GB) or `ml.g5.xlarge` (A10G, 24 GB)."""),
    code(SETUP),
    md("## 1. Raw diagnostics"),
    code("!nvidia-smi"),
    code("!{PY} --version\n!{PY} -m pip show torch transformers accelerate 2>/dev/null | grep -E '^(Name|Version)'\n!df -h"),
    code('''import torch

print("PyTorch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())

if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        p = torch.cuda.get_device_properties(i)
        free, total = torch.cuda.mem_get_info(i)

        print("GPU:", p.name)
        print("Total VRAM:", total / 1024**3)
        print("Free VRAM:", free / 1024**3)'''),
    md("## 2. Install missing packages (does not replace a working CUDA PyTorch) and write `requirements-lock.txt`"),
    code("!bash setup_env.sh"),
    md("## 3. Hugging Face token and cache location"),
    code(HF_TOKEN),
    md("## 4. Full server report → `results/server_report.json`"),
    code("!{PY} scripts/server_report.py"),
    code('''import json
rep = json.load(open("results/server_report.json"))
{k: rep[k] for k in ("sagemaker_environment", "aws_region", "instance_type", "python")}'''),
    md("## 5. Can Qwen2.5-Coder-7B-Instruct run here at 8K? (estimate; notebook 02 verifies)"),
    code("!{PY} scripts/check_gpu.py --model Qwen/Qwen2.5-Coder-7B-Instruct --context 8192"),
    md("## 6. Unit tests that need no GPU model download (evaluation, results, config)"),
    code("!{PY} -m pytest tests/test_gpu.py tests/test_evaluation.py tests/test_results.py -q"),
])

write("02_model_loading.ipynb", [
    md("""# 02 · Model loading and first coding inference
Downloads `Qwen/Qwen2.5-Coder-7B-Instruct` (~15 GB, once), loads it in bf16 with the standard full KV cache, and runs
one coding task. **Shut this notebook's kernel down before running notebook 03**, otherwise two copies of the
model compete for the 24 GB GPU."""),
    code(SETUP),
    code(HF_TOKEN),
    code("!{PY} scripts/download_model.py --model Qwen/Qwen2.5-Coder-7B-Instruct"),
    code('''from utils.config import load_config
from models.model_loader import load_model
from evaluation.memory_metrics import snapshot, gib

cfg = load_config()                      # configs/model_config.yaml + configs/experiment_config.yaml
model, tok, info = load_model(cfg)
info'''),
    code('''snap = snapshot()
print("model weights on GPU:", gib(info["gpu_model_memory_bytes"]), "GiB")
print("free after load     :", gib(snap["free_bytes"]), "of", gib(snap["total_bytes"]), "GiB")
print("advertised max context (max_position_embeddings):", info["max_position_embeddings"])
print("KV heads / layers / head_dim:", info["num_key_value_heads"], info["num_hidden_layers"], info["head_dim"])'''),
    md("### Chat template check"),
    code('''from inference.generation import build_messages, encode_messages
from utils.dataset import load_samples
samples, _ = load_samples("datasets/sample_tasks.json")
s = samples[0]
ids = encode_messages(tok, build_messages(s))
print(len(ids), "prompt tokens\\n")
print(tok.decode(ids))'''),
    md("### First full-cache coding inference"),
    code('''from inference.generation import prepare_prompt
from inference.inference_runner import InferenceRunner
from evaluation.correctness import evaluate_sample

runner = InferenceRunner(model, tok, cfg)
ids, pinfo = prepare_prompt(tok, s, cfg)
r = runner.run(ids, pinfo["max_new_tokens"])
print(r["response"])'''),
    code('''ev = evaluate_sample(s, r["response"])
keys = ["output_tokens", "prefill_time_seconds", "ttft_seconds", "tpot_ms", "total_inference_seconds",
        "kv_cache_bytes", "kv_cache_bytes_theoretical_full", "peak_allocated_gpu_bytes", "peak_reserved_gpu_bytes"]
{**pinfo, **{k: r[k] for k in keys}, **{k: ev[k] for k in ("syntax_valid", "functional_correctness", "unit_test_pass_rate")}}'''),
    code('''import json
from utils.paths import results_dirs
out = results_dirs()["json"] / "first_inference.json"
out.write_text(json.dumps({"sample_id": s["sample_id"], "model_info": info, **pinfo, **r, **ev}, indent=2, default=str))
print("saved", out)'''),
    md("### Model and tokenizer tests (real model)"),
    code('''# Frees this notebook's copy first so the test process can load its own
del runner, model; import gc, torch; gc.collect(); torch.cuda.empty_cache()
!KVR_TEST_MODEL=Qwen/Qwen2.5-Coder-7B-Instruct {PY} -m pytest tests/test_model.py tests/test_inference.py -q'''),
    md("**Now: Kernel → Shut Down Kernel** before opening notebook 03."),
])

write("03_full_cache_baseline.ipynb", [
    md("""# 03 · Full KV-cache baseline over the dataset
Runs `scripts/run_experiments.py` (model loaded once, one record per sample saved immediately to
`results/raw_logs/<experiment_id>.jsonl`, JSON + CSV rebuilt after each sample). If the kernel dies, re-run the
same cell with `--resume`. Make sure notebook 02's kernel is shut down."""),
    code(SETUP),
    code(HF_TOKEN),
    code('''import torch
free, total = torch.cuda.mem_get_info()
print(f"free GPU memory: {free/1024**3:.1f} of {total/1024**3:.1f} GiB")
assert free / total > 0.8, "Another process is using the GPU (notebook 02 still running?). Shut it down first."'''),
    md("## 1. Quick check: 3 samples (~1–2 min)"),
    code("!{PY} scripts/run_experiments.py --max-samples 3 --experiment-id smoke_3 --overwrite"),
    md("""## 2. Full baseline: 30 samples, 8K budget, 256 new tokens (~10–20 min on L4)
For a long run you can also start it from a terminal so it survives a closed browser tab:
`nohup python scripts/run_experiments.py > results/raw_logs/baseline_001.log 2>&1 &`"""),
    code("!{PY} scripts/run_experiments.py --experiment-id baseline_001 --resume"),
    code('''import pandas as pd
df = pd.read_csv("results/csv/baseline_001.csv")
cols = ["sample_id", "workflow", "status", "input_tokens", "output_tokens", "total_inference_seconds", "ttft_seconds",
        "tpot_ms", "kv_cache_bytes", "peak_allocated_gpu_bytes", "syntax_valid", "functional_correctness"]
df[[c for c in cols if c in df]]'''),
    md("""## 3. Optional: same tasks padded to ~8K tokens (task code in the middle)
Still the full KV cache. This is the long-context baseline that later compression methods are compared against."""),
    code("!{PY} scripts/run_experiments.py --config configs/experiment_longctx.yaml --resume"),
])

write("04_result_analysis.ipynb", [
    md("""# 04 · Result analysis
Charts and summary for the baseline. 30 samples demonstrate the pipeline; they are **not** statistically conclusive,
and no compression method has been run, so there are no memory-reduction or quality-retention numbers yet."""),
    code(SETUP),
    code('''EXPERIMENT = "baseline_001"
from analyze_results import analyze
summary, charts = analyze(EXPERIMENT)'''),
    code('''from IPython.display import Image, display
for c in charts:
    display(Image(filename=str(c)))'''),
    code('''import pandas as pd
pd.read_csv(f"results/csv/{EXPERIMENT}_by_workflow.csv")'''),
    code('''# Long-context run, if you did it in notebook 03
import os
if os.path.exists("results/json/baseline_longctx_8k_d50.json"):
    s2, c2 = analyze("baseline_longctx_8k_d50")
    for c in c2[:2]:
        display(Image(filename=str(c)))'''),
    md("""## Copy results to S3 (existing bucket only) and shut down
Set the bucket name, dry-run, then upload. Model weights are never uploaded."""),
    code('''import os
os.environ["KVR_S3_BUCKET"] = ""          # <- your existing bucket name
os.environ["KVR_S3_PREFIX"] = "kv_cache_research"
if os.environ["KVR_S3_BUCKET"]:
    !{PY} scripts/sync_to_s3.py --dry-run
    # !{PY} scripts/sync_to_s3.py'''),
    md("""### Shutdown checklist (GPU time is billed until the space is stopped)
1. Results saved: `results/json`, `results/csv`, `results/charts`, `results/server_report.json`, `requirements-lock.txt`.
2. Optional S3 copy done (cell above) or files downloaded from the file browser.
3. `git add results requirements-lock.txt && git commit && git push` (from a terminal).
4. Shut down all kernels (Running Terminals and Kernels panel).
5. **Stop the space**: SageMaker Studio → JupyterLab / Code Editor → your space → **Stop**. Closing the browser tab does not stop billing.
6. Check SageMaker → Inference → Endpoints and Training/Processing jobs show nothing running (this project never creates them)."""),
])

# How the pipeline works, step by step

This is the simple companion to `PIPELINE_EXPLAINED.md`. That file explains the research and every
module. This one explains the **pipeline itself**: what happens, in what order, what goes in and what
comes out at each stage. It follows one real task from the start to the final chart.

---

## 1. The pipeline in one picture

```
 ┌──────────────┐   ┌──────────────┐   ┌──────────────┐   ┌──────────────┐   ┌──────────────┐
 │ 1. SETUP     │──►│ 2. INPUTS    │──►│ 3. RUN       │──►│ 4. SCORE     │──►│ 5. REPORT    │
 │ machine +    │   │ tasks +      │   │ model answers│   │ is the answer│   │ tables +     │
 │ model ready  │   │ settings     │   │ each task,   │   │ right?       │   │ charts +     │
 │              │   │              │   │ timed        │   │              │   │ GitHub/S3    │
 └──────────────┘   └──────────────┘   └──────────────┘   └──────────────┘   └──────────────┘
   once per            once per            once per task (loop)                 once per
   machine             experiment                                               experiment
```

Stages 3 and 4 repeat for every task. Everything else happens once.
One command drives each stage. They all go through `main.py`.

---

## 2. Stage 1: Setup (make the machine ready)

**Goal:** a GPU machine with the right software and the model downloaded.

| Step | Command | What happens | Output |
|---|---|---|---|
| 1a | `bash setup_env.sh` | Checks which Python packages exist, installs only the missing ones, then records exact versions | `requirements-lock.txt` |
| 1b | `python main.py report` | Records the machine: SageMaker space, region, Python, PyTorch, GPU name and memory, disk | `results/server_report.json` |
| 1c | `python main.py check-gpu` | Confirms a GPU is visible and estimates whether a 7B model plus an 8K-token memory fits | printed on screen |
| 1d | `python main.py download` | Downloads Qwen2.5-Coder-7B-Instruct (about 15 GB) to the machine's disk | files in `$HF_HOME` |
| 1e | `python main.py first` | Runs **one** task end to end as a smoke test | `results/json/first_inference.json` |

What we got: an NVIDIA A10G GPU (22 GiB), Python 3.13 and PyTorch 2.12, and the first task answered correctly.

---

## 3. Stage 2: Inputs (what goes into a run)

A run needs two things: **a list of tasks** and **a settings file**.

### 3a. A task
Every task is one entry in a JSON file. Here is the real first task (`python_001`):

```json
{
  "sample_id": "python_001",
  "workflow": "completion",
  "task":    "Complete the function so it satisfies its docstring. Return only the complete Python code in one ```python block.",
  "context": "def binary_search(items, target):\n    \"\"\"Return the index of target in the sorted list items, or -1 if absent.\n    Must run in O(log n).\"\"\"\n",
  "tests": [
    "assert binary_search([1, 3, 5, 7, 9], 7) == 3",
    "assert binary_search([1, 3, 5, 7, 9], 1) == 0",
    "assert binary_search([1, 3, 5, 7, 9], 9) == 4",
    "assert binary_search([1, 3, 5], 4) == -1",
    "assert binary_search([], 1) == -1"
  ],
  "reference_solution": "..."
}
```

There are two kinds of fields:
- **Shown to the model:** `task` (the instruction) and `context` (the code).
- **Kept secret:** `tests`, `reference_solution`, buggy versions and expected answers. These are used only
  in the scoring stage. An automated test checks that they never leak into the prompt.

The three task sets used:

| Task set | File | Count | Size of each input |
|---|---|---|---|
| Custom tasks | `datasets/sample_tasks.json` | 30 (6 each: completion, debugging, testgen, longcode, repoqa) | about 100–500 tokens |
| Position test | same 30 tasks, padded with filler code | 30 × 5 positions = 150 | about 7,600 tokens |
| LongCodeU | `datasets/processed/longcodeu_8k.json` (made by `prepare-longcodeu`) | 80 (10 each of 8 task types) | about 4,500 tokens |

### 3b. The settings file
`configs/experiment_config.yaml` sets everything about a run, so two runs differ only where the
file says so:

| Setting | Value | Why |
|---|---|---|
| `experiment_id` | `baseline_001` | Names every output file |
| `cache_method` | `full` | Normal, uncompressed memory (later: `snapkv`, `kivi`, ...) |
| `context_length` | 8192 | Max tokens for prompt + answer |
| `max_new_tokens` | 256 (512 for LongCodeU) | Longest answer allowed |
| `do_sample: false`, `temperature: 0` | greedy | Same input gives the same answer every time |
| `overflow_policy` | `skip` | Too-long prompts are recorded as `context_overflow`, never silently cut |
| `warmup` | true | One untimed run first, so GPU start-up doesn't distort the first timing |

The settings are checked before anything runs. For example, asking for `cache_method: snapkv` today stops
immediately with "not implemented yet", instead of quietly running the full cache.

---

## 4. Stage 3: Run (the model answers each task)

Command: `python main.py run --experiment-id baseline_001`

### 4a. Before the loop (once)
1. **Read and check the settings** and the task list (30 selected, 0 invalid).
2. **GPU-busy check.** If something else already uses more than 2 GiB of the GPU, stop. This prevents two runs colliding.
3. **Load the model onto the GPU** (about 80 seconds, 14.2 GiB). It is loaded once and reused for every task.
4. **Warm-up.** One short, untimed generation.
5. **Resume check.** If a previous run was interrupted, the finished tasks are skipped.

### 4b. For each task (the loop)
Here is what happens to `python_001`, with the real numbers from the run:

| # | Step | What happens | python_001 |
|---|---|---|---|
| 1 | **Build the prompt** | A fixed system message ("You are a careful senior Python engineer...") plus the code in a ```python block plus the instruction | |
| 2 | **Turn it into tokens** | Qwen's chat template converts the text into token numbers | **103 input tokens** |
| 3 | **Check the budget** | 103 + 256 ≤ 8192, so it fits. Otherwise the status would be `context_overflow` | fits |
| 4 | **Prefill** | The model reads all 103 tokens in one pass and fills its KV cache (its memory of the prompt) | 0.058 s, KV cache 5.6 MiB |
| 5 | **Generate** | The model writes the answer one token at a time, and each new token is added to the KV cache. A stopwatch records the time of every token | **117 tokens** in **4.11 s** |
| 6 | **Measure speed** | First token after **0.046 s** (TTFT). Then one token every **35.1 ms** (TPOT), about 28.5 tokens per second | |
| 7 | **Measure memory** | Peak GPU memory **14.21 GiB** (mostly the model itself). KV cache at the end **12.0 MiB** for 219 tokens, which matches the formula exactly | |
| 8 | **Hand the answer to scoring** | (Stage 4) | |
| 9 | **Save the row immediately** | One line appended to `results/raw_logs/baseline_001.jsonl` | |
| 10 | **Print a progress line** | `1/30 python_001 completion ok in=103 out=117 t=4.11s peak=14.21GiB correct=True` | |

The model's actual answer:

```python
def binary_search(items, target):
    left, right = 0, len(items) - 1
    while left <= right:
        mid = (left + right) // 2
        if items[mid] == target:
            return mid
        elif items[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    return -1
```

### 4c. When something goes wrong
A failure doesn't stop the run. It is recorded with a **status**, and the loop moves on:

| Status | Meaning |
|---|---|
| `ok` | Normal |
| `oom` | GPU ran out of memory |
| `context_overflow` | Prompt too long for the budget |
| `tokenizer_error` / `inference_error` | Something broke while preparing or generating |
| `invalid_sample` | Task file entry is missing fields |
| `model_load_failed` | Model couldn't load |

Failed rows stay in the results and **count as wrong**, so the numbers can never look better by
hiding failures. In all our runs, every sample finished with status `ok` (30/30, 150/150 and 80/80).

---

## 5. Stage 4: Score (is the answer right?)

Each workflow is scored differently, because "correct" means different things:

| Workflow | How the answer is checked | python_001 |
|---|---|---|
| **completion, debugging** | (1) Pull the code out of the ```python block. (2) Check it is valid Python. (3) Run the hidden `assert` tests in a **separate, locked-down Python process** with a 10 s time limit and a memory limit | valid; **5/5 tests passed** → correct |
| **testgen** | Run the model's tests with real pytest twice: they must **pass on the correct code** and **fail on 2 buggy versions** | n/a |
| **longcode, repoqa** | Compare with the expected answer (exact match) or check that key terms appear (keyword coverage) | n/a |
| **LongCodeU** | Benchmark metrics: name matching, line matching, CodeBLEU or BLEU, depending on the task type | n/a |

The scores are added to the same saved row:
`syntax_valid: true, tests_passed: 5, tests_total: 5, functional_correctness: true`.

**Re-scoring:** answers are saved, so scoring can be improved later and rerun with
`python main.py rescore`, **without using the GPU again**. That was done when the LongCodeU scorer was
fixed (mean 0.22 → 0.52, same answers).

To check failures by hand, run `python main.py show-failures`. It prints each wrong answer with its test error.
That is how testgen's 0% was confirmed to be real model mistakes, not a pipeline bug.

---

## 6. Stage 5: Report (turn rows into results)

Command: `python main.py analyze --experiment-id baseline_001`

| Output | What it is |
|---|---|
| `results/json/baseline_001.json` | All 30 rows with every field (prompt size, answer, timings, memory, scores, status) |
| `results/json/baseline_001_run_meta.json` | Exact settings, model details and software versions of the run |
| `results/csv/baseline_001.csv` | The same rows as a spreadsheet |
| `results/csv/baseline_001_by_workflow.csv` | Averages per workflow |
| `results/csv/baseline_001_summary.csv` + `results/baseline_001_summary.md` | One headline table |
| `results/charts/*.png` | Inference time, peak GPU memory, input tokens vs. time, output tokens vs. time, correctness by workflow |

Then the results are saved off the machine:
- `git add results ... && git commit && git push` sends them to GitHub;
- `python main.py s3` copies them to an existing S3 bucket (optional);
- **stop the SageMaker space** so billing stops.

---

## 7. The same pipeline, three experiments

All three experiments use the exact same five stages. Only the inputs and settings change:

| | Baseline | Position test | LongCodeU |
|---|---|---|---|
| Command | `run` | `position` | `prepare-longcodeu`, then `run --config configs/experiment_longcodeu.yaml` |
| What changes | nothing (plain tasks) | each task's code is hidden inside about 7,600 tokens of unrelated code at 0/25/50/75/100% | real repository code from a published benchmark |
| Samples | 30 | 150 (30 × 5) | 80 |
| Model loads | 1 | 1 (reused for all 5 positions) | 1 |
| Analysis | `analyze` | `analyze-position` | `analyze` (adds a by-task chart) |
| Headline | pass@1 61% | correct 50–69% by position | mean score 0.52 |

How the position test hides the code: the pipeline generates filler Python files (functions, classes,
comments) with a fixed random seed. It inserts the real task code at the chosen point and adds filler until the prompt
reaches about 7,600 tokens. The question stays at the end. That tests whether the model still finds code at the start,
middle or end of a long input.

---

## 8. Safety nets built into the pipeline

| Problem it prevents | How |
|---|---|
| Losing hours of work if the browser closes or the run crashes | Each row is saved the moment it finishes. `--resume` skips done tasks. Long runs use `nohup` |
| Two runs on one GPU at the same time | GPU-busy check at start |
| The model seeing the answers | Only `task` and `context` reach the prompt (tested) |
| Bad generated code harming the machine or freezing the run | Separate process, time limit, memory limit |
| Misleading numbers | Failures kept and counted as wrong. KV size measured, not guessed. Results not from the 7B bf16 model are labelled *PRELIMINARY* |
| Unfair comparisons later | Greedy decoding, fixed seeds, the same tasks, and settings saved with every run |
| Leaking secrets | Hugging Face token only as an environment variable. No AWS keys in code. Nothing in the code creates AWS resources |
| Not knowing whether the pipeline works | 45 automated tests run on a CPU with a tiny fake model before any GPU time is spent |

---

## 9. How compression will plug in (the next phase)

The pipeline was built so that adding a compression method changes **only one step**, step 5 of
Stage 3 (how the KV cache is stored during generation):

```
Stage 3, step 5:   cache_method: full     → keep every token's memory      (done)
                   cache_method: snapkv   → keep only important tokens       (planned)
                   cache_method: kivi     → store memory in 2 or 4 bits      (planned)
```

Each new method is a small class in `cache_methods/`. Everything else (tasks, timing, scoring,
reports) stays the same. The comparison then runs automatically:

- **Quality retention** = compressed score / full-cache score × 100
- **Memory reduction** = (full KV − compressed KV) / full KV × 100

The order planned in `docs/ROADMAP.md` is StreamingLLM → SnapKV → H2O → PyramidKV → KIVI → EvolKV.

---

## 10. One-paragraph summary to say out loud

> The pipeline has five stages: set up the GPU machine, load tasks and settings, let the model
> answer each task while timing every token and measuring memory, score each answer by actually running
> the code or comparing it with the expected answer, then produce tables and charts. It ran three experiments
> on an A10G GPU: 30 custom coding tasks (pass@1 61%), a position test where the code is hidden in about 7,600
> tokens (50–69% correct by position) and 80 LongCodeU samples (mean 0.52). All of them used the normal,
> uncompressed KV cache. That is the baseline. Next, each compression method replaces one step of the
> pipeline, and I measure how much quality it keeps and how much memory it saves.

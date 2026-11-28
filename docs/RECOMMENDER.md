# The quality-aware recommender, in plain words

## What question it answers

> "I have this kind of coding job, this much GPU memory, and I can accept at most this much
> quality loss and this much slowdown. Which KV-cache method should I use?"

It answers with **the method that leaves the least KV cache on the GPU while meeting every
limit**, and shows why each other method passed or failed. It uses only the benchmark
results already in `results/json/`, so it runs in a second on any laptop, without a GPU.

```bash
python main.py recommend --workload longcodeu --free-gpu-gib 7 --max-slowdown 1.3 --min-quality 95
```

```
Recommendation: kivi2: 80% less KV cache on the GPU (51 MiB instead of 259 MiB per request),
quality 96% of the full cache, 1.18x the full cache's time per token.

| method        | quality kept               | GPU KV  | ... | slowdown | verdict                     |
| offload       | 100% (exact)               | 9 MiB   |     | 1.60x    | 1.60x slower > 1.30x limit  |
| kivi2         | 96% (measured 96, low 89)  | 51 MiB  |     | 1.18x    | chosen                      |
| streaming1024 | 38% (measured 38, low 29)  | 56 MiB  |     | 0.98x    | quality 38% < 95%           |
| ...
```

## The inputs

| flag | meaning | example |
|---|---|---|
| `--workload` | the kind of job: `custom` (short coding tasks), `longcodeu` (understand a long code file), `position` (find the relevant code in an ~8K-token file) | `longcodeu` |
| `--prompt-tokens` | how long the prompt is (default: the measured average for the workload) | `6000` |
| `--output-tokens` | how many tokens the model will write (default: measured average) | `300` |
| `--free-gpu-gib` | GPU memory left for KV caches after loading the model. On the A10G (24 GB) with the 7B model this is about 7 GiB | `7` |
| `--batch` | how many requests run at the same time (they share that memory) | `32` |
| `--min-quality` | the least quality you accept, as % of the full cache (default 95) | `95` |
| `--max-slowdown` | the speed limit, as a multiple of the full cache's time per token | `1.3` = at most 30 % slower |
| `--max-tpot-ms` | the same limit in milliseconds per generated token | `45` |
| `--strict` | only trust a method if it is *reliably* good enough (see rule 1) | |
| `--depth` | position workload only: where the relevant code sits (0, 25, 50, 75, 100 %) | `50` |
| `--json` | machine-readable output, e.g. for a dashboard | |

## How it decides (four rules)

**1. Quality.** For each method it looks up the *quality retention* measured on that workload:
the method's score divided by the full cache's score on the same samples, × 100.
Some cases are exact by construction:
- the full cache and CPU offloading (offloading gave the same answers as the full cache on every sample);
- SnapKV and StreamingLLM when the prompt is shorter than their budget (1024 or 2048 tokens), because then nothing is thrown away.

Because the test sets are small (18–150 scored samples), the measured average can be lucky or
unlucky. So the recommender also computes a **95 % lower confidence bound** with a *paired
bootstrap*: it resamples the samples 2,000 times and looks at how low retention could
plausibly be. With `--strict` a method must pass using this lower bound, not the average.
Example: on LongCodeU, 2-bit KIVI keeps 96 % on average but its lower bound is 89 %, so a
strict 95 % limit switches the choice to 4-bit KIVI (average 98 %, lower bound 96 %).
Averages above 100 % (e.g. SnapKV 109 % on the position test) are noise and are counted as 100 %.

**2. Memory.** It predicts how many bytes of KV cache each method leaves on the GPU for the
given prompt and output length, from one formula per method. The formulas come from how each
method works; their constants come from the measurements (the model stores 57,344 bytes per
token in 28 layers):

| method | KV left on the GPU |
|---|---|
| full | every token: 57,344 × tokens |
| offload | one layer of 28: full ÷ 28 (the rest sits in CPU memory) |
| kivi4 / kivi2 | a straight line fitted to all KIVI runs (packed 4/2-bit codes + a 128-token fp16 window) |
| snapkv N | N prompt tokens + every generated token |
| streaming N | a fixed window of N tokens |

`results/recommender/recommender_report.md` checks these formulas against every measured run:
they are exact for all methods except KIVI on very short prompts (within 7 %).

The memory limit counts `batch × KV per request`, plus one detail: SnapKV and StreamingLLM
first build the *full* cache for the prompt and only then throw tokens away, so one request
briefly needs the full size. The recommender adds that, so it never recommends something that
would run out of memory during prefill.

**3. Speed.** The slowdown per generated token, compared with the full cache, is fitted
against prompt length from all runs (e.g. offloading: 1.06x at 240 tokens, 1.6x at 4.5K, 2x
at 7.6K, because it copies the whole cache from CPU memory on every token).

**4. Choice.** Drop every method that breaks a limit. From the rest, pick the one leaving the
least KV cache on the GPU. If two are within 2 % of each other, the one with higher quality,
then the faster one, wins. If nothing fits, it says so and names the closest method and
which limit it breaks.

## What "Pareto front" means

A method is on the Pareto front when no other method is at the same time smaller on the GPU,
at least as good in quality and at least as fast. Methods off the front are never worth
choosing: something else beats them on every count. `python main.py recommend --report` writes the fronts for all
three workloads, a chart (`results/recommender/pareto_fronts.png`) and a table of choices for
typical deployments.

## What it currently recommends (A10G, 7 GiB free for KV)

| scenario | custom (~240 tokens) | longcodeu (~4.5K) | position (~7.6K) |
|---|---|---|---|
| One user, no speed limit | offload | offload | offload |
| Interactive, ≤ 30 % slower | offload | kivi2 | snapkv1024 |
| Same, quality guaranteed (`--strict`) | offload | kivi4 | snapkv1024 |
| 32 requests at once, ≤ 30 % slower, ≥ 90 % | offload | kivi2 | snapkv1024 |
| Quality may drop to 75 %, ≤ 10 % slower | offload | snapkv2048 | snapkv1024 |

What this tells you:
- **No single method is best.** The answer depends on the workload and the limits, which is
  exactly why a recommender is needed instead of a fixed choice.
- **Quantization (KIVI) is the safe choice** for understanding long code: it keeps every token,
  just with fewer bits.
- **Eviction (SnapKV) is good when the question says what matters** (the position test), but
  loses badly on LongCodeU, where the answer needs code from all over the file.
- **StreamingLLM is never chosen at 90–95 %**: it keeps only the start and the end of the context
  and forgets the middle (see the position-by-depth table in `results/comparison/`).
- **Offloading keeps exact answers and frees the most GPU memory, but costs speed** on long prompts.

## Limits (say these at the review)

- One model (Qwen2.5-Coder-7B), one GPU (A10G), three workloads, prompts up to ~8K tokens.
- For prompts longer than what was measured, quality is an extrapolation; the output marks it
  with ⚠. Running the 16K / 32K benchmarks fixes this: the recommender reads new results
  automatically, with no code change.
- It is rule-based on purpose (ROADMAP section 4): training a model to predict quality needs
  many more measurements than we have.

## Where the code is

- `recommender/profiles.py` reads `results/json/`, computes retention + bootstrap bounds, fits the memory and speed formulas
- `recommender/engine.py` applies the four rules and the Pareto front
- `recommender/report.py` writes `results/recommender/`
- `scripts/recommend.py` the command line (`python main.py recommend`)
- `tests/test_recommender.py` tests on synthetic results, plus a check against the real ones

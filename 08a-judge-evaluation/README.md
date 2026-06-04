# Judge Evaluation

A system for evaluating LLM judges against reward-bench-style datasets. Given a benchmark of (prompt, response\_A, response\_B, label) pairs, each judge scores both responses independently; accuracy is how often the higher-scored response matches the label.

## Relationship to `08-judge-evaluation`

`08-judge-evaluation` and `08a-judge-evaluation` serve different evaluation goals:

| | `08-judge-evaluation` | `08a-judge-evaluation` (this directory) |
|---|---|---|
| **Benchmark source** | Synthetic, internally-generated preference data (completions degraded from best→worst) | External, public reward-bench-style datasets (JudgeBench, RewardBench2, RM-Bench) |
| **Evaluation task** | Ranking/scoring/pairwise across 9 quality tiers | Binary preference accuracy: does the judge pick the human-labelled winner? |
| **Primary purpose** | Exploratory — iterate on judge prompts and scoring methods against known-quality synthetic pairs | Standardised benchmarking — measure judges against published leaderboard datasets |
| **Infrastructure** | Standalone Python scripts with a shared `lib.py` framework | SLURM-orchestrated server + judging jobs; fixed four-step pipeline per benchmark |

Use `08-judge-evaluation` when you want to compare judge configurations against a synthetic quality signal you control. Use `08a-judge-evaluation` (this directory) when you want to measure a judge's absolute accuracy on established external benchmarks.

## Directory structure

```
08a-judge-evaluation/
├── judges/                        # Judge configs (one .py per judge)
│   ├── activeuf.py                # Shared model config, prompts, and scoring helpers
│   ├── 00-ActiveUF.py             # Composite judge (mean of 01–04)
│   ├── 01-ActiveUF-Helpfulness.py
│   ├── ...
│   └── README.md                  # Judge interface spec + how to add a judge
│
├── benchmarks/                    # One subdirectory per benchmark dataset
│   ├── JudgeBench-gpt/
│   ├── RewardBench2-test/
│   ├── RM-Bench-train/
│   └── README.md                  # Benchmark structure + how to add a benchmark
│
├── src/                           # Core evaluation engine
│   ├── judge.py                   # Async judge runner (phase 1: own answers; phase 2: scoring)
│   └── utils.py                   # OpenAI client, module loader, prompt stringifier
│
├── results/                       # Aggregated HTML/notebook across all benchmarks
│   ├── overall.html
│   └── overall.ipynb
│
├── logs/                          # SLURM job logs (one subdirectory per job ID)
│
├── init_judge_server.py           # Submits the LLM serving job; prints job ID to stdout
├── run_judge_on_benchmark.py      # Polls for server URL, runs src/judge.py, cancels server
├── run_judges_on_benchmarks.sh    # Batch SLURM orchestration (edit arrays at the top)
└── compile_overall_results.py     # Merges per-benchmark overall.html → results/overall.html
```

## Evaluation workflow

Each benchmark follows a fixed four-step pipeline. Steps 1 and 2 are benchmark-specific (scripts live under `benchmarks/<Name>/src/`); steps 3 and 4 use the shared engine.

```
0-reformat-benchmark     →  1-reformatted/   (one row per response)
                         ↓
run_judges_on_benchmarks →  2-judged/<judge>/  (score_distribution per row)
                         ↓
2-reformat-post-judging  →  3-rereformatted/<judge>/  (paired rows with numeric scores)
                         ↓
3-run-metrics            →  4-results/<judge>.json  (accuracy per source)
                         ↓
4-compile-results-html   →  overall.html  (colour-coded table)
                         ↓
compile_overall_results  →  results/overall.html + results/overall.ipynb
```

### Step 0 — reformat the benchmark

Run the benchmark-specific script once per benchmark (not per judge):

```bash
python -m benchmarks.JudgeBench-gpt.src.0-reformat-benchmark \
    --input-path ScalerLab/JudgeBench \
    --output-path benchmarks/JudgeBench-gpt/1-reformatted \
    --split gpt
```

Output is a HuggingFace `datasets` Arrow directory with one row per response. Required columns: `prompt`, `response`, `prompt_id`. See `benchmarks/README.md` for the full column spec.

### Step 1 — run judges (SLURM)

Edit the arrays at the top of `run_judges_on_benchmarks.sh` to select benchmarks and judges, then submit:

```bash
bash run_judges_on_benchmarks.sh
```

This submits one LLM serving job and one dependent judging job per (benchmark, judge) pair. Output lands in `benchmarks/<Name>/2-judged/<judge-name>/`.

To run a single (benchmark, judge) pair manually:

```bash
# 1. Start the model server; note the printed job ID
SERVER_JOB_ID=$(python init_judge_server.py \
    --input-dir benchmarks/JudgeBench-gpt/1-reformatted \
    --judge-cfg-path judges/01-ActiveUF-Helpfulness.py \
    --job-time 00:30:00)

# 2. Submit the judging job, dependent on the server coming up
sbatch --dependency=after:${SERVER_JOB_ID} \
    --wrap="python -u run_judge_on_benchmark.py \
        --input-dir benchmarks/JudgeBench-gpt/1-reformatted \
        --output-dir benchmarks/JudgeBench-gpt/2-judged/01-ActiveUF-Helpfulness \
        --judge-cfg-path judges/01-ActiveUF-Helpfulness.py \
        --job-time 00:30:00 \
        --server-job-id ${SERVER_JOB_ID}"
```

### Step 2 — post-process and compute metrics

Run these three commands from the repo root after all judges finish:

```bash
# Re-pair per-response scores into (response_A, response_B, label) rows
python -m benchmarks.JudgeBench-gpt.src.2-reformat-post-judging

# Compute accuracy per source → 4-results/<judge>.json
python -m benchmarks.JudgeBench-gpt.src.3-run-metrics

# Build colour-coded HTML table → benchmarks/JudgeBench-gpt/overall.html
python -m benchmarks.JudgeBench-gpt.src.4-compile-results-html
```

All three scripts process every judge found under `2-judged/` by default. Pass `--judge-name <name>` to process a single judge.

### Step 3 — aggregate across benchmarks

```bash
python compile_overall_results.py
# → results/overall.html
# → results/overall.ipynb
```

## Adding a new judge

See [judges/README.md](judges/README.md).

## Adding a new benchmark

See [benchmarks/README.md](benchmarks/README.md).

## Environment

Scripts expect:
- `$SCRATCH/model-launch/.venv/bin/python` for `init_judge_server.py`
- The `activeuf` container environment for judging jobs (set in `run_judges_on_benchmarks.sh`)
- HuggingFace `datasets`, `openai`, `uvloop`, `tqdm`, `pandas` in the Python path

SLURM account and reservation are set at the top of `run_judges_on_benchmarks.sh`.

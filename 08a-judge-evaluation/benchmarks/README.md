# Benchmarks

Each benchmark is a dataset of (prompt, responses, label) tuples where the label indicates the preferred response(s). Judges are evaluated by how often their scores agree with the labels.

## Existing benchmarks

| Name | Source dataset | Split | Notes |
|------|---------------|-------|-------|
| `JudgeBench-gpt` | `ScalerLab/JudgeBench` | `gpt` | Pairwise A/B; sources: mmlu-pro, livebench-reasoning, livebench-math, livecodebench |
| `RewardBench2-test` | `allenai/reward-bench-2` | `test` | Multi-response (1 chosen + 3 rejected per prompt) |
| `RM-Bench-train` | `THU-KEG/RM-Bench` | `train` | Multi-response with concise/plain/markdown formatting variants |

## Benchmark directory structure

```
benchmarks/<Name>/
├── src/
│   ├── 0-reformat-benchmark.py    # Download + reformat raw dataset → 1-reformatted/
│   ├── 2-reformat-post-judging.py # Pair per-response scores → 3-rereformatted/
│   ├── 3-run-metrics.py           # Compute accuracy → 4-results/
│   └── 4-compile-results-html.py  # Colour-coded HTML table → overall.html
│
├── 1-reformatted/   (gitignored)  # One row per response; input to judges
├── 2-judged/        (gitignored)  # One subdirectory per judge
│   ├── <judge-name>/              # HuggingFace Arrow dataset + judge_responses.jsonl
│   └── logs/                      # Per-judge SLURM logs
├── 3-rereformatted/ (gitignored)  # Benchmark-specific; one subdirectory per judge
├── 4-results/       (gitignored)  # <judge-name>.json with accuracy per source
└── overall.html                   # Compiled results table (committed for preview)
```

## Shared data schemas

Only `1-reformatted/` and `2-judged/` have a shared schema. Everything downstream is benchmark-specific.

### `1-reformatted/` — input to `src/judge.py`

Required columns (others are passed through unchanged):

| Column | Type | Description |
|--------|------|-------------|
| `prompt` | `str` or `list` | User prompt (string or list of chat-turn dicts) |
| `response` | `str` | One candidate response |
| `prompt_id` | `str` | Unique per-response identifier that encodes group membership so that `2-reformat-post-judging.py` can re-group rows |

### `2-judged/` — output of `src/judge.py`

Adds two columns to the `1-reformatted` schema:

| Column | Type | Description |
|--------|------|-------------|
| `score_distribution` | `dict[str, float]` | Token probabilities over `scoring_range` |
| `judge_response_texts` | `str \| None` | Raw model text (non-null for text-generation judges) |

### `4-results/<judge>.json`

```json
{
    "source-name": {"score": 73.4, "none_rate": 0.0},
    "overall":     {"score": 75.1, "none_rate": 0.0}
}
```

`score` is accuracy (0–100). `none_rate` is the fraction of prompts where the judge returned a None score for at least one response, and therefore, resorted to random scoring.

## Adding a new benchmark

### 1. Create the benchmark directory

```
benchmarks/<NewName>/
└── src/
    ├── 0-reformat-benchmark.py
    ├── 2-reformat-post-judging.py
    ├── 3-run-metrics.py
    └── 4-compile-results-html.py
```

Copy from the existing benchmark whose data format is closest to yours and adapt.

### 2. Write `0-reformat-benchmark.py`

Produces `1-reformatted/` with the shared schema above. Key requirements:

- Output must be a HuggingFace `datasets` Arrow directory (`dataset.save_to_disk(...)`).
- `prompt_id` must encode group membership so that `2-reformat-post-judging.py` can re-group rows later. Common conventions: `<pair_id>_A` / `<pair_id>_B` (JudgeBench), `<subset>_<id>_chosen_<i>` / `<subset>_<id>_rejected_<i>` (RewardBench2).
- Multi-turn prompts can be stored as a list of `{"role": ..., "content": ...}` dicts; `src/utils.stringify_prompt` flattens them when calling the judge.

```bash
python -m benchmarks.NewName.src.0-reformat-benchmark \
    --input-path <hf-dataset-id-or-local-path> \
    --output-path benchmarks/NewName/1-reformatted
```

### 3. Write `2-reformat-post-judging.py`

Re-groups the per-response judged rows and calls `judge_args.get_score_from_distribution(row["score_distribution"])` to convert each distribution to a numeric score. The output schema is up to you — design it to match what `3-run-metrics.py` expects. See the existing scripts for examples of how different benchmarks handle grouping and tie-breaking.

### 4. Write `3-run-metrics.py`

Computes accuracy per source and writes `4-results/<judge>.json` in the format above. The metric depends on your benchmark format (pairwise accuracy, top-1 ranking, etc.).

### 5. Write `4-compile-results-html.py`

Builds the per-benchmark `overall.html`. Copy from an existing benchmark and update:
- The `folder` path pointing to `4-results/`
- The `SCORE_COLS` list to match your source names
- The `out_path` pointing to `benchmarks/NewName/overall.html`
- The title strings

### 6. Register in `compile_overall_results.py`

Add an entry to the `BENCHMARKS` list:

```python
BENCHMARKS = [
    ...
    ("NewName", "benchmarks/NewName/overall.html"),
]
```

### 7. Run the pipeline

```bash
python -m benchmarks.NewName.src.0-reformat-benchmark ...

# Edit BENCHMARK_DIRS in run_judges_on_benchmarks.sh, then:
bash run_judges_on_benchmarks.sh

# After judging finishes:
python -m benchmarks.NewName.src.2-reformat-post-judging
python -m benchmarks.NewName.src.3-run-metrics
python -m benchmarks.NewName.src.4-compile-results-html
python compile_overall_results.py
```
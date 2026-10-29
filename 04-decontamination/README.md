# Decontamination

Cross-checks the prompts of a training dataset against the prompts of the evaluation benchmarks.
If significant overlap is detected, the sample is removed from the training data.

## How it works

1. Every user message of a training sample (`initial_prompt` and the user turns of all
   `conversation_branches`) and every benchmark prompt are normalised (Unicode NFKC, runs of a repeated non-letter character such as `━━━━━` collapsed to one,
   whitespace collapsed to one space), tokenized (`swiss-ai/Apertus-8B-Instruct-2509`, without `<s>`)
   and split into units. A unit is a token, except that a number (`1,023`, `16.482`, `2'832`) counts
   as one unit; the tokenizer would split it into single digits (`matching_units.py`).
2. Each user message is compared on its own with each benchmark prompt. The rule depends on the length of the
   benchmark prompt:

   | Benchmark prompt (after normalisation) | Rule | A training sample is removed if, for one of its user messages (the training prompt), … |
   |---|---|---|
   | **0 units** (empty or only whitespace) | skipped, counted as empty | never |
   | **1–3 units** (e.g. "5!=", "Paris", "Yes.") | exact equality only | the training prompt is **identical** to the benchmark prompt |
   | **4–7 units** (e.g. "Compute $\cos 225^\circ$.") | complete containment + coverage guard | the **whole** benchmark prompt appears as one continuous run in the training prompt **and** makes up **at least 50%** of the training prompt's units |
   | **8+ units** | 8-gram candidates + share of matching runs | the training prompt shares **at least one run of 8 consecutive units** with the benchmark prompt (candidate), **and** the runs of **at least 5 consecutive units** that both have in common (`difflib.SequenceMatcher`) cover **at least 50%** of the benchmark prompt's units |

   Prompts shorter than 8 units have no 8-gram, so they need their own rules. The number of
   benchmark prompts in the classes 0, 1–3 and 4–7 is recorded in the processing log
   (`short_benchmark_prompts`).
3. For each benchmark a report `<benchmark>__contamination_report.json` is written, mapping each
   contaminated `conversation_id` to the matched benchmark prompt. A sample is contaminated if any of its
   user messages matches.
4. The final filter removes every `conversation_id` that appears in any report.

Only near-verbatim overlap in user messages is detected; system prompts and assistant responses are not
checked.

## Test set for the decontamination strategy

`validation/contamination_dataset.jsonl` is a synthetically generated, labelled test set for measuring how many false positives
and false negatives the decontamination produces. It has 720 samples (`id`, `question`, `choices`, `answer`,
`item_type`):

- **360 contaminated** (`is_contaminated: true`): 60 each from MMLU, MATH (`EleutherAI/hendrycks_math`),
  ARC-Challenge, TruthfulQA, HellaSwag and GSM8K. `benchmark_hf_dir`, `benchmark_config`,
  `benchmark_split` and `benchmark_row` point to the source item, and `contamination_type` says how it was
  altered: `exact_copy` (110), `surface_change` (120), `light_rewording` (80) or `option_shuffle` (50).
- **360 clean** (`is_contaminated: false`): 60 per benchmark, written in the style of that benchmark
  (`item_type` names it) but not taken from it, so they are near misses that test for false positives.

A clean sample that is removed is a false positive; a contaminated sample that is kept is a false
negative. The file is not in the unified schema, so convert it first with
`02-standardisation/convert-contamination-test.py` (`question` as `initial_prompt`, the test set's `id` kept in
`original_metadata`).

`validation/contamination_dataset_v3.jsonl` is a larger set in the same format: 19,722 samples, 9,361
contaminated from 72 benchmarks (also `partial`, a part of a benchmark item) and 10,361 clean, most of them
wrapped in chat templates, JSON records, few-shot prompts or quizzes. Its `answer` can be null.

An earlier version, `contamination_dataset_v1_old.jsonl` (removed, still in commit `7564cae`), has the
same 360 contaminated samples, but its 360 clean samples are short, generic math, multiple-choice, QA and
completion items (`item_type` `math`, `mcq`, `qa`, `completion`).

## Files

### Pipeline (in run order)
- **`gather-decontamination-prompts.py`** - Gathers the evaluation benchmark prompts listed in `benchmarks.yaml`
- **`decontamination-parallel.py`** - Writes reports for a subset of benchmarks (one array job), or for all
- **`merge-decontamination-reports.py`** - Combines the reports and filters the dataset
- **`submit-parallel-decontamination.sh`** - SLURM job array + merge job, or one job running both (`single`)

### Shared modules
- **`matching_units.py`** - Normalisation and units compared by the check
- **`decontamination_common.py`** - N-gram matching, verification and benchmark n-gram cache
- **`benchmark_filters.py`** - Benchmark exclusion patterns and the expected benchmark list
- **`conversation_id_checks.py`** - Validation of `conversation_id`s
- **`removal_log.py`** - Writes the removal log (removed samples with their matched benchmark prompts)

### Configuration
- **`benchmarks.yaml`** - The benchmarks to decontaminate against (HF dataset, configs, splits, prompt column)
- **`decontamination_settings.py`** - Tokenizer, prompt and cache paths, matching settings (see [Settings](#settings))
- **`slurm_config.sh`** - Slurm settings of the submit script: account, partition, time limits, logs, Python env

## Requirements and guarantees

- **Unique string IDs.** Every sample needs a unique, non-empty string `conversation_id`. They are
  assigned in `02-standardisation` (`conversation_ids.py`); all decontamination scripts refuse to run
  on empty, duplicated or non-string IDs.
- **Every benchmark checked.** The merge refuses to filter unless every benchmark of the prompt set
  (minus `DECONTAM_EXCLUDE_BENCHMARK_PATTERNS`) has a readable report. Reports for benchmarks that are
  not in the prompt set are ignored.
- **Re-runs reuse reports.** Existing reports are not recomputed and are always applied in the final
  filter, so a failed or interrupted run can simply be relaunched.
- **No silent no-op filter.** Filtering fails without saving if the number of removed rows differs from
  the number of flagged IDs (reports from a different version of the data).
- **Recorded in metadata.** The output's `dataset_metadata.json` gets a processing-log entry with
  `benchmarks_processed`, `benchmark_names`, `contaminated_ids_flagged` and
  `contaminated_samples_removed`. `07-dataset-aggregation/concatenate-datasets.py` only accepts inputs
  whose latest decontamination entry covers every benchmark.
- **Removal log.** The removed samples are saved unchanged to `<output>/decontamination_removed/` (same
  splits as the input), with a `decontamination_matches` column holding one entry per flagging benchmark:
  `benchmark`, `eval_idx`, `benchmark_prompt`, `benchmark_prompt_tokens`, `match_ratio`, `longest_block`
  `matched_text` (the matched runs, joined with ` [...] `) and `user_turn` (the user message with the
  highest `match_ratio`, 0 = `initial_prompt`). `match_ratio` is recomputed at filter
  time with the merge's tokenizer, which must be the one the reports were made with (the merge checks
  `decontamination_settings.json`); the processing-log entry records `logged_matches_below_threshold`. Reports keep only the
  first benchmark prompt above the threshold, so `benchmark_prompt` is a match, not necessarily the
  closest one. The entry also gets `removed_by_benchmark` and `removal_log_path`.

## Settings

All decontamination settings are in `decontamination_settings.py`; every script takes its defaults from
there, and the submit script reads the tokenizer and the paths from it. The tokenizer and the paths can be
overridden from the environment; the matching settings are changed in the file only, so that all scripts
use the same values.

| Setting | Default | Meaning |
|---|---|---|
| `TOKENIZER_NAME` (env) | `swiss-ai/Apertus-8B-Instruct-2509` | tokenizer the units are built from |
| `DECONTAMINATION_PROMPTS` (env) | `/capstor/store/cscs/swissai/infra01/posttrain_data/04_decontaminated/decontamination_prompts` | prompt set |
| `DECONTAMINATION_CACHE_DIR` (env) | `/iopsstor/scratch/cscs/$USER/posttraining-data/decontamination_cache` | benchmark n-gram cache |
| `NGRAM_LENGTH` | 8 | n-gram length of the candidate search |
| `DIFF_THRESHOLD` | 0.5 | share of the benchmark prompt that matched runs must cover |
| `MIN_BLOCK` | 5 | minimum length of a matched run that counts towards that share |
| `EXACT_MAX_UNITS` | 3 | benchmark prompts up to this length must match the training prompt exactly |
| `SHORT_MIN_COVERAGE` | 0.5 | longer short prompts must make up this share of the training prompt |

**Reports record their settings.** The first job writes the matching settings (with `UNITS_VERSION` and
`MATCHING_VERSION`) to `<reports>/decontamination_settings.json`. A relaunch or merge with other settings
on the same reports stops with an error instead of reusing reports made with the old ones; use a new output
path. Reports from before this file existed are assumed to match (with a warning). The merge also records
the settings in the processing log (`matching_settings`).

The Slurm settings are in `slurm_config.sh`; every value can be overridden from the environment:

| Variable | Default |
|---|---|
| `SLURM_ACCOUNT` | `infra01` |
| `SLURM_PARTITION` | `normal` |
| `SLURM_QOS` | `normal` (also valid on `debug`; set to `""` for the partition default) |
| `SLURM_RESERVATION` | unset |
| `SLURM_LOG_DIR` | `/iopsstor/scratch/cscs/$USER/posttraining-data/slurm_logs` |
| `DECONTAM_TIME_LIMIT` / `PARALLEL_TIME_LIMIT` / `MERGE_TIME_LIMIT` | `12:00:00` / `8:00:00` / `2:00:00` |
| `PYTHON_ENV_ACTIVATE` | `<repo>/.venv/bin/activate` (created by `uv sync`) |

**Tokenizer: adjust it for new model versions.** The n-grams are built from token IDs, so
`TOKENIZER_NAME` should be the tokenizer of the model being trained (currently Apertus 1.5:
`swiss-ai/Apertus-8B-Instruct-2509`). For a new model version with a different tokenizer, update the
default in `decontamination_settings.py`. The benchmark n-gram cache is keyed by tokenizer and `UNITS_VERSION` (`matching_units.py`), so
a new tokenizer or a change of the units builds its own cache entries; results from runs with different tokenizers are not directly comparable (v1.0 used
`alehc/swissai-tokenizer`).

`HF_HOME` is not set by the scripts; jobs inherit it from your shell. The `debug` partition allows at most
2 submitted jobs per user, so use the single-job mode there (`chunk_size` `single`).
`DECONTAM_TIME_LIMIT` is the time limit of that job.

## Usage

### 1. Gather the benchmark prompts (only when benchmarks change)

Edit `benchmarks.yaml`, then gather the prompts. The n-gram cache is keyed by the benchmark name, not by
its prompts, so give an entry whose prompts change a new `name` (or use a new `--cache_dir`).

```bash
python 04-decontamination/gather-decontamination-prompts.py \
  --output "/capstor/store/cscs/swissai/infra01/posttrain_data/04_decontaminated/decontamination_prompts"
```

### 2. Decontaminate a dataset

```bash
./04-decontamination/submit-parallel-decontamination.sh <input_dataset> <output_dataset> [chunk_size|single] [max_parallel_jobs]
```

Both modes write the reports to `<output_dataset>_parallel_reports/` and reuse existing reports there, so a
relaunch (in either mode) recomputes only the missing ones.

**Parallel (job array + merge job)** — the default, for full runs:

```bash
./04-decontamination/submit-parallel-decontamination.sh \
  /path/to/02_standardised/my-dataset /path/to/04_decontaminated/my-dataset [chunk_size] [max_parallel_jobs]
```

- `chunk_size`: benchmarks per array job (default: 20)
- `max_parallel_jobs`: maximum array jobs running at once (default: 64)

The script submits a job array and a merge job that runs after it (`afterany`). The merge checks that
every benchmark has a readable report (reports are written atomically); if an array job failed, it stops
without saving.

**Single job** — one job that writes the reports of all benchmarks and then runs the merge, e.g. on the
debug partition (at most 2 submitted jobs per user):

```bash
SLURM_PARTITION=debug ./04-decontamination/submit-parallel-decontamination.sh \
  /path/to/small-dataset /path/to/output single
```

Its time limit is `DECONTAM_TIME_LIMIT`; `max_parallel_jobs` is ignored.

Manual merge (e.g. after fixing a failed array job):

```bash
python 04-decontamination/merge-decontamination-reports.py \
  <input_dataset> <output_dataset> <output_dataset>_parallel_reports \
  --decontamination_prompts "$DECONTAMINATION_PROMPTS"
```

### 3. Without SLURM (e.g. interactively)

```bash
python 04-decontamination/decontamination-parallel.py <input_dataset> \
  --report_path <output_dataset>_parallel_reports --num_proc 16
python 04-decontamination/merge-decontamination-reports.py \
  <input_dataset> <output_dataset> <output_dataset>_parallel_reports
```

The tokenizer, prompt set, cache and matching settings come from `decontamination_settings.py` (or the
environment variables above). Without `--benchmark-start`/`--benchmark-end` the first command checks all
benchmarks. The merge requires a report for every benchmark.

Useful flags of `decontamination-parallel.py`: `--overwrite` recomputes existing reports,
`--show_contaminated` prints examples of contaminated pairs, `--benchmark_name` restricts the run to
specific benchmarks (comma-separated).

### Monitoring

```bash
squeue -u $USER
tail -f "$SLURM_LOG_DIR"/pdecontam_<dataset>_<timestamp>_*.out   # array jobs
tail -f "$SLURM_LOG_DIR"/merge_<dataset>_<timestamp>.out          # merge job
tail -f "$SLURM_LOG_DIR"/decontam_<dataset>_<timestamp>.out       # single-job mode
```

### Utilities

```bash
# List the benchmarks of a prompt set (after exclusions)
python -c 'import sys; sys.path.insert(0, "04-decontamination"); from benchmark_filters import expected_benchmark_names; print("\n".join(expected_benchmark_names(sys.argv[1])))' /path/to/decontamination_prompts

# Exclude benchmarks (comma-separated name patterns, case/punctuation-insensitive)
export DECONTAM_EXCLUDE_BENCHMARK_PATTERNS="wmdp,hallulens"
```

## History

The per-dataset commands used for the v1.0 SFT mixture (inputs from `03_license_filtered`, tokenizer
`alehc/swissai-tokenizer`, ~400 benchmarks) are in the git history of this README. Outputs from those runs
and from the v1.5 runs (`04_decontaminated/sft-1.1/`) predate the checks above and are rejected by
`concatenate-datasets.py` unless `--skip-decontamination-check` is passed.

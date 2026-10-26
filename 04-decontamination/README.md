# Decontamination

Cross-checks the prompts of a training dataset against the prompts of the evaluation benchmarks.
If significant overlap is detected, the sample is removed from the training data.

## How it works

1. The first user turn (`initial_prompt`) of every training sample and every benchmark prompt are
   normalised (Unicode NFKC, runs of a repeated non-letter character such as `━━━━━` collapsed to one,
   whitespace collapsed to one space), tokenized (`swiss-ai/Apertus-8B-Instruct-2509`, without `<s>`)
   and split into units. A unit is a token, except that a number (`1,023`, `16.482`, `2'832`) counts
   as one unit; the tokenizer would split it into single digits (`matching_units.py`).
2. Each training prompt is compared with each benchmark prompt. The rule depends on the length of the
   benchmark prompt:

   | Benchmark prompt (after normalisation) | Rule | A training sample is removed if … |
   |---|---|---|
   | **0 units** (empty or only whitespace) | skipped, counted as empty | never |
   | **1–3 units** (e.g. "5!=", "Paris", "Yes.") | exact equality only | the training prompt is **identical** to the benchmark prompt |
   | **4–7 units** (e.g. "Compute $\cos 225^\circ$.") | complete containment + coverage guard | the **whole** benchmark prompt appears as one continuous run in the training prompt **and** makes up **at least 50%** of the training prompt's units |
   | **8+ units** | 8-gram candidates + share of matching runs | the training prompt shares **at least one run of 8 consecutive units** with the benchmark prompt (candidate), **and** the runs of **at least 5 consecutive units** that both have in common (`difflib.SequenceMatcher`) cover **at least 50%** of the benchmark prompt's units |

   Prompts shorter than 8 units have no 8-gram, so they need their own rules. The number of
   benchmark prompts in the classes 0, 1–3 and 4–7 is recorded in the processing log
   (`short_benchmark_prompts`, single-job mode only).
3. For each benchmark a report `<benchmark>__contamination_report.json` is written, mapping each
   contaminated `conversation_id` to the matched benchmark prompt.
4. The final filter removes every `conversation_id` that appears in any report.

Only near-verbatim overlap in the first user turn is detected; system prompts, later turns and
responses are not checked.

## Test set for the decontamination strategy

`contamination_dataset.jsonl` is a synthetically generated, labelled test set for measuring how many false positives and false
negatives the decontamination produces. It has 720 samples (`id`, `question`, `choices`, `answer`,
`item_type`):

- **360 contaminated** (`is_contaminated: true`): 60 each from MMLU, MATH (`EleutherAI/hendrycks_math`),
  ARC-Challenge, TruthfulQA, HellaSwag and GSM8K. `benchmark_hf_dir`, `benchmark_config`,
  `benchmark_split` and `benchmark_row` point to the source item, and `contamination_type` says how it was
  altered: `exact_copy` (110), `surface_change` (120), `light_rewording` (80) or `option_shuffle` (50).
- **360 clean** (`is_contaminated: false`): 60 per benchmark, written in the style of that benchmark
  (`item_type` names it) but not taken from it, so they are near misses that test for false positives.

A clean sample that is removed is a false positive; a contaminated sample that is kept is a false
negative. The file is not in the unified schema, so convert it first (`question` as `initial_prompt`,
`id` as `conversation_id`).

An earlier version, `contamination_dataset_v1_old.jsonl` (removed, still in commit `7564cae`), has the
same 360 contaminated samples, but its 360 clean samples are short, generic math, multiple-choice, QA and
completion items (`item_type` `math`, `mcq`, `qa`, `completion`).

## Available Scripts

### Python Scripts
- **`decontamination.py`** - Single-job decontamination: all benchmarks in one process, then filtering
- **`decontamination-parallel.py`** - Writes reports for a subset of benchmarks (one array job)
- **`merge-decontamination-reports.py`** - Combines the reports of all array jobs and filters the dataset
- **`matching_units.py`** - Normalisation and units compared by the check (shared by all scripts)
- **`gather-decontamination-prompts.py`** - Gathers the evaluation benchmark prompts
- **`list-benchmarks.py`** - Lists/counts the benchmarks used for the job array
- **`benchmark_filters.py`** - Benchmark exclusion patterns and the expected benchmark list
- **`conversation_id_checks.py`** - Validation of `conversation_id`s
- **`removal_log.py`** - Writes the removal log (removed samples with their matched benchmark prompts)

### Shell Scripts
- **`submit-parallel-decontamination.sh`** - SLURM job array + merge job (recommended)
- **`submit-decontamination.sh`** - Single SLURM job running `decontamination.py`
- **`slurm_config.sh`** - Shared settings for both submit scripts

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
  and `matched_text` (the matched runs, joined with ` [...] `). `match_ratio` is recomputed at filter
  time, so pass the tokenizer the reports were made with (also to the merge script); the processing-log
  entry records `logged_matches_below_threshold`, which is non-zero when it was not. Reports keep only the
  first benchmark prompt above the threshold, so `benchmark_prompt` is a match, not necessarily the
  closest one. The entry also gets `removed_by_benchmark` and `removal_log_path`.

## Settings

Both submit scripts read their settings from `slurm_config.sh`. Every value can be overridden from the
environment:

| Variable | Default |
|---|---|
| `SLURM_ACCOUNT` | `infra01` |
| `SLURM_PARTITION` | `normal` |
| `SLURM_QOS` | `normal` (also valid on `debug`; set to `""` for the partition default) |
| `SLURM_RESERVATION` | unset |
| `SLURM_LOG_DIR` | `/iopsstor/scratch/cscs/$USER/posttraining-data/slurm_logs` |
| `DECONTAM_TIME_LIMIT` / `PARALLEL_TIME_LIMIT` / `MERGE_TIME_LIMIT` | `12:00:00` / `8:00:00` / `2:00:00` |
| `PYTHON_ENV_ACTIVATE` | `<repo>/venv/bin/activate` |
| `DECONTAMINATION_PROMPTS` | `/capstor/store/cscs/swissai/infra01/posttrain_data/04_decontaminated/decontamination_prompts` |
| `DECONTAMINATION_CACHE_DIR` | `/capstor/store/cscs/swissai/infra01/posttrain_data/decontamination_cache` |
| `TOKENIZER_NAME` | `swiss-ai/Apertus-8B-Instruct-2509` |

**Tokenizer: adjust it for new model versions.** The n-grams are built from token IDs, so
`TOKENIZER_NAME` should be the tokenizer of the model being trained (currently Apertus 1.5:
`swiss-ai/Apertus-8B-Instruct-2509`). For a new model version with a different tokenizer, update the
default in `slurm_config.sh` (and pass the same `--tokenizer_name` when calling the Python scripts
directly). The benchmark n-gram cache is keyed by tokenizer and `UNITS_VERSION` (`matching_units.py`), so
a new tokenizer or a change of the units builds its own cache entries; results from runs with different tokenizers are not directly comparable (v1.0 used
`alehc/swissai-tokenizer`).

`HF_HOME` is not set by the scripts; jobs inherit it from your shell. The `debug` partition allows at most
2 submitted jobs per user, so use a chunk size that gives a single array job there (e.g. `chunk_size` ≥
number of benchmarks).

## Usage

### 1. Gather the benchmark prompts (only when benchmarks change)

```bash
python 04-decontamination/gather-decontamination-prompts.py \
  --output "/capstor/store/cscs/swissai/infra01/posttrain_data/04_decontaminated/decontamination_prompts"
```

### 2. Parallel decontamination (recommended)

```bash
./04-decontamination/submit-parallel-decontamination.sh <input_dataset> <output_dataset> [chunk_size] [max_parallel_jobs]
```

- `chunk_size`: benchmarks per array job (default: 20)
- `max_parallel_jobs`: maximum array jobs running at once (default: 64)

The script submits a job array (reports go to `<output_dataset>_parallel_reports/`) and a merge job that
runs after the array (`afterany`). The merge checks the completion markers (`--expected-jobs`) and that
every benchmark has a report; if an array job failed, it stops without saving. Relaunching the same command
recomputes only the missing reports.

Example, and a quick test on the debug partition:

```bash
./04-decontamination/submit-parallel-decontamination.sh \
  /path/to/02_standardised/my-dataset /path/to/04_decontaminated/my-dataset

SLURM_PARTITION=debug ./04-decontamination/submit-parallel-decontamination.sh \
  /path/to/small-dataset /path/to/output 1000
```

Manual merge (e.g. after fixing a failed array job):

```bash
python 04-decontamination/merge-decontamination-reports.py \
  <input_dataset> <output_dataset> <output_dataset>_parallel_reports \
  --decontamination_prompts "$DECONTAMINATION_PROMPTS" \
  --expected-jobs <number_of_array_jobs> \
  --tokenizer_name "swiss-ai/Apertus-8B-Instruct-2509"
```

### 3. Single-job decontamination

```bash
./04-decontamination/submit-decontamination.sh <input_dataset> <output_dataset>
```

Runs `decontamination.py` in one job; reports go to `<output_dataset>/contamination_reports/`.
Directly (e.g. interactively):

```bash
python 04-decontamination/decontamination.py <input_dataset> \
  --output <output_dataset> \
  --decontamination_prompts /capstor/store/cscs/swissai/infra01/posttrain_data/04_decontaminated/decontamination_prompts \
  --tokenizer_name "swiss-ai/Apertus-8B-Instruct-2509" \
  --report_path <output_dataset>/contamination_reports \
  --cache_dir /capstor/store/cscs/swissai/infra01/posttrain_data/decontamination_cache \
  --ngram_length 8 --diff_threshold 0.5 --num_proc 16
```

Useful flags: `--overwrite` recomputes existing reports, `--show_contaminated` prints examples of
contaminated pairs, `--benchmark_name` restricts the run to specific benchmarks. `--base_report_path`
takes the 8-gram matches from the reports of an earlier run with the same units but without the
short benchmark prompt check, and runs only that check (much faster than a full run).

### Monitoring

```bash
squeue -u $USER
tail -f "$SLURM_LOG_DIR"/pdecontam_<dataset>_<timestamp>_*.out   # array jobs
tail -f "$SLURM_LOG_DIR"/merge_<dataset>_<timestamp>.out          # merge job
```

### Utilities

```bash
# List / count the benchmarks and estimate the number of array jobs
python 04-decontamination/list-benchmarks.py --prompts-path /path/to/decontamination_prompts --chunk-size 20

# Exclude benchmarks (comma-separated name patterns, case/punctuation-insensitive)
export DECONTAM_EXCLUDE_BENCHMARK_PATTERNS="wmdp,hallulens"
```

## History

The per-dataset commands used for the v1.0 SFT mixture (inputs from `03_license_filtered`, tokenizer
`alehc/swissai-tokenizer`, ~400 benchmarks) are in the git history of this README. Outputs from those runs
and from the v1.5 runs (`04_decontaminated/sft-1.1/`) predate the checks above and are rejected by
`concatenate-datasets.py` unless `--skip-decontamination-check` is passed.

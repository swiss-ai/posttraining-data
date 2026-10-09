# Decontamination

Cross-checks the prompts of a training dataset against the prompts of the evaluation benchmarks.
If significant overlap is detected, the sample is removed from the training data.

## How it works

1. The first user turn (`initial_prompt`) of every training sample and every benchmark prompt are
   tokenized (`swiss-ai/Apertus-8B-Instruct-2509`) and split into 8-grams.
2. Training samples that share at least one 8-gram with a benchmark prompt are candidates.
3. A candidate is contaminated if at least 50% of the benchmark prompt's tokens appear in the training
   prompt in aligned runs of 5 or more tokens (`difflib.SequenceMatcher`).
4. For each benchmark a report `<benchmark>__contamination_report.json` is written, mapping each
   contaminated `conversation_id` to the matched benchmark prompt.
5. The final filter removes every `conversation_id` that appears in any report.

Only near-verbatim overlap in the first user turn is detected; system prompts, later turns and
responses are not checked.

## Available Scripts

### Python Scripts
- **`decontamination.py`** - Single-job decontamination: all benchmarks in one process, then filtering
- **`decontamination-parallel.py`** - Writes reports for a subset of benchmarks (one array job)
- **`merge-decontamination-reports.py`** - Combines the reports of all array jobs and filters the dataset
- **`gather-decontamination-prompts.py`** - Gathers the evaluation benchmark prompts
- **`list-benchmarks.py`** - Lists/counts the benchmarks used for the job array
- **`benchmark_filters.py`** - Benchmark exclusion patterns and the expected benchmark list
- **`conversation_id_checks.py`** - Validation of `conversation_id`s

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
directly). The benchmark n-gram cache is keyed by tokenizer, so a new tokenizer builds its own cache
entries; results from runs with different tokenizers are not directly comparable (v1.0 used
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
contaminated pairs, `--benchmark_name` restricts the run to specific benchmarks.

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

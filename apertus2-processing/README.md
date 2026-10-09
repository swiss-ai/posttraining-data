# Apertus 2 data processing

Check and tokenize **native Apertus 2 datasets** with
[apertus-common](https://github.com/swiss-ai/apertus-common), locally or across Slurm
nodes. Checking produces per-sample issue reports; tokenization produces Megatron
indexed datasets (`.bin`/`.idx`) with optional loss weights. Interrupted runs resume
from completed shards.

**Prepare native data first.** Read the
[format specification](https://github.com/swiss-ai/apertus-2-format-spec-proposal/blob/v1/spec.md)
and [Apertus 2 profile](https://github.com/swiss-ai/apertus-2-format-spec-proposal/blob/v1/profile/apertus_2/apertus_2.md),
then write a mapper such as [mappers/no_robots.py](mappers/no_robots.py). The processing
commands do not convert other chat formats.

## Install

Requires Python 3.13+, [uv](https://docs.astral.sh/uv/), Git, and GitHub read access to
the private `apertus-common` repository. [pyproject.toml](pyproject.toml) pins a
revision from its [apertus2 branch](https://github.com/swiss-ai/apertus-common/tree/apertus2).
For a workstation:

```sh
cd apertus2-processing
uv sync --locked
uv run --no-sync apertus-data --help
```

### CSCS Alps setup

The launcher defaults to the official [Alps Extended Image](https://docs.cscs.ch/software/alps-extended-images/)
in [slurm/alps.toml](slurm/alps.toml), currently `ngc-pytorch:26.02-py3-alps6`
from the CSCS registry. No personal SquashFS image or `~/.edf` entry is needed.
`slurm/setup.sh` installs the locked dependencies and managed Python 3.13 on a
compute node; it does not use the container's older Python. The default environment
is `$SCRATCH/apertus2-processing/venv`. uv's cache and Python installation also live
on scratch and must remain available for subsequent jobs.

On the login node, discover an authorized account and cache the private Git
dependency. This avoids relying on login-only credential helpers inside jobs:

```sh
cd apertus2-processing
sacctmgr -nP show assoc where user="$(id -un)" format=Account,QOS
scontrol show partition debug
export JOB_ACCOUNT=...  # authorized account from the output above
mkdir -p "$SCRATCH/apertus2-processing/logs"
export APERTUS_COMMON_MIRROR="$SCRATCH/apertus2-processing/apertus-common.git"
git clone --mirror https://github.com/swiss-ai/apertus-common.git "$APERTUS_COMMON_MIRROR"
# If the mirror already exists, refresh it instead:
# git -C "$APERTUS_COMMON_MIRROR" fetch --prune origin
sbatch --account="$JOB_ACCOUNT" \
  --output="$SCRATCH/apertus2-processing/logs/setup-%j.out" slurm/setup.sh
```

Wait for the setup job to complete successfully before processing. It requests
one debug node for at most 20 minutes, with QoS `normal`. The mirror is only needed
for installation; the dependency URL and commit stay pinned by `uv.lock`.
`uv` must be available at `$HOME/.local/bin/uv`, or set `UV_BIN` to its absolute path
on a shared filesystem. Network access is needed for the registry and public Python
packages. A first image pull may take several minutes; retry setup if it times out.

Override `UV_PROJECT_ENVIRONMENT` for a separate scratch environment and
`APERTUS_ENVIRONMENT` for a different EDF **absolute path**, consistently for setup
and processing. Do not pass `--environment` to `sbatch`: each launcher starts its
compute steps in the container. Outside Alps, install in your site's environment
and set `APERTUS_ENVIRONMENT=none` and `UV_PROJECT_ENVIRONMENT` when using `run.sh`.
Scratch is subject to automatic purging; rerun setup if the environment or its
managed interpreter has been removed. Store durable datasets/results in project
storage; keep caches, temporary data, and logs on scratch.

Tokenization also requires the Apertus 2 instruct tokenizer from
[apertus-omni-tokenizer](https://github.com/swiss-ai/apertus-omni-tokenizer/tree/feat/apertus-2-text-tokenizers):

```sh
git clone --depth 1 --branch feat/apertus-2-text-tokenizers \
  https://github.com/swiss-ai/apertus-omni-tokenizer.git ../../apertus-omni-tokenizer
export APERTUS_ENCODING="$(cd ../../apertus-omni-tokenizer && pwd)/tokenizers/Apertus_2_instruct"
```

## Native format

Each conversation has one system prompt and items in observed order:

```json
{
  "schema_version": 2,
  "system": {"thinking": "medium", "tools": []},
  "items": [
    {"type": "user", "payload": "What is 6 times 7?"},
    {"type": "think", "payload": "Multiply the numbers."},
    {"type": "reply", "payload": "42"},
    {"type": "wait"}
  ]
}
```

The complete [examples/native.json](examples/native.json) also demonstrates system
settings, documents and citations, parallel tool calls and results, host notices,
and a gradable claim:

| Part | Meaning |
|---|---|
| `system` | Required `thinking`: `low`, `medium` (standard), or `high`. Optional `identity`, `behavior`, `environment`, and `tools`; each tool has a `name`, `description`, argument `schema`, and optional `policy`. |
| Input items | `user`; a `document` with supplier `from` and citation `id` (e.g. `a1`, cited as `[a1]`); a `host` notice from the harness; tool `result`s. |
| Output items | `think`, `call`, `claim` (a gradable answer before its `reply`), and `reply`. Call `counter`s increase across the whole conversation; matching results carry the same `name` and `counter`. Calls may run in parallel. |
| `wait` | The model stops for new input: after each turn's outputs, including the last, and after calls that await results. Further output requires new input. |

Input arriving mid-turn, such as a tool result, is valid but produces a check warning.
Build conversations with `apertus-common` and serialize with `Conversation.to_json()`.
Messages use `type` alone: the library derives `direction`, which must not appear in
JSON. Only schema v2 is accepted. Regenerate v1 or direction-bearing records with
your mapper, check them again, and use new run directories.

### Loss weights

Defaults train on all output tokens and waits, with no loss on system or input
tokens. An item's `loss_weight` overrides only that item's default; unannotated
items and samples keep their defaults. Weights must be finite and non-negative;
fractions and values above 1 are allowed.

| Tokens | Default | With `loss_weight` |
|---|---|---|
| BOS; system and input opening, header, and closing tokens | 0 | Always 0 |
| System and input payloads | 0 | Item's weight |
| Output items, including control tokens | 1 | Item's weight |
| Each wait | 1 | Wait's weight |

[examples/weighted.json](examples/weighted.json) disables loss on reasoning and
halves the reply's weight:

```json
{
  "schema_version": 2,
  "system": {"thinking": "medium", "tools": []},
  "items": [
    {"type": "user", "payload": "What is 6 times 7?"},
    {"type": "think", "payload": "Multiply the numbers.", "loss_weight": 0},
    {"type": "reply", "payload": "42", "loss_weight": 0.5},
    {"type": "wait", "loss_weight": 1}
  ]
}
```

For annotated data, pass `--loss-weights` to **both** `check` and `tokenize`:

| Mode | With `--loss-weights` | Without it |
|---|---|---|
| `check` | Accepts annotations | Annotations fail with `profile/loss-weight` |
| `tokenize` | Writes tokens and `loss_weights.bin/.idx`: float32, one weight per token in every document, including defaults | Ignores annotations and writes only tokens; the trainer applies the defaults |

Use `RUN_DIR/loss_weights` as the Megatron loss-weight prefix.

## Prepare a dataset

### Storing conversations

Store one native JSON **string** per row in a column named `conversation_json`
(or choose another with `--column`). Supported inputs:

| Input | Read order and selection |
|---|---|
| `.parquet` file or directory | Files are discovered recursively in sorted path order; hidden files and directories are skipped. Every discovered Parquet file is processed. |
| Saved HF `Dataset` or `DatasetDict` | Dataset row order; choose a `DatasetDict` split with `--split` (default `train`). |

For example, save conversations with `datasets`:

```python
from datasets import Dataset

Dataset.from_dict({"conversation_json": [c.to_json() for c in conversations]}).save_to_disk(
    "/data/my-native"
)
```

For native data on the Hugging Face Hub, download the Parquet files of **one split**
and pass the local directory:

```sh
uv run --no-sync hf download --repo-type dataset NAME \
  --include "data/train-*" --local-dir DIR
```

### Mapping a dataset

[mappers/no_robots.py](mappers/no_robots.py) maps the small
[HuggingFaceH4/no_robots](https://huggingface.co/datasets/HuggingFaceH4/no_robots)
dataset using the `apertus-common` builder. It sets `thinking` to `medium`, maps a
source system message to `behavior`, and adds a wait after each assistant answer.
It skips rows whose user/assistant roles do not alternate and saves
`conversation_json` plus the source `prompt_id` as an HF dataset.

`--split` selects `train` (default) or `test`; `--num-proc` controls filtering and
mapping workers (default 1). Run from `apertus2-processing`, on a laptop or an
allocated compute node, never a cluster login node. After the Alps setup, the
cluster example starts a one-node debug allocation (omit account/partition/time
when using an existing allocation):

```sh
# Laptop
uv run --no-sync python mappers/no_robots.py /data/no_robots-native --num-proc 8
# Cluster; caches and output stay on scratch
export HF_HOME="$SCRATCH/.cache/huggingface"
srun --account="$JOB_ACCOUNT" --partition=debug --qos=normal --time=00:20:00 \
  --nodes=1 --ntasks=1 --cpus-per-task=32 \
  --environment="${APERTUS_ENVIRONMENT:-$PWD/slurm/alps.toml}" --container-workdir="$PWD" \
  "${UV_PROJECT_ENVIRONMENT:-$SCRATCH/apertus2-processing/venv}/bin/python" \
  mappers/no_robots.py "$SCRATCH/no_robots-native" \
  --num-proc 32
```

In your own mapper, pass `load_from_cache_file=False` to `datasets.map`: its cache
ignores library upgrades and can return an older native format. The example
mapper disables caching for both filtering and mapping.

## Check and tokenize

**Check first**, fix the mapper until checking passes, then tokenize the same input
with the same options. Tokenization assumes checked data and stops on records it
cannot parse or encode. Run processing on a laptop or allocated compute nodes,
never a cluster login node.

Submit [slurm/run.sh](slurm/run.sh) from `apertus2-processing`, adding your site's
`--account`, `--partition`, and `--qos` before the script path. On Alps the launcher
uses the image and scratch environment described above. The installation, `INPUT`, and `RUN_DIR` must be shared across
nodes. For a new run, choose a new or empty `RUN_DIR` outside the input:

```sh
sbatch --account="$JOB_ACCOUNT" --partition=debug --qos=normal --nodes=2 --time=00:20:00 \
  --output="$SCRATCH/apertus2-processing/logs/check-%j.out" \
  slurm/run.sh check "$SCRATCH/no_robots-native" "$SCRATCH/no_robots-check" --shard-rows 500
# After the check job succeeds:
sbatch --account="$JOB_ACCOUNT" --partition=debug --qos=normal --nodes=2 --time=00:20:00 \
  --output="$SCRATCH/apertus2-processing/logs/tokenize-%j.out" \
  slurm/run.sh tokenize "$SCRATCH/no_robots-native" "$SCRATCH/no_robots-tokens" \
  --shard-rows 500 --tokenizer "$APERTUS_ENCODING"
```

These examples use at most two debug nodes for 20 minutes. For production, select
an authorized partition/QoS and time limit appropriate to the workload; project
reservations are no longer used. Never choose `highprio` without an explicit user
request. Each job runs one mode. By default, logs go to `apertus-data-<job id>.out` in the
submit directory. Exit codes: **0** success, **1** check found failing samples,
**2** processing error.

| Option | Meaning |
|---|---|
| `--column NAME` | Native JSON column (default `conversation_json`). |
| `--split NAME` | Saved `DatasetDict` split (default `train`). |
| `--loss-weights` | Accept annotations in check; export weights in tokenize. See [Loss weights](#loss-weights). |
| `--tokenizer DIR` | Required for tokenize; ignored by check, so both modes can share an option list. |
| `--shard-rows N` | Target rows per shard (default 10000). HF shards use this size; Parquet shards group whole row groups until at least this many rows. A final shard may be smaller. |

### Without Slurm

Run the same three steps locally:

```sh
uv run --no-sync apertus-data prepare check /data/no_robots-native /tmp/no_robots-check \
  --shard-rows 500
uv run --no-sync apertus-data encode /tmp/no_robots-check
uv run --no-sync apertus-data merge /tmp/no_robots-check
```

For tokenization, use `prepare tokenize` with `--tokenizer "$APERTUS_ENCODING"`.
`encode` uses all available CPUs by default; `--workers` and `--threads` control
parallelism as described below. Run `prepare` again before resuming.

### Tuning

| Setting | Meaning |
|---|---|
| `sbatch --nodes=N` | Allocated nodes (default 1), with one task per node. |
| `WORKERS` | Worker processes per node (default: CPUs divided by `THREADS`, rounded down, at least 1). Lower it if memory runs short. |
| `THREADS` | Tokenizer threads per worker (default 1). Only tokenization uses the threads; prefer scaling worker processes first. |
| `--shard-rows` | Aim for at least four shards per worker. Write Parquet with row groups of this size so shards can be distributed effectively. |

For example, prefix either submission above with `WORKERS=4 THREADS=2` for a small
validation run. Encoding logs include the Slurm task rank to identify each node.
Allow a few seconds of startup per worker; measure throughput with representative
conversations on your hardware.

### How a job runs

Both modes follow the same sequence:

```text
prepare   first node, once → RUN_DIR/run.json
encode    every node       → RUN_DIR/shards/NNNNNN/
            workers claim a free shard, process it, and repeat
merge     first node       → final outputs in RUN_DIR, in input order
```

- **`prepare`** plans shards from input row-count metadata and records options,
  library identity, and input/tokenizer file sizes, modification times, and sampled
  content hashes in `run.json`. Shard IDs refer to the same rows on every node and resume.
  An existing plan is verified and reused; stale claims and temporary shards are
  removed.
- **`encode`** uses one worker process per CPU by default. Workers coordinate through
  atomic directory creation under `claims/`: only one can claim each shard. Each
  writes to a temporary directory, then renames it into `shards/NNNNNN/`. There is
  no coordinator, and faster nodes take more shards.
- **`merge`** verifies all shards are complete and combines them in input order.
  Each shard contains the same file types as the [final outputs](#outputs), plus
  its summary.

### Resume after a crash or timeout

Resubmit the **identical command**. Completed shards are kept; only shards in
progress are lost after a crash. Missing shards are processed and merged. You may
change nodes, `WORKERS`, `THREADS`, and the time limit between submissions.

Changed inputs, options, tokenizer files, or `apertus-common` version/Git revision
require a new run directory. `encode` and `merge` also verify inputs, tokenizer
files, and library identity before starting.

Keep inputs and tokenizer files immutable throughout processing and resume. The
identity check hashes up to the first and last 64 KiB of each file, in addition to
its size and modification time. This catches same-size rewrites hidden by Lustre's
whole-second timestamps without rereading the entire dataset on every node. It is
**not a full content checksum** for files larger than 128 KiB: changes confined to
the middle with unchanged metadata can escape detection. Older run plans without
`sample_sha256` must use a new run directory after upgrading.

Never run two jobs on the same run directory concurrently. Use
`--job-name=<run name> --dependency=singleton` to queue a resubmission behind the
earlier job. After a successful merge, you may delete `RUN_DIR/shards/` and
`RUN_DIR/claims/`.

## Outputs

### Check

`RUN_DIR/issues.jsonl` contains one line per issue in input order; samples without
issues are omitted. Each issue identifies the `source` file/dataset and zero-based
`row` (within the selected split for HF data). `location.item`, when present,
indexes the conversation's `items`:

```json
{"source": "/data/part-00000.parquet", "row": 4, "rule": "conversation/system",
 "severity": "error",
 "location": {"scope": "system", "item": null, "field": null, "source": "value"},
 "message": "a complete history requires a system prompt"}
```

Errors and unevaluated findings fail a sample. Warnings, including
`training/final-wait` and `training/input-mid-burst`, do not, but often indicate
missing waits. Checker exceptions become `check/exception` findings with severity
`unevaluated`: the sample fails and processing continues with the next row.
`RUN_DIR/summary.json` aggregates the counts, for example:

```json
{"mode": "check", "rows": 8, "failed_rows": 2, "rows_with_issues": 4,
 "issues": [
   {"rule": "conversation/system", "severity": "error", "count": 2},
   {"rule": "training/final-wait", "severity": "warning", "count": 2}
 ]}
```

[`issue_rows`](src/apertus2_processing/report.py) returns `{source: [row, ...]}`
with sorted, unique row indices. By default it includes all samples with issues;
`failed_only=True` includes only failing samples. Run this inspection/filtering
recipe with `uv run --no-sync python` from `apertus2-processing`:

```python
from datasets import load_from_disk

from apertus2_processing.report import issue_rows

failed = issue_rows("/scratch/no_robots-check", failed_only=True)
for source, rows in failed.items():
    data = load_from_disk(source)  # For a DatasetDict, add [split] of the checked split.
    for sample in data.select(rows):
        print(sample["conversation_json"])
    bad = set(rows)
    data.select([i for i in range(len(data)) if i not in bad]).save_to_disk(f"{source}-passed")
```

For a Parquet source, use `load_dataset("parquet", data_files=source, split="train")`
to load that file in row order. A filtered copy is a new input: check it again in a
new run directory.

### Tokenize

`RUN_DIR/tokens.bin/.idx` contain int32 token IDs, one document per conversation in
input order. Use `RUN_DIR/tokens` as the Megatron data prefix. Each document starts
with BOS and ends with the conversation's last item, normally a wait. Nothing is
padded, packed, or shifted. With `--loss-weights`, an aligned
`RUN_DIR/loss_weights.bin/.idx` pair is also written; see [Loss weights](#loss-weights).
`RUN_DIR/summary.json` records the totals:

```json
{"mode": "tokenize", "documents": 9485, "tokens": 2949133, "loss_weights": false}
```

## Development

On a workstation, or inside an allocated compute environment:

```sh
uv run --no-sync pytest
uv run --no-sync ruff check && uv run --no-sync ruff format --check
APERTUS_ENCODING=... uv run --no-sync pytest -k real_tokenizer
```

### Real two-node validation on Alps

After setup, set `APERTUS_ENCODING` to the real `Apertus_2_instruct` directory and
submit from this directory:

```sh
sbatch --account="$JOB_ACCOUNT" \
  --output="$SCRATCH/apertus2-processing/logs/test-%j.out" slurm/test.sh
```

[slurm/test.sh](slurm/test.sh) is limited to **two debug nodes and 20 minutes**. It
runs Ruff, the full pytest suite (including the real tokenizer), then actual
two-node `run.sh` launches with four workers/node and two tokenizer threads/worker.
The integration fixture contains 1,024 conversations: multilingual text, tools,
documents, citations, and fractional/zero loss weights. Assertions cover:

- Recursive Parquet and saved HF `DatasetDict` inputs, custom column and split.
- Successful checks and per-row errors/warnings with the expected failing exit code.
- Every token and weight against direct single-conversation encoding using the real
  tokenizer; identical token `.bin`/`.idx` files across the two input formats.
- Recovery after removing one generated shard and leaving temporary output;
  completed shards retain their contents and modification times.
- A second, fully completed resume that performs no encoding or shard rewrites.

Artifacts are saved under `$SCRATCH/apertus2-processing/validation-$SLURM_JOB_ID`
(override with a **new** `VALIDATION_DIR`): pytest JUnit XML, node names, resume logs,
fixtures, all processing outputs, and `validation.json` on success. These are
functional correctness tests, not a production-scale throughput or OOM benchmark.

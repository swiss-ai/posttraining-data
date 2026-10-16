# Apertus 2 data processing

Check and tokenize **native Apertus 2 datasets** with
[apertus-common](https://github.com/swiss-ai/apertus-common), locally or across Slurm
nodes. Checking produces per-sample issue reports; tokenization produces Megatron
indexed datasets (`.bin`/`.idx`) with optional loss weights, which `inspect` reads back
as text. Interrupted runs resume from completed shards.

The commands do not convert other chat formats: first produce native data with a
mapper ([Prepare a dataset](#prepare-a-dataset)).

## Install

Requires Python 3.13+, [uv](https://docs.astral.sh/uv/), Git, and GitHub read access to
the private `apertus-common` repository. [pyproject.toml](pyproject.toml) pins a
revision from its [apertus2 branch](https://github.com/swiss-ai/apertus-common/tree/apertus2).
Run every command from `apertus2-processing`, on a workstation or an allocated compute
node, never a cluster login node. For a workstation:

```sh
cd apertus2-processing
uv sync --locked
uv run --no-sync apertus-data --help
```

### CSCS Alps setup

`slurm/setup.sh` installs the locked dependencies and a managed Python 3.13 into
`$SCRATCH/apertus2-processing/venv` from a compute node, inside the official
[Alps Extended Image](https://docs.cscs.ch/software/alps-extended-images/) named in
[slurm/alps.toml](slurm/alps.toml) (currently `ngc-pytorch:26.02-py3-alps6` from the
CSCS registry; no personal SquashFS image or `~/.edf` entry is needed). It does not use
the container's older Python. `uv` must be available at `$HOME/.local/bin/uv`, or set
`UV_BIN` to its absolute path on a shared filesystem; the job needs network access for
the registry and public Python packages.

On the login node, discover an authorized account and cache the private Git
dependency, which avoids relying on login-only credential helpers inside jobs:

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
one debug node for at most 20 minutes, with QoS `normal`; a first image pull may take
several minutes, so retry if it times out. The mirror is only needed for installation;
the dependency URL and commit stay pinned by `uv.lock`.

Override `UV_PROJECT_ENVIRONMENT` for a separate scratch environment and
`APERTUS_ENVIRONMENT` for a different EDF **absolute path**, consistently for setup
and processing. Do not pass `--environment` to `sbatch`: each launcher starts its
compute steps in the container. Outside Alps, install in your site's environment
and set `APERTUS_ENVIRONMENT=none` and `UV_PROJECT_ENVIRONMENT` when using `run.sh`.
Everything on scratch (environment, `uv` cache, managed interpreter, logs) is purged
automatically: keep caches and temporary data there, durable datasets and results in
project storage, and rerun setup if the environment has been removed.

### Tokenizer

Tokenization requires the Apertus 2 instruct tokenizer from
[apertus-omni-tokenizer](https://github.com/swiss-ai/apertus-omni-tokenizer/tree/feat/apertus-2-text-tokenizers):

```sh
git clone --depth 1 --branch feat/apertus-2-text-tokenizers \
  https://github.com/swiss-ai/apertus-omni-tokenizer.git ../../apertus-omni-tokenizer
export APERTUS_ENCODING="$(cd ../../apertus-omni-tokenizer && pwd)/tokenizers/Apertus_2_instruct"
```

## Native format

Each conversation (see the [format specification](https://github.com/swiss-ai/apertus-2-format-spec-proposal/blob/v1/spec.md)
and [Apertus 2 profile](https://github.com/swiss-ai/apertus-2-format-spec-proposal/blob/v1/profile/apertus_2/apertus_2.md))
has one system prompt and items in observed order:

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
| `wait` | Ends an output burst (consecutive output items) and hands control to the harness for new input: after each turn's outputs, including the last, and after calls that await results. |

Only schema v2 is accepted; regenerate older conversations with your mapper.

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

### Input layout

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

Three mappers built on the foundation described in [Writing a mapper](#writing-a-mapper)
ship with the repository; each docstring states what a row becomes:

- [mappers/no_robots.py](mappers/no_robots.py) maps the small
  [HuggingFaceH4/no_robots](https://huggingface.co/datasets/HuggingFaceH4/no_robots)
  chat dataset, keeping the source `prompt_id`; `--split` selects `train` (default) or
  `test`.
- [mappers/nemotron_math.py](mappers/nemotron_math.py) maps
  [nvidia/Nemotron-SFT-Math-v4](https://huggingface.co/datasets/nvidia/Nemotron-SFT-Math-v4)
  with `thinking: high` and a `claim` holding the bare answer from the final reply's
  `\boxed{}`; `--input DIR` is the local download and `--subset` selects `cot` or `tir`.
- [mappers/apertus_tool_use.py](mappers/apertus_tool_use.py) maps the tool-use and
  terminal SFT mixes of swiss-ai/apertus-tool-use, keeping `teacher`, `subset`, `task`,
  `language` and `license` as columns;
  `--input FILE...` takes the mix files (JSON Lines), and `--max-tokens N --tokenizer DIR`
  skips samples longer than N tokens. Check and tokenize with `--loss-weights`.

`--help` lists the options every mapper shares (`--num-proc`, `--limit`, `--check`,
`--strict`, `--loss-weights`, `--thinking-convention`), described in
[Writing a mapper](#writing-a-mapper). After the Alps setup, the cluster example starts a
one-node debug allocation (omit account/partition/time when using an existing allocation):

```sh
# Laptop
uv run --no-sync python mappers/no_robots.py /data/no_robots-native --num-proc 8 --check
# Cluster; mappers have no launcher script and run in the container directly
export HF_HOME="$SCRATCH/.cache/huggingface"
srun --account="$JOB_ACCOUNT" --partition=debug --qos=normal --time=00:20:00 \
  --nodes=1 --ntasks=1 --cpus-per-task=32 \
  --environment="${APERTUS_ENVIRONMENT:-$PWD/slurm/alps.toml}" --container-workdir="$PWD" \
  "${UV_PROJECT_ENVIRONMENT:-$SCRATCH/apertus2-processing/venv}/bin/python" \
  mappers/no_robots.py "$SCRATCH/no_robots-native" \
  --num-proc 32 --check
```

### Writing a mapper

The profile is the authority on the format; this section gives the process and the
rules of thumb that `apertus-data check` enforces.
[`apertus2_processing.mapping`](src/apertus2_processing/mapping.py) provides the
building blocks.

1. Decide what each source row becomes: system fields, input items, output bursts,
   and where the model waits.
2. Write `to_native(row) -> Conversation` in `mappers/<name>.py`. For OpenAI-style
   message lists call `from_chat(messages, tools=..., system=...)`, a thin wrapper over
   `apertus_common.from_openai`: it maps `system` messages to `behavior`,
   `reasoning_content` to `think`, `tool_calls` to `call`, tool messages to `result`, and
   inserts a wait after every assistant turn. Dataset quirks are normalised: `None`
   fields and tool-message `name`s are dropped, empty `content` on tool-call turns is
   dropped, and tool `parameters` given as JSON strings are parsed. It has no path for
   `document`, `host`, `claim`, or item loss weights (insert those into its result, as
   `mappers/nemotron_math.py` does for the claim), and rejects `developer` or repeated
   system messages unless `merge_system=True`. For sources that are not message lists,
   use `Conversation.builder()` from `apertus_common` and add waits yourself. `thinking`
   defaults to `medium` in both; pass a source effort level and fixed `identity` or
   `environment` values through `system=`, e.g. `system={"thinking": row["effort"]}`.
   Raise `ValueError` for rows to skip; its message, like the library's `AdapterError`
   reason, is counted in the summary. Any other exception is a mapper bug and stops the
   run.
3. Add `main()`: `parser = mapper_parser(description)`, your own flags, then
   `run_mapper(dataset, to_native, args, keep_columns=[...])`. The driver checks with
   `--check`, skips failing rows with `--strict`, writes the [Output](#output) with the
   `datasets` cache disabled (it ignores library upgrades and could return an older
   native format), and exits with code 2 when no conversation was written.
4. Iterate with `--limit 200 --check` until `mapping.json` lists no rule counts and no
   unexpected skips; then map everything.
5. Run `apertus-data` check and tokenize on the output as described below. Inline
   checking is a development aid; the check run remains the gate.

#### Output

`OUTPUT` is a saved HF `Dataset` with the `keep_columns` of the source followed by
`conversation_json`, one compact native JSON string per surviving row in source order,
ready for `apertus-data prepare`. `OUTPUT/mapping.json` records the run; `check` is
`null` unless `--check` or `--strict` was given, and `skipped` keys are the skip reasons,
or `check/<rule>` under `--strict`:

```json
{"rows": 8, "written": 7,
 "skipped": {"roles must alternate user, assistant after an optional system message": 1},
 "convention": "think-first",
 "check": {"rows_with_issues": 1, "failed_rows": 0,
           "issues": [{"rule": "training/final-wait", "severity": "warning", "count": 1}]}}
```

#### Thinking convention

By default (`think-first`) every output burst starts with a `think`; where the source
has no reasoning, the driver inserts an empty one so the model learns the think slot at
every effort level. `source` keeps think items only where the source has reasoning.

This is a **project convention**, not a profile rule: the profile makes `think`
optional and the checker does not flag a missing or empty one. It lives in one place,
`DEFAULT_CONVENTION` in [`mapping.py`](src/apertus2_processing/mapping.py), overridable
per run with `--thinking-convention`. The driver applies it to the conversation
`to_native` returns, so mappers never insert empty thinks themselves; it never adds or
moves waits. The profile warns that the harness may drop think items from earlier
bursts, so every conclusion the model needs later must also appear in a `reply` or
`call`.

#### Waits

Treat the two training warnings as errors for turn-based SFT data; they are warnings
only because mid-burst input is valid in agentic traces.

| Rule of thumb | Caught by |
|---|---|
| Turn-based chat: `user`, `think`, `reply`, `wait`, repeated per turn. | `training/input-mid-burst` (warning) when the wait before the next user message is missing |
| The conversation ends with a `wait`. | `training/final-wait` (warning) |
| Calls that await results end the burst: `call`(s), `wait`, `result`(s), then the next burst. Parallel calls share one wait. | `training/input-mid-burst` (warning) |

#### Checker rules

`apertus-data check` and `--check` run the library's rules below, except
`arguments/tool-name`, which applies to standalone argument checking only. Errors and
unevaluated findings fail a sample; warnings do not.

| Rule | Severity | Checks |
|---|---|---|
| `profile/known-output` | error | output types belong to the Apertus 2 profile |
| `profile/nonnegative-counter` | error | call and result counters are non-negative |
| `profile/document-provenance` | error | documents declare supplying party and citation ID |
| `profile/unique-tool-name` | error | declared tool names are unique |
| `profile/object-schema` | error | tool schemas declare object-root arguments |
| `profile/loss-weight` | error | training loss annotations require opt-in (`--loss-weights`) |
| `conversation/system` | error | complete histories have a system prompt |
| `conversation/declared-tool` | error | calls name a declared tool |
| `conversation/counter` | error | calls use consecutive conversation-wide counters |
| `conversation/result` | error | each result closes exactly one matching pending call |
| `conversation/host-closure` | error | explicit host closures close matching pending calls |
| `conversation/document-id` | error | document IDs are unique across the history |
| `conversation/wait` | error | output or another wait after a wait requires intervening input |
| `conversation/claim-placement` | warning | claims normally precede the replies they support |
| `arguments/json` | error | call arguments are strict JSON objects |
| `arguments/schema` | error | arguments conform to the declared tool schema |
| `arguments/broken-schema` | error | every declared schema is valid and locally resolvable |
| `arguments/tool-name` | error | standalone argument checking uses the named tool |
| `training/input-mid-burst` | warning | an output burst closes with a wait before the next input |
| `training/final-wait` | warning | the final output burst closes with a wait |

## Check and tokenize

**Check first**, fix the mapper until checking passes, then tokenize the same input
with the same options. Tokenization assumes checked data and stops on conversations it
cannot parse or encode.

Submit [slurm/run.sh](slurm/run.sh), adding your site's `--account`, `--partition`,
and `--qos` before the script path. The installation, `INPUT`, and `RUN_DIR` must be
shared across nodes. For a new run, choose a new or empty `RUN_DIR` outside the input:

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

These examples use at most two debug nodes for 20 minutes. For production, pick an
authorized partition, QoS, and time limit for the workload; `highprio` is not for
routine runs. Each job runs one mode. By default, logs go to `apertus-data-<job id>.out`
in the submit directory; encoding log lines carry the Slurm task rank to identify the
node. Exit codes: **0** success, **1** check found failing samples, **2** processing
error.

| Option | Meaning |
|---|---|
| `--column NAME` | Native JSON column (default `conversation_json`). |
| `--split NAME` | Saved `DatasetDict` split (default `train`). |
| `--loss-weights` | Accept annotations in check; export weights in tokenize. See [Loss weights](#loss-weights). |
| `--tokenizer DIR` | Required for tokenize; ignored by check, so both modes can share an option list. |
| `--shard-rows N` | Target rows per shard (default 10000). HF shards use this size; Parquet shards group whole row groups until at least this many rows. A final shard may be smaller. |

### How a job runs

Both modes follow the same sequence:

```text
prepare   first node, once → RUN_DIR/run.json
encode    every node       → RUN_DIR/shards/NNNNNN/
            workers claim a free shard, process it, and repeat
merge     first node       → final outputs in RUN_DIR, in input order
```

- **`prepare`** plans shards from input row-count metadata and records options and
  the input, tokenizer, and library identity in `run.json` (see
  [Resume](#resume-after-a-crash-or-timeout)). Shard IDs refer to the same rows on
  every node and resume. An existing plan is verified and reused; stale claims and
  temporary shards are removed.
- **`encode`** runs `WORKERS` processes per node ([Tuning](#tuning)). Workers
  coordinate through atomic directory creation under `claims/`: only one can claim each
  shard. Each writes to a temporary directory, then renames it into `shards/NNNNNN/`.
  There is no coordinator, and faster nodes take more shards.
- **`merge`** verifies all shards are complete and combines them in input order.
  Each shard contains the same file types as the [final outputs](#outputs), plus
  its summary.

### Without Slurm

Run the same three steps locally:

```sh
uv run --no-sync apertus-data prepare check /data/no_robots-native /tmp/no_robots-check \
  --shard-rows 500
uv run --no-sync apertus-data encode /tmp/no_robots-check
uv run --no-sync apertus-data merge /tmp/no_robots-check
```

For tokenization, use `prepare tokenize` with `--tokenizer "$APERTUS_ENCODING"`.
`encode` takes `--workers` and `--threads` directly. Run `prepare` again before resuming.

### Tuning

| Setting | Meaning |
|---|---|
| `sbatch --nodes=N` | Allocated nodes (default 1), with one task per node. |
| `WORKERS` | Worker processes per node (default: CPUs divided by `THREADS`, rounded down, at least 1). Lower it if memory runs short. |
| `THREADS` | Tokenizer threads per worker (default 1). Only tokenization uses the threads; prefer scaling worker processes first. |
| `--shard-rows` | Aim for at least four shards per worker. Write Parquet with row groups of this size so shards can be distributed effectively. |

For example, prefix either submission above with `WORKERS=4 THREADS=2` for a small
validation run. Allow a few seconds of startup per worker.

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

Checker exceptions become `check/exception` findings with severity `unevaluated`: the
sample fails and processing continues with the next row.
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
{"mode": "tokenize", "documents": 9485, "tokens": 2993761, "loss_weights": false,
 "length": {"min": 32, "mean": 315.6, "p50": 258, "p90": 640, "p99": 1489, "max": 3732,
            "over": {"4096": {"documents": 0, "share": 0.0}, "8192": {"documents": 0, "share": 0.0},
                     "16384": {"documents": 0, "share": 0.0}, "32768": {"documents": 0, "share": 0.0}}}}
```

`length` describes the document lengths in tokens, for choosing a training sequence
length or a length filter: percentiles (linearly interpolated, truncated to whole tokens;
`mean` rounded to one decimal) and, for 4k, 8k, 16k, and 32k tokens, how many documents
exceed the limit and which share. It is `null` for an empty run.

Read documents back with `apertus-data inspect RUN_DIR`. The output starts with a
`=== run` line repeating those totals and statistics, read from the index so it also
covers runs merged before `length` existed; `--sample 0` prints only that line. Each
document then prints as decoded text with its control tokens under a header
`=== document I (of N): T tokens, L with loss (weights ..., trainer defaults)`. Without
`--weights`, the weights are the trainer defaults derived from the control tokens (see
[Loss weights](#loss-weights)); `--weights` reads the written `loss_weights` instead.
Errors, such as a directory that is not a merged tokenize run, an index out of range, or
changed sources, exit with code 2:

| Option | Meaning |
|---|---|
| `--index I` | Zero-based document to show; repeatable. Without it, `--sample N` random documents (default 3), reproducible with `--seed S`. |
| `--tokens` | Text output only: one line per token with position, ID, loss weight, and text (partial for multi-byte characters). |
| `--weights` | Use the written `loss_weights` instead of the trainer defaults everywhere (header, `--tokens`, `report.html`). Requires a `--loss-weights` run whose weight index matches the tokens. |
| `--source` | Append `--- source PATH row R` (`R` as in [Check](#check)) and the row's native JSON. Fails if the input files have moved or changed since tokenization. |
| `--tokenizer DIR` | Override the run's tokenizer. |
| `--output DIR` | Write to `DIR` instead of printing: one `document-NNNNNN.txt` per document (the printed block; earlier ones in `DIR` are removed) and a self-contained `report.html` for any browser, no server needed: each document token by token, coloured by loss weight, control and partial-character tokens marked, hover for position, ID, and weight, toggles for token boundaries and light or dark theme, and the source row in a collapsible panel with `--source`. |

```sh
uv run --no-sync apertus-data inspect /tmp/no_robots-tokens --sample 2 --seed 1 --source
uv run --no-sync apertus-data inspect /tmp/no_robots-tokens --index 0 --tokens --weights | head
uv run --no-sync apertus-data inspect /tmp/no_robots-tokens --sample 5 --weights --source \
  --output /tmp/no_robots-report  # open report.html in a browser
```

## Development

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

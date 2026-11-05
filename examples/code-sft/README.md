# Checked Apertus 2 native coding SFT

This recipe connects the existing downloader to full native chat standardisation
in stage 02 and duplicate/benchmark filtering in stage 04.

| Location | Responsibility |
| --- | --- |
| Existing `01-hf-download/hf-download.py` | Save an HF Dataset or DatasetDict, with metadata |
| [02-standardisation/code-sft](../../02-standardisation/code-sft/) | Column mapping, checked chat conversion, reasoning/tools, native mapping, validation and rendering |
| [04-decontamination/code-sft](../../04-decontamination/code-sft/) | Exact/optional fuzzy dedup, separate evaluation references and retained native shards |

## Setup

Run recipe commands from this directory:

```bash
uv sync --python 3.13 --locked
uv run pytest -q
```

The environment pins `apertus-common` to an immutable Git revision. This reference
library supplies native conversations, checking, formatting and optional tokenization.
The legacy downloader keeps its existing environment and CLI:

```bash
# Run with the existing download environment, from the repository root:
python 01-hf-download/hf-download.py OWNER/DATASET --split train \
  --download-folder /path/to/downloads
```

## Full native standardisation

```bash
uv run python pipeline.py standardise \
  --input /path/to/downloads/DATASET --output /path/to/native
```

Accepts saved HF Dataset/DatasetDict directories, JSONL or Parquet files. Repository
and revision are read from downloader metadata when available. Supply
`--repo owner/name --revision COMMIT` for local files or missing metadata. An absent
revision is recorded as `unrecorded`; no immutable revision is invented. Multiple
DatasetDict splits require an explicit `--split train` (or another selected split).

For custom source columns:

```bash
uv run python pipeline.py standardise --input /path/to/source.jsonl \
  --repo example/source --revision source-version --output /path/to/native \
  --column-map column-map.example.json
```

The column map is source name to canonical name, e.g.
`{"task_text":"prompt","completion_text":"response"}`. Known message containers
and role aliases are normalized too. The standardiser then:

1. Converts source traces to checked structured chats, retaining reasoning,
   tool-schema provenance and original objects in `source_record_json`.
2. Validates message structure, tool definitions/argument schemas, unique call IDs,
   call/result links and reasoning boundaries. Ambiguous mappings and unresolved
   inferred schemas are quarantined.
3. Maps reasoning into native `think` items and answers into `reply` items, with
   calls, results and waits. Leading system/developer instructions can be merged
   with originals retained; mid-dialogue instruction changes are quarantined.
4. Runs full reference-library checks including training rules, and uses
   `render_str` for the Apertus 2 control-token sequence. The default `think-first`
   convention adds an empty think when source reasoning is absent; `--convention
   source` uses only source think items. The system defaults to medium thinking.

Revision-guarded source adapters can recover released reference answers or wrap
source execution protocols. These changes are recorded in provenance alongside
the original record. Declared adapter tools still require the corresponding
inference harness; native format validation does not supply that runtime.

Accepted Parquet rows contain `conversation_json`, `native_text`, source IDs,
original objects, checked-chat provenance and auxiliary matching fields. Shards
are written under `CATEGORY/OWNER/DATASET/file=*/part-*.parquet`. Per-partition
`rejected.jsonl` ledgers retain rejected source objects and failure reasons.
Completion reports count read, written and quarantined rows: completion means a
finished audit, not acceptance of every row.

This is native Apertus 2, separate from the legacy conversation-branches schema.
Use the appropriate downstream native reader. Source/split selection is explicit;
this standardiser does not guess domain or perform license/quality/mixture filtering.

## Trajectories packed in tar shards (Terminus-2 / ATIF)

Some releases keep the agent trajectories in tar shards, one directory per trial, with
only a catalog table next to them (`Recursive-Task-Synthesis-Trajectories`, the GLM-5.3
rollouts). `load_dataset` never reads those tars. Download the whole repository as raw
files (`01-hf-download/download-coding-raw-datasets.sh`), then rehydrate the
trajectories into JSONL rows with a `messages` array and standardise those:

```bash
uv run python pipeline.py atif --input /path/to/download --output /path/to/atif-jsonl
uv run python pipeline.py standardise --input /path/to/atif-jsonl --output /path/to/native \
  --repo owner/name --revision COMMIT --category code_agentic
```

Both ATIF-v1.7 layouts are read: `trajectories/<id>/trajectory.json` (the agent message is
the raw Terminus-2 JSON reply) and `<execution>/agent/trajectory.json` (the commands are
structured `bash_command` / `mark_task_complete` tool calls and the reasoning is a
separate field). The second layout stores what the harness parsed, not the model's reply,
so a step with tool calls is rebuilt as the JSON reply the prompt asks for: `analysis` and
`plan` from the `Analysis: ... / Plan: ...` message, the `bash_command` arguments as
`commands` in call order, and `task_complete: true` for `mark_task_complete`. The ATIF
message and calls stay on the message as `atif_message` / `atif_tool_calls`, and a trial
with a step that does not fit that shape is counted as `not_rebuilt` and skipped. Other
agent messages are kept as they are, and each observation becomes the next user turn.
Trials without an agent trajectory (the gold-solution runs of the rollouts) are
counted and skipped. The verifier reward is kept as `reward` in the row and is not used
to filter. `--limit N` stops after N trajectories, which is what the smoke test uses.

## Two versions of terminal trajectories

`--terminal-harness` picks the version of a Terminus-2 trajectory, and each version needs its
own `--output`. Other rows are the same in both.

| Value | Harness the model sees | Module |
|---|---|---|
| `opencode` (default) | OpenCode: `bash` / `read` / `write` calls with plain output | `opencode_terminal.py` |
| `terminus` | Terminus-2 as recorded: its description, JSON replies and terminal screens | `terminus_terminal.py` |

```bash
uv run python pipeline.py standardise --input DATASET --output native-opencode
uv run python pipeline.py standardise --input DATASET --output native-terminus --terminal-harness terminus
```

The smoke test takes the same option, and its results go to `smoke_all_swe_terminal_terminus`:
`./smoke-all-swe-terminal.sh --terminal-harness terminus`.

### Terminus version

The conversation is the source, in Apertus 2 items: the harness description stays where the
source has it, each JSON reply is a `reply` item, and terminal output is the next `user` item.
There are no tool calls. `<think>` blocks and `reasoning_content` become `think` items. A reply
whose JSON is inside a `<think>` block is the one structural repair; the module docstring has
the details. Trailing user turns move to `trailing_context` in `provenance_json`, since a
training row ends on an assistant turn.

### OpenCode version

A converted row has OpenCode's system prompt and tool schemas
(`opencode_harness.json`), `bash` / `read` / `write` calls with plain output, and ends in a
plain text turn. The module docstring lists every mapping. Conversion counts are in
`provenance_json` under `harness_conversion`, and the source stays in `source_record_json`.

Rows that cannot be mapped honestly are quarantined with
`terminal_not_convertible` and a reason in `rejected.jsonl`: interactive programs, control
keys, keystrokes never submitted, handoff prompts, rows with no task, and output that
belongs to no command.

`opencode_harness.json` is a snapshot from `nvidia/Nemotron-SFT-OpenCode-v1` (January 2026),
so converted rows match that dataset. Current OpenCode releases may name tools and arguments
differently; replace the snapshot to target another version.

## Optional native tokenization

```bash
uv run python pipeline.py standardise \
  --input /path/to/downloads/DATASET --output /path/to/native-tokenized \
  --tokenizer /path/to/approved-apertus2-tokenizer
```

`load_encoding` verifies a local native tokenizer artifact and writes aligned
`tokens` and `loss_weights`. No tokenizer is downloaded or guessed. Without this
option, token columns remain null; checked conversations and native rendering are
still produced. No generic Jinja template substitutes for native formatting.

## Resume and validation

Identical reruns verify completed partition checksums and skip them. Interrupted,
unpublished partitions restart. Changed input snapshots, code/resources, pinned
library, settings or tokenizer require fresh output. A process lock prevents
competing writers. `--rows-per-shard` controls accepted-row shard size. Processing
streams one partition at a time; concurrent source jobs need distinct repository
output paths.

Tests cover downloader output, split selection, native roundtrips/rendering,
reasoning, tool schemas/links, quarantine, provenance and resume integrity.
Token/loss-weight plumbing has fixture coverage; a production tokenizer and full
corpus are not bundled in tests.

Hosted CI runs public chat/schema/reasoning/dedup checks independently. The full
native job also needs the `APERTUS_COMMON_READ_TOKEN` Actions secret with read access
to the private `swiss-ai/apertus-common` dependency. Without that secret, the native
job is explicitly skipped and the workflow summary states the missing access.

## Deduplicate and decontaminate native rows

```bash
uv run python pipeline.py benchmarks --output /path/to/evaluation --workers 8
uv run python pipeline.py dedup --input /path/to/native \
  --benchmarks /path/to/evaluation --output /path/to/dedup \
  --categories code_sft code_agentic --workers 16 --memory-limit 32GB --no-fuzzy
```

Evaluation manifests/locks live in stage 04. References stay separate from training
and incomplete coverage is an error. Exact fingerprints include checked messages,
reasoning, tools and tests; case/code whitespace are preserved with line endings
normalized. Benchmark matching also checks normalized text equality/containment
(minimum 80 characters). `--fuzzy` adds LSH candidates with token-Jaccard
verification; it can miss similar pairs and does not establish semantic equivalence.
`--report-only` calculates decisions first. Optional `--exclusions PATH` reads an
explicit repository-exclusion JSON.

Clean shards under `/path/to/dedup/clean/` retain native conversations, rendering,
optional tokens/weights and provenance. `report.json` and `_work/` retain removal
counts and decision evidence. Inputs/settings are frozen for resume.

## Slurm debug partition

```bash
sbatch debug.sh standardise --input /path/to/downloads/DATASET \
  --output /path/to/native
```

Uses account `infra01`, partition `debug`, 16 CPUs and one hour. Submit from this
directory with an environment available on compute nodes. `CODE_SFT_PYTHON` can
select another interpreter. Resubmit after a time limit to resume completed
partitions.

Source adapters and checked conversion are adapted from
[code-data-processing](https://github.com/User3574/code-data-processing/tree/be3ecc1).
Full-corpus reprocessing remains a separate validation run.

## Code Alchemy

`process_code_alchemy.py` processes a locally rehydrated, validated Code Alchemy
collection at revision `d367da91def5024929d0fa8d46d47d4ef616b467`:

```bash
uv run python process_code_alchemy.py --input /path/to/code_alchemy_rehydrated \
  --output /path/to/prepared-code-alchemy --benchmarks /path/to/benchmarks \
  --reference /path/to/existing-native-decontaminated --workers 96
```

It covers all five training subsets. Explicit User/Assistant conversations and
educational QA headings become checked native SFT. Role boundaries are parsed
before source placeholders are substituted, excluding code fences and reasoning
spans. Missing required source context, ambiguous boundaries and malformed reasoning
are quarantined. Original records, scores, source-status fields and language metadata
are retained; no quality-score filter is applied.

`code-enhance` and MultiPL-E completion-only rows from `code-qa` remain separate
pretraining text records. A chat prompt is never invented for these records.
`dev-eval` and `trace-eval` are evaluation references only; their source blob IDs
are held out across training subsets, including exact duplicate groups.

Native conversations and pretraining text receive separate exact fingerprints.
With `--reference`, existing native rows take precedence over new duplicates.
Benchmark equality/containment checks use the registered revision lock plus Code
Alchemy evaluation prompts. Accepted output is fully materialized and read back,
with per-shard hashes, decisions and source-row accounting. `--limit-files N` is a
smoke-run option and never publishes the collection completion marker.

Completed source shards resume by checksum; interrupted source shards restart.
The final summary exposes separate `training_path` and `pretraining_path` values.

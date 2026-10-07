# Apertus 2 data preparation

Standalone checking, tokenization and indexed export for datasets already converted
to native Apertus conversations. This project does not modify the repository's
numbered stages or convert their legacy chat format automatically.

`apertus-common` owns format validation, conformance checks and tokenization.
This project owns dataset iteration, reports, workers, deterministic shards and
resumption. Packing, training target shifts and training loaders remain downstream.

## Before you start

**The required handoff is native conversation JSON.** If your data already has
that format, no numbered pipeline stage is required by these commands. Otherwise,
finish the source-specific preparation and convert the selected conversations
before running `check`.

For data using this repository's legacy pipeline, use this sequence:

```text
Acquire data -> standardise for legacy scripts -> apply recipe-specific curation
  -> optionally filter/assemble a mixture -> convert selected branches to native JSON
  -> check the complete native corpus -> fix issues and recheck
  -> tokenize the same unchanged native corpus -> optional indexed export
```

| Existing stage | What to do before native checking/tokenization |
| --- | --- |
| [01 Download](../01-hf-download/) | Acquire local data if needed; the native CLI does not download Hub datasets. |
| [02 Standardisation](../02-standardisation/) | Use for legacy curation scripts. Its `conversation_branches`/`parts` output still needs native conversion. |
| [03 License/source filtering](../03-license-based-filtering/) | Complete the filtering required by your dataset recipe. Native format checks do not perform it. |
| [04 Decontamination](../04-decontamination/) | Run when required by the recipe. Its overlap-detection tokenization does not produce final Apertus token IDs. |
| [05 Annotations](../05-annotations/) | Produce annotations needed for selection, scoring or training objectives. |
| [06 Field filtering](../06-field-based-filtering/) | Apply required selection and partitioning while their source fields are still available. |
| [07 Aggregation](../07-dataset-aggregation/) | Optionally assemble/filter the mixture, then convert to native JSON before the legacy `linearise-dataset.py` step. |
| [08 Judge evaluation](../08-judge-evaluation/) | Optional evaluation of legacy data. If scores affect selection, apply that selection before freezing the native corpus. It does not consume native token output automatically. |

Stage 07 contains separate scripts rather than a mandatory sequence. Its legacy
linearizer selects the first branch and writes a different `messages` schema;
it is not needed for native encoding. `create-mixture.py` and
`concatenate-datasets.py` combine input splits, so select the intended training
splits first. Their normalization can remove or rewrite metadata: apply filters
that depend on it first and retain a source copy for conversion/provenance.

There is no generic legacy-to-native converter in this integration. The dataset
owner must define and review branch selection, source-field mapping, tool-result
association and any loss weights. Renaming an existing column does not perform
that conversion. [Input construction below](#native-input-format) shows the exact
target representation and an executable storage example.

## Install

Use Python 3.13 or newer and `uv` on Linux or macOS (POSIX file locks are required).
Use a Linux environment such as WSL on Windows. This project has its own
environment; the legacy root `requirements.txt` is not its installation method.
From the repository root:

```sh
cd apertus2-processing
make install
uv run --no-sync apertus-data --help
```

The lockfile pins `apertus-common` to
`7265179b3a033a607865ec1df3468f6c5a68829d`. That commit must be available in its
Git remote for a fresh locked installation. While working with the local,
unpublished library branch, use the explicit development installation:

```sh
make install-dev LIBRARY_PATH=../../apertus-common
```

The development installation is editable and resolves its dependencies separately;
use `make install` for the committed dependency lock. No tokenizer files are vendored.
Acquire the Apertus 2 instruct artifact separately and pass its local directory.
Only tokenization needs it; use the instruct artifact with Apertus 2 control roles,
not the base tokenizer or stage 04's decontamination tokenizer. Before the
tokenization examples, set:

```sh
export APERTUS_ENCODING=/absolute/path/to/tokenizers/Apertus_2_instruct
```

## Native input format

A complete, unweighted native record looks like this (pretty-printed here only):

```json
{
  "schema_version": 2,
  "system": {"thinking": "medium", "tools": []},
  "items": [
    {"direction": "in", "type": "user", "payload": "Hello"},
    {"direction": "out", "type": "reply", "payload": "Hi"},
    {"type": "wait"}
  ]
}
```

For complete-history checking, supply one system prompt with `thinking` equal
to `low`, `medium` or `high`. Native messages require `direction`, `type` and a
string `payload`; waits have `type: "wait"` and no direction or payload. Keep
messages in their observed order. A wait marks the end of an output burst, not
every individual output message. Another output after a wait requires input.
Pending calls and unfinished conversations may be valid; the checker does not
require an artificial final wait or a completed tool call.

Construct with the library and use `Conversation.to_json()` to get correct
discriminators, aliases and escaping. Contributors must make these mappings
explicit in their source adapter:

| Source information | Native representation |
| --- | --- |
| Standing system instructions | `SystemPrompt.behavior`; choose `thinking` and tool declarations explicitly. |
| User text / reasoning / assistant answer | `User` / `Think` / `Reply`, each with a string payload. |
| Tool declarations | `system.tools` entries with `name`, `description` and object-root JSON `schema`. Legacy `parameters` often needs decoding/mapping. |
| Tool call | `Call(name=..., counter=..., payload=...)`; payload is JSON object text. `Call.build(arguments=...)` accepts an object and serializes it. |
| Tool result | `Result` with the matching name and counter, and a string payload. Preserve or recover the association from reliable source evidence. |
| Retrieved or attached text | `Document` with supplying-party `from` and citation `id` provenance. |
| Structured answer / verifier target | Choose a `Claim` payload under the training recipe; legacy answer lists do not have a universal automatic mapping. |

Complete histories use conversation-wide tool counters starting at zero, with
each new call advancing the counter. Parallel calls remain separate messages.
Tool results can arrive out of order but must identify the correct call. Legacy
parts embedded under an assistant role can still be input tool results in the
native format. Do not infer ambiguous result associations from text.

Select the intended legacy branches and produce one native record per selected
linear conversation, with distinct stable IDs. Preserve preference labels and
source annotations in external metadata under the recipe; do not concatenate
chosen/rejected branches into one transcript. Fragments need explicit
[checking context](#partial-histories-and-checking-context).

Supported containers:

| Container | Required content | CLI setting |
| --- | --- | --- |
| Bare JSONL | One native conversation object per physical UTF-8 line, with escaped payload newlines. No top-level array. | Default for JSONL |
| Wrapped JSONL | A wrapper such as `{"record_id":"example-1","conversation_json":"...native JSON string..."}` | `--conversation-column conversation_json` |
| Saved HF Dataset/Dict | A string column containing each `Conversation.to_json()` result | Default column `conversation_json` |

Save HF inputs with `save_to_disk()` and pass the saved directory. A plain
Dataset is treated as split `train`; a DatasetDict retains its split names.
Uncompressed JSONL directories are read non-recursively in lexical filename
order. Hub identifiers, arbitrary Parquet/Arrow files, diagnostic `<|in|>...`
text and OpenAI `role`/`content` lists are not native CLI inputs. `--format`
selects the output format only.

The [input-writing example](examples/write_native_inputs.py) creates all three
supported layouts from the same small typed conversation. It demonstrates the
storage contract; adapt your dataset separately using the mapping decisions above:

```sh
export APERTUS_DEMO_ROOT="$(mktemp -d)"
uv run --no-sync python examples/write_native_inputs.py "$APERTUS_DEMO_ROOT/input"
# input/native.jsonl, input/wrapped.jsonl, input/hf/
```

## Check, then tokenize

Use the same native input path for both commands, and gate tokenization on a
successful full check with `&&`:

```sh
uv run --no-sync apertus-data check "$APERTUS_DEMO_ROOT/input/native.jsonl" \
  --output "$APERTUS_DEMO_ROOT/check" &&
uv run --no-sync apertus-data tokenize "$APERTUS_DEMO_ROOT/input/native.jsonl" \
  --output "$APERTUS_DEMO_ROOT/tokens" --tokenizer "$APERTUS_ENCODING" \
  --check none --format hf
```

For your own corpus, replace the demo input with your native JSONL path or saved
native HF directory and select fresh report/token output directories. Checking
writes a report, not a repaired dataset. Inspect `summary.json` and the issue
shards; fix conversion/data problems upstream and check again before tokenizing.
Outputs must be outside the input dataset. Reuse an unchanged run with `--resume`;
changed input requires a new output directory.

The other demo containers can be checked with:

```sh
uv run --no-sync apertus-data check "$APERTUS_DEMO_ROOT/input/hf" \
  --output "$APERTUS_DEMO_ROOT/check-hf"
uv run --no-sync apertus-data check "$APERTUS_DEMO_ROOT/input/wrapped.jsonl" \
  --conversation-column conversation_json --output "$APERTUS_DEMO_ROOT/check-wrapped"
```

The bundled [weighted example](examples/weighted.jsonl) masks reasoning, weights
the reply by 0.5 and gives the final wait its own weight of 1.

```sh
uv run --no-sync apertus-data check examples/weighted.jsonl \
  --output "$APERTUS_DEMO_ROOT/check-weighted" --allow-loss-weights &&
uv run --no-sync apertus-data tokenize examples/weighted.jsonl \
  --output "$APERTUS_DEMO_ROOT/tokens-weighted" --tokenizer "$APERTUS_ENCODING" \
  --format hf --emit-loss-weights
```

Checking runs every built-in conformance check and structural validation. It needs
no tokenizer. Tokenization always validates native structure, defaults to
`--check none`, and can repeat full checking with `--check full`. The separate
check report is not consumed by tokenization: callers must keep data, selected
splits and checking context unchanged between the steps. `--check full` is useful
for a single checked tokenization run or when repeating checks is desired.

Default profile checking rejects explicit loss annotations. `check --allow-loss-weights`
opts in; `tokenize --emit-loss-weights` also enables acceptance during full checking.
Without weight emission, tokenization writes token IDs only. With full checking
enabled, annotated data therefore requires `--emit-loss-weights`.

| Region | Default weight | Override |
| --- | --- | --- |
| BOS | 0 | None |
| Input/system frame and header | 0 | None |
| Input/system payload | 0 | Message's `loss_weight` |
| Entire output frame | 1 | Message's `loss_weight` |
| Wait | 1 | Wait's own `loss_weight` |

Weights must be finite, non-negative numbers. Fractional values and values above
one are allowed. They never alter token IDs or enter rendered content. Output
weights are float64 and align with token IDs before training shifts or packing.
Native v2 preserves annotations; unweighted v1 input is upgraded by the library.

## Inputs, outputs and reports

Accepted inputs are a saved HF `Dataset`/`DatasetDict`, a JSONL file, or a directory
of uncompressed `*.jsonl` files. JSONL records are bare native conversations by
default. HF defaults to a `conversation_json` string column. For wrapped JSONL or
another HF column, pass `--conversation-column NAME`; its value must be native
JSON encoded as a string, not a nested object. `--input-format hf|jsonl` overrides
automatic detection. `--split NAME` selects an HF split or labels JSONL data.

Wrapped input may contain `record_id`, `conversation_id` or `id`, in that priority
order. The selected ID is preserved as a string. Output includes `source`, `split`,
zero-based `row`, `record_id`, `token_ids`, `token_count`, and optional `loss_weights`.
Other wrapper columns and source conversation JSON are not copied into token output.

A completed run contains:

- `run.json`: immutable semantic configuration, input/tokenizer/code fingerprints
  and deterministic shard plan.
- `shards/NNNNNN/`: checksummed issue reports, summary and optional token files.
- `summary.json`: aggregate processed/accepted/rejected counts and issue totals.
- `dataset/`: finalized token data, produced only when the run passes its policy.

Detailed `issues.jsonl` files retain each record's identity, rule, severity,
location and message. Aggregates distinguish issue occurrences from affected
records by rule, severity, source and split. Structural failures and unevaluated
checks remain visible. A record passes only when checking is both OK and complete;
warnings do not fail it. Summary counts cover every scanned record, even under
the default tokenization failure policy. Issue files contain findings only;
successful records without findings do not get an issue entry.

HF output is a saved `DatasetDict`, including empty splits. Parquet output has a
`splits.json` mapping logical splits to ordered shard paths; directory names are
opaque identifiers. Both formats include `manifest.json` with checksums.

```python
import os
from pathlib import Path

from datasets import load_from_disk

data = load_from_disk(Path(os.environ["APERTUS_DEMO_ROOT"]) / "tokens-weighted/dataset")
first = data["train"][0]
assert len(first["token_ids"]) == first["token_count"]
assert len(first["loss_weights"]) == first["token_count"]
```

`--on-error fail` is the tokenization default: rejected records prevent final
dataset publication. `--on-error skip` publishes accepted rows and records all
rejections. Unexpected implementation failures and I/O failures abort under
either policy. Exit codes are 0 for completion under the selected policy, 1 for
data failures, and 2 for operational failures. An external shard's exit 0 means
that shard completed; only a successful merge establishes whole-run completion.

## Workers, prepared jobs and resume

Use bounded batches and more shards than workers:

```sh
uv run --no-sync apertus-data tokenize /data/native-dataset \
  --output /data/token-run --tokenizer "$APERTUS_ENCODING" \
  --format parquet --workers 4 --num-shards 32 \
  --batch-rows 64 --batch-bytes 4194304 --emit-loss-weights
```

Workers use spawn and initialize one encoding and reusable checker tuple each.
Default worker and tokenizer thread counts are both 1; `--tokenizer-threads`
controls backend parallelism. Batch limits include conversation and per-record
context bytes. One oversized record is processed alone. Token writers stream
bounded batches, while issue reports are incremental. Memory also depends on
the largest record, tokenizer, schema cache and shard metadata.

For externally scheduled jobs, prepare once to avoid rescanning the full corpus
for every array task:

```sh
uv run --no-sync apertus-data tokenize /data/native-dataset \
  --output /data/token-run --tokenizer "$APERTUS_ENCODING" \
  --format parquet --num-shards 32 --emit-loss-weights --prepare-only

uv run --no-sync apertus-data run-shard /data/token-run --shard-index 0
# Run indices 1 through 31, using an external scheduler or separate processes.
uv run --no-sync apertus-data merge /data/token-run
```

The [SLURM example](examples/slurm_array.sh) runs that prepared plan. This project
does not submit or manage jobs. `--shard-index` also works directly on `check`
and `tokenize`, but those invocations repeat input discovery; prefer a prepared
run for large arrays.

Keep source data and tokenizer artifacts immutable during a run. Prepared workers
check input file guards; merge verifies full source content and exact shard
coverage. Changed semantic options, library/tokenizer revisions or input content
invalidate reuse. Worker counts, batch limits and tokenizer thread counts are
execution settings and may change on resume.

Add `--resume` to reuse completed, verified shards. Interrupted temporary output
is not treated as complete. Merge rejects missing, unexpected, incompatible or
corrupt shards and preserves source/split order. Use a fresh output directory
when input or semantic configuration changes.

## Partial histories and checking context

Use `--context-file context.json` for shared `CheckContext` evidence, or
`--context-column check_context` for per-record evidence in wrapped JSONL/HF.
An explicitly named column must exist. A non-null per-record value overrides the
shared context; null falls back to the shared context. Values may be objects or
JSON strings. Bare JSONL cannot carry a separate context column.

For example, a retained fragment whose prior state is known can use:

```json
{
  "history": "partial",
  "checkpoint": {
    "system": {"thinking": "medium", "tools": []},
    "next_call_counter": 0,
    "calls": [],
    "document_ids": [],
    "waiting": false
  },
  "host_closures": []
}
```

The optional `prefix` is a native conversation object. Host closures identify a
retained host by `source` (`value` or `prefix`), `item`, `counter` and `name`.
Unknown checkpoint fields remain unknown; empty collections mean known empty.
Missing evidence produces unevaluated findings, never an assumed pass.

## Optional Megatron indexed export

Use an environment with the consuming Megatron checkout's dependencies. Ordinary
checking/tokenization does not import Megatron or require Torch.

```sh
make install-megatron
uv run --no-sync apertus-data export-megatron /data/token-run \
  --output /data/indexed-run --megatron-path /path/to/SwissAI-Megatron-LM \
  --documents-per-shard 10000 --tokens-per-shard 10000000
```

The checked checkout revision is
`04edde000b189470836d6fe85cda9a9f3c0b3161`. The exporter calls its genuine
`IndexedDatasetBuilder` without retokenizing. One conversation becomes one
sequence/document. Token data uses int32; optional weights use a separate
float64 `.bin`/`.idx` pair with identical document boundaries. No extra BOS, EOS,
padding or shift is added. Manifests record writer identity, provenance and
checksums. Export shards are bounded by documents and tokens, except a single
oversized conversation; `--resume` verifies and reuses complete shards.

The paired files provide indexed storage compatibility. The existing Swiss AI
SFT loader does not automatically consume this new pair; training-loader
integration is separate work. Do not treat this as its legacy concatenated
token/mask format.

## Verification

```sh
make all
make check-artifact APERTUS_ENCODING=/path/to/Apertus_2_instruct
make check-megatron MEGATRON_PATH=/path/to/SwissAI-Megatron-LM
make bench-memory
uv run --no-sync python examples/bench_memory.py --records 40000 \
  --tokenizer "$APERTUS_ENCODING"
```

Regular tests use a synthetic tokenizer and exercise both storage formats,
weighted/unweighted rows, checking policies, spawned workers, contexts, shard
coverage, corruption and resume. The artifact check requires the real tokenizer.
Megatron tests require the real consumer implementation and its dependencies;
they verify IDs, weights, dtypes and document boundaries through its reader.

The memory example generates a corpus on disk and reports incremental Python
allocations during processing, including lazy imports and run initialization.
It excludes native allocations and is not a total-RAM measurement. One local
run checked 5,000 records (41.81 MB input) with a 2.98 MB Python peak; checking
and weighted tokenization of 40,000 records (334.48 MB) peaked at 266.43 MB,
including tokenizer/runtime initialization. A 1,000-record tokenization run
had the same approximately 266 MB Python peak, indicating startup dominated
this measurement. These are observations for the provided repetitive fixture,
not throughput or memory guarantees.

# Apertus 2 data preparation

Standalone checking, tokenization and indexed export for datasets already converted
to native Apertus conversations. This project does not modify the repository's
numbered stages or convert their legacy chat format automatically.

`apertus-common` owns format validation, conformance checks and tokenization.
This project owns dataset iteration, reports, workers, deterministic shards and
resumption. Packing, training target shifts and training loaders remain downstream.

## Install

Use Python 3.13 or newer and `uv` from this directory:

```sh
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

## Check, then tokenize

The bundled [weighted example](examples/weighted.jsonl) masks reasoning, weights
the reply by 0.5 and gives the final wait its own weight of 1.

```sh
uv run --no-sync apertus-data check examples/weighted.jsonl \
  --output /tmp/apertus-check --allow-loss-weights

uv run --no-sync apertus-data tokenize examples/weighted.jsonl \
  --output /tmp/apertus-tokens --tokenizer "$APERTUS_ENCODING" \
  --format hf --emit-loss-weights
```

Checking runs every built-in conformance check and structural validation. It needs
no tokenizer. Tokenization always validates native structure, defaults to
`--check none`, and can repeat full checking with `--check full`. The separate
check report is not an authorization token: callers must keep inputs unchanged
between checking and tokenization. Tokenization does not search for an earlier report.

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
warnings do not fail it. Reports include all records, even under the default
tokenization failure policy.

HF output is a saved `DatasetDict`, including empty splits. Parquet output has a
`splits.json` mapping logical splits to ordered shard paths; directory names are
opaque identifiers. Both formats include `manifest.json` with checksums.

```python
from datasets import load_from_disk

data = load_from_disk("/tmp/apertus-tokens/dataset")
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

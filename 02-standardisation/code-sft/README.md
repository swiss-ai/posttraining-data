# Checked Apertus 2 native standardisation

`standardise_native.py` consumes the existing downloader's saved Dataset/DatasetDict,
JSONL or Parquet. Column mapping and source-specific chat conversion feed independent
chat/schema checks, then pinned `apertus-common` maps, validates and renders native
conversations. Reasoning becomes think items; calls, results and waits are preserved.
Rejected rows retain the original source and explicit reasons.

Output includes native `conversation_json`, rendered `native_text`, original records,
provenance, matching fields and optional token IDs/loss weights. `native_format.py`
contains the native adapter; `openai_*` modules validate source trace semantics.
`unified.py` and `standardize.py` provide an intermediate source view, not the final
training format.

See the [recipe](../../examples/code-sft/README.md) for the locked environment,
inputs, validation scope, native formatting/tokenization, quarantine and resume.

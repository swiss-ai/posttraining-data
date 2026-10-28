# Coding SFT standardisation

`standardize.py` defines the common Parquet columns and alias mappings. `unified.py`
adds original-source records, per-file provenance, selection accounting and resume
checkpoints. `preprocessors/` contains source adapters and `adapter-policies.json`
declares known source selection policies. `unify_local.py` handles existing local
JSONL/Parquet files with an optional source-to-canonical column-name map.

This schema preserves messages, prompt, response, reasoning, tools, tests and the
original source object. It is separate from the existing stage-02 HF DatasetDict
conversation-branches schema. See the [recipe](../../examples/code-sft/README.md)
for mappings, limitations and commands.

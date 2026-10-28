# Native coding SFT deduplication and decontamination

`dedup.py` fingerprints checked stage-02 rows, chooses deterministic global duplicate
representatives, checks separate evaluation references and materializes retained rows
with reports. `fuzzy.py` optionally generates/verifies near-duplicate candidates on
disk. Native conversations, rendering, optional tokens/weights, original records and
provenance are preserved.

`benchmarks.py` prepares references using this directory's manifest/revision lock.
Evaluation data stays outside training. Parent-directory decontamination scripts
continue to consume HF DatasetDict; this path consumes checked native Parquet with
auxiliary matching fields.

See the [recipe](../../examples/code-sft/README.md) for policies, reference coverage,
explicit exclusions, commands and limitations.

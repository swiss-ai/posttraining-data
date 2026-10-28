# Coding SFT deduplication and decontamination

`dedup.py` fingerprints complete stage-02 Parquet sources, chooses deterministic
global duplicate representatives, checks separate evaluation references, and
materializes retained rows with removal decisions and per-source reports.
`fuzzy.py` optionally generates and verifies near-duplicate candidates on disk.

This path consumes the code-SFT Parquet schema. The existing decontamination
scripts in the parent directory continue to consume their HF DatasetDict inputs.
See the [recipe](../../examples/code-sft/README.md) for exact/fuzzy policies,
benchmark coverage, explicit exclusions and commands.

# Verbatim + template dedup pre-step

Goal: collapse the heavily-templated vision-SFT prompts before decontamination
so each unique template is scored once, the matcher's denominator effect goes
away for the worst FPs, and the contamination report becomes
"this template hit N benchmarks, M training rows are affected" instead of
N near-identical lines.

## Three hash views per row

- `prompt_hash` — lowercase + whitespace-collapse + `<image>` strip → verbatim
  duplicates (e.g. the OCR-this-image-section cluster).
- `prompt_skeleton_hash` — also replaces any numeric run with `<N>` and any
  bbox-like `[<N>,<N>,<N>,<N>]` with `<BBOX>`; collapses the 40-turn region
  description rows into one cluster regardless of coordinates.
- `prompt_prefix8_hash` — hash of the first 8 normalized tokens only; collapses
  slot-filled stems like `"What is the main focus of the …"` regardless of slot.

Plus per-row `response_hash` and token counts, kept for future Q+A matching and
length-floor decisions downstream.

## Pipeline

```
compute-prompt-hashes.py    one source shard  -> one tiny hash parquet
submit-dedup-array.sbatch   SLURM array fan-out across all shards of a dataset
aggregate-clusters.py       polars merge -> clusters_*.parquet + summary.json
```

## Run on Innovator-VL

```bash
export SRC_GLOB="/capstor/store/cscs/swissai/infra01/vision-datasets/raw/sft/hf___InnovatorLab___Innovator-VL-Instruct-46M/data*/SFT_*.parquet"
export DST_DIR="/iopsstor/scratch/cscs/schlag/apertus1p5-decontam/dedup/hf___InnovatorLab___Innovator-VL-Instruct-46M"

sbatch --array=0-99 --export=ALL,ARRAY_SIZE=100 submit-dedup-array.sbatch
# then once all tasks finish:
python aggregate-clusters.py "$DST_DIR/shards" "$DST_DIR"
```

Outputs `clusters_prompt.parquet`, `clusters_prompt_skeleton.parquet`,
`clusters_prompt_prefix8.parquet`, and `summary.json` next to the shards.

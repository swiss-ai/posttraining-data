#!/bin/bash
#SBATCH --job-name=smoke-decon
#SBATCH --partition=normal
#SBATCH --account=infra01
#SBATCH --nodes=1
#SBATCH --cpus-per-task=288
#SBATCH --time=2:00:00
#SBATCH --output=/iopsstor/scratch/cscs/schlag/apertus1p5-decontam/smoke/slurm-decon-%j.out

set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"
export TOKENIZERS_PARALLELISM=true
export OMP_NUM_THREADS=1
export HF_HOME=/iopsstor/scratch/cscs/schlag/.cache/huggingface

SMOKE_DIR=/iopsstor/scratch/cscs/schlag/apertus1p5-decontam/smoke
REPO=$HOME/apertus1p5-decontam/posttraining-data
WRAP=$HOME/apertus1p5-decontam/vision-sft-decontam
SHARD=/capstor/store/cscs/swissai/infra01/vision-datasets/raw/sft/hf___InnovatorLab___Innovator-VL-Instruct-46M/data1/SFT_000001.parquet
PROMPTS=/capstor/store/cscs/swissai/infra01/posttrain_data/04_decontaminated/decontamination_prompts
CACHE=/capstor/store/cscs/swissai/infra01/posttrain_data/decontamination_cache

cd "$WRAP"
mkdir -p "$SMOKE_DIR/decontam-output/reports"

echo "=== run patched decontaminate.py: parquet-direct, in-memory, no filtered copy ==="
echo "input shard : $SHARD"
echo "benchmarks  : $(ls $PROMPTS | wc -l) (minus excluded)"

uv run --quiet --python 3.11 --with-requirements "$REPO/requirements.txt" \
    python "$WRAP/decontaminate.py" \
    "$SHARD" \
    --input-format sharegpt-vision \
    --decontamination_prompts "$PROMPTS" \
    --tokenizer_name "alehc/swissai-tokenizer" \
    --report_path "$SMOKE_DIR/decontam-output/reports" \
    --cache_dir "$CACHE" \
    --ngram_length 8 \
    --diff_threshold 0.5 \
    --num_proc 16 \
    --exclude-benchmarks polyglotoxicity \
    --show_contaminated

echo
echo "=== done ==="

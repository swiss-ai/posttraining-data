#!/bin/bash
#SBATCH --job-name=innov-decon
#SBATCH --partition=normal
#SBATCH --account=infra01
#SBATCH --nodes=1
#SBATCH --cpus-per-task=288
#SBATCH --time=10:00:00
#SBATCH --output=/iopsstor/scratch/cscs/schlag/apertus1p5-decontam/dedup/logs/decontam-innovator-%j.out

set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"
export TOKENIZERS_PARALLELISM=true
export OMP_NUM_THREADS=1
export HF_HOME=/iopsstor/scratch/cscs/schlag/.cache/huggingface

REPO=$HOME/apertus1p5-decontam/posttraining-data
WRAP=$HOME/apertus1p5-decontam/vision-sft-decontam
DEDUP=/iopsstor/scratch/cscs/schlag/apertus1p5-decontam/dedup/hf___InnovatorLab___Innovator-VL-Instruct-46M
INPUT=$DEDUP/decontam-input.parquet
REPORTS=$DEDUP/decontam-reports-13gram07
PROMPTS=/capstor/store/cscs/swissai/infra01/posttrain_data/04_decontaminated/decontamination_prompts
CACHE=/capstor/store/cscs/swissai/infra01/posttrain_data/decontamination_cache

mkdir -p "$REPORTS"

echo "input        : $INPUT"
echo "reports      : $REPORTS"

uv run --quiet --python 3.11 --with-requirements "$REPO/requirements.txt" \
    python "$WRAP/decontaminate.py" \
    "$INPUT" \
    --input-format sharegpt-vision \
    --decontamination_prompts "$PROMPTS" \
    --tokenizer_name "alehc/swissai-tokenizer" \
    --report_path "$REPORTS" \
    --cache_dir "$CACHE" \
    --ngram_length 13 \
    --diff_threshold 0.7 \
    --num_proc 8 \
    --exclude-benchmarks polyglotoxicity,toxicity,polygloto

echo "done"

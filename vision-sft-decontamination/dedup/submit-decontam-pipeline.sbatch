#!/bin/bash
#SBATCH --job-name=decon-pipe
#SBATCH --partition=normal
#SBATCH --account=infra01
#SBATCH --nodes=1
#SBATCH --cpus-per-task=288
#SBATCH --time=12:00:00
#SBATCH --output=/iopsstor/scratch/cscs/schlag/apertus1p5-decontam/dedup/logs/decon-pipe-%j.out

set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"
export TOKENIZERS_PARALLELISM=true
export OMP_NUM_THREADS=1
export HF_HOME=/iopsstor/scratch/cscs/schlag/.cache/huggingface

: "${DS_FOLDER:?DS_FOLDER required (dataset output dir name)}"
: "${SOURCE_GLOB:?SOURCE_GLOB required}"
: "${COMPUTE_SCRIPT:?COMPUTE_SCRIPT required (file under dedup/)}"

DEDUP=/iopsstor/scratch/cscs/schlag/apertus1p5-decontam/dedup/$DS_FOLDER
REPO=$HOME/apertus1p5-decontam/posttraining-data
WRAP=$HOME/apertus1p5-decontam/vision-sft-decontam
DEDUP_DIR=$HOME/apertus1p5-decontam/vision-sft-decontam/dedup
INPUT=$DEDUP/decontam-input.parquet
CLUSTER=$DEDUP/clusters_prompt_prefix8.parquet
REPORTS=$DEDUP/decontam-reports-13gram07
PROMPTS=/capstor/store/cscs/swissai/infra01/posttrain_data/04_decontaminated/decontamination_prompts
CACHE=/capstor/store/cscs/swissai/infra01/posttrain_data/decontamination_cache

mkdir -p "$REPORTS"

echo "dataset    : $DS_FOLDER"
echo "cluster    : $CLUSTER"
echo "source     : $SOURCE_GLOB"
echo "compute    : $COMPUTE_SCRIPT"
echo "input out  : $INPUT"
echo "reports    : $REPORTS"

if [ ! -f "$INPUT" ]; then
    echo "=== step 1: prepare-decontam-input ==="
    uv run --quiet --python 3.11 --with 'pyarrow==21.0.0' --with 'polars>=1.0' --with 'ijson' --with 'datasets==3.3.2' \
        python "$DEDUP_DIR/prepare-decontam-input.py" \
        --cluster-table "$CLUSTER" \
        --source-glob "$SOURCE_GLOB" \
        --compute-script "$DEDUP_DIR/$COMPUTE_SCRIPT" \
        --output "$INPUT" \
        --num-proc 32
else
    echo "(skipping prepare — $INPUT already exists)"
fi

echo
echo "=== step 2: decontamination (13-gram / 0.7) ==="
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
    --num_proc "${NUM_PROC:-8}" \
    --exclude-benchmarks polyglotoxicity,toxicity,polygloto

echo "done"

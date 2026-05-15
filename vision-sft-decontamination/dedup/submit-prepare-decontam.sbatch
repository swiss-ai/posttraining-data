#!/bin/bash
#SBATCH --job-name=prep-decon
#SBATCH --partition=normal
#SBATCH --account=infra01
#SBATCH --nodes=1
#SBATCH --time=1:00:00
#SBATCH --output=/iopsstor/scratch/cscs/schlag/apertus1p5-decontam/dedup/logs/prep-decon-%j.out

set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"

: "${CLUSTER_TABLE:?CLUSTER_TABLE required}"
: "${SOURCE_GLOB:?SOURCE_GLOB required}"
: "${OUTPUT:?OUTPUT required}"

WRAP="$HOME/apertus1p5-decontam/vision-sft-decontam/dedup"

echo "cluster table : $CLUSTER_TABLE"
echo "source glob   : $SOURCE_GLOB"
echo "output        : $OUTPUT"

uv run --quiet --python 3.11 --with 'pyarrow==21.0.0' --with 'polars>=1.0' \
    python "$WRAP/prepare-decontam-input.py" \
    --cluster-table "$CLUSTER_TABLE" \
    --source-glob "$SOURCE_GLOB" \
    --output "$OUTPUT" \
    --num-proc 32

echo "done"

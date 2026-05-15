#!/bin/bash
#SBATCH --job-name=dedup-agg
#SBATCH --partition=normal
#SBATCH --account=infra01
#SBATCH --nodes=1
#SBATCH --time=0:45:00
#SBATCH --output=/iopsstor/scratch/cscs/schlag/apertus1p5-decontam/dedup/logs/aggregate-%j.out

set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"

: "${DST_DIR:?DST_DIR required, e.g. /iopsstor/scratch/.../dedup/<dataset>}"

WRAP="$HOME/apertus1p5-decontam/vision-sft-decontam/dedup"
mkdir -p /iopsstor/scratch/cscs/schlag/apertus1p5-decontam/dedup/logs

echo "Aggregating clusters for $DST_DIR"
uv run --quiet --python 3.11 --with 'polars>=1.0' \
    python "$WRAP/aggregate-clusters.py" "$DST_DIR/shards" "$DST_DIR"
echo "Done"

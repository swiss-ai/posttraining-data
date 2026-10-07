#!/usr/bin/env bash
#SBATCH --job-name=apertus-tokenize
#SBATCH --array=0-31
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=02:00:00

# Run from apertus2-processing after --prepare-only created a 32-shard plan:
# sbatch examples/slurm_array.sh /data/token-run
# After every task succeeds: uv run --no-sync apertus-data merge /data/token-run
# Select partition/account and resources for your site and largest records.
set -euo pipefail

run_dir=${1:?Supply the prepared run directory}
task_index=${SLURM_ARRAY_TASK_ID:?Run this script as a SLURM array task}
uv run --no-sync apertus-data run-shard "$run_dir" \
  --shard-index "$task_index" --workers 1 --tokenizer-threads 1 --resume

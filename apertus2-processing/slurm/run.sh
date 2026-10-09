#!/bin/bash
#SBATCH --job-name=apertus-data
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --exclusive
#SBATCH --time=04:00:00
#SBATCH --output=%x-%j.out
#
# Check or tokenize native data on all allocated CPUs. Submit from apertus2-processing
# with your site's --account and --partition (on CSCS Alps, also --environment):
#
#   sbatch slurm/run.sh check    INPUT RUN_DIR [options]
#   sbatch slurm/run.sh tokenize INPUT RUN_DIR --tokenizer DIR [options]
#
# INPUT: Parquet file/directory or saved HF dataset. Options: `apertus-data prepare`
# (--column, --split, --loss-weights, --shard-rows). INPUT and RUN_DIR must be shared
# across nodes. Default log: apertus-data-<job id>.out in the submit directory.
# Resume by resubmitting the same command after the previous job ends.
#
# Parallelism (may change on resume): THREADS = tokenizer threads per worker
# (default 1; tokenize only); WORKERS = processes per node (default CPUs / THREADS).
# See README.md for input format, outputs, tuning, and resume constraints.
set -euo pipefail

if [[ $# -lt 3 || -z ${SLURM_JOB_ID:-} ]]; then
  echo "usage: sbatch [--nodes=N] slurm/run.sh check|tokenize INPUT RUN_DIR [options]" >&2
  exit 2
fi
mode=$1 input=$2 run=$3
shift 3
apertus_data=.venv/bin/apertus-data  # created by `uv sync`; compute nodes need no uv

"$apertus_data" prepare "$mode" "$input" "$run" "$@"
srun --ntasks-per-node=1 --cpus-per-task="$SLURM_CPUS_ON_NODE" \
  "$apertus_data" encode "$run" --threads "${THREADS:-1}" ${WORKERS:+--workers "$WORKERS"}
"$apertus_data" merge "$run"

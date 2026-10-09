#!/bin/bash
#SBATCH --job-name=apertus-data
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --exclusive
#SBATCH --time=04:00:00
#SBATCH --output=%x-%j.out
#
# Check or tokenize native data on all allocated CPUs. Submit from apertus2-processing
# with your site's --account, --partition and --qos:
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
# Uses slurm/alps.toml and the scratch environment installed by slurm/setup.sh.
# Override APERTUS_ENVIRONMENT (EDF path; "none" outside Alps) and
# UV_PROJECT_ENVIRONMENT as needed. See README.md for the complete setup.
set -euo pipefail

if [[ $# -lt 3 || -z ${SLURM_JOB_ID:-} ]]; then
  echo "usage: sbatch [--nodes=N] slurm/run.sh check|tokenize INPUT RUN_DIR [options]" >&2
  exit 2
fi
mode=$1 input=$2 run=$3
shift 3
apertus_data=${UV_PROJECT_ENVIRONMENT:-${SCRATCH:?}/apertus2-processing/venv}/bin/apertus-data
container=()
if [[ ${APERTUS_ENVIRONMENT:-} != none ]]; then
  container=(--environment="${APERTUS_ENVIRONMENT:-$PWD/slurm/alps.toml}"
    --container-workdir="$PWD" --container-name="apertus-data-$SLURM_JOB_ID")
fi
workers=()
if [[ -n ${WORKERS:-} ]]; then workers=(--workers "$WORKERS"); fi

srun --nodes=1 --ntasks=1 --kill-on-bad-exit=1 "${container[@]}" \
  "$apertus_data" prepare "$mode" "$input" "$run" "$@"
srun --nodes="$SLURM_JOB_NUM_NODES" --ntasks="$SLURM_JOB_NUM_NODES" \
  --ntasks-per-node=1 --cpus-per-task="$SLURM_CPUS_ON_NODE" --kill-on-bad-exit=1 --label \
  "${container[@]}" "$apertus_data" encode "$run" --threads "${THREADS:-1}" "${workers[@]}"
srun --nodes=1 --ntasks=1 --kill-on-bad-exit=1 "${container[@]}" \
  "$apertus_data" merge "$run"

#!/bin/bash
#SBATCH --job-name=code-sft
#SBATCH --account=infra01
#SBATCH --partition=debug
#SBATCH --qos=normal
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --output=code-sft-%j.out
set -euo pipefail
umask 0027
cd "$SLURM_SUBMIT_DIR"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONUNBUFFERED=1
exec "${CODE_SFT_PYTHON:-.venv/bin/python}" pipeline.py "$@"

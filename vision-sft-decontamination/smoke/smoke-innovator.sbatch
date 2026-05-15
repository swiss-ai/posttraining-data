#!/bin/bash
#SBATCH --job-name=smoke-innov
#SBATCH --partition=normal
#SBATCH --account=infra01
#SBATCH --nodes=1
#SBATCH --time=0:10:00
#SBATCH --output=/iopsstor/scratch/cscs/schlag/apertus1p5-decontam/smoke/slurm-%j.out

set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"
mkdir -p /iopsstor/scratch/cscs/schlag/apertus1p5-decontam/smoke
cd "$SLURM_SUBMIT_DIR"
exec uv run --quiet --with pyarrow python smoke-innovator.py

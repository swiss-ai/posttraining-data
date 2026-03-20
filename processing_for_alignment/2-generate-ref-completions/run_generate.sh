#!/bin/bash
#SBATCH --job-name=gen-ref-completions
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=normal
#SBATCH --account=infra01
#SBATCH --reservation=PA-2338-RL
#SBATCH --time=12:00:00
#SBATCH --output=logs/%j/gen-ref-completions.out
#SBATCH --error=logs/%j/gen-ref-completions.err
set -euo pipefail

# ── Configuration ──────────────────────────────────────────────────
DATASET_PATH="/iopsstor/scratch/cscs/dmelikidze/posttraining-data/processing_for_alignment/datasets/MaxMin-Filtered"
OUTPUT_DIR="/iopsstor/scratch/cscs/dmelikidze/posttraining-data/processing_for_alignment/datasets/MaxMin-Filtered-Ref-Completions-30"
MODEL_PATH="/iopsstor/scratch/cscs/dmelikidze/huggingface/hub/models--swiss-ai--Apertus-8B-Instruct-2509-SFT/snapshots/d57e4f1a3baa6315c60707346b5498b48b40a364"
SERVED_MODEL_NAME="swissai-apertus8b-sft-$(whoami)"

SLURM_SERVER_NODES=32
WORKERS=32
NODES_PER_WORKER=1
TP_SIZE=1
DP_SIZE=4
ROUTER_ENV="$SCRATCH/model-launch/legacy/serving/envs/sglang.toml"
# ───────────────────────────────────────────────────────────────────

SCRIPT_DIR="$SCRATCH/posttraining-data/processing_for_alignment/2-generate-ref-completions"

echo "Job ID: ${SLURM_JOB_ID:-local}"
echo "Node: $(hostname)"
echo "Start: $(date)"
echo "Dataset: ${DATASET_PATH}"
echo "Output: ${OUTPUT_DIR}"
echo "Model: ${SERVED_MODEL_NAME}"

srun --environment=./response_generation/env/alignment.toml --container-writable --container-workdir="${SCRIPT_DIR}" \
    bash -c "python -u ${SCRIPT_DIR}/generate_completions.py \
    --dataset-path ${DATASET_PATH} \
    --output-dir ${OUTPUT_DIR} \
    --model-name-or-path ${MODEL_PATH} \
    --served-model-name ${SERVED_MODEL_NAME} \
    --slurm-nodes ${SLURM_SERVER_NODES} \
    --workers ${WORKERS} \
    --nodes-per-worker ${NODES_PER_WORKER} \
    --tensor-parallel-size ${TP_SIZE} \
    --data-parallel-size ${DP_SIZE} \
    --router-environment ${ROUTER_ENV}"

echo "Done: $(date)"

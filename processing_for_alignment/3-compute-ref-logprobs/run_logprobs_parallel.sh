#!/bin/bash
set -euo pipefail

# ── Environment Setup ──────────────────────────────────────────────
# Source the exact same setup script used by the trainer to guarantee 
# container image, environment, and mount parity.
export PROJECT_ROOT_AT=$HOME/projects/posttraining/run
export ENABLE_RETRY=0
source $PROJECT_ROOT_AT/installation/docker-arm64-cuda/CSCS-Clariden-setup/shared-submit-scripts/setup.sh
# ───────────────────────────────────────────────────────────────────

# ── Configuration ──────────────────────────────────────────────────
DATASET_PATH="/iopsstor/scratch/cscs/dmelikidze/posttraining-data/processing_for_alignment/datasets/MaxMin_3600-Filtered"
# DATASET_PATH="/iopsstor/scratch/cscs/dmelikidze/ActiveUltraFeedback/datasets/uf_qwen_235b_w_features_preference/drts_filtered"
# OUTPUT_BASE="/iopsstor/scratch/cscs/dmelikidze/ActiveUltraFeedback/datasets/uf_qwen_235b_w_features_preference/drts_logprobs_simpo"
OUTPUT_BASE="/iopsstor/scratch/cscs/dmelikidze/posttraining-data/processing_for_alignment/datasets/MaxMin_3600-Filtered-reflogprobs"
MODEL_PATH="/iopsstor/scratch/cscs/dmelikidze/huggingface/hub/models--swiss-ai--Apertus-8B-Instruct-2509-SFT/snapshots/d57e4f1a3baa6315c60707346b5498b48b40a364"
# MODEL_PATH="/iopsstor/scratch/cscs/dmelikidze/huggingface/hub/models--allenai--Llama-3.1-Tulu-3-8B-SFT/snapshots/f2a0b46b0cfda21003c6141b1ff837b7e165524d"
# MODEL_PATH="/iopsstor/scratch/cscs/dmelikidze/ActiveUltraFeedback/models/cpo2/1748535-drts-simpo-lr5e-06-sg1.2-b2.0-seed42-loraR64-loraA16"
MAX_SEQ_LEN=3600
SPLIT="train_split"
DATASET_SIZE=259230

# Parallelism
PARTITION_SIZE=16384           # rows per node-job (8192 for 8B, 1024 for 70B)
NUM_GPUS_PER_NODE=4
TENSOR_PARALLEL_SIZE=1        # GPUs per model instance
SAVE_INTERVAL=4096            # checkpoint every N rows per subpartition

# SLURM
SLURM_TIME="03:00:00"
SLURM_PARTITION="normal"
SLURM_ACCOUNT="infra01"
SLURM_RESERVATION="SD-69241-apertus-1-5-3"
# ───────────────────────────────────────────────────────────────────

SCRIPT_DIR="$SCRATCH/posttraining-data/processing_for_alignment/3-compute-ref-logprobs"
LOGS_DIR="$SCRATCH/posttraining-data/processing_for_alignment/logs/logprobs-$(date +%Y-%m-%d-%H-%M)"
NUM_SUBPARTITIONS=$((NUM_GPUS_PER_NODE / TENSOR_PARALLEL_SIZE))

if [ "${DATASET_SIZE}" -eq 0 ]; then
    echo "ERROR: Set DATASET_SIZE in the script before running."
    exit 1
fi

echo "Dataset size: ${DATASET_SIZE}"
echo "Partition size: ${PARTITION_SIZE}"
echo "Subpartitions per node (GPUs used in parallel): ${NUM_SUBPARTITIONS}"

mkdir -p "${LOGS_DIR}"

JOB_IDS=()

for START in $(seq 0 ${PARTITION_SIZE} $((DATASET_SIZE - 1))); do
    END=$((START + PARTITION_SIZE))
    if [ $END -gt $DATASET_SIZE ]; then
        END=$DATASET_SIZE
    fi

    PART_OUTPUT="${OUTPUT_BASE}/partitions/${START}-${END}"

    # Updated sbatch wrap to mirror recursive-unattended-accelerate.sh
    JOB_ID=$(sbatch \
        --job-name="logprobs-${START}-${END}" \
        --nodes=1 \
        --ntasks-per-node=${NUM_SUBPARTITIONS} \
        --gpus-per-node=${NUM_GPUS_PER_NODE} \
        --partition="${SLURM_PARTITION}" \
        --account="${SLURM_ACCOUNT}" \
        --reservation="${SLURM_RESERVATION}" \
        --time="${SLURM_TIME}" \
        --output="${LOGS_DIR}/logprobs-${START}-${END}-%t.out" \
        --error="${LOGS_DIR}/logprobs-${START}-${END}-%t.err" \
        --parsable \
        --wrap="srun \
            --container-image=${CONTAINER_IMAGE} \
            --environment=${CONTAINER_ENV_FILE} \
            --container-mounts=${PROJECT_ROOT_AT},${HOME}/projects/posttraining/dev,${SCRATCH},${SHARED_SCRATCH},${STORE},/iopsstor,${WANDB_API_KEY_FILE_AT} \
            --container-workdir=${PROJECT_ROOT_AT} \
            --no-container-mount-home \
            --no-container-remap-root \
            --no-container-entrypoint \
            --container-writable \
            /opt/template-entrypoints/pre-entrypoint.sh \
            bash -c 'python -u ${SCRIPT_DIR}/compute_logprobs.py \
            --dataset-path ${DATASET_PATH} \
            --output-dir ${PART_OUTPUT}/subpart_\${SLURM_PROCID} \
            --model-name-or-path ${MODEL_PATH} \
            --max-seq-len ${MAX_SEQ_LEN} \
            --partition-start ${START} \
            --partition-end ${END} \
            --num-gpus-per-node ${NUM_GPUS_PER_NODE} \
            --tensor-parallel-size ${TENSOR_PARALLEL_SIZE} \
            --save-interval ${SAVE_INTERVAL} \
            --split ${SPLIT} \
            --debug'")

    JOB_IDS+=("$JOB_ID")
    echo "Submitted partition [${START}, ${END}) -> job ${JOB_ID} (${NUM_SUBPARTITIONS} GPUs)"
done

echo ""
echo "All jobs submitted. Job IDs: ${JOB_IDS[*]}"
echo ""
echo "Monitor with:"
echo "  squeue -u $(whoami) | grep logprobs"
echo ""
echo "Once all jobs complete, run:"
echo "  python ${SCRIPT_DIR}/combine_logprobs.py \\"
echo "    --partitions-dir ${OUTPUT_BASE}/partitions \\"
echo "    --output-dir ${OUTPUT_BASE}/merged"
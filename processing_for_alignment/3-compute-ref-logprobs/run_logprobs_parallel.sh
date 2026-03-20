#!/bin/bash
set -euo pipefail

# ── Configuration ──────────────────────────────────────────────────
DATASET_PATH="/iopsstor/scratch/cscs/dmelikidze/posttraining-data/processing_for_alignment/datasets/MaxMin-Filtered"
OUTPUT_BASE="/iopsstor/scratch/cscs/dmelikidze/posttraining-data/processing_for_alignment/datasets/MaxMin-Filtered-Logprobs"
MODEL_PATH="/iopsstor/scratch/cscs/dmelikidze/huggingface/hub/models--swiss-ai--Apertus-8B-Instruct-2509-SFT/snapshots/d57e4f1a3baa6315c60707346b5498b48b40a364"
CONTAINER_ENV="./response_generation/env/alignment.toml"
MAX_SEQ_LEN=4096
SPLIT="train_split"
DATASET_SIZE=259453                # Set this! (or run: python -c "from datasets import load_from_disk; print(len(load_from_disk('$DATASET_PATH')['$SPLIT']))")

# Parallelism (adapted from swiss_alignment generate_submit.py)
# Reference numbers for 8B with 2 completions (chosen + rejected) per row:
#   ~2048 rows per hour per GPU
# We need N nodes for X rows in H hours:
#   N = X / (PARTITION_SIZE * H)
PARTITION_SIZE=8192           # rows per node-job (8192 for 8B, 1024 for 70B)
NUM_GPUS_PER_NODE=4
TENSOR_PARALLEL_SIZE=1        # GPUs per model instance
SAVE_INTERVAL=2048            # checkpoint every N rows per subpartition

# SLURM
SLURM_TIME="12:00:00"
SLURM_PARTITION="normal"
SLURM_ACCOUNT="infra01"
SLURM_RESERVATION="PA-2338-RL"
# ───────────────────────────────────────────────────────────────────

SCRIPT_DIR="$SCRATCH/posttraining-data/processing_for_alignment/3-compute-ref-logprobs"
LOGS_DIR="$SCRATCH/posttraining-data/processing_for_alignment/logs/logprobs-$(date +%Y-%m-%d-%H-%M)"
NUM_SUBPARTITIONS=$((NUM_GPUS_PER_NODE / TENSOR_PARALLEL_SIZE))

if [ "${DATASET_SIZE}" -eq 0 ]; then
    echo "ERROR: Set DATASET_SIZE in the script before running."
    echo "  Hint: python -c \"from datasets import load_from_disk; print(len(load_from_disk('${DATASET_PATH}')['${SPLIT}']))\""
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

    # Each job runs NUM_SUBPARTITIONS tasks (one per GPU group).
    # Each task picks up SLURM_PROCID from the environment automatically.
    # Output dir per subpartition: ${PART_OUTPUT}/subpart_${SLURM_PROCID}
    #
    # The Python script reads SLURM_PROCID from env to determine:
    #   - which GPU(s) to use
    #   - which slice of the partition to process
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
        --wrap="srun --environment=${CONTAINER_ENV} --container-writable --container-workdir=${SCRIPT_DIR} \
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
    --split ${SPLIT}'")

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

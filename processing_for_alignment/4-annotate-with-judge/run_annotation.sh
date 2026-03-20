#!/bin/bash

# Fixed model and cluster configuration for Qwen3-235B
MODEL="Qwen/Qwen3-235B-A22B-Instruct-2507"
NNODES=4
WORKERS=2
NPW=2
DP=1
TP=8
FRAMEWORK="sglang"
OCF_FLAG="--disable-ocf"

# Completion splits from split_by_completion.py
DATASETS_DIR="$SCRATCH/posttraining-data/processing_for_alignment/datasets/MaxMin-Filtered-Ref-Completions-30-split"

BASE_OUTPUT_DIR="$SCRATCH/posttraining-data/processing_for_alignment/datasets/MaxMin-Filtered-Ref-Completions-30-split-Annotated"
PROMPT_COLUMN_NAME="chosen"
REMOVE_LAST_MESSAGE=1
JOB_TIME="09:00:00"

ACCOUNT="infra01"
RESERVATION="PA-2338-RL"
REMOVE_LAST_MESSAGE_FLAG=""
if [ "$REMOVE_LAST_MESSAGE" -eq 1 ]; then REMOVE_LAST_MESSAGE_FLAG="--remove-last-message"; fi

ANNOTATION_SCRIPT_DIR="$SCRATCH/posttraining-data/response_annotation"
LOGS_DIR="$SCRATCH/posttraining-data/processing_for_alignment/logs/annotation"

mkdir -p $LOGS_DIR

for DATASET_PATH in ${DATASETS_DIR}/completion_*; do
    SAFE_DATASET_NAME=$(basename "$DATASET_PATH")

    sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=ann_${SAFE_DATASET_NAME}
#SBATCH --account=${ACCOUNT}
#SBATCH --output=${LOGS_DIR}/${SAFE_DATASET_NAME}_%j.log
#SBATCH --time=${JOB_TIME}
#SBATCH --reservation=${RESERVATION}
#SBATCH --partition=normal
#SBATCH --nodes=1

srun --environment="./response_generation/env/alignment.toml" --container-writable --container-workdir="$SCRATCH/posttraining-data" \
    bash -c "unset SSL_CERT_FILE && python -u ${ANNOTATION_SCRIPT_DIR}/run_annotation.py \
    --base-output-dir '${BASE_OUTPUT_DIR}' \
    --logs-dir '${LOGS_DIR}' \
    --dataset '${DATASET_PATH}' \
    --prompt-column-name '${PROMPT_COLUMN_NAME}' \
    --model '${MODEL}' \
    --slurm-nodes ${NNODES} \
    --workers ${WORKERS} \
    --nodes-per-worker ${NPW} \
    --dp-size ${DP} \
    --tp-size ${TP} \
    --framework '${FRAMEWORK}' \
    ${OCF_FLAG} ${REMOVE_LAST_MESSAGE_FLAG}"
EOF

    echo "Submitted annotation for ${SAFE_DATASET_NAME}"
done

echo "All jobs submitted."

#!/bin/bash

# Fixed model and cluster configuration for Qwen3-235B
MODEL="/capstor/store/cscs/swissai/infra01/hf_models/models/Qwen/Qwen3.6-27B"
NNODES=1
WORKERS=1
NPW=1
DP=1
TP=4
FRAMEWORK="vllm"
# FRAMEWORK="vllm"
OCF_FLAG="--disable-ocf"

# Array of absolute paths to your datasets
DATASETS=(
    # /capstor/store/cscs/swissai/infra01/datasets/alignment/completions/Dolci-Instruct-DPO/EuroLLM-1.7B-Instruct
    # /capstor/store/cscs/swissai/infra01/datasets/alignment/completions/Dolci-Instruct-DPO/EuroLLM-22B-Instruct-2512
    # /capstor/store/cscs/swissai/infra01/datasets/alignment/completions/Dolci-Instruct-DPO/EuroLLM-9B-Instruct-2512
    # /capstor/store/cscs/swissai/infra01/datasets/alignment/completions/Dolci-Instruct-DPO/Ministral-3-14B-Instruct-2512
    # /capstor/store/cscs/swissai/infra01/datasets/alignment/completions/Dolci-Instruct-DPO/Ministral-3-3B-Instruct-2512
    # /capstor/store/cscs/swissai/infra01/datasets/alignment/completions/Dolci-Instruct-DPO/Ministral-3-8B-Instruct-2512
    /capstor/store/cscs/swissai/infra01/datasets/alignment/completions/Dolci-Instruct-DPO/Mistral-Large-3-675B-Instruct-2512
    # /capstor/store/cscs/swissai/infra01/datasets/alignment/completions/Dolci-Instruct-DPO/Mistral-Small-24B-Instruct-2501
    # /capstor/store/cscs/swissai/infra01/datasets/alignment/completions/Dolci-Instruct-DPO/Mixtral-8x22B-Instruct-v0.1
    # /capstor/store/cscs/swissai/infra01/datasets/alignment/completions/Dolci-Instruct-DPO/Phi-4-mini-instruct
    # /capstor/store/cscs/swissai/infra01/datasets/alignment/completions/Dolci-Instruct-DPO/Qwen2.5-0.5B-Instruct
    # /capstor/store/cscs/swissai/infra01/datasets/alignment/completions/Dolci-Instruct-DPO/Qwen2.5-1.5B-Instruct
    # /capstor/store/cscs/swissai/infra01/datasets/alignment/completions/Dolci-Instruct-DPO/Qwen3-0.6B
    # /capstor/store/cscs/swissai/infra01/datasets/alignment/completions/Dolci-Instruct-DPO/Qwen3-1.7B
    # /capstor/store/cscs/swissai/infra01/datasets/alignment/completions/Dolci-Instruct-DPO/Qwen3-235B-A22B-Instruct-2507
    # /capstor/store/cscs/swissai/infra01/datasets/alignment/completions/Dolci-Instruct-DPO/Qwen3-30B-A3B-Instruct-2507
    # /capstor/store/cscs/swissai/infra01/datasets/alignment/completions/Dolci-Instruct-DPO/Qwen3-32B
    # /capstor/store/cscs/swissai/infra01/datasets/alignment/completions/Dolci-Instruct-DPO/Qwen3-4B-Instruct-2507
    # /capstor/store/cscs/swissai/infra01/datasets/alignment/completions/Dolci-Instruct-DPO/Qwen3.5-397B-A17B
    # /capstor/store/cscs/swissai/infra01/datasets/alignment/completions/Dolci-Instruct-DPO/Qwen3-8B
    # /capstor/store/cscs/swissai/infra01/datasets/alignment/completions/Dolci-Instruct-DPO/Qwen3-Next-80B-A3B-Instruct
    # /capstor/store/cscs/swissai/infra01/datasets/alignment/completions/Dolci-Instruct-DPO/Qwen3-Omni-30B-A3B-Instruct
    # /capstor/store/cscs/swissai/infra01/datasets/alignment/completions/Dolci-Instruct-DPO/SmolLM3-3B
    # /capstor/store/cscs/swissai/infra01/datasets/alignment/completions/Dolci-Instruct-DPO/Trinity-Mini
    # /capstor/store/cscs/swissai/infra01/datasets/alignment/completions/Dolci-Instruct-DPO/Trinity-Nano-Preview
)

BASE_OUTPUT_DIR="$SCRATCH/posttraining-data/response_annotation/datasets/alignment/completions/Dolci-Instruct-DPO/"
PROMPT_COLUMN_NAME="chosen"
REMOVE_LAST_MESSAGE=1  # Set to 1 if you want to remove the last message from the conversation history, e.g. if you take it from a "chosen" column
JOB_TIME="08:00:00"

ACCOUNT="infra01"
RESERVATION="SD-69241-apertus-1-5-0"
EXCLUDE="nid006239"
REMOVE_LAST_MESSAGE_FLAG=""
if [ "$REMOVE_LAST_MESSAGE" -eq 1 ]; then REMOVE_LAST_MESSAGE_FLAG="--remove-last-message"; fi

LOGS_DIR="./logs/annotation"


mkdir -p $LOGS_DIR

for DATASET_PATH in "${DATASETS[@]}"; do
    # Extract just the folder name of the dataset for cleaner job names and logs
    SAFE_DATASET_NAME=$(basename "$DATASET_PATH")

    sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=ann_${SAFE_DATASET_NAME}
#SBATCH --account=${ACCOUNT}
#SBATCH --output=${LOGS_DIR}/${SAFE_DATASET_NAME}_%j.log
#SBATCH --time=${JOB_TIME}
#SBATCH --reservation=${RESERVATION}                 # Uncomment if you have a reservation to use
#SBATCH --partition=normal
#SBATCH --nodes=1

srun --environment="./response_generation/env/alignment.toml" --container-writable --container-workdir="$PWD" \\
    bash -c "unset SSL_CERT_FILE && python -u response_annotation/run_annotation.py \\
    --base-output-dir '${BASE_OUTPUT_DIR}' \\
    --logs-dir '${LOGS_DIR}' \\
    --dataset '${DATASET_PATH}' \\
    --prompt-column-name '${PROMPT_COLUMN_NAME}' \\
    --model '${MODEL}' \\
    --slurm-nodes ${NNODES} \\
    --job-time ${JOB_TIME} \\
    --workers ${WORKERS} \\
    --nodes-per-worker ${NPW} \\
    --dp-size ${DP} \\
    --tp-size ${TP} \\
    --framework '${FRAMEWORK}' \\
    --slurm-exclude '${EXCLUDE}' \\
    ${OCF_FLAG} ${REMOVE_LAST_MESSAGE_FLAG}"
EOF
done

echo "✅ All jobs submitted."
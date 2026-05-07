#!/bin/bash

# Format: "ModelName TotalNodes Workers NodesPerWorker DP TP DisableOCF(true/false) Framework NoReasoningKwargs(true/false)"
JOBS=(
    "Qwen/Qwen2.5-0.5B-Instruct 1 1 1 4 1 false sglang false"
    "Qwen/Qwen2.5-1.5B-Instruct 1 1 1 4 1 false sglang false"
    "Qwen/Qwen3-0.6B 1 1 1 4 1 false sglang false"
    "Qwen/Qwen3-1.7B 1 1 1 4 1 false sglang false"
    "Qwen/Qwen3-4B-Instruct-2507 1 1 1 4 1 false sglang false"
    "Qwen/Qwen3-8B 1 1 1 4 1 false sglang false"
    "Qwen/Qwen3-32B 1 1 1 4 1 false sglang false"
    "Qwen/Qwen3-30B-A3B-Instruct-2507 1 1 1 1 4 false sglang false"
    "Qwen/Qwen3-Omni-30B-A3B-Instruct 1 1 1 1 4 false vllm false"
    "Qwen/Qwen3-Next-80B-A3B-Instruct 1 1 1 1 4 false vllm false" 
    "Qwen/Qwen3-235B-A22B-Instruct-2507 16 8 2 1 8 true vllm false" 
    "microsoft/Phi-4-mini-instruct 1 1 1 4 1 false sglang false"
    "mistralai/Mistral-Small-24B-Instruct-2501 1 1 1 1 4 false sglang false"
    "mistralai/Mixtral-8x22B-Instruct-v0.1 2 1 2 1 8 true sglang false"
    "mistralai/Ministral-3-3B-Instruct-2512 1 1 1 4 1 false vllm true"
    "mistralai/Ministral-3-8B-Instruct-2512 1 1 1 4 1 false vllm true"
    "mistralai/Ministral-3-14B-Instruct-2512 1 1 1 4 1 false vllm true"
    "arcee-ai/Trinity-Mini 1 1 1 4 1 false vllm false"
    "arcee-ai/Trinity-Nano-Preview 1 1 1 4 1 false vllm false"
    "HuggingFaceTB/SmolLM3-3B 1 1 1 4 1 false sglang false"
    "utter-project/EuroLLM-1.7B-Instruct 1 1 1 4 1 false sglang false"
    "utter-project/EuroLLM-9B-Instruct-2512 1 1 1 4 1 false sglang false"
    "utter-project/EuroLLM-22B-Instruct-2512 1 1 1 4 1 false sglang false"
    "Qwen/Qwen3.5-397B-A17B 4 1 4 1 16 true vllm false"
    "mistralai/Mistral-Large-3-675B-Instruct-2512 4 1 4 1 16 true vllm true"
)


INPUT_DATASET="allenai/Dolci-Instruct-DPO"
BASE_OUTPUT_DIR="./response_generation/datasets/inference_results_final"
PROMPT_COLUMN_NAME="chosen"
REMOVE_LAST_MESSAGE=1  # Set to 1 if you want to remove the last message from the conversation history, e.g. if you take it from a "chosen" column
JOB_TIME="12:00:00"

ACCOUNT="infra01"
RESERVATION="SD-69241-apertus-1-5-3"
LOGS_DIR="./response_generation/logs/generation"

mkdir -p $LOGS_DIR

for ENTRY in "${JOBS[@]}"; do
    read -r MODEL NNODES WORKERS NPW DP TP DOCF FRAMEWORK NO_REASONING <<< "$ENTRY"
    SAFE_MODEL_NAME=$(echo "$MODEL" | tr '/' '_')

    OCF_FLAG=""
    if [ "$DOCF" = "true" ]; then OCF_FLAG="--disable-ocf"; fi

    REASONING_FLAG=""
    if [ "$NO_REASONING" = "true" ]; then REASONING_FLAG="--no-reasoning-kwargs"; fi

    REMOVE_LAST_MESSAGE_FLAG=""
    if [ "$REMOVE_LAST_MESSAGE" -eq 1 ]; then REMOVE_LAST_MESSAGE_FLAG="--remove-last-message"; fi

    sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=gen_${SAFE_MODEL_NAME}
#SBATCH --account=${ACCOUNT}
#SBATCH --output=${LOGS_DIR}/client/${SAFE_MODEL_NAME}_%j.log
#SBATCH --time=${JOB_TIME}
#SBATCH --reservation=${RESERVATION}                 # Uncomment if you have a reservation to use
#SBATCH --partition=normal
#SBATCH --nodes=1

srun --environment="./response_generation/env/alignment.toml" --container-writable --container-workdir="$PWD" \\
    bash -c "unset SSL_CERT_FILE && pip install mistral_common && python -u response_generation/run_generation.py \\
    --dataset '${INPUT_DATASET}' \\
    --prompt-column-name '${PROMPT_COLUMN_NAME}' \\
    --base-output-dir '${BASE_OUTPUT_DIR}' \\
    --logs-dir '${LOGS_DIR}/server' \\
    --model '${MODEL}' \\
    --slurm-nodes ${NNODES} \\
    --workers ${WORKERS} \\
    --nodes-per-worker ${NPW} \\
    --dp-size ${DP} \\
    --tp-size ${TP} \\
    --framework '${FRAMEWORK}' \\
    ${OCF_FLAG} ${REASONING_FLAG} ${REMOVE_LAST_MESSAGE_FLAG}"
EOF
done

echo "✅ All jobs submitted."
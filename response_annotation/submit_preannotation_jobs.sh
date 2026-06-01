#!/bin/bash

MODEL="/capstor/store/cscs/swissai/infra01/hf_models/models/Qwen/Qwen3.6-27B"
NNODES=1
WORKERS=1
DP=4
TP=1
FRAMEWORK="vllm"
DISABLE_OCF=1

DATASETS=(
    $SCRATCH/posttraining-data/response_annotation/datasets/alignment/annotations/Dolci-Instruct-DPO-Qwen3.6-27B-combined
)

BASE_OUTPUT_DIR="$SCRATCH/posttraining-data/response_annotation/datasets/alignment/preannotation"
PROMPT_COLUMN_NAME="prompt"
MAX_TOKENS=4096
JOB_TIME="12:00:00"

ACCOUNT="infra01"
RESERVATION="SD-69241-apertus-1-5-0"

WORKDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

LOGS_DIR="${WORKDIR}/logs/preannotation"
mkdir -p "${LOGS_DIR}"

DISABLE_OCF_FLAG=""
if [ "${DISABLE_OCF}" -eq 1 ]; then DISABLE_OCF_FLAG="--disable-ocf"; fi

cd "${WORKDIR}"

for DATASET_PATH in "${DATASETS[@]}"; do
    DATASET_NAME="$(basename "${DATASET_PATH}")"

    echo "🚀 Submitting server job for ${DATASET_NAME}..."
    SERVER_JOB_ID=$(bash "${WORKDIR}/init_llm_server.sh" \
        --model "${MODEL}" \
        --nnodes ${NNODES} \
        --dp-size ${DP} \
        --tp-size ${TP} \
        --framework "${FRAMEWORK}" \
        --job-time "${JOB_TIME}" \
        ${DISABLE_OCF_FLAG})

    if [[ -z "${SERVER_JOB_ID}" ]]; then
        echo "❌ Failed to get server job ID for ${DATASET_NAME}, skipping."
        continue
    fi
    echo "✅ Server job submitted: ${SERVER_JOB_ID}"

    sbatch --dependency=after:${SERVER_JOB_ID} <<EOF
#!/bin/bash
#SBATCH --job-name=preann_${DATASET_NAME}
#SBATCH --account=${ACCOUNT}
#SBATCH --reservation=${RESERVATION}
#SBATCH --output=${LOGS_DIR}/${DATASET_NAME}_%j.log
#SBATCH --time=${JOB_TIME}
#SBATCH --partition=normal
#SBATCH --nodes=1

cd "${WORKDIR}"

srun --environment="${WORKDIR}/../response_generation/env/alignment.toml" --container-writable --container-workdir="${WORKDIR}" \\
    bash -c "unset SSL_CERT_FILE && python -u preannotate.py \\
    --model '${MODEL}' \\
    --dataset-path '${DATASET_PATH}' \\
    --output-dir '${BASE_OUTPUT_DIR}' \\
    --server-job-id '${SERVER_JOB_ID}' \\
    --workers ${WORKERS} \\
    --prompt-column-name '${PROMPT_COLUMN_NAME}' \\
    --max-tokens ${MAX_TOKENS}"
EOF

done

echo "✅ All jobs submitted."

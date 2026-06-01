#!/bin/bash
# Submits the preannotation LLM server job from the login node.
# Prints the server job ID to stdout; all other output goes to stderr.

MODEL=""
NNODES=4
TP=4
DP=4
FRAMEWORK="vllm"
JOB_TIME="2:00:00"
DISABLE_OCF=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --model)       MODEL="$2";     shift 2 ;;
        --nnodes)      NNODES="$2";    shift 2 ;;
        --tp-size)     TP="$2";        shift 2 ;;
        --dp-size)     DP="$2";        shift 2 ;;
        --framework)   FRAMEWORK="$2"; shift 2 ;;
        --job-time)    JOB_TIME="$2";  shift 2 ;;
        --disable-ocf) DISABLE_OCF=1;  shift ;;
        *) echo "Unknown argument: $1" >&2; exit 1 ;;
    esac
done

if [[ -z "${MODEL}" ]]; then
    echo "Error: --model is required and cannot be empty." >&2
    exit 1
fi

ML_PYTHON="${SCRATCH}/model-launch/.venv/bin/python"

CMD=(
    "${ML_PYTHON}" "${SCRATCH}/model-launch/legacy/serving/submit_job.py"
    "--slurm-nodes" "${NNODES}"
    "--slurm-time" "${JOB_TIME}"
    "--serving-framework" "${FRAMEWORK}"
    "--worker-port" "8080"
    "--slurm-environment" "${SCRATCH}/model-launch/legacy/serving/envs/${FRAMEWORK}.toml"
)
if [ "${DISABLE_OCF}" -eq 1 ]; then CMD+=("--disable-ocf"); fi


if [ "${FRAMEWORK}" = "vllm" ]; then
    FW_ARGS="--model ${MODEL} --host 0.0.0.0 --port 8080 --served-model-name ${MODEL} --data-parallel-size ${DP} --tensor-parallel-size ${TP} --trust-remote-code"
    if [ "${NNODES}" -eq 1 ]; then FW_ARGS="${FW_ARGS} --distributed-executor-backend mp"; fi
elif [ "${FRAMEWORK}" = "sglang" ]; then
    FW_ARGS="--model-path ${MODEL} --host 0.0.0.0 --port 8080 --served-model-name ${MODEL} --dp-size ${DP} --tp-size ${TP} --trust-remote-code"
    if [ "${NNODES}" -gt 1 ]; then FW_ARGS="${FW_ARGS} --enable-dp-attention"; fi
    CMD+=("--pre-launch-cmds" "export SGLANG_DISABLE_CUDNN_CHECK=1")
else
    echo "Unknown framework: ${FRAMEWORK}" >&2
    exit 1
fi
CMD+=("--framework-args" "${FW_ARGS}")

echo "Submitting: ${CMD[*]}" >&2
OUTPUT=$("${CMD[@]}" 2>&1)
echo "${OUTPUT}" >&2

JOB_ID=$(echo "${OUTPUT}" | grep "Job submitted successfully with ID:" | awk '{print $NF}')
if [[ -z "${JOB_ID}" ]]; then
    echo "Failed to parse job ID from output." >&2
    exit 1
fi

echo "${JOB_ID}"

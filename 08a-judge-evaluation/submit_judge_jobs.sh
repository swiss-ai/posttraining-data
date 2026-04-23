#!/bin/bash
ACCOUNT="infra01"
RESERVATION="SD-69241-apertus-1-5"
JOB_TIME="08:00:00"

MODEL="Qwen/Qwen3-235B-A22B-Instruct-2507"
NNODES=4
WORKERS=2
NPW=2
DP=1
TP=8
FRAMEWORK="sglang"
OCF_FLAG="--disable-ocf"
CONCURRENT=32

BENCHMARK_DIRS=(
    # "$SCRATCH/posttraining-data/08a-judge-evaluation/benchmarks/JudgeBench-gpt"
    "$SCRATCH/posttraining-data/08a-judge-evaluation/benchmarks/RM-Bench-train"
)
JUDGE_ARGS_PATHS=(
    # "$SCRATCH/posttraining-data/08a-judge-evaluation/judges/01.py"
    # "$SCRATCH/posttraining-data/08a-judge-evaluation/judges/02.py"
    # "$SCRATCH/posttraining-data/08a-judge-evaluation/judges/03.py"
    # "$SCRATCH/posttraining-data/08a-judge-evaluation/judges/04.py"

    # "$SCRATCH/posttraining-data/08a-judge-evaluation/judges/05.py"

    # "$SCRATCH/posttraining-data/08a-judge-evaluation/judges/11.py"
    "$SCRATCH/posttraining-data/08a-judge-evaluation/judges/12.py" # this judge takes much longer than the others, ~1h for 1K samples
)

WORKDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

for BENCHMARK_DIR in "${BENCHMARK_DIRS[@]}"; do
    BENCHMARK_NAME="$(basename "$BENCHMARK_DIR")"
    INPUT_DIR="${BENCHMARK_DIR}/1-reformatted"

    for JUDGE_ARGS_PATH in "${JUDGE_ARGS_PATHS[@]}"; do
        JUDGE_NAME="$(basename "$JUDGE_ARGS_PATH" .py)"
        OUTPUT_DIR="${BENCHMARK_DIR}/2-judged/${JUDGE_NAME}"
        mkdir -p "$OUTPUT_DIR"

        sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=judge_${BENCHMARK_NAME}_${JUDGE_NAME}
#SBATCH --account=${ACCOUNT}
#SBATCH --reservation=${RESERVATION}
#SBATCH --output=${BENCHMARK_DIR}/2-judged/logs/${JUDGE_NAME}_%j.log
#SBATCH --time=${JOB_TIME}
#SBATCH --partition=normal
#SBATCH --nodes=1

cd "${WORKDIR}"

srun --environment=activeuf --container-writable --container-workdir="${WORKDIR}" \\
    bash -c "unset SSL_CERT_FILE && python -u run_judge.py \\
    --input-dir '${INPUT_DIR}' \\
    --output-dir '${OUTPUT_DIR}' \\
    --judge-args-path '${JUDGE_ARGS_PATH}' \\
    --job-time '${JOB_TIME}' \\
    --model '${MODEL}' \\
    --slurm-nodes ${NNODES} \\
    --workers ${WORKERS} \\
    --nodes-per-worker ${NPW} \\
    --dp-size ${DP} \\
    --tp-size ${TP} \\
    --framework '${FRAMEWORK}' \\
    --concurrent ${CONCURRENT} \\
    ${OCF_FLAG}"
EOF
    done
done

echo "✅ All jobs submitted."

#!/bin/bash
ACCOUNT="infra01"
RESERVATION="SD-69241-apertus-1-5-0"
JOB_TIME="00:30:00"

BENCHMARK_DIRS=(
    # "$SCRATCH/posttraining-data/08a-judge-evaluation/benchmarks/JudgeBench-gpt"
    # "$SCRATCH/posttraining-data/08a-judge-evaluation/benchmarks/RM-Bench-train"
    "$SCRATCH/posttraining-data/08a-judge-evaluation/benchmarks/RewardBench2-test"
)
JUDGE_CFG_PATHS=(
    #"$SCRATCH/posttraining-data/08a-judge-evaluation/judges/00-ActiveUF.py                        
    #"$SCRATCH/posttraining-data/08a-judge-evaluation/judges/01-ActiveUF-Helpfulness.py
    #"$SCRATCH/posttraining-data/08a-judge-evaluation/judges/02-ActiveUF-Instruction_Following.py  
    #"$SCRATCH/posttraining-data/08a-judge-evaluation/judges/03-ActiveUF-Honesty.py
    #"$SCRATCH/posttraining-data/08a-judge-evaluation/judges/04-ActiveUF-Truthfulness.py

    #"$SCRATCH/posttraining-data/08a-judge-evaluation/judges/10-General-Quality.py
    #"$SCRATCH/posttraining-data/08a-judge-evaluation/judges/11-SwissAI_Charter.py

    #"$SCRATCH/posttraining-data/08a-judge-evaluation/judges/20-Qwen3.5_35B-Helpfulness.py
    #"$SCRATCH/posttraining-data/08a-judge-evaluation/judges/21-Qwen3.6_27B-Helpfulness.py

    #"$SCRATCH/posttraining-data/08a-judge-evaluation/judges/30-ArenaHard-regex.py
    #"$SCRATCH/posttraining-data/08a-judge-evaluation/judges/31-Qwen3.6_27B-ArenaHard-regex.py
    #"$SCRATCH/posttraining-data/08a-judge-evaluation/judges/32-Qwen3.6_27B-ArenaHard.py
)

WORKDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ML_PYTHON="${SCRATCH}/model-launch/.venv/bin/python"

# Must cd to WORKDIR so that server job logs land in ./logs/<job_id>/log.out
# relative to here, which is where run_judge_on_benchmark.py looks for them.
cd "${WORKDIR}"

for BENCHMARK_DIR in "${BENCHMARK_DIRS[@]}"; do
    BENCHMARK_NAME="$(basename "$BENCHMARK_DIR")"
    INPUT_DIR="${BENCHMARK_DIR}/1-reformatted"

    for JUDGE_CFG_PATH in "${JUDGE_CFG_PATHS[@]}"; do
        JUDGE_NAME="$(basename "$JUDGE_CFG_PATH" .py)"
        OUTPUT_DIR="${BENCHMARK_DIR}/2-judged/${JUDGE_NAME}"
        mkdir -p "$OUTPUT_DIR"
        mkdir -p "${BENCHMARK_DIR}/2-judged/logs"

        echo "🚀 Submitting server job for ${BENCHMARK_NAME}_${JUDGE_NAME}..."
        SERVER_JOB_ID=$(${ML_PYTHON} init_judge_server.py \
            --input-dir "${INPUT_DIR}" \
            --judge-cfg-path "${JUDGE_CFG_PATH}" \
            --job-time "${JOB_TIME}")

        if [[ -z "${SERVER_JOB_ID}" ]]; then
            echo "❌ Failed to get server job ID for ${BENCHMARK_NAME}_${JUDGE_NAME}, skipping."
            continue
        fi
        echo "✅ Server job submitted: ${SERVER_JOB_ID}"

        sbatch --dependency=after:${SERVER_JOB_ID} <<EOF
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
    bash -c "unset SSL_CERT_FILE && python -u run_judge_on_benchmark.py \\
    --input-dir '${INPUT_DIR}' \\
    --output-dir '${OUTPUT_DIR}' \\
    --judge-cfg-path '${JUDGE_CFG_PATH}' \\
    --job-time '${JOB_TIME}' \\
    --server-job-id '${SERVER_JOB_ID}'"
EOF

    done
done

echo "✅ All jobs submitted."

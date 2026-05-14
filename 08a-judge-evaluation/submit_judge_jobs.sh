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
    "$SCRATCH/posttraining-data/08a-judge-evaluation/judges/01.py"
    # "$SCRATCH/posttraining-data/08a-judge-evaluation/judges/02.py"
    # "$SCRATCH/posttraining-data/08a-judge-evaluation/judges/03.py"
    # "$SCRATCH/posttraining-data/08a-judge-evaluation/judges/04.py"

    # "$SCRATCH/posttraining-data/08a-judge-evaluation/judges/05.py"

    # "$SCRATCH/posttraining-data/08a-judge-evaluation/judges/11.py"
    "$SCRATCH/posttraining-data/08a-judge-evaluation/judges/12.py" # this judge takes much longer than the others, ~1h for 1K samples
    
    # "$SCRATCH/posttraining-data/08a-judge-evaluation/judges/21.py"
    # "$SCRATCH/posttraining-data/08a-judge-evaluation/judges/22.py"
)

WORKDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

for BENCHMARK_DIR in "${BENCHMARK_DIRS[@]}"; do
    BENCHMARK_NAME="$(basename "$BENCHMARK_DIR")"
    INPUT_DIR="${BENCHMARK_DIR}/1-reformatted"

    for JUDGE_CFG_PATH in "${JUDGE_CFG_PATHS[@]}"; do
        JUDGE_NAME="$(basename "$JUDGE_CFG_PATH" .py)"
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
    --judge-cfg-path '${JUDGE_CFG_PATH}' \\
    --job-time '${JOB_TIME}'"
EOF
    done
done

echo "✅ All jobs submitted."

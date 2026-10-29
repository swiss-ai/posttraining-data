#!/bin/bash

# Submits decontamination as a SLURM job array plus a merge job, or as one job running both ('single')
# Usage: ./submit-parallel-decontamination.sh <input_dataset_path> <output_dataset_path> [chunk_size|single] [max_parallel_jobs]

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
# Get the repository root (parent of 04-decontamination)
REPO_ROOT="$( cd "${SCRIPT_DIR}/.." && pwd )"
# Account, partition, QOS, log folder, Python env, ... (overridable via environment)
source "${SCRIPT_DIR}/slurm_config.sh"

# Default parameters
DEFAULT_CHUNK_SIZE=20
DEFAULT_MAX_PARALLEL=64

# Check if correct number of arguments provided
if [ $# -lt 2 ] || [ $# -gt 4 ]; then
    echo "Usage: $0 <input_dataset_path> <output_dataset_path> [chunk_size|single] [max_parallel_jobs]"
    echo ""
    echo "Arguments:"
    echo "  input_dataset_path   Path to input dataset directory"
    echo "  output_dataset_path  Path for final output dataset directory"
    echo "  chunk_size          Benchmarks per parallel job (default: $DEFAULT_CHUNK_SIZE), or 'single' for one"
    echo "                      job that checks all benchmarks and then merges (e.g. on the debug partition)"
    echo "  max_parallel_jobs   Maximum parallel jobs (default: $DEFAULT_MAX_PARALLEL)"
    echo ""
    echo "Example: $0 /path/to/input/dataset /path/to/output/dataset 25 16"
    exit 1
fi

# Get input arguments
INPUT_PATH="$1"
OUTPUT_PATH="$2"
CHUNK_SIZE=${3:-$DEFAULT_CHUNK_SIZE}
MAX_PARALLEL=${4:-$DEFAULT_MAX_PARALLEL}

# Validate input path exists
if [ ! -d "$INPUT_PATH" ]; then
    echo "Error: Input dataset path does not exist: $INPUT_PATH"
    exit 1
fi

check_slurm_config

# Extract dataset name from path for job naming
DATASET_NAME=$(basename "$INPUT_PATH")

# Create output directory if it doesn't exist
OUTPUT_DIR=$(dirname "$OUTPUT_PATH")
mkdir -p "$OUTPUT_DIR"

# Create reports directory
REPORTS_DIR="${OUTPUT_PATH}_parallel_reports"
mkdir -p "$REPORTS_DIR"

# Generate unique timestamp for this run
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

cd "$REPO_ROOT"
source "$PYTHON_ENV_ACTIVATE"

# Prompts and cache directory (decontamination_settings.py, overridable via environment); the tokenizer and
# the matching settings are the defaults of the Python scripts
SETTINGS=$(python 04-decontamination/decontamination_settings.py) || exit 1
eval "$SETTINGS"
if [ ! -d "$DECONTAMINATION_PROMPTS" ]; then
    echo "Error: Decontamination prompts not found: $DECONTAMINATION_PROMPTS"
    echo "Run gather-decontamination-prompts.py first to create benchmark prompts"
    exit 1
fi
mkdir -p "$DECONTAMINATION_CACHE_DIR"
echo "Prompts:   $DECONTAMINATION_PROMPTS"
echo "Cache:     $DECONTAMINATION_CACHE_DIR"

# Get total number of benchmarks and calculate job array size
echo "Counting available benchmarks..."

TOTAL_BENCHMARKS=$(python -c 'import sys; sys.path.insert(0, "04-decontamination"); from benchmark_filters import expected_benchmark_names; print(len(expected_benchmark_names(sys.argv[1])))' "$DECONTAMINATION_PROMPTS")
if [ -z "$TOTAL_BENCHMARKS" ] || [ "$TOTAL_BENCHMARKS" -eq 0 ]; then
    echo "Error: Could not determine number of benchmarks"
    exit 1
fi

# Single-job mode: one chunk with all benchmarks, run together with the merge in one job
SINGLE_JOB=false
if [ "$CHUNK_SIZE" = "single" ]; then
    SINGLE_JOB=true
    CHUNK_SIZE=$TOTAL_BENCHMARKS
fi

NUM_JOBS=$(( (TOTAL_BENCHMARKS + CHUNK_SIZE - 1) / CHUNK_SIZE ))  # Ceiling division
if [ "$NUM_JOBS" -gt "$MAX_PARALLEL" ]; then
    echo "Warning: Need $NUM_JOBS jobs for $TOTAL_BENCHMARKS benchmarks, but max parallel is $MAX_PARALLEL"
    echo "Some jobs will queue until others complete"
fi

echo "=== PARALLEL DECONTAMINATION SETUP ==="
echo "Dataset: $DATASET_NAME"
echo "Input:   $INPUT_PATH"
echo "Output:  $OUTPUT_PATH"
echo "Reports: $REPORTS_DIR"
echo "Total benchmarks: $TOTAL_BENCHMARKS"
if [ "$SINGLE_JOB" = true ]; then
    echo "Single job: all benchmarks, then the merge"
else
    echo "Chunk size: $CHUNK_SIZE benchmarks per job"
    echo "Array jobs: $NUM_JOBS jobs (0-$((NUM_JOBS-1)))"
    echo "Max parallel: $MAX_PARALLEL jobs"
fi
echo "======================================="

# Body of a job that writes the reports of one benchmark chunk (TASK_ID: array index, 0 in single-job mode)
decontamination_body() {
cat << EOF
# Set environment variables
export TOKENIZERS_PARALLELISM=true
export OMP_NUM_THREADS=1

TASK_ID=\${SLURM_ARRAY_TASK_ID:-0}

# Calculate benchmark range for this job
BENCHMARK_START=\$((TASK_ID * $CHUNK_SIZE))
BENCHMARK_END=\$((BENCHMARK_START + $CHUNK_SIZE))

# Don't exceed total benchmarks
if [ \$BENCHMARK_END -gt $TOTAL_BENCHMARKS ]; then
    BENCHMARK_END=$TOTAL_BENCHMARKS
fi

echo "=== JOB \$TASK_ID/$((NUM_JOBS-1)) ==="
echo "Processing benchmarks \$BENCHMARK_START to \$((\$BENCHMARK_END-1))"
echo "Chunk size: \$((\$BENCHMARK_END - \$BENCHMARK_START)) benchmarks"

# Change to project directory
cd ${REPO_ROOT}
source "${PYTHON_ENV_ACTIVATE}"

# Run parallel decontamination for this benchmark subset
python 04-decontamination/decontamination-parallel.py \\
      "${INPUT_PATH}" \\
      --decontamination_prompts "${DECONTAMINATION_PROMPTS}" \\
      --report_path "${REPORTS_DIR}" \\
      --cache_dir "${DECONTAMINATION_CACHE_DIR}" \\
      --benchmark-start \$BENCHMARK_START \\
      --benchmark-end \$BENCHMARK_END \\
      --num_proc 4
STATUS=\$?
if [ \$STATUS -eq 0 ]; then
    echo "Parallel job \$TASK_ID completed successfully."
else
    echo "Parallel job \$TASK_ID failed with exit code \$STATUS"
    exit 1
fi
EOF
}

# Body of the job that combines all reports and writes the filtered dataset
merge_body() {
cat << EOF
echo "=== MERGING PARALLEL DECONTAMINATION RESULTS ==="
echo "Dataset: ${DATASET_NAME}"
echo "Reports directory: ${REPORTS_DIR}"
echo "Final output: ${OUTPUT_PATH}"

# Change to project directory
cd ${REPO_ROOT}
source "${PYTHON_ENV_ACTIVATE}"

# Run merge script to combine all reports and create final filtered dataset
python 04-decontamination/merge-decontamination-reports.py \\
      "${INPUT_PATH}" \\
      "${OUTPUT_PATH}" \\
      "${REPORTS_DIR}" \\
      --decontamination_prompts "${DECONTAMINATION_PROMPTS}"
STATUS=\$?
if [ \$STATUS -eq 0 ]; then
    echo "Merge completed successfully."
    echo "Final filtered dataset saved to: ${OUTPUT_PATH}"
else
    echo "Merge failed with exit code \$STATUS"
    exit 1
fi
EOF
}

if [ "$SINGLE_JOB" = true ]; then
    # One job: the reports of all benchmarks, then the merge
    JOB_SCRIPT="${SLURM_LOG_DIR}/single_decontam_${DATASET_NAME}_${TIMESTAMP}.slurm"
    LOG_FILE="${SLURM_LOG_DIR}/decontam_${DATASET_NAME}_${TIMESTAMP}.out"

    cat > "$JOB_SCRIPT" << EOF
#!/bin/bash

#SBATCH -J decontam_${DATASET_NAME}
#SBATCH -t ${DECONTAM_TIME_LIMIT}
#SBATCH --account=${SLURM_ACCOUNT}
#SBATCH --partition=${SLURM_PARTITION}
${SBATCH_QOS_LINE}
${SBATCH_RESERVATION_LINE}
#SBATCH --output=${LOG_FILE}
#SBATCH --error=${LOG_FILE}
#SBATCH --nodes 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=72

$(decontamination_body)

$(merge_body)
EOF

    echo "Submitting single decontamination job..."
    JOB_ID=$(sbatch --parsable "$JOB_SCRIPT")
    if [ $? -eq 0 ]; then
        echo "Job submitted successfully (Job ID: $JOB_ID)"
        echo ""
        echo "=== MONITORING COMMANDS ==="
        echo "Monitor job:  squeue -j $JOB_ID"
        echo "View log:     tail -f ${LOG_FILE}"
    else
        echo "Error: Failed to submit job"
        exit 1
    fi
    exit 0
fi

# Create the parallel processing job array script
JOB_SCRIPT="${SLURM_LOG_DIR}/parallel_decontam_${DATASET_NAME}_${TIMESTAMP}.slurm"

cat > "$JOB_SCRIPT" << EOF
#!/bin/bash

#SBATCH -J pdecontam_${DATASET_NAME}
#SBATCH -t ${PARALLEL_TIME_LIMIT}
#SBATCH --account=${SLURM_ACCOUNT}
#SBATCH --partition=${SLURM_PARTITION}
${SBATCH_QOS_LINE}
${SBATCH_RESERVATION_LINE}
#SBATCH --output=${SLURM_LOG_DIR}/pdecontam_${DATASET_NAME}_${TIMESTAMP}_%a.out
#SBATCH --error=${SLURM_LOG_DIR}/pdecontam_${DATASET_NAME}_${TIMESTAMP}_%a.out
#SBATCH --array=0-$((NUM_JOBS-1))%${MAX_PARALLEL}
#SBATCH --nodes 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --exclusive

$(decontamination_body)
EOF

# Create the merge job script (runs after all parallel jobs complete)
MERGE_SCRIPT="${SLURM_LOG_DIR}/merge_decontam_${DATASET_NAME}_${TIMESTAMP}.slurm"

cat > "$MERGE_SCRIPT" << EOF
#!/bin/bash

#SBATCH -J merge_${DATASET_NAME}
#SBATCH -t ${MERGE_TIME_LIMIT}
#SBATCH --account=${SLURM_ACCOUNT}
#SBATCH --partition=${SLURM_PARTITION}
${SBATCH_QOS_LINE}
${SBATCH_RESERVATION_LINE}
#SBATCH --output=${SLURM_LOG_DIR}/merge_${DATASET_NAME}_${TIMESTAMP}.out
#SBATCH --error=${SLURM_LOG_DIR}/merge_${DATASET_NAME}_${TIMESTAMP}.out
#SBATCH --nodes 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=36

$(merge_body)
EOF

# Submit the parallel job array
echo "Submitting parallel decontamination job array..."
PARALLEL_JOB_ID=$(sbatch --parsable "$JOB_SCRIPT")

if [ $? -eq 0 ]; then
    echo "Parallel jobs submitted successfully (Job ID: $PARALLEL_JOB_ID)"
    
    # Submit merge job with dependency on parallel jobs
    echo "Submitting merge job with dependency..."
    MERGE_JOB_ID=$(sbatch --parsable --dependency=afterany:$PARALLEL_JOB_ID "$MERGE_SCRIPT")
    
    if [ $? -eq 0 ]; then
        echo "Merge job submitted successfully (Job ID: $MERGE_JOB_ID)"
        echo ""
        echo "=== MONITORING COMMANDS ==="
        echo "Monitor parallel jobs:  squeue -j $PARALLEL_JOB_ID"
        echo "Monitor merge job:      squeue -j $MERGE_JOB_ID" 
        echo "Watch parallel progress (reports written):"
        echo "  watch -n 5 'ls ${REPORTS_DIR}/*__contamination_report.json 2>/dev/null | wc -l; echo \"/ $TOTAL_BENCHMARKS benchmarks\"'"
        echo "View parallel job logs:"
        echo "  tail -f ${SLURM_LOG_DIR}/pdecontam_${DATASET_NAME}_${TIMESTAMP}_*.out"
        echo "View merge job log:"
        echo "  tail -f ${SLURM_LOG_DIR}/merge_${DATASET_NAME}_${TIMESTAMP}.out"
    else
        echo "Error: Failed to submit merge job"
        echo "You can run the merge manually after parallel jobs complete:"
        echo "  python 04-decontamination/merge-decontamination-reports.py $INPUT_PATH $OUTPUT_PATH $REPORTS_DIR --decontamination_prompts $DECONTAMINATION_PROMPTS"
    fi
else
    echo "Error: Failed to submit parallel jobs"
    exit 1
fi

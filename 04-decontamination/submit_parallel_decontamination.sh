#!/bin/bash

# Script to submit parallel decontamination SLURM job arrays
# Usage: ./submit_parallel_decontamination.sh <input_dataset_path> <output_dataset_path> [chunk_size] [max_parallel_jobs]

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
# Get the repository root (parent of 04-decontamination)
REPO_ROOT="$( cd "${SCRIPT_DIR}/.." && pwd )"

# Default parameters
DEFAULT_CHUNK_SIZE=20
DEFAULT_MAX_PARALLEL=20
DECONTAMINATION_PROMPTS="/capstor/store/cscs/swissai/infra01/posttrain_data/04_decontaminated/decontamination_prompts"

# Check if correct number of arguments provided
if [ $# -lt 2 ] || [ $# -gt 4 ]; then
    echo "Usage: $0 <input_dataset_path> <output_dataset_path> [chunk_size] [max_parallel_jobs]"
    echo ""
    echo "Arguments:"
    echo "  input_dataset_path   Path to input dataset directory"
    echo "  output_dataset_path  Path for final output dataset directory"
    echo "  chunk_size          Benchmarks per parallel job (default: $DEFAULT_CHUNK_SIZE)"
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

# Validate decontamination prompts exist
if [ ! -d "$DECONTAMINATION_PROMPTS" ]; then
    echo "Error: Decontamination prompts not found: $DECONTAMINATION_PROMPTS"
    echo "Run gather_decontamination_prompts.py first to create benchmark prompts"
    exit 1
fi

# Extract dataset name from path for job naming
DATASET_NAME=$(basename "$INPUT_PATH")

# Create output directory if it doesn't exist
OUTPUT_DIR=$(dirname "$OUTPUT_PATH")
mkdir -p "$OUTPUT_DIR"

# Create reports directory
REPORTS_DIR="${OUTPUT_PATH}_parallel_reports"
mkdir -p "$REPORTS_DIR"

# Create slurm logs directory if it doesn't exist
mkdir -p slurm_logs

# Generate unique timestamp for this run
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Get total number of benchmarks and calculate job array size
echo "Counting available benchmarks..."
cd "$REPO_ROOT"
source venv/bin/activate

TOTAL_BENCHMARKS=$(python 04-decontamination/list_benchmarks.py --prompts-path "$DECONTAMINATION_PROMPTS" --count-only | head -n 1 | cut -d' ' -f1)
if [ -z "$TOTAL_BENCHMARKS" ] || [ "$TOTAL_BENCHMARKS" -eq 0 ]; then
    echo "Error: Could not determine number of benchmarks"
    exit 1
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
echo "Chunk size: $CHUNK_SIZE benchmarks per job"  
echo "Array jobs: $NUM_JOBS jobs (0-$((NUM_JOBS-1)))"
echo "Max parallel: $MAX_PARALLEL jobs"
echo "======================================="

# Create the parallel processing job array script
JOB_SCRIPT="slurm_logs/parallel_decontam_${DATASET_NAME}_${TIMESTAMP}.slurm"

cat > "$JOB_SCRIPT" << EOF
#!/bin/bash

#SBATCH -J pdecontam_${DATASET_NAME}
#SBATCH -t 8:00:00
#SBATCH -A a-infra01-1
#SBATCH --output=slurm_logs/pdecontam_${DATASET_NAME}_${TIMESTAMP}_%a.out
#SBATCH --error=slurm_logs/pdecontam_${DATASET_NAME}_${TIMESTAMP}_%a.out
#SBATCH --array=0-$((NUM_JOBS-1))%${MAX_PARALLEL}
#SBATCH --nodes 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=144
#SBATCH --partition=large512

# Set environment variables
export TOKENIZERS_PARALLELISM=true
export OMP_NUM_THREADS=1

# Set cache directory on capstor
export DECONTAMINATION_CACHE_DIR="/capstor/store/cscs/swissai/infra01/posttrain_data/decontamination_cache"

# Create cache directory if it doesn't exist
mkdir -p \${DECONTAMINATION_CACHE_DIR}

# Calculate benchmark range for this job
BENCHMARK_START=\$((SLURM_ARRAY_TASK_ID * $CHUNK_SIZE))
BENCHMARK_END=\$((BENCHMARK_START + $CHUNK_SIZE))

# Don't exceed total benchmarks
if [ \$BENCHMARK_END -gt $TOTAL_BENCHMARKS ]; then
    BENCHMARK_END=$TOTAL_BENCHMARKS
fi

echo "=== JOB \$SLURM_ARRAY_TASK_ID/$((NUM_JOBS-1)) ==="
echo "Processing benchmarks \$BENCHMARK_START to \$((\$BENCHMARK_END-1))"
echo "Chunk size: \$((\$BENCHMARK_END - \$BENCHMARK_START)) benchmarks"

# Change to project directory
cd ${REPO_ROOT}
source venv/bin/activate

# Run parallel decontamination for this benchmark subset
python 04-decontamination/decontamination_parallel.py \\
      "${INPUT_PATH}" \\
      --decontamination_prompts "${DECONTAMINATION_PROMPTS}" \\
      --tokenizer_name "alehc/swissai-tokenizer" \\
      --report_path "${REPORTS_DIR}" \\
      --cache_dir "\${DECONTAMINATION_CACHE_DIR}" \\
      --benchmark-start \$BENCHMARK_START \\
      --benchmark-end \$BENCHMARK_END \\
      --ngram_length 8 \\
      --diff_threshold 0.5 \\
      --num_proc 8 \\
      --show_contaminated

# Check exit status
if [ \$? -eq 0 ]; then
    echo "Parallel job \$SLURM_ARRAY_TASK_ID completed successfully."
    # Create completion marker
    touch "${REPORTS_DIR}/job_\${SLURM_ARRAY_TASK_ID}.completed"
else
    echo "Parallel job \$SLURM_ARRAY_TASK_ID failed with exit code \$?"
    exit 1
fi
EOF

# Create the merge job script (runs after all parallel jobs complete)
MERGE_SCRIPT="slurm_logs/merge_decontam_${DATASET_NAME}_${TIMESTAMP}.slurm"

cat > "$MERGE_SCRIPT" << EOF
#!/bin/bash

#SBATCH -J merge_${DATASET_NAME}
#SBATCH -t 2:00:00
#SBATCH -A a-infra01-1
#SBATCH --output=slurm_logs/merge_${DATASET_NAME}_${TIMESTAMP}.out
#SBATCH --error=slurm_logs/merge_${DATASET_NAME}_${TIMESTAMP}.out
#SBATCH --nodes 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --partition=large512

echo "=== MERGING PARALLEL DECONTAMINATION RESULTS ==="
echo "Dataset: ${DATASET_NAME}"
echo "Reports directory: ${REPORTS_DIR}"
echo "Final output: ${OUTPUT_PATH}"

# Change to project directory
cd ${REPO_ROOT}
source venv/bin/activate

# Run merge script to combine all reports and create final filtered dataset
python 04-decontamination/merge_decontamination_reports.py \\
      "${INPUT_PATH}" \\
      "${OUTPUT_PATH}" \\
      "${REPORTS_DIR}" \\
      --tokenizer_name "alehc/swissai-tokenizer" \\
      --ngram_length 8 \\
      --diff_threshold 0.5

# Check exit status
if [ \$? -eq 0 ]; then
    echo "Merge completed successfully."
    echo "Final filtered dataset saved to: ${OUTPUT_PATH}"
else
    echo "Merge failed with exit code \$?"
    exit 1
fi
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
        echo "Watch parallel progress:"
        echo "  watch -n 5 'ls ${REPORTS_DIR}/*.completed 2>/dev/null | wc -l; echo \"/ $NUM_JOBS jobs completed\"'"
        echo "View parallel job logs:"
        echo "  tail -f slurm_logs/pdecontam_${DATASET_NAME}_${TIMESTAMP}_*.out"
        echo "View merge job log:"
        echo "  tail -f slurm_logs/merge_${DATASET_NAME}_${TIMESTAMP}.out"
    else
        echo "Error: Failed to submit merge job"
        echo "You can run the merge manually after parallel jobs complete:"
        echo "  python 04-decontamination/merge_decontamination_reports.py $INPUT_PATH $OUTPUT_PATH $REPORTS_DIR"
    fi
else
    echo "Error: Failed to submit parallel jobs"
    exit 1
fi
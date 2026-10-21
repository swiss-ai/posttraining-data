# Shared settings for the decontamination submit scripts (sourced, not executed).
# Every value can be overridden from the environment, e.g.
#   SLURM_PARTITION=debug SLURM_QOS=debug-qos ./04-decontamination/submit-decontamination.sh <in> <out>
# HF_HOME is not set here: jobs inherit it from the submitting shell.
# Requires REPO_ROOT to be set by the calling script.

SLURM_ACCOUNT="${SLURM_ACCOUNT:-infra01}"
SLURM_PARTITION="${SLURM_PARTITION:-normal}"
SLURM_QOS="${SLURM_QOS:-normal}"                 # set to "" to use the partition's default QOS
SLURM_RESERVATION="${SLURM_RESERVATION:-}"       # optional
SLURM_LOG_DIR="${SLURM_LOG_DIR:-/iopsstor/scratch/cscs/${USER}/posttraining-data/slurm_logs}"

DECONTAM_TIME_LIMIT="${DECONTAM_TIME_LIMIT:-12:00:00}"   # single-job decontamination
PARALLEL_TIME_LIMIT="${PARALLEL_TIME_LIMIT:-8:00:00}"    # each parallel array job
MERGE_TIME_LIMIT="${MERGE_TIME_LIMIT:-2:00:00}"          # merge job

PYTHON_ENV_ACTIVATE="${PYTHON_ENV_ACTIVATE:-${REPO_ROOT}/venv/bin/activate}"
DECONTAMINATION_PROMPTS="${DECONTAMINATION_PROMPTS:-/capstor/store/cscs/swissai/infra01/posttrain_data/04_decontaminated/decontamination_prompts}"
DECONTAMINATION_CACHE_DIR="${DECONTAMINATION_CACHE_DIR:-/capstor/store/cscs/swissai/infra01/posttrain_data/decontamination_cache}"
TOKENIZER_NAME="${TOKENIZER_NAME:-swiss-ai/Apertus-8B-Instruct-2509}"

# Optional #SBATCH lines (empty when the setting is empty)
SBATCH_QOS_LINE=""
[ -n "$SLURM_QOS" ] && SBATCH_QOS_LINE="#SBATCH --qos=${SLURM_QOS}"
SBATCH_RESERVATION_LINE=""
[ -n "$SLURM_RESERVATION" ] && SBATCH_RESERVATION_LINE="#SBATCH --reservation=${SLURM_RESERVATION}"

check_slurm_config() {
    if [ ! -f "$PYTHON_ENV_ACTIVATE" ]; then
        echo "Error: Python environment not found: $PYTHON_ENV_ACTIVATE (set PYTHON_ENV_ACTIVATE)"
        exit 1
    fi
    if [ ! -d "$DECONTAMINATION_PROMPTS" ]; then
        echo "Error: Decontamination prompts not found: $DECONTAMINATION_PROMPTS"
        echo "Run gather-decontamination-prompts.py first to create benchmark prompts"
        exit 1
    fi
    mkdir -p "$SLURM_LOG_DIR" "$DECONTAMINATION_CACHE_DIR"
    echo "SLURM: account=$SLURM_ACCOUNT partition=$SLURM_PARTITION qos=${SLURM_QOS:-<default>} reservation=${SLURM_RESERVATION:-<none>}"
    echo "Logs:  $SLURM_LOG_DIR"
    echo "Env:   $PYTHON_ENV_ACTIVATE"
}

# Slurm settings for submit-parallel-decontamination.sh (sourced, not executed). The tokenizer, the paths
# of the prompts and the cache, and the matching settings are in decontamination_settings.py.
# Every value can be overridden from the environment, e.g.
#   SLURM_PARTITION=debug SLURM_QOS=debug-qos ./04-decontamination/submit-parallel-decontamination.sh <in> <out> single
# HF_HOME is not set here: jobs inherit it from the submitting shell.
# Requires REPO_ROOT to be set by the calling script.

SLURM_ACCOUNT="${SLURM_ACCOUNT:-infra01}"
SLURM_PARTITION="${SLURM_PARTITION:-normal}"
SLURM_QOS="${SLURM_QOS:-normal}"                 # set to "" to use the partition's default QOS
SLURM_RESERVATION="${SLURM_RESERVATION:-}"       # optional
SLURM_LOG_DIR="${SLURM_LOG_DIR:-/iopsstor/scratch/cscs/${USER}/posttraining-data/slurm_logs}"

DECONTAM_TIME_LIMIT="${DECONTAM_TIME_LIMIT:-12:00:00}"   # single-job mode (reports and merge in one job)
PARALLEL_TIME_LIMIT="${PARALLEL_TIME_LIMIT:-8:00:00}"    # each parallel array job
MERGE_TIME_LIMIT="${MERGE_TIME_LIMIT:-2:00:00}"          # merge job

PYTHON_ENV_ACTIVATE="${PYTHON_ENV_ACTIVATE:-${REPO_ROOT}/.venv/bin/activate}"   # uv sync in the repo root

# Optional #SBATCH lines (empty when the setting is empty)
SBATCH_QOS_LINE=""
[ -n "$SLURM_QOS" ] && SBATCH_QOS_LINE="#SBATCH --qos=${SLURM_QOS}"
SBATCH_RESERVATION_LINE=""
[ -n "$SLURM_RESERVATION" ] && SBATCH_RESERVATION_LINE="#SBATCH --reservation=${SLURM_RESERVATION}"

check_slurm_config() {
    if [ ! -f "$PYTHON_ENV_ACTIVATE" ]; then
        echo "Error: Python environment not found: $PYTHON_ENV_ACTIVATE (run uv sync in ${REPO_ROOT} or set PYTHON_ENV_ACTIVATE)"
        exit 1
    fi
    mkdir -p "$SLURM_LOG_DIR"
    echo "SLURM: account=$SLURM_ACCOUNT partition=$SLURM_PARTITION qos=${SLURM_QOS:-<default>} reservation=${SLURM_RESERVATION:-<none>}"
    echo "Logs:  $SLURM_LOG_DIR"
    echo "Env:   $PYTHON_ENV_ACTIVATE"
}

#!/usr/bin/env bash
#SBATCH --job-name=sft-qwen3-judge
#SBATCH --partition=mi300
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=18
#SBATCH --time=01:00:00
#SBATCH --output=slurm_logs/%x-%j.out

set -euo pipefail
ulimit -c 0

: "${MIXTURE_INPUT:?set MIXTURE_INPUT to an absolute JSON or JSONL input path}"
: "${MIXTURE_OUTPUT:?set MIXTURE_OUTPUT to a new absolute output directory}"

if (( SLURM_JOB_NUM_NODES < 1 || SLURM_JOB_NUM_NODES > 4 )); then
    echo "This experiment supports between one and four Beverin MI300A nodes." >&2
    exit 2
fi

export REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:?submit this job from the repository root}}"
export BEVERIN_ENV_FILE="${BEVERIN_ENV_FILE:-${REPO_ROOT}/05-annotations/slurm/beverin_sglang_rocm.toml}"
RANK_LAUNCHER="${REPO_ROOT}/05-annotations/slurm/run_beverin_hf_judge_rank.sh"
if [[ ! -f "${BEVERIN_ENV_FILE}" || ! -f "${RANK_LAUNCHER}" ]]; then
    echo "Could not find the Beverin EDF or rank launcher under REPO_ROOT=${REPO_ROOT}." >&2
    exit 2
fi

if [[ -z "${HF_HOME:-}" ]]; then
    export HF_HOME="/capstor/scratch/cscs/${USER}/huggingface"
fi
mkdir -p "${HF_HOME}" "${MIXTURE_OUTPUT}"

export JUDGE_MODEL_ID="Qwen/Qwen3-14B"
export JUDGE_MODEL_REVISION="40c069824f4251a91eefaf281ebe4c544efd3e18"
export TOKENIZERS_PARALLELISM=false
export HF_HUB_DISABLE_TELEMETRY=1
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-18}"

echo "Preloading ${JUDGE_MODEL_ID}@${JUDGE_MODEL_REVISION} into ${HF_HOME}"
srun \
    --nodes=1 \
    --ntasks=1 \
    --gpus-per-task=1 \
    --cpus-per-task=1 \
    --environment="${BEVERIN_ENV_FILE}" \
    python -c 'import os; from huggingface_hub import snapshot_download; snapshot_download(repo_id=os.environ["JUDGE_MODEL_ID"], revision=os.environ["JUDGE_MODEL_REVISION"])'
export HF_HUB_OFFLINE=1

echo "Launching ${SLURM_NTASKS} independent ROCm model replicas (${SLURM_JOB_NUM_NODES} node(s), four MI300A APUs per node)."
srun \
    --cpu-bind=cores \
    --environment="${BEVERIN_ENV_FILE}" \
    bash "${RANK_LAUNCHER}"

#!/usr/bin/env bash
#SBATCH --job-name=sft-phi4-judge
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=72
#SBATCH --time=01:00:00
#SBATCH --output=slurm_logs/%x-%j.out

set -euo pipefail
ulimit -c 0

: "${MIXTURE_INPUT:?set MIXTURE_INPUT to an absolute JSON or JSONL input path}"
: "${MIXTURE_OUTPUT:?set MIXTURE_OUTPUT to a new absolute output directory}"

if (( SLURM_JOB_NUM_NODES != 1 )); then
    echo "The Phi-4 comparison preset uses exactly one Clariden node (four GH200 GPUs)." >&2
    exit 2
fi

if [[ -z "${HF_HOME:-}" ]]; then
    : "${SCRATCH:?set SCRATCH or HF_HOME for the shared Hugging Face cache}"
    export HF_HOME="${SCRATCH}/huggingface"
fi
mkdir -p "${HF_HOME}" "${MIXTURE_OUTPUT}"

export JUDGE_MODEL_ID="microsoft/phi-4"
export JUDGE_MODEL_REVISION="2db69c1c3e91a05d2c64a3185acfbaf36f744e25"
export JUDGE_MAX_INPUT_TOKENS="${JUDGE_MAX_INPUT_TOKENS:-8192}"
export JUDGE_MAX_OUTPUT_TOKENS="${JUDGE_MAX_OUTPUT_TOKENS:-384}"
export JUDGE_BATCH_SIZE="${JUDGE_BATCH_SIZE:-2}"

# Keep the batch shell on system libraries. The immutable uenv is applied only
# to srun steps so the site Slurm client continues to use host libraries.
export CLARIDEN_UENV_LABEL="pytorch/v2.9.1:v2@clariden%gh200"
export CLARIDEN_UENV_IMAGE_ID="c05f143d5fbf092714650697c821ee9fcf3d00c3580d56dabeacb2370a824b0f"
export CLARIDEN_UENV_SQUASHFS_SHA256="b9cb3fd2e59c4e71643c1505d2f57b31438ef4b3843c32972a3534160d116b00"
readonly CLARIDEN_UENV_LABEL CLARIDEN_UENV_IMAGE_ID CLARIDEN_UENV_SQUASHFS_SHA256

uenv_step=(
    /usr/bin/srun
    --uenv="${CLARIDEN_UENV_IMAGE_ID}:/user-environment"
    --no-default-view
    --view=default
)

export TOKENIZERS_PARALLELISM=false
export HF_HUB_DISABLE_TELEMETRY=1
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-72}"

export REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:?submit from the repository root}}"
export MIXTURE_TAXONOMY="${MIXTURE_TAXONOMY:-${REPO_ROOT}/05-annotations/taxonomies/swe_terminal_tool_use_v0.2.0.yaml}"
rank_launcher="${REPO_ROOT}/05-annotations/slurm/run_clariden_hf_judge_rank.sh"
if [[ ! -f "${rank_launcher}" || ! -f "${MIXTURE_TAXONOMY}" ]]; then
    echo "Could not find the rank launcher or compact taxonomy under REPO_ROOT=${REPO_ROOT}." >&2
    exit 2
fi

echo "Clariden uenv: ${CLARIDEN_UENV_LABEL}"
echo "Taxonomy: ${MIXTURE_TAXONOMY}"
echo "Preloading ${JUDGE_MODEL_ID}@${JUDGE_MODEL_REVISION} into ${HF_HOME}"
"${uenv_step[@]}" \
    --nodes=1 \
    --ntasks=1 \
    --gpus-per-task=1 \
    --cpus-per-task=1 \
    python -c 'import os; from huggingface_hub import snapshot_download; snapshot_download(repo_id=os.environ["JUDGE_MODEL_ID"], revision=os.environ["JUDGE_MODEL_REVISION"])'
export HF_HUB_OFFLINE=1

echo "Launching ${SLURM_NTASKS} independent Phi-4 replicas."
"${uenv_step[@]}" \
    --cpu-bind=cores \
    bash "${rank_launcher}"

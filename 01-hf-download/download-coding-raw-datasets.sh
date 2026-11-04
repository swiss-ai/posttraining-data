#!/usr/bin/env bash
# Submit one Slurm job per dataset that has to be downloaded as raw files with
# hf-download-raw.py, on the beverin cluster. These datasets cannot go through
# load_dataset (hf-download.py): their files have different columns, their tool-call
# schemas differ between rows, or the repo's PNG files make the loader build an
# image dataset. The files are copied exactly as they are on the Hub, checked
# against the Hub's file sizes, and recorded in dataset_metadata.json together
# with the commit hash.
#
# Usage:
#   ./download-coding-raw-datasets.sh [--dry-run] [DOWNLOAD_FOLDER]
#
# DOWNLOAD_FOLDER defaults to the shared SWE/terminal SFT folder below. Entries that
# already have a dataset_metadata.json in their folder are skipped, so the script
# can be re-run to resubmit only what is missing or failed.
#
# Environment overrides:
#   SLURM_ACCOUNT    (default infra01)
#   SLURM_PARTITION  (default mi300)
#   SLURM_TIME       (default 24:00:00, the partition maximum)
#   CONDA_ROOT       (default /users/kpitas/beverin/miniconda3)
#   CONDA_ENV        (default hf_download; only needs huggingface_hub)
#   LOG_DIR          (default <this folder>/slurm_logs, git-ignored)
#   HF_HOME          (default <DOWNLOAD_FOLDER>/.hf_home, keeps the Hub cache off $HOME)
#   HF_TOKEN_FILE    (default ~/.huggingface_token; exported as HF_TOKEN inside each job)

set -euo pipefail

DRY_RUN=0
if [[ "${1:-}" == "--dry-run" ]]; then
    DRY_RUN=1
    shift
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DOWNLOAD_FOLDER="${1:-/capstor/store/cscs/swissai/infra01/datasets/swe_terminal_sft_apertus_v2}"
HF_DOWNLOAD_RAW="$SCRIPT_DIR/hf-download-raw.py"

SLURM_ACCOUNT="${SLURM_ACCOUNT:-infra01}"
SLURM_PARTITION="${SLURM_PARTITION:-mi300}"
SLURM_TIME="${SLURM_TIME:-24:00:00}"
CONDA_ROOT="${CONDA_ROOT:-/users/kpitas/beverin/miniconda3}"
CONDA_ENV="${CONDA_ENV:-hf_download}"
LOG_DIR="${LOG_DIR:-$SCRIPT_DIR/slurm_logs}"
HF_HOME="${HF_HOME:-$DOWNLOAD_FOLDER/.hf_home}"
HF_TOKEN_FILE="${HF_TOKEN_FILE:-$HOME/.huggingface_token}"

if [[ -s "$HF_TOKEN_FILE" ]]; then
    echo "Using Hugging Face token from $HF_TOKEN_FILE (read by each job at run time)"
else
    echo "Warning: no token found at $HF_TOKEN_FILE; downloads will be unauthenticated and rate-limited" >&2
    HF_TOKEN_FILE=""
fi

# One entry per download: "dataset|subset|glob patterns of the files to copy".
# The subset (may be empty) only names the folder: <name> or <name>-<subset>.
ENTRIES=(
    # jsonl files with different columns cannot share one split
    "nvidia/Nemotron-Competitive-Programming-v1||data/*.jsonl"
    "nvidia/Nemotron-SFT-Competitive-Programming-v2||data/*.jsonl"
    "nvidia/Nemotron-SFT-SWE-v2||data/*.jsonl"

    # The repo's PNG files make the loader build an image dataset; take only the data file
    "codefuse-ai/CodeExercise-Python-27k||CodeExercise-Python-27k.json"

    # Irregular nested columns: pyarrow cannot convert them in one chunk
    "nvidia/Nemotron-Terminal-Corpus|dataset_adapters|dataset_adapters/*"

    # The trajectories are in tar shards that load_dataset does not read; the catalogs
    # come along. Convert with: pipeline.py atif --input <folder> --output <jsonl folder>
    "Zhongzhi1228/Recursive-Task-Synthesis-Trajectories||*"
    "open-athena/recursive-task-synthesis-glm-5.3-rollouts||*"

    # Tool-call argument schemas differ between rows, so load_dataset cannot cast them
    "openbmb/UltraData-SFT-Agent-2609|Code-Agent|data/Code_Agent/*.jsonl"
    "openbmb/UltraData-SFT-Agent-2609|General-Agent|data/General_Agent/*.jsonl"
)

mkdir -p "$DOWNLOAD_FOLDER" "$LOG_DIR"
JOBS_LOG="$LOG_DIR/submitted-jobs.txt"

# Names of this user's queued/running jobs, so they are not submitted twice.
queued_jobs="$(squeue -u "$USER" -h -o %j 2>/dev/null || true)"

submitted=0
skipped=0
failed=()
total=${#ENTRIES[@]}
i=0

for entry in "${ENTRIES[@]}"; do
    i=$((i + 1))
    IFS='|' read -r ds subset patterns <<<"$entry"
    short="${ds##*/}"

    folder="$short"
    label="$ds"
    subset_arg=""
    if [[ -n "$subset" ]]; then
        folder="$short-$subset"
        label="$ds ($subset)"
        subset_arg="--subset $subset"
    fi

    if [[ -f "$DOWNLOAD_FOLDER/$folder/dataset_metadata.json" ]]; then
        echo "[$i/$total] Skipping $label (already downloaded)"
        skipped=$((skipped + 1))
        continue
    fi
    if grep -qxF "hfdl_${folder}" <<<"$queued_jobs"; then
        echo "[$i/$total] Skipping $label (job already queued or running)"
        skipped=$((skipped + 1))
        continue
    fi

    # The token is read inside the job, so it never appears in the job script.
    token_line=""
    if [[ -n "$HF_TOKEN_FILE" ]]; then
        token_line="export HF_TOKEN=\"\$(cat ${HF_TOKEN_FILE})\""
    fi

    job_script=$(cat <<JOB
#!/bin/bash
#SBATCH --job-name=hfdl_${folder}
#SBATCH --output=${LOG_DIR}/${folder}.out
#SBATCH --error=${LOG_DIR}/${folder}.err
#SBATCH --time=${SLURM_TIME}
#SBATCH --account=${SLURM_ACCOUNT}
#SBATCH --partition=${SLURM_PARTITION}

set -euo pipefail

# Jobs inherit the submitting shell's environment, which can make
# \`conda activate\` a no-op, so call the env's interpreter explicitly.
source ${CONDA_ROOT}/etc/profile.d/conda.sh
conda activate ${CONDA_ENV}
PY="${CONDA_ROOT}/envs/${CONDA_ENV}/bin/python"

export HF_HOME="${HF_HOME}"
${token_line}

\$PY "${HF_DOWNLOAD_RAW}" "${ds}" --download-folder "${DOWNLOAD_FOLDER}" ${subset_arg} --allow-patterns '${patterns}'
JOB
)

    if (( DRY_RUN )); then
        echo "[$i/$total] DRY RUN: would submit $label ($patterns) -> $DOWNLOAD_FOLDER/$folder"
        continue
    fi

    if job_id=$(sbatch --parsable <<<"$job_script"); then
        echo "[$i/$total] Submitted $label as job $job_id"
        echo "$job_id $label" >> "$JOBS_LOG"
        submitted=$((submitted + 1))
    else
        echo "!! sbatch failed for $label"
        failed+=("$label")
    fi
done

echo
echo "Done: $submitted submitted, $skipped skipped, ${#failed[@]} failed to submit."
echo "Downloads: $DOWNLOAD_FOLDER"
echo "Logs: $LOG_DIR (job ids in $JOBS_LOG)"
if (( ${#failed[@]} > 0 )); then
    printf '  %s\n' "${failed[@]}"
    exit 1
fi

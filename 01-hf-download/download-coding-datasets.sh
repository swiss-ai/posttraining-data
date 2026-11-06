#!/usr/bin/env bash
# Submit one Slurm job per dataset on the beverin cluster. Each job runs
# hf-download.py in the `posttraining_data` conda env.
#
# Usage:
#   ./download-coding-datasets.sh [--dry-run] [DOWNLOAD_FOLDER]
#
# DOWNLOAD_FOLDER defaults to the shared SWE/terminal SFT folder below. Datasets
# that already have a dataset_metadata.json in their folder are skipped, so the
# script can be re-run to resubmit only what is missing or failed.
#
# Environment overrides:
#   SLURM_ACCOUNT    (default infra01)
#   SLURM_PARTITION  (default mi300)
#   SLURM_TIME       (default 24:00:00, the partition maximum)
#   CONDA_ROOT       (default /users/kpitas/beverin/miniconda3)
#   CONDA_ENV        (default posttraining_data; needs datasets==3.3.2)
#   LOG_DIR          (default <this folder>/slurm_logs, git-ignored)
#   HF_HOME          (default <DOWNLOAD_FOLDER>/.hf_home, keeps the Hub cache off $HOME)
#   HF_TOKEN_FILE    (default ~/.huggingface_token; exported as HF_TOKEN inside each job)
#
# Datasets that load_dataset cannot build are in download-coding-raw-datasets.sh.
# bigcode/commitpackft (277 language configs) has its own script, hf-download-commitpackft.py.
#
# Datasets with several configs (e.g. rStar-Coder,
# Mixture-of-Thoughts) may fail without --subset; add an entry to SUBSETS below.

set -euo pipefail

DRY_RUN=0
if [[ "${1:-}" == "--dry-run" ]]; then
    DRY_RUN=1
    shift
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DOWNLOAD_FOLDER="${1:-/capstor/store/cscs/swissai/infra01/datasets/swe_terminal_sft_apertus_v2}"
HF_DOWNLOAD="$SCRIPT_DIR/hf-download.py"

SLURM_ACCOUNT="${SLURM_ACCOUNT:-infra01}"
SLURM_PARTITION="${SLURM_PARTITION:-mi300}"
SLURM_TIME="${SLURM_TIME:-24:00:00}"
CONDA_ROOT="${CONDA_ROOT:-/users/kpitas/beverin/miniconda3}"
CONDA_ENV="${CONDA_ENV:-posttraining_data}"
LOG_DIR="${LOG_DIR:-$SCRIPT_DIR/slurm_logs}"
HF_HOME="${HF_HOME:-$DOWNLOAD_FOLDER/.hf_home}"
HF_TOKEN_FILE="${HF_TOKEN_FILE:-$HOME/.huggingface_token}"

if [[ -s "$HF_TOKEN_FILE" ]]; then
    echo "Using Hugging Face token from $HF_TOKEN_FILE (read by each job at run time)"
else
    echo "Warning: no token found at $HF_TOKEN_FILE; downloads will be unauthenticated and rate-limited" >&2
    HF_TOKEN_FILE=""
fi

# Subsets to download per dataset (space separated). Each subset becomes its own
# job and its own dataset folder: <DOWNLOAD_FOLDER>/<name>-<subset>.
declare -A SUBSETS=(
    ["microsoft/rStar-Coder"]="synthetic_sft seed_sft"
    ["nvidia/OpenCodeReasoning"]="split_0"
    ["OpenCoder-LLM/opc-sft-stage1"]="filtered_infinity_instruct largescale_diverse_instruct realuser_instruct"
    ["OpenCoder-LLM/opc-sft-stage2"]="educational_instruct evol_instruct mceval_instruct package_instruct"
    ["GAIR/daVinci-Dev"]="llm_enhanced_prs"
    ["openbmb/UltraData-SFT-2605"]="Code"
    ["nvidia/Nemotron-Terminal-Corpus"]="skill_based_easy skill_based_medium skill_based_mixed"
    ["ai-and/termgrade-trajectories"]="trajectories_messages"
)

# Datasets whose subsets are saved side by side as <name>/<subset> (one folder,
# one job per subset) instead of <name>-<subset>.
declare -A NESTED=(
    ["ai-and/termgrade-trajectories"]=1
)

# Conda env per dataset (or "dataset:subset") when the default env does not work.
# hf_download has datasets>=4, which reads the `Json` feature type.
declare -A ENV_FOR=(
    # Json feature type (unknown to datasets 3.3.2)
    ["nvidia/Nemotron-SFT-CUDA-v1"]="hf_download"
    ["nvidia/Nemotron-SFT-SWE-v3.5"]="hf_download"
    # Irregular nested columns / mixed JSON schemas: newer datasets and pyarrow
    ["nvidia/Nemotron-SFT-OpenCode-v1"]="hf_download"
    ["ukisai/Qwen3.8-27B-multi-turn-agent-sft"]="hf_download"
)

# Datasets whose loading script has to be run (needs the default datasets 3.x env).
declare -A TRUST_REMOTE_CODE=(
)

# Datasets that are known to fail with hf-download.py and are not submitted.
# Remove an entry once it is fixed.
declare -A SKIP=(
    # Individual subsets can be keyed "dataset:subset"
)

DATASETS=(
    # --- Code SFT ---
    ajibawa-2023/Code-290k-ShareGPT
    b-mc2/sql-create-context
    bigcode/self-oss-instruct-sc2-exec-filter-50k
    CIIRC-NLP/alquistcoder2025_SFT_dataset
    CodeResearch/Code-Evol-Instruct-OSS
    glaiveai/glaive-code-assistant-v2
    gretelai/synthetic_text_to_sql
    hkust-nlp/CodeIO-PyEdu-Reasoning
    IIGroup/X-Coder-SFT-376k
    inclusionAI/Ling-Coder-SFT
    ise-uiuc/Magicoder-Evol-Instruct-110K
    ise-uiuc/Magicoder-OSS-Instruct-75K
    m-a-p/CodeFeedback-Filtered-Instruction
    microsoft/rStar-Coder
    Modotte/CodeX-2M-Thinking
    Modotte/CodeX-7M-Non-Thinking
    nvidia/AceReason-1.1-SFT
    nvidia/OpenCodeInstruct
    nvidia/OpenCodeReasoning
    nvidia/OpenCodeReasoning-2
    open-thoughts/OpenThoughts3-1.2M
    openbmb/UltraData-SFT-2605
    OpenCoder-LLM/opc-sft-stage1
    OpenCoder-LLM/opc-sft-stage2
    PrimeIntellect/SYNTHETIC-1-SFT-Data
    PrimeIntellect/SYNTHETIC-2-SFT-verified
    tudor-iustin22/maple

    # --- SWE / agentic coding ---
    nvidia/Nemotron-Cascade-SFT-SWE
    open-thoughts/OpenThoughts-Agent-SFT-ColdStartForRL-10K
    open-thoughts/OpenThoughts-Agent-v1-SFT
    ukisai/Qwen3.8-27B-multi-turn-agent-sft
    allenai/Sera-4.5A-Full-T1
    allenai/Sera-4.5A-Full-T2
    allenai/Sera-4.6-Lite-T1
    allenai/Sera-4.6-Lite-T2
    GAIR/daVinci-Dev
    Lego-X/SWE-Lego-Real-Data
    Lego-X/SWE-Lego-Synthetic-Data
    nvidia/Nemotron-SFT-CUDA-v1
    nvidia/Nemotron-SFT-OpenCode-v1
    nvidia/Nemotron-SFT-SWE-v3
    nvidia/Nemotron-SFT-SWE-v3.5
    nvidia/Nemotron-SWE-v1
    nvidia/Open-SWE-Traces
    nvidia/SWE-Hero-openhands-trajectories
    nvidia/SWE-Zero-openhands-trajectories
    TIGER-Lab/SWE-Next-SFT-Trajectories

    # --- Terminal / agent trajectories ---
    Lego-X/Terminal-Lego-Traj-Deepseek-V3-2-15k
    Lite-Coder/LiteCoder-Terminal-SFT
    nvidia/Nemotron-Terminal-Corpus
    open-thoughts/AgentTrove
    open-thoughts/OpenThoughts-Agent-SFT-100K
    ai-and/termgrade-trajectories
    shunzou05/TerminalHorizon-3K
    m-a-p/TerminalTraj
)

mkdir -p "$DOWNLOAD_FOLDER" "$LOG_DIR"
JOBS_LOG="$LOG_DIR/submitted-jobs.txt"

# Names of this user's queued/running jobs, so they are not submitted twice.
queued_jobs="$(squeue -u "$USER" -h -o %j 2>/dev/null || true)"

submitted=0
skipped=0
failed=()
skipped_known=()
total=${#DATASETS[@]}
i=0

for ds in "${DATASETS[@]}"; do
    i=$((i + 1))
    short="${ds##*/}"

    if [[ -n "${SKIP[$ds]:-}" ]]; then
        skipped_known+=("$ds: ${SKIP[$ds]}")
        continue
    fi

    # One job per subset; a dataset without subsets gets a single job.
    read -r -a subsets <<<"${SUBSETS[$ds]:-}"
    if (( ${#subsets[@]} == 0 )); then
        subsets=("")
    fi

    for subset in "${subsets[@]}"; do
        folder="$short"
        jobname="$short"
        subset_arg=""
        label="$ds"
        if [[ -n "$subset" ]]; then
            jobname="$short-$subset"
            folder="$short-$subset"
            subset_arg="--subset $subset"
            label="$ds ($subset)"
            if [[ -n "${NESTED[$ds]:-}" ]]; then
                folder="$short/$subset"
                subset_arg="--subset $subset --nest-subset"
            fi
        fi

        if [[ -n "$subset" && -n "${SKIP[$ds:$subset]:-}" ]]; then
            skipped_known+=("$label: ${SKIP[$ds:$subset]}")
            continue
        fi
        if [[ -f "$DOWNLOAD_FOLDER/$folder/dataset_metadata.json" ]]; then
            echo "[$i/$total] Skipping $label (already downloaded)"
            skipped=$((skipped + 1))
            continue
        fi
        if grep -qxF "hfdl_${jobname}" <<<"$queued_jobs"; then
            echo "[$i/$total] Skipping $label (job already queued or running)"
            skipped=$((skipped + 1))
            continue
        fi

        env_name="${ENV_FOR[$ds:$subset]:-${ENV_FOR[$ds]:-$CONDA_ENV}}"
        trust_arg=""
        if [[ -n "${TRUST_REMOTE_CODE[$ds]:-}" ]]; then
            trust_arg="--trust-remote-code"
        fi

        # The token is read inside the job, so it never appears in the job script.
        token_line=""
        if [[ -n "$HF_TOKEN_FILE" ]]; then
            token_line="export HF_TOKEN=\"\$(cat ${HF_TOKEN_FILE})\""
        fi

        job_script=$(cat <<JOB
#!/bin/bash
#SBATCH --job-name=hfdl_${jobname}
#SBATCH --output=${LOG_DIR}/${jobname}.out
#SBATCH --error=${LOG_DIR}/${jobname}.err
#SBATCH --time=${SLURM_TIME}
#SBATCH --account=${SLURM_ACCOUNT}
#SBATCH --partition=${SLURM_PARTITION}

set -euo pipefail

# Jobs inherit the submitting shell's environment, which can make
# \`conda activate\` a no-op, so call the env's interpreter explicitly.
source ${CONDA_ROOT}/etc/profile.d/conda.sh
conda activate ${env_name}
PY="${CONDA_ROOT}/envs/${env_name}/bin/python"

export HF_HOME="${HF_HOME}"
${token_line}

\$PY "${HF_DOWNLOAD}" "${ds}" --download-folder "${DOWNLOAD_FOLDER}" ${subset_arg} ${trust_arg}
JOB
)

        if (( DRY_RUN )); then
            echo "[$i/$total] DRY RUN: would submit $label -> $DOWNLOAD_FOLDER/$folder"
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
done

echo
echo "Done: $submitted submitted, $skipped skipped, ${#skipped_known[@]} known-bad not submitted, ${#failed[@]} failed to submit."
if (( ${#skipped_known[@]} > 0 )); then
    echo "Not submitted (see SKIP in this script):"
    printf '  %s\n' "${skipped_known[@]}"
fi
echo "Downloads: $DOWNLOAD_FOLDER"
echo "Logs: $LOG_DIR (job ids in $JOBS_LOG)"
if (( ${#failed[@]} > 0 )); then
    printf '  %s\n' "${failed[@]}"
    exit 1
fi

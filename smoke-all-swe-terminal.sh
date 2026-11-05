#!/usr/bin/env bash
# Smoke test of the code preprocessing steps on every downloaded SWE/terminal dataset.
#
# Submits ONE Slurm job per dataset on the beverin cluster. Each job takes the first
# 5 rows of the dataset (of every split, or of every raw file) and runs
#   standardise  ->  dedup + benchmark decontamination
# from examples/code-sft, so you can see which datasets can be processed and which not.
# The job body is smoke-all-swe-terminal-run.py.
#
# Usage:
#   ./smoke-all-swe-terminal.sh [options] [DATASET ...]
#
#   DATASET           dataset folder names under the source folder; shell-style patterns
#                     work, e.g. 'Nemotron-*'. Without any, ALL datasets are submitted.
#   --datasets-file F one dataset name or pattern per line (lines starting with # ignored)
#   --list            list the datasets that can be submitted and exit
#   --dry-run         show what would be submitted, submit nothing
#   --report          print a table of the results so far and exit
#   --force           also resubmit datasets that already have an "ok" summary
#   --include-excluded  also take the datasets in smoke-all-swe-terminal-exclude.txt
#                     (benchmarks and prompt-only sets that are not processing targets)
#   --rows N          rows per sample (default 5)
#   --max-subsets N   sub-datasets sampled per dataset folder, e.g. commitpackft has 277
#                     languages (default 5)
#   --max-files N     raw files sampled per raw dataset (default 8)
#   --category C      auto (default), code_sft or code_agentic. auto looks the dataset up
#                     in the shared reference folder (CATEGORY_REF), else code_sft.
#   --terminal-harness H  version of Terminus-2 trajectories: opencode (default) or terminus
#                     (the source format ported to Apertus 2)
#   --src DIR         downloaded datasets (default below)
#   --out DIR         all samples, intermediate and final artifacts (default below; the
#                     terminus version defaults to smoke_all_swe_terminal_terminus)
#
# Output, per dataset:  <out>/<dataset>/[<subset>/]<sample>/{_samples,native,dedup}/ and
# result.json, plus <out>/<dataset>/summary.json and Slurm logs in <out>/_slurm_logs.
# Datasets with a "ok" summary are skipped on a rerun (use --force), and so are datasets
# whose job is still queued or running.
#
# Environment overrides:
#   SLURM_ACCOUNT (infra01)   SLURM_PARTITION (mi200)   SLURM_TIME (03:00:00)
#   CONDA_ROOT (/users/kpitas/beverin/miniconda3)   CONDA_ENV (code_sft)
#   BENCHMARKS (<repo>/_code_sft/evaluation, built with: pipeline.py benchmarks --output ...)
#   CATEGORY_REF (/capstor/store/cscs/swissai/infra01/datasets/code_posttrain/sft)

set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUNNER="$REPO_DIR/smoke-all-swe-terminal-run.py"

SRC="/capstor/store/cscs/swissai/infra01/datasets/swe_terminal_sft_apertus_v2"
OUT=""
TERMINAL_HARNESS="opencode"
ROWS=5
MAX_SUBSETS=5
MAX_FILES=8
CATEGORY="auto"

SLURM_ACCOUNT="${SLURM_ACCOUNT:-infra01}"
SLURM_PARTITION="${SLURM_PARTITION:-mi200}"
SLURM_TIME="${SLURM_TIME:-03:00:00}"
CONDA_ROOT="${CONDA_ROOT:-/users/kpitas/beverin/miniconda3}"
CONDA_ENV="${CONDA_ENV:-code_sft}"
BENCHMARKS="${BENCHMARKS:-$REPO_DIR/_code_sft/evaluation}"
CATEGORY_REF="${CATEGORY_REF:-/capstor/store/cscs/swissai/infra01/datasets/code_posttrain/sft}"
PY="$CONDA_ROOT/envs/$CONDA_ENV/bin/python"

EXCLUDE_FILE="${EXCLUDE_FILE:-$REPO_DIR/smoke-all-swe-terminal-exclude.txt}"
DRY_RUN=0; LIST=0; REPORT=0; FORCE=0; INCLUDE_EXCLUDED=0
PATTERNS=()

while (( $# )); do
    case "$1" in
        --dry-run) DRY_RUN=1 ;;
        --list) LIST=1 ;;
        --report) REPORT=1 ;;
        --force) FORCE=1 ;;
        --include-excluded) INCLUDE_EXCLUDED=1 ;;
        --rows) ROWS="$2"; shift ;;
        --max-subsets) MAX_SUBSETS="$2"; shift ;;
        --max-files) MAX_FILES="$2"; shift ;;
        --category) CATEGORY="$2"; shift ;;
        --terminal-harness) TERMINAL_HARNESS="$2"; shift ;;
        --src) SRC="$2"; shift ;;
        --out) OUT="$2"; shift ;;
        --datasets-file)
            [[ -f "$2" ]] || { echo "No such file: $2" >&2; exit 1; }
            while IFS= read -r line; do
                line="${line%%#*}"; line="${line//[[:space:]]/}"
                [[ -n "$line" ]] && PATTERNS+=("$line")
            done < "$2"
            shift ;;
        -h|--help) sed -n '2,/^set -euo/p' "${BASH_SOURCE[0]}" | sed '$d' | sed 's/^# \{0,1\}//'; exit 0 ;;
        -*) echo "Unknown option: $1 (see --help)" >&2; exit 1 ;;
        *) PATTERNS+=("$1") ;;
    esac
    shift
done

# The versions have separate outputs and job names, so they can be generated side by side.
case "$TERMINAL_HARNESS" in
    opencode) JOB_PREFIX="smoke_"; OUT="${OUT:-$REPO_DIR/smoke_all_swe_terminal}" ;;
    terminus) JOB_PREFIX="smoke_terminus_"; OUT="${OUT:-$REPO_DIR/smoke_all_swe_terminal_terminus}" ;;
    *) echo "Unknown --terminal-harness: $TERMINAL_HARNESS (opencode or terminus)" >&2; exit 1 ;;
esac

if (( REPORT )); then
    report_args=(--report --out "$OUT" --src "$SRC")
    (( INCLUDE_EXCLUDED )) || report_args+=(--exclude-file "$EXCLUDE_FILE")
    exec "$PY" "$RUNNER" "${report_args[@]}"
fi

# Datasets that can be submitted: folders with a dataset_metadata.json here or one level down.
mapfile -t ALL < <(for d in "$SRC"/*/; do
    n="$(basename "$d")"
    if [[ -f "$d/dataset_metadata.json" ]] || compgen -G "$d/*/dataset_metadata.json" > /dev/null; then
        echo "$n"
    fi
done | sort)

# Datasets that are not processing targets (see the exclude file)
EXCLUDED=()
if [[ -f "$EXCLUDE_FILE" ]]; then
    while IFS= read -r line; do
        line="${line%%#*}"; line="${line//[[:space:]]/}"
        [[ -n "$line" ]] && EXCLUDED+=("$line")
    done < "$EXCLUDE_FILE"
fi
is_excluded() { [[ " ${EXCLUDED[*]:-} " == *" $1 "* ]]; }

TARGETS=()
for n in "${ALL[@]}"; do
    if (( INCLUDE_EXCLUDED )) || ! is_excluded "$n"; then TARGETS+=("$n"); fi
done

if (( LIST )); then
    for n in "${TARGETS[@]}"; do
        if [[ -f "$SRC/$n/dataset_metadata.json" ]]; then
            kind="dataset"
            grep -q '"snapshot_download' "$SRC/$n/dataset_metadata.json" && kind="raw files"
            echo "$n  ($kind)"
        else
            echo "$n  ($(ls -d "$SRC/$n"/*/ | wc -l) sub-datasets)"
        fi
    done
    echo "${#TARGETS[@]} datasets in $SRC"
    if (( ${#TARGETS[@]} < ${#ALL[@]} )); then
        echo "$(( ${#ALL[@]} - ${#TARGETS[@]} )) more are excluded (see $EXCLUDE_FILE, --include-excluded)"
    fi
    exit 0
fi

# Choose the datasets
SELECTED=()
if (( ${#PATTERNS[@]} == 0 )); then
    SELECTED=("${TARGETS[@]}")
else
    for pat in "${PATTERNS[@]}"; do
        matched=0
        for n in "${ALL[@]}"; do
            # shellcheck disable=SC2053
            if [[ "$n" == $pat ]]; then
                matched=1
                if (( ! INCLUDE_EXCLUDED )) && is_excluded "$n"; then
                    echo "Not submitting $n: it is in the exclude file (--include-excluded to submit it)"
                    continue
                fi
                [[ " ${SELECTED[*]:-} " == *" $n "* ]] || SELECTED+=("$n")
            fi
        done
        (( matched )) || { echo "No dataset matches '$pat' (try --list)" >&2; exit 1; }
    done
fi

if [[ ! -d "$BENCHMARKS" ]] || [[ -z "$(ls -A "$BENCHMARKS" 2>/dev/null)" ]]; then
    echo "Benchmark references not found at $BENCHMARKS" >&2
    echo "Build them once with: (cd $REPO_DIR/examples/code-sft && $PY pipeline.py benchmarks --output $BENCHMARKS --workers 16)" >&2
    exit 1
fi
[[ -x "$PY" ]] || { echo "Python not found: $PY" >&2; exit 1; }

mkdir -p "$OUT/_slurm_logs"
JOBS_LOG="$OUT/_slurm_logs/submitted-jobs.txt"
queued_jobs="$(squeue -u "$USER" -h -o %j 2>/dev/null || true)"

echo "Source:      $SRC"
echo "Output:      $OUT"
echo "Terminal:    $TERMINAL_HARNESS"
echo "Benchmarks:  $BENCHMARKS"
echo "Per dataset: $ROWS rows per sample, up to $MAX_SUBSETS sub-datasets and $MAX_FILES raw files"
echo

submitted=0; skipped=0; failed=()
total=${#SELECTED[@]}; i=0
for name in "${SELECTED[@]}"; do
    i=$((i + 1))
    if (( ! FORCE )) && grep -q '"overall": "ok"' "$OUT/$name/summary.json" 2>/dev/null; then
        echo "[$i/$total] Skipping $name (already ok; --force to rerun)"
        skipped=$((skipped + 1)); continue
    fi
    if grep -qxF "${JOB_PREFIX}${name}" <<<"$queued_jobs"; then
        echo "[$i/$total] Skipping $name (job already queued or running)"
        skipped=$((skipped + 1)); continue
    fi

    job_script=$(cat <<JOB
#!/bin/bash
#SBATCH --job-name=${JOB_PREFIX}${name}
#SBATCH --output=${OUT}/_slurm_logs/${name}.out
#SBATCH --error=${OUT}/_slurm_logs/${name}.err
#SBATCH --time=${SLURM_TIME}
#SBATCH --account=${SLURM_ACCOUNT}
#SBATCH --partition=${SLURM_PARTITION}

set -uo pipefail

# The env's interpreter is called explicitly: jobs inherit the submitting shell's environment.
"${PY}" "${RUNNER}" --name "${name}" --src "${SRC}" --out "${OUT}" --rows ${ROWS} --max-subsets ${MAX_SUBSETS} --max-files ${MAX_FILES} --python "${PY}" --benchmarks "${BENCHMARKS}" --category ${CATEGORY} --category-ref "${CATEGORY_REF}" --terminal-harness ${TERMINAL_HARNESS}
JOB
)

    if (( DRY_RUN )); then
        echo "[$i/$total] DRY RUN: would submit $name"
        continue
    fi
    if job_id=$(sbatch --parsable <<<"$job_script"); then
        echo "[$i/$total] Submitted $name as job $job_id"
        echo "$job_id $name" >> "$JOBS_LOG"
        submitted=$((submitted + 1))
    else
        echo "!! sbatch failed for $name"
        failed+=("$name")
    fi
done

echo
echo "Done: $submitted submitted, $skipped skipped, ${#failed[@]} failed to submit."
echo "When the jobs are finished: $0 --report --terminal-harness $TERMINAL_HARNESS"
if (( ${#failed[@]} > 0 )); then
    printf '  %s\n' "${failed[@]}"
    exit 1
fi

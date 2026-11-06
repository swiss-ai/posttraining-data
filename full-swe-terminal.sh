#!/usr/bin/env bash
# Full processing of every downloaded SWE/terminal dataset: all rows, not a sample.
#
# Submits ONE Slurm job per dataset on the beverin cluster, each on a whole node. A job
# standardises every split and file of its dataset with up to 64 processes in parallel, then
# deduplicates and decontaminates that dataset's rows (within the dataset, no fuzzy matching).
# Terminal and SWE-agent trajectories are written as OpenCode sessions. The job body is
# full-swe-terminal-run.py.
#
# Usage:
#   ./full-swe-terminal.sh [options] [DATASET ...]
#
#   DATASET           dataset folder names under the source folder; shell-style patterns
#                     work, e.g. 'Nemotron-*'. Without any, ALL datasets are submitted.
#   --list            list the datasets that can be submitted and exit
#   --dry-run         show what would be submitted, submit nothing
#   --report          print a table of the results so far and exit
#   --force           also resubmit datasets that already have an "ok" summary
#   --include-excluded  also take the datasets in smoke-all-swe-terminal-exclude.txt
#   --category C      auto (default), code_sft or code_agentic
#   --src DIR         downloaded datasets (default below)
#   --out DIR         all artifacts (default: <repo>/all_swe_terminal)
#
# Output, per dataset:  <out>/<dataset>/[<subset>/]{native,dedup,logs}/ and result.json, plus
# <out>/<dataset>/summary.json and Slurm logs in <out>/_slurm_logs. A resubmitted dataset
# continues where it stopped: finished partitions are kept.
#
# Environment overrides:
#   SLURM_ACCOUNT (infra01)   SLURM_PARTITION (mi200)   SLURM_TIME (12:00:00)
#   CONDA_ROOT (/users/kpitas/beverin/miniconda3)   CONDA_ENV (code_sft)
#   BENCHMARKS (<repo>/_code_sft/evaluation)   MAX_WORKERS (64)   GB_PER_WORKER (2; per-dataset
#   values in GB_PER_WORKER_FOR below take precedence)
#   CATEGORY_REF (/capstor/store/cscs/swissai/infra01/datasets/code_posttrain/sft)

set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUNNER="$REPO_DIR/full-swe-terminal-run.py"
SMOKE_RUNNER="$REPO_DIR/smoke-all-swe-terminal-run.py"

SRC="/capstor/store/cscs/swissai/infra01/datasets/swe_terminal_sft_apertus_v2"
OUT="$REPO_DIR/all_swe_terminal"
CATEGORY="auto"

SLURM_ACCOUNT="${SLURM_ACCOUNT:-infra01}"
SLURM_PARTITION="${SLURM_PARTITION:-mi200}"
SLURM_TIME="${SLURM_TIME:-12:00:00}"
CONDA_ROOT="${CONDA_ROOT:-/users/kpitas/beverin/miniconda3}"
CONDA_ENV="${CONDA_ENV:-code_sft}"
BENCHMARKS="${BENCHMARKS:-$REPO_DIR/_code_sft/evaluation}"
MAX_WORKERS="${MAX_WORKERS:-64}"
GB_PER_WORKER="${GB_PER_WORKER:-2}"
CATEGORY_REF="${CATEGORY_REF:-/capstor/store/cscs/swissai/infra01/datasets/code_posttrain/sft}"
PY="$CONDA_ROOT/envs/$CONDA_ENV/bin/python"

# Per-dataset GB_PER_WORKER, replacing the default above for these datasets only.
# Workers = min(MAX_WORKERS, ceil(input GB / GB_PER_WORKER)), all running at once on one node.
# Every row is still processed: a higher value means fewer workers, each taking more rows.
# A worker holds up to --rows-per-shard rows (5000) in memory, about 7-8 GB for long rows,
# so fewer workers keep the node within its 500 GB, at the cost of a longer standardise.
declare -A GB_PER_WORKER_FOR=(
    # Ran out of memory on 2026-11-06 with GB_PER_WORKER=2, also with 5000 rows per shard
    ["Nemotron-SFT-Competitive-Programming-v2"]=6   # 97 GB: 49 -> 17 workers
    ["OpenCodeReasoning-2"]=6                       # 117 GB: 59 -> 20 workers
    ["Open-SWE-Traces"]=6                           # 98 GB: 49 -> 17 workers
)
# Changing a dataset's value changes its number of workers: delete its native/ folder before
# resubmitting it, or standardise refuses to continue the earlier run.

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
        --category) CATEGORY="$2"; shift ;;
        --src) SRC="$2"; shift ;;
        --out) OUT="$2"; shift ;;
        -h|--help) sed -n '2,/^set -euo/p' "${BASH_SOURCE[0]}" | sed '$d' | sed 's/^# \{0,1\}//'; exit 0 ;;
        -*) echo "Unknown option: $1 (see --help)" >&2; exit 1 ;;
        *) PATTERNS+=("$1") ;;
    esac
    shift
done

if (( REPORT )); then
    report_args=(--report --out "$OUT" --src "$SRC")
    (( INCLUDE_EXCLUDED )) || report_args+=(--exclude-file "$EXCLUDE_FILE")
    exec "$PY" "$SMOKE_RUNNER" "${report_args[@]}"
fi

# Datasets: folders with a dataset_metadata.json here or one level down
mapfile -t ALL < <(for d in "$SRC"/*/; do
    n="$(basename "$d")"
    if [[ -f "$d/dataset_metadata.json" ]] || compgen -G "$d/*/dataset_metadata.json" > /dev/null; then
        echo "$n"
    fi
done | sort)

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
    printf '%s\n' "${TARGETS[@]}"
    echo "${#TARGETS[@]} datasets in $SRC ($(( ${#ALL[@]} - ${#TARGETS[@]} )) excluded)"
    exit 0
fi

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
    exit 1
fi
[[ -x "$PY" ]] || { echo "Python not found: $PY" >&2; exit 1; }

mkdir -p "$OUT/_slurm_logs"
JOBS_LOG="$OUT/_slurm_logs/submitted-jobs.txt"
queued_jobs="$(squeue -u "$USER" -h -o %j 2>/dev/null || true)"

echo "Source:      $SRC"
echo "Output:      $OUT"
echo "Benchmarks:  $BENCHMARKS"
echo "Slurm:       $SLURM_PARTITION, $SLURM_TIME per dataset, up to $MAX_WORKERS processes per node"
echo

submitted=0; skipped=0; failed=()
total=${#SELECTED[@]}; i=0
for name in "${SELECTED[@]}"; do
    i=$((i + 1))
    if (( ! FORCE )) && grep -q '"overall": "ok"' "$OUT/$name/summary.json" 2>/dev/null; then
        echo "[$i/$total] Skipping $name (already ok; --force to rerun)"
        skipped=$((skipped + 1)); continue
    fi
    if grep -qxF "full_${name}" <<<"$queued_jobs"; then
        echo "[$i/$total] Skipping $name (job already queued or running)"
        skipped=$((skipped + 1)); continue
    fi
    gb_per_worker="${GB_PER_WORKER_FOR[$name]:-$GB_PER_WORKER}"
    setting="GB_PER_WORKER=$gb_per_worker"
    [[ -n "${GB_PER_WORKER_FOR[$name]:-}" ]] && setting+=" (per-dataset override)"

    job_script=$(cat <<JOB
#!/bin/bash
#SBATCH --job-name=full_${name}
#SBATCH --output=${OUT}/_slurm_logs/${name}.out
#SBATCH --error=${OUT}/_slurm_logs/${name}.err
#SBATCH --time=${SLURM_TIME}
#SBATCH --account=${SLURM_ACCOUNT}
#SBATCH --partition=${SLURM_PARTITION}
#SBATCH --nodes=1
#SBATCH --exclusive

set -uo pipefail

# The env's interpreter is called explicitly: jobs inherit the submitting shell's environment.
"${PY}" "${RUNNER}" --name "${name}" --src "${SRC}" --out "${OUT}" --python "${PY}" --benchmarks "${BENCHMARKS}" --category ${CATEGORY} --category-ref "${CATEGORY_REF}" --max-workers ${MAX_WORKERS} --gb-per-worker ${gb_per_worker}
JOB
)

    if (( DRY_RUN )); then
        echo "[$i/$total] DRY RUN: would submit $name, $setting"
        continue
    fi
    if job_id=$(sbatch --parsable <<<"$job_script"); then
        echo "[$i/$total] Submitted $name as job $job_id, $setting"
        echo "$job_id $name $setting" >> "$JOBS_LOG"
        submitted=$((submitted + 1))
    else
        echo "!! sbatch failed for $name"
        failed+=("$name")
    fi
done

echo
echo "Done: $submitted submitted, $skipped skipped, ${#failed[@]} failed to submit."
echo "When the jobs are finished: $0 --report"
if (( ${#failed[@]} > 0 )); then
    printf '  %s\n' "${failed[@]}"
    exit 1
fi

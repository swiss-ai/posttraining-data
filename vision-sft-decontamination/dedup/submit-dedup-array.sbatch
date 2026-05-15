#!/bin/bash
#SBATCH --job-name=vsft-dedup
#SBATCH --partition=normal
#SBATCH --account=infra01
#SBATCH --nodes=1
#SBATCH --time=0:30:00
#SBATCH --output=/iopsstor/scratch/cscs/schlag/apertus1p5-decontam/dedup/logs/dedup-%A_%a.out

# Per-task: process a contiguous chunk of source parquet shards through
# compute-prompt-hashes.py, writing tiny hash parquets to the dataset's
# shards/ output dir. Environment knobs:
#   SRC_GLOB   - shell glob matching source shards
#   DST_DIR    - destination root for one dataset (shards/ created inside)
#   ARRAY_SIZE - total array size (default: $SLURM_ARRAY_TASK_COUNT)

set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"

: "${SRC_GLOB:?SRC_GLOB required, e.g. /capstor/.../Innovator-VL.../data*/SFT_*.parquet}"
: "${DST_DIR:?DST_DIR required, e.g. /iopsstor/scratch/.../dedup/<dataset>}"
: "${COMPUTE_SCRIPT:=compute-prompt-hashes.py}"

WRAP="$HOME/apertus1p5-decontam/vision-sft-decontam/dedup"
SHARDS_OUT="$DST_DIR/shards"
mkdir -p "$SHARDS_OUT" /iopsstor/scratch/cscs/schlag/apertus1p5-decontam/dedup/logs

# Gather all source shards once per task (cheap on Lustre directory listing).
mapfile -t ALL_SRC < <(ls $SRC_GLOB | sort)
N_SRC=${#ALL_SRC[@]}
N_TASKS=${ARRAY_SIZE:-${SLURM_ARRAY_TASK_COUNT:-1}}
TASK=${SLURM_ARRAY_TASK_ID:-0}

echo "task $TASK / $N_TASKS, total $N_SRC source shards"

# Contiguous chunking: shards [TASK * chunk, (TASK+1) * chunk)
CHUNK=$(( (N_SRC + N_TASKS - 1) / N_TASKS ))
START=$(( TASK * CHUNK ))
END=$(( START + CHUNK ))
if [ "$END" -gt "$N_SRC" ]; then END=$N_SRC; fi
echo "processing shards [$START, $END)"

# uv first call resolves the env; subsequent calls reuse the cache.
# We bundle multiple shards per Python invocation to amortize startup.
for (( i=START; i<END; i++ )); do
    SRC="${ALL_SRC[$i]}"
    # Output naming: avoid collisions across nested subfolders that share
    # filenames. If SRC_ROOT is set, flatten everything under it (handles
    # LLaVA-OneVision's <subdataset>/0.0.0/<shared-hash>/file.arrow layout).
    # Else fall back to parent__base (handles mathnet/by-country etc).
    if [ -n "${SRC_ROOT:-}" ]; then
        REL="${SRC#"$SRC_ROOT"}"
        REL="${REL#/}"
        DST_BASE="${REL//\//__}"
        DST="$SHARDS_OUT/${DST_BASE%.*}.parquet"
    else
        PARENT=$(basename "$(dirname "$SRC")")
        BASE=$(basename "$SRC")
        BASE="${BASE%.*}"
        DST="$SHARDS_OUT/${PARENT}__${BASE}.parquet"
    fi
    if [ -f "$DST" ]; then continue; fi
    uv run --quiet --python 3.11 --with 'pyarrow==21.0.0' --with 'xxhash' --with 'ijson' --with 'datasets==3.3.2' \
        python "$WRAP/$COMPUTE_SCRIPT" "$SRC" "$DST"
done

echo "task $TASK done"

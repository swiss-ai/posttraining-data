#!/bin/bash
#SBATCH --job-name=apertus-test
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --exclusive
#SBATCH --partition=debug
#SBATCH --qos=normal
#SBATCH --time=00:20:00
#SBATCH --output=%x-%j.out
# Submit from apertus2-processing after setup.sh succeeds; set APERTUS_ENCODING.
set -euo pipefail
[[ ${SLURM_JOB_NUM_NODES:-0} == 2 ]] || { echo 'Tests require two nodes.' >&2; exit 2; }
: "${APERTUS_ENCODING:?Set the real Apertus 2 instruct tokenizer directory}"
export UV_PROJECT_ENVIRONMENT=${UV_PROJECT_ENVIRONMENT:-${SCRATCH:?}/apertus2-processing/venv}
export HF_HOME=${HF_HOME:-${SCRATCH:?}/.cache/huggingface}
export RUFF_CACHE_DIR=${SCRATCH:?}/.cache/ruff
export WORKERS=4 THREADS=2
run=${VALIDATION_DIR:-${SCRATCH:?}/apertus2-processing/validation-$SLURM_JOB_ID}
mkdir -p "$run"
container=(--environment="${APERTUS_ENVIRONMENT:-$PWD/slurm/alps.toml}"
  --container-workdir="$PWD" --container-name="apertus-data-$SLURM_JOB_ID")
one() { srun --nodes=1 --ntasks=1 --cpus-per-task=16 --kill-on-bad-exit=1 "${container[@]}" "$@"; }
python=$UV_PROJECT_ENVIRONMENT/bin/python
one "$python" --version
srun --nodes=2 --ntasks=2 --ntasks-per-node=1 --label "${container[@]}" hostname | tee "$run/nodes.txt"
one "$UV_PROJECT_ENVIRONMENT/bin/ruff" check
one "$UV_PROJECT_ENVIRONMENT/bin/ruff" format --check
one "$UV_PROJECT_ENVIRONMENT/bin/pytest" -ra --basetemp="$run/pytest" --junitxml="$run/pytest.xml"
one "$python" tests/cluster_smoke.py prepare "$run"
bash slurm/run.sh check "$run/parquet" "$run/check" --loss-weights --shard-rows 32
status=0
bash slurm/run.sh check "$run/bad" "$run/bad-check" --split validation --column native --shard-rows 1 || status=$?
[[ $status == 1 ]] || { echo "Expected failing check exit 1, got $status" >&2; exit 1; }
bash slurm/run.sh tokenize "$run/parquet" "$run/tokens" --tokenizer "$APERTUS_ENCODING" --loss-weights --shard-rows 32
bash slurm/run.sh tokenize "$run/hf" "$run/hf-tokens" --tokenizer "$APERTUS_ENCODING" --split validation --column native --shard-rows 32
one "$python" tests/cluster_smoke.py interrupt "$run"
bash slurm/run.sh tokenize "$run/parquet" "$run/tokens" --tokenizer "$APERTUS_ENCODING" --loss-weights --shard-rows 32 2>&1 | tee "$run/resume.log"
one "$python" tests/cluster_smoke.py resumed "$run"
bash slurm/run.sh tokenize "$run/parquet" "$run/tokens" --tokenizer "$APERTUS_ENCODING" --loss-weights --shard-rows 32 2>&1 | tee "$run/noop.log"
one "$python" tests/cluster_smoke.py verify "$run"
echo "Cluster validation passed: $run"

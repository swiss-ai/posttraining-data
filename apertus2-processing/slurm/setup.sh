#!/bin/bash
#SBATCH --job-name=apertus-setup
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --exclusive
#SBATCH --partition=debug
#SBATCH --qos=normal
#SBATCH --time=00:20:00
#SBATCH --output=%x-%j.out
# Submit from apertus2-processing, supplying --account and a scratch --output.
set -euo pipefail
export UV_PROJECT_ENVIRONMENT=${UV_PROJECT_ENVIRONMENT:-${SCRATCH:?}/apertus2-processing/venv}
export UV_CACHE_DIR=${UV_CACHE_DIR:-${SCRATCH:?}/.cache/uv}
export UV_PYTHON_INSTALL_DIR=${UV_PYTHON_INSTALL_DIR:-${SCRATCH:?}/.local/share/uv/python}
uv_bin=${UV_BIN:-$HOME/.local/bin/uv}
# Login-node credential helpers (for example VS Code) may not work on compute nodes.
# A bare mirror avoids forwarding credentials; uv still records the pinned Git URL.
if [[ -n ${APERTUS_COMMON_MIRROR:-} ]]; then
  config_index=${GIT_CONFIG_COUNT:-0}
  export "GIT_CONFIG_KEY_$config_index=url.file://$APERTUS_COMMON_MIRROR.insteadOf"
  export "GIT_CONFIG_VALUE_$config_index=https://github.com/swiss-ai/apertus-common.git"
  export GIT_CONFIG_COUNT=$((config_index + 1))
fi
srun --nodes=1 --ntasks=1 --kill-on-bad-exit=1 \
  --environment="${APERTUS_ENVIRONMENT:-$PWD/slurm/alps.toml}" \
  --container-workdir="$PWD" \
  "$uv_bin" sync --locked --python 3.13

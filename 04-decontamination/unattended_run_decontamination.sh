#!/bin/bash

#SBATCH -J decontamination_run
#SBATCH -t 12:00:00
#SBATCH -A a-infra01-1
#SBATCH --output=sdecontamination_run.out
#SBATCH --nodes 1

# Variables used by the entrypoint script
export PROJECT_ROOT_AT=$HOME/projects/post-training-scripts/04-decontamination
source $PROJECT_ROOT_AT/container-scripts/env-vars.sh $@
export SLURM_ONE_ENTRYPOINT_SCRIPT_PER_NODE=1
export HF_TOKEN_AT=$HOME/.hf-token

export OMP_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false

srun \
  --container-image=/capstor/store/cscs/swissai/infra01/swiss-alignment/container-images/swiss-alignment+apertus-vllm.sqsh \
  --environment="${PROJECT_ROOT_AT}/container-scripts/edf.toml" \
  --container-mounts=\
$PROJECT_ROOT_AT,\
$SCRATCH,\
$SWISS_AI_STORAGE,\
/iopsstor/scratch/cscs/${USER}/projects/swiss-alignment/,\
$WANDB_API_KEY_FILE_AT,\
$HF_TOKEN_AT\
  --container-workdir=$PROJECT_ROOT_AT \
  --no-container-mount-home \
  --no-container-remap-root \
  --no-container-entrypoint \
  --container-writable \
  /opt/template-entrypoints/pre-entrypoint.sh \
  bash -c "\
      bash ${PROJECT_ROOT_AT}/container-scripts/hot-pip-install.sh && \
      bash ${PROJECT_ROOT_AT}/run_decontamination.sh"

exit 0

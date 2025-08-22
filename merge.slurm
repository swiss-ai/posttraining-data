#!/bin/bash
#SBATCH --job-name=merge-mix-8
#SBATCH --container-writable
#SBATCH --environment=/iopsstor/scratch/cscs/schlag/post-training-scripts/mix.toml
#SBATCH --time=6:00:00
#SBATCH --ntasks-per-node=1
#SBATCH --dependency=singleton
#SBATCH --account=a-infra01
#SBATCH --output=mix-8-%j.out
#SBATCH --error=mix-8-%j.err
#SBATCH --partition=normal

cd /iopsstor/scratch/cscs/schlag/post-training-scripts
source venv/bin/activate
python 07-dataset-aggregation/dataset-aggregation-newformat.py \
    --mixture_name apertus-sft-mixture-8 \
    --config_path 07-dataset-aggregation/data-mixtures.yaml \
    --output /capstor/store/cscs/swissai/infra01/posttrain_data/06_sft_mixtures_newformat/apertus-sft-mixture-8-merged
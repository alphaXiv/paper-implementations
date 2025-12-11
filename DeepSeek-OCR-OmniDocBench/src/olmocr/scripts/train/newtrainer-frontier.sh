#!/usr/bin/env bash

#SBATCH -A csc652
#SBATCH -J olmocr-train
#SBATCH -o logs/%j.out
#SBATCH -N 1
#SBATCH -t 02:00:00

module reset
module load PrgEnv-gnu/8.6.0
module load rocm/6.3
module load craype-accel-amd-gfx90a
module load miniforge3/23.11.0-0

# Run in offline mode to make sure it doesn't timeout loading the model
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1

export HF_DATASETS_CACHE="/lustre/orion/csc652/proj-shared/huggingface-shared/datasets"
export HF_HUB_CACHE="/lustre/orion/csc652/proj-shared/huggingface-shared/hub"

# Was getting MIOpen errors with caching, had to disable for now
export MIOPEN_DISABLE_CACHE=1

source activate /lustre/orion/csc652/proj-shared/jakep/conda_env_312_olmocr_train

# Run in offline mode to make sure it doesn't timeout loading the model
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

python -m olmocr.train.train --config olmocr/train/configs/example_config_frontier.yaml
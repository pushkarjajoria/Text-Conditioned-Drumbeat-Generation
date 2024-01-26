#!/bin/bash

# Change directory to the project root
cd ~/Git/BeatBrewer/

# Activate the conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate ddpm

# Load necessary modules
module load cuda/11.4

# Set wandb to offline mode
export WANDB_MODE=offline
echo "Weights and bias env variable = $WANDB_MODE"

# Run the training script
python -m unsupervised_pretraining.main

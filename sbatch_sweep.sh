#!/bin/bash

# The partition we want (short==24 hours, long=7 days)
#SBATCH --partition=short
# One node
#SBATCH -N 1
# One job on that node
#SBATCH -n 1
# Request 4 CPU Cores
#SBATCH -c 8
# Please give me a GPU
#SBATCH --gres=gpu:1
# Ask for memory
#SBATCH --mem=128gb

# Run a python program using our local virtual environment
cd /home/bfmorissette/biofilm-cnn-pipeline
/home/bfmorissette/.local/bin/uv run -- wandb agent brianmorissette-worcester-polytechnic-institute/biofilm-cnn-pipeline-sweep-spinning-disk-v1/stw8nv7p --count 100

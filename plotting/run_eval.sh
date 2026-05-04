#!/bin/bash

# SBATCH (resource allocation):
#SBATCH --job-name=cell_eval
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:30:00
#SBATCH --output=slurm_%j.out

# Environment (activation):
source .venv/bin/activate

# Execution (correct path to script):
python3 plotting/ModelEvals.py
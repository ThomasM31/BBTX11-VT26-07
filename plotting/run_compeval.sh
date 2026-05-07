#!/bin/bash

# SBATCH (resource allocation):
#SBATCH --job-name=comp_eval
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:10:00
#SBATCH --output=slurm_%j.out

# Environment (activation):
cd /data/users/lucasant/kandidatarbete/BBTX11-VT26-07
source .venv/bin/activate

# Execution (correct path to script):
python3 plotting/ComprehensiveModelEval.py
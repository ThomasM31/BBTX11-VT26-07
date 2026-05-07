#!/bin/bash

# SBATCH (resource allocation):
#SBATCH --job-name=hier_clust
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:30:00
#SBATCH --output=slurm_%j.out

# Environment:
source .venv/bin/activate

# Execution:
python3 -u plotting/HierarcClustering.py
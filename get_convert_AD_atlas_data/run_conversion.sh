#!/bin/bash
#SBATCH --job-name=rds_array
#SBATCH --time=00:30:00
#SBATCH --array=0-6
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --output=logs/proc_%a_%A.log  # %a = array index, %A = main job ID

FILES=(
    Excitatory_neurons_set1.rds 
    Excitatory_neurons_set2.rds 
    Excitatory_neurons_set3.rds 
    Oligodendrocytes.rds 
    OPCs.rds 
    Astrocytes.rds 
    Immune_cells.rds
    Vasculature_cells.rds
)


FULL_FILENAME=${FILES[$SLURM_ARRAY_TASK_ID]}
# Strip the .rds extension
CLEAN_NAME=${FULL_FILENAME%.rds}

# Activation and Logging
echo "------------------------------------------------"
echo "Job ID: $SLURM_JOB_ID"
echo "Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Processing File: $FULL_FILENAME"
echo "Argument passed to Python: $CLEAN_NAME"
echo "Started at: $(date)"
echo "------------------------------------------------"

# Activate environment
source ~/miniforge3/bin/activate r_env

# Run python script
python3 convert_rds_to_h5ad.py "$CLEAN_NAME"

# Check exit status
if [ $? -eq 0 ]; then
    echo "------------------------------------------------"
    echo "Success: $CLEAN_NAME finished at $(date)"
else
    echo "------------------------------------------------"
    echo "ERROR: $CLEAN_NAME failed with exit code $?"
fi
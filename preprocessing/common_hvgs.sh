#!/bin/bash
#SBATCH --job-name=hvgs_batch
#SBATCH --partition=short
#SBATCH --time=00:15:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --array=0-1,3-8

# --- LOGGING SETUP ---
# %A = Master Job ID | %a = Array Index ID
# This creates one log file per task (e.g., logs/hvgs_12345_0.out)
#SBATCH --output=logs/hvgs_%A_%a.out
# This captures errors in a separate file for quicker debugging
#SBATCH --error=logs/hvgs_%A_%a.err

# Ensure directory exists
mkdir -p logs

# Define a central status log for the whole array
STATUS_LOG="logs/array_${SLURM_ARRAY_JOB_ID}_summary.log"

# Log Start Time
START_TIME=$(date)
echo "------------------------------------------------------------"
echo "TASK ID: $SLURM_ARRAY_TASK_ID"
echo "STARTING: $START_TIME"
echo "NODE: $SLURM_NODENAME"
echo "------------------------------------------------------------"

# Append a quick status line to the summary log
echo "Task $SLURM_ARRAY_TASK_ID started on $SLURM_NODENAME at $START_TIME" >> $STATUS_LOG

# Run the Python script
# We use 'time' to see how long the Python process actually took
time python3 preprocess_pipeline_pseudobulk_common_hvgs.py $SLURM_ARRAY_TASK_ID

# Capture the exit code of the Python script
EXIT_CODE=$?

# Log Finish Time and Result
END_TIME=$(date)
if [ $EXIT_CODE -eq 0 ]; then
    echo "SUCCESS: Task $SLURM_ARRAY_TASK_ID finished at $END_TIME" >> $STATUS_LOG
else
    echo "FAILURE: Task $SLURM_ARRAY_TASK_ID exited with code $EXIT_CODE at $END_TIME" >> $STATUS_LOG
fi

echo "------------------------------------------------------------"
echo "FINISHED: $END_TIME"
echo "------------------------------------------------------------"
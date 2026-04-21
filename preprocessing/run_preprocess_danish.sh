#!/bin/bash
#SBATCH --job-name=preprocessing_danish
#SBATCH --partition=long
#SBATCH --time=03:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --array=0-8

# --- LOGGING SETUP ---
# %A = Master Job ID | %a = Array Index ID
# This creates one log file per task (e.g., logs/hvgs_12345_0.out)
#SBATCH --output=logs/%A_%a_preproc_danish.out
# This captures errors in a separate file for quicker debugging
#SBATCH --error=logs/%A_%a_preproc_danish.err

# Ensure directory exists
mkdir -p logs

# Define a central status log for the whole array
STATUS_LOG="logs/${SLURM_ARRAY_JOB_ID}_summary.log"

# Log Start Time
START_TIME=$(date)
echo "------------------------------------------------------------"
echo "TASK ID: $SLURM_ARRAY_TASK_ID"
echo "STARTING: $START_TIME"
echo "NODE: $SLURM_NODENAME"
echo "------------------------------------------------------------"

# Append a quick status line to the summary log
echo "Task $SLURM_ARRAY_TASK_ID started on $SLURM_NODENAME at $START_TIME" >> $STATUS_LOG

export PYTHONUNBUFFERED=1

# Run the Python script
# We use 'time' to see how long the Python process actually took
time python3 -u pipeline_preprocess_danish.py $SLURM_ARRAY_TASK_ID

# Capture the exit code of the Python script
EXIT_CODE=$?

# Log Finish Time and Result
END_TIME=$(date)
if [ $EXIT_CODE -eq 0 ]; then
    echo "SUCCESS: Task $SLURM_ARRAY_TASK_ID finished at $END_TIME" >> $STATUS_LOG
else
    echo "FAILURE: Task $SLURM_ARRAY_TASK_ID exited with code $EXIT_CODE at $END_TIME" >> $STATUS_LOG
fi

# print snapshot of error to terminal
if [ $EXIT_CODE -ne 0 ]; then
    echo "--- ERROR LOG SNAPSHOT ---"
    cat "logs/hvgs_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.err"
    echo "--------------------------"
fi

echo "------------------------------------------------------------"
echo "FINISHED: $END_TIME"
echo "------------------------------------------------------------"
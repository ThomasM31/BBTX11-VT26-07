#!/bin/bash
#SBATCH --job-name=create_masks
#SBATCH --partition=long
#SBATCH --time=03:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

# --- LOGGING SETUP ---
# %A = Master Job ID | %a = Array Index ID
# This creates one log file per task (e.g., logs/hvgs_12345_0.out)
#SBATCH --output=logs/masking_%A_%a.out
# This captures errors in a separate file for quicker debugging
#SBATCH --error=logs/masking_%A_%a.err

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

export PYTHONUNBUFFERED=1

INPUT_DIR="/data/shared/alzgene26/data/processed_data/completed/danish"
OUTPUT_DIR="/data/shared/alzgene26/PathwayData/MaskMatrixLayers/danish"
CONNECTIVITY_FILE="/data/shared/alzgene26/PathwayData/binn_connectivity.csv"

# Run the Python script
time python3 -u MaskCreator.py "$INPUT_DIR" "$OUTPUT_DIR" "$CONNECTIVITY_FILE" --mask_label=vasc

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
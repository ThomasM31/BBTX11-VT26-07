#!/bin/bash
#SBATCH --job-name=draw_sankey
#SBATCH --partition=short
#SBATCH --time=00:10:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G


# --- LOGGING SETUP ---
# %A = Master Job ID | %a = Array Index ID
# This creates one log file per task
#SBATCH --output=logs/%A_draw_sankey.out
# This captures errors in a separate file for quicker debugging
#SBATCH --error=logs/%A_draw_sankey.err

# Ensure directory exists
mkdir -p logs

# Define a central status log for the whole array
STATUS_LOG="logs/${SLURM_ARRAY_JOB_ID}_summary.log"

# Log Start Time
START_TIME=$(date)
echo "------------------------------------------------------------"
echo "STARTING: $START_TIME"
echo "NODE: $SLURM_NODENAME"
echo "------------------------------------------------------------"

# Append a quick status line to the summary log
echo "Task started on $SLURM_NODENAME at $START_TIME" >> $STATUS_LOG

export PYTHONUNBUFFERED=1

# Run the Python script
# We use 'time' to see how long the Python process actually took
time python3 -u -m plotting.sankey.plot_sankey

# Capture the exit code of the Python script
EXIT_CODE=$?

# Log Finish Time and Result
END_TIME=$(date)
if [ $EXIT_CODE -eq 0 ]; then
    echo "SUCCESS: Task finished at $END_TIME" >> $STATUS_LOG
else
    echo "FAILURE: Task exited with code $EXIT_CODE at $END_TIME" >> $STATUS_LOG
fi

# print snapshot of error to terminal
if [ $EXIT_CODE -ne 0 ]; then
    echo "--- ERROR LOG SNAPSHOT ---"
    cat "logs/${SLURM_ARRAY_JOB_ID}_draw_sankey.err"
    echo "--------------------------"
fi

echo "------------------------------------------------------------"
echo "FINISHED: $END_TIME"
echo "------------------------------------------------------------"
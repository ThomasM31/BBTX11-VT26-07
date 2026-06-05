#!/bin/bash
#SBATCH --job-name=draw_sankey
#SBATCH --partition=short
#SBATCH --time=00:10:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

# Fix 1: Point directly to the existing nested logs directory
#SBATCH --output=plotting/sankey/logs/%A_draw_sankey.out
#SBATCH --error=plotting/sankey/logs/%A_draw_sankey.err

# Fix 2: Explicitly set the working directory to the project root
#SBATCH --chdir=/home/isanor/kand/proj/BBTX11-VT26-07

# Define a central status log for the whole array
STATUS_LOG="plotting/sankey/logs/${SLURM_ARRAY_JOB_ID}_summary.log"

# Log Start Time
START_TIME=$(date)
echo "------------------------------------------------------------"
echo "STARTING: $START_TIME"
echo "NODE: $SLURM_NODENAME"
echo "------------------------------------------------------------"

echo "Task started on $SLURM_NODENAME at $START_TIME" >> $STATUS_LOG

export PYTHONUNBUFFERED=1
export PYTHONPATH="${PYTHONPATH}:${PWD}"

# Run the Python script
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
    cat "plotting/sankey/logs/${SLURM_ARRAY_JOB_ID}_draw_sankey.err"
    echo "--------------------------"
fi

echo "------------------------------------------------------------"
echo "FINISHED: $END_TIME"
echo "------------------------------------------------------------"
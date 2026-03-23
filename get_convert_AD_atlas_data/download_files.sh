#!/bin/bash
#SBATCH --job-name=download-files
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --output=download_%j.log

echo "Starting downloads..."

# List of URLs, replace with your URL
URLS=(
    "https://personal.broadinstitute.org/cboix/ad427_data/Data/Metadata/individual_metadata_deidentified.tsv"
    "https://personal.broadinstitute.org/cboix/ad427_data/Data/Metadata/Cell_types.csv"
    "https://personal.broadinstitute.org/cboix/ad427_data/Data/Metadata/Inhibitory_neuron_subclass_assignment_PFC.xlsx"
    "https://personal.broadinstitute.org/cboix/ad427_data/Data/Raw_data/batch_mapping_deidentified.tsv"
)

# Loop through and download one by one
for url in "${URLS[@]}"; do
    echo "Downloading: $url"
    wget -c "$url"   # In case of disconnection, mark for restart
done

echo "Finished at $(date)"

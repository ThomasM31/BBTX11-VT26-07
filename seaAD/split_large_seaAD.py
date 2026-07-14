# Inspection of .h5ad file
import scanpy as sc, numpy as np, pandas as pd, anndata as ad

shared_path = "/data/shared/alzgene26/data/seaAD/"
my_path = "/data/users/thomath/kand/data/seaAD/split_data/"

def load_file(data_path: str) -> ad.AnnData:
    """
        Load massive seaAD anndata file 
    """
    # Load .h5ad file from path
    file_path = data_path + "SEAAD_DFC_RNAseq_final-nuclei.2026-06-22.h5ad"

    # ensure that the file was properly saved
    print("Reading full seaAD-data")
    adata = ad.read_h5ad(file_path) # backed=True
    print(f"Successfully read and saved {adata.n_obs} cells")

    return adata

def show_file(adata: ad.AnnData) -> None:
    """
        Give basic information about the input anndata
    """
    # Show information about data file
    print("Data information: ")
    print(adata)

    # Show example
    print("Data example: ")
    print(adata.obs.head())

    # Print all available category names you can use for plotting
    print("Columns: ")
    print(adata.obs.columns)

    # Sparsity Check
    print(f"Sparse AnnData: {adata.n_obs} n_obs x {adata.n_vars} n_vars")
    #sparsity = 1 - adata.X.nnz / (adata.n_obs * adata.n_vars)
    #print(f"Sparsity: {sparsity:.2%}")

def split_file(adata_full: ad.AnnData, split_path: str, n_chunks=4) -> dict: 
    """
        Split large files into parts based on "Donor ID"
    """
    data_sets = {}

    # Fetch unique donors & split into list of chunks
    donors = np.sort(adata_full.obs["Donor ID"].unique())
    donor_chunks = np.array_split(donors, n_chunks)

    # step through chunks of adata to write to .h5ad
    for i, donor_chunk in enumerate(donor_chunks, start=1):
        # Condition on adata
        print(f"Creating donor mask for adata chunk: {i}")
        donor_mask = adata_full.obs["Donor ID"].isin(donor_chunk)
        
        # Create the actual chunk of adata
        print(f"Creating chunk from adata_full ")
        adata_chunk = adata_full[donor_mask].copy()

        # Add to dict
        data_sets[i] = adata_chunk
        
        # Rename problematic columns when writing ('/' not allowed)
        adata_chunk.obs.columns = [c.replace("/", "_") for c in adata_chunk.obs.columns]

        # Write to .h5ad file
        print(f"Writing chunk nr {i} to .h5ad with {len(donor_chunk)} donors") # & {s} cells")
        adata_chunk.write_h5ad(split_path + f"adata_chunk_{i}.h5ad")

    return data_sets

def main():
    adata_full = load_file(data_path=shared_path)
    show_file(adata_full)

    # shared directory path for split data
    split_path = shared_path + "split_data/"
    split_data_sets = split_file(adata_full, split_path, n_chunks=4)

main()
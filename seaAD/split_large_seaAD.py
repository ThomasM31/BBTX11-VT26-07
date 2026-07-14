# Inspection of .h5ad file
import scanpy as sc

shared_path = "/data/shared/alzgene26/data/seaAD/"

def load_file(data_path):
    # Load .h5ad file from path
    file_path = data_path + "SEAAD_DFC_RNAseq_final-nuclei.2026-06-22.h5ad"

    # ensure that the file was properly saved
    print("Reading full seaAD-data")
    adata = sc.read_h5ad(file_path) # backed=True
    print(f"Successfully read and saved {adata.n_obs} cells")

    return adata

def show_file(adata):
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

def split_file(adata_full):

    return

def main():
    adata_full = load_file(data_path=shared_path)
    show_file(adata_full)

    split_file(adata_full)

main()
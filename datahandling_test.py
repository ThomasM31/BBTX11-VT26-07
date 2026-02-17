import scanpy as sc, anndata as ad
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, csc_matrix

#print(sc.__version__)
vasc_path = "Vasculature_cells.h5ad"

def check_ok(f):
    # ensure that the file was properly saved
    test_load = sc.read_h5ad(f)
    print(f"Successfully read and saved {test_load.n_obs} cells")
    return test_load

adata = check_ok(vasc_path)

def show_data(adata):
    print(adata)

    # See cell clusters and batch info
    print("\n + cell clusters and batch info:")
    print(adata.obs.head())

    # List all available metadata columns
    print("\n + metadata cols:")
    print(adata.obs_names)

    # This prints all available category names you can use for plotting
    print(adata.obs.columns.tolist())

    # Check the type and shape
    print("\n + type and shape:")
    print(type(adata.X))
    print(adata.X.shape)

    # View a small slice of raw counts (first 5 cells, first 5 genes)
    # If it's sparse, you might need to use .toarray() to read it
    print("\n + slice of raw counts:")
    print(adata.X[:10, :10].toarray())

    # See gene names and any HVG (Highly Variable Gene) info
    print("\n + gene names, HVG info:")
    print(adata.var.head())

    # List the names of all available reductions
    print("\n + available reductions:")
    print(adata.obsm.keys())

    # View the actual UMAP coordinates
    print("\n + umap coords:")
    print(adata.obsm['X_umap'][:5])

    # This prints all available category names you can use for plotting
    print(adata.obs.columns.tolist())

    # Plot UMAP colored by clusters
    print("\n + plotting umap:")
    sc.pl.umap(adata, color='ident', frameon=False, title="Imported Seurat Clusters")


def check_sparsity(adata):
    print(f"Sparse AnnData: {adata.n_obs} n_obs x {adata.n_vars} n_vars")
    sparsity = 1 - adata.X.nnz / (adata.n_obs * adata.n_vars)
    print(f"Sparsity: {sparsity:.2%}")
    return sparsity

#sparsity = check_sparsity(adata)

def create_layers(adata):
    adata.layers["raw"] = adata.X.copy()
    print(f"Available layers: {list(adata.layers.keys())}")

    return adata

#create_layers(adata)
#show_data(adata)

# BACKED MODE
def read_in_backed(path):
    adata_backed = sc.read_h5ad(path, backed="r")   

    assert adata_backed.isbacked, "Should be in backed mode"

    print(f"Backed mode: {adata_backed.isbacked}")
    return adata_backed

adata_backed = read_in_backed(path=vasc_path)

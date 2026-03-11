import scanpy as sc
import anndata
import os

def inspect_data(adata):
    #### To inspect the data in the converted file

    #Summary
    print(adata)

    # See cell clusters and batch info
    print("cell clusters and batch info:")
    print(adata.obs.head())

    # List all available metadata columns
    print("metadata cols:")
    print(adata.obs_names)

    # See gene names and any HVG (Highly Variable Gene) info
    print("gene names, HVG info:")
    print(adata.var.head())

    # List the names of all available reductions
    print("av. reductions:")
    print(adata.obsm.keys())

    # View the actual UMAP coordinates
    print("umap coords:")
    print(adata.obsm['X_umap'][:5])

    # This prints all available category names you can use for plotting
    print(adata.obs.columns.tolist())

    # Plot UMAP colored by clusters
    print("plotting umap:")
    sc.pl.umap(adata, color='ident', frameon=False, title="Imported Seurat Clusters")

    # Check the type and shape
    print("type and shape:")
    print(type(adata.X))
    print(adata.X.shape)

    # View a small slice of raw counts (first 5 cells, first 5 genes)
    # If it's sparse, you might need to use .toarray() to read it
    print("slice of raw counts:")
    print(adata.X[:5, :5])


def main():
    ### ADD FILE PATH AND NAME HERE
    file = "conv_data/Astrocytes.h5ad"
    
    if os.path.exists(file):
        print("file found, loading...")
        test_load = sc.read_h5ad(file)
        inspect_data(test_load)
    else:
        print("file not found")

main()
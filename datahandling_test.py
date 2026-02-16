import scanpy as sc
#import anndata as ad

#print(sc.__version__)
#print(adata.obs.head())
#print(adata.var.head())

def check_ok(f):
    # ensure that the file was properly saved
    test_load = sc.read_h5ad(f)
    print(f"Successfully saved {test_load.n_obs} cells")
    return test_load

adata = check_ok("Vasculature_cells.h5ad")

print(adata)

# See cell clusters and batch info
print("\n cell clusters and batch info:")
print(adata.obs.head())

# List all available metadata columns
print("\n metadata cols:")
print(adata.obs_names)

# This prints all available category names you can use for plotting
print(adata.obs.columns.tolist())

# Check the type and shape
print("\n type and shape:")
print(type(adata.X))
print(adata.X.shape)

# View a small slice of raw counts (first 5 cells, first 5 genes)
# If it's sparse, you might need to use .toarray() to read it
print("\n + slice of raw counts:")
print(adata.X[:10, :10].toarray())

# See gene names and any HVG (Highly Variable Gene) info
print("\n gene names, HVG info:")
print(adata.var.head())

# List the names of all available reductions
print("\n av. reductions:")
print(adata.obsm.keys())

# View the actual UMAP coordinates
print("\n umap coords:")
print(adata.obsm['X_umap'][:5])

# This prints all available category names you can use for plotting
print(adata.obs.columns.tolist())

# Plot UMAP colored by clusters
print("\n plotting umap:")
sc.pl.umap(adata, color='ident', frameon=False, title="Imported Seurat Clusters")


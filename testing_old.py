import scanpy as sc
#import anndata as ad

#print(sc.__version__)
adata = sc.read_h5ad("df2.h5ad")

#print(adata)

#print(adata.obs.head())
#print(adata.var.head())

mets = sc.pp.calculate_qc_metrics(adata)
print(mets)

#sc.pl.violin(adata, ['n_genes_by_counts', 'total_counts'], multi_panel=True)

#print(adata.var_names)
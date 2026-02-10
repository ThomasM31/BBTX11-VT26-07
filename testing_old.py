import scanpy as sc
#import anndata as ad

#print(sc.__version__)
adata = sc.read_h5ad("df2.h5ad")

#print(adata)

print(adata.obs.head())
#print(adata.var.head())

#sc.pp.calculate_qc_metrics(adata, inplace=True)
#sc.pl.violin(adata, ['n_genes_by_counts', 'total_counts', 'pct_counts_mt'], jitter=0.4, multi_panel=True)

#print(adata.var_names)
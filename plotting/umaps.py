import scanpy as sc
from pathlib import Path
import anndata as ad

def draw_umap(adata, 
              label: str, 
              filepath: Path,
              stage: str, 
              color_by:str) -> None:
    print(f'Drawing umaps for {label} ({stage}, {color_by}).')

    f = filepath / f'{label}_{stage}_{color_by}'
    title = f'{label.capitalize()} ({stage}) colored by "{color_by}".'

    sc.tl.pca(adata, svd_solver='arpack')
    n_pcs = 30 if adata.n_obs > 30 else adata.n_obs - 1
    sc.pp.neighbors(adata, n_neighbors=20, n_pcs=n_pcs)
    sc.tl.umap(adata)
    sc.pl.umap(adata, 
                title=title,
                color=[color_by], 
                show=False,
                legend_loc="upper left").figure.savefig(f)
import scanpy as sc
from pathlib import Path

def draw_umaps_logic(adata_orig, adata_proc, label: str, filepath: Path) -> None:
    print(f'Drawing umaps for {label}')

    fname1 = filepath / f'{label}_orig'
    fname2 = filepath / f'{label}_proc'

    # UMAP for original
    sc.tl.pca(adata_orig, svd_solver='arpack')
    sc.pp.neighbors(adata_orig, n_neighbors=20, n_pcs = 30)
    sc.tl.umap(adata_orig)
    sc.pl.umap(adata_orig, 
                title=f"{label} original",
                color=['cell_type_high_resolution'], 
                show=False,
                legend_loc="upper left").figure.savefig(fname1)

    # UMAP for preprocessed
    sc.tl.pca(adata_proc, svd_solver='arpack')
    sc.pp.neighbors(adata_proc, n_neighbors=20, n_pcs = 30)
    sc.tl.umap(adata_proc)
    sc.pl.umap(adata_proc, 
                title=f"{label} pre-processed",
                color=['cell_type_high_resolution'], 
                show=False,
                legend_loc="upper left").figure.savefig(fname2)
    
    print(f"Drawing completed for {label}")


def draw_umaps(datasets_orig: dict, datasets_proc: dict, filepath: Path) -> None:
    '''
    To draw UMAPS when original data and processed data are in the main 
    layers of two separate ad objects
    '''
    for label in datasets_orig.keys():
        adata_orig = datasets_orig[label]
        adata_proc = datasets_proc[label]
        draw_umaps_logic(adata_orig, adata_proc, label, filepath)
import os
from pathlib import Path
import argparse

import scanpy as sc
import pandas as pd
import numpy as np

import anndata as ad
from anndata.experimental import AnnCollection
from anndata.experimental.pytorch import AnnLoader
ad.settings.allow_write_nullable_strings = True

def get_labels(to_include: list) -> list:
    labels = ['astro', 'exc1', 'exc2', 'exc3', 'immune', 'inhi', 'oligo', 'opcs', 'vasc']
    included_labels = [labels[i] for i in to_include]
    print(f'Labels to include: {included_labels}')

    return included_labels

def get_datasets(included_labels: list) -> dict:
    # inner dict holds the original data, hvg subset, and pseudobatched data
    # keeping all of them helps us compare them later to see the effects of pre-processing
    datasets = {
        label: {'orig': None, 'subset': None, 'pseudo': None} 
        for label in included_labels
    }
    return datasets

def read_files(filepath: str, datasets: dict) -> None:

    files = {'astro'  : 'Astrocytes.h5ad',
             'exc1'   : 'Excitatory_neurons_set1.h5ad',
             'exc2'   : 'Excitatory_neurons_set2.h5ad',
             'exc3'   : 'Excitatory_neurons_set3.h5ad',   
             'immune' : 'Immune_cells.h5ad',   
             'inhi'   : 'Inhibitory_neurons.h5ad',   
             'oligo'  : 'Oligodendrocytes.h5ad',   
             'opcs'   : 'OPCs.h5ad',
             'vasc'   : 'Vasculature_cells.h5ad'}

    p = 'conv_data'
    p = os.path.join(filepath, p)

    for label in list(datasets.keys()):
        print(f'Reading: {label}')
        f = os.path.join(p,files[label])
        datasets[label]['orig'] = ad.read_h5ad(f)

    return datasets
def filter_cells(datasets: dict, min_genes: int=200, min_cells: int=200):
    for label in list(datasets.keys()):
        adata = datasets[label]['orig']
        print(f"AnnData loaded, shape={adata.shape}\n")
        
        print(f"Filtering cells (with <{min_genes} genes) and genes (detected in <{min_cells} cells)...")
        sc.pp.filter_cells(adata, min_genes=min_genes)
        sc.pp.filter_genes(adata, min_cells=min_cells)
        print(f"Filtered data, shape={adata.shape}\n")

        print("Calculating QC metrics")
        adata.var["mt"] = np.array(adata.var_names.str.startswith("MT-"), dtype=bool)
        print("MT genes detected:", adata.var["mt"].sum())

        sc.pp.calculate_qc_metrics(adata, qc_vars=["mt"], percent_top=None, log1p=False, inplace=True)

        print("Dropping cells with >5% mitochondrial counts...")
        datasets[label]['orig'] = adata[adata.obs.pct_counts_mt < 5, :].copy()
        

def prep_for_hvg_sel(datasets: dict) -> None:
    for label in list(datasets.keys()):
        dataset = datasets.get(label)['orig']
        # need to log normalize, some datasets seem to not be properly normalized in the logcounts layer
        # if we just use the logcounts layer it causes crashes when extracting HVGs

        # using the log-norm data just for feature selection, 
        # give raw counts to pseudobatch (don't overwrite data with lognorm)
        dataset.uns['log1p'] = sc.pp.log1p(dataset, copy=True)
        print(f'\nDataset "{label}" after log normalization')
        print(dataset)

        dataset_log = dataset.uns['log1p']
        print(f'min before: {dataset.X.min()}, max before: {dataset.X.max()}')
        print(f'min after: {dataset_log.X.min()}, max after: {dataset_log.X.max():.1f}')


def extract_common_hvgs(datasets: dict, n_top_genes: int = 2000) -> None:
    # Keep only highly variable genes (HVGs)
    hvgs = []

    # extract HVGs (NOTE: expects normalized data)
    for label in list(datasets.keys()):
        print(f'Extracting HVGS from "{label}"')

        # use log1p layer
        adata = datasets[label]['orig'].uns['log1p']

        sc.pp.highly_variable_genes(adata, n_top_genes=2000)
        g = adata.var_names[adata.var.highly_variable]
        hvgs.append(g)

    # find the genes that are HVGs for all datasets
    common_hvgs = set(hvgs[0])
    for i in range(1, len(hvgs)):
        common_hvgs = common_hvgs & set(hvgs[i])
    common_hvgs = list(common_hvgs)

    print(f'Nr common hvgs: {len(common_hvgs)}')

    # keep only HVGs
    for label in list(datasets.keys()):
        datasets[label]['subset'] = datasets[label]['orig'][:, common_hvgs].copy()
        print(f'HVGs for {label}')
        print(datasets[label]['subset'])


def pseudobulk(datasets: dict) -> None:
    # Perform pseudobulk by test subject and high res celltype

    for label in list(datasets.keys()):
        print(f'Pseudobulking "{label}"')

        pseudo = sc.get.aggregate(
            datasets[label]['subset'], 
            by=['subject', 'cell_type_high_resolution'], 
            func='sum'
            )

        # moves pseudobulk to the main layer
        pseudo.X = pseudo.layers['sum'].copy()
        datasets[label]['pseudo'] = pseudo


def normalize(datasets: dict) -> None:
    for label in list(datasets.keys()):
        print(f'Normalizing: "{label}"')
        
        adata = datasets[label]['pseudo']

        # Normalizes counts per pseudobulk sample
        # There are possible options, e.g. exclude highly expressed genes from computation

        # Using Counts per million
        sc.pp.normalize_total(adata, exclude_highly_expressed=False, target_sum=1e6)

        # Especially useful if range is large?
        sc.pp.log1p(adata)

        # Scaling centers genes at 0
        # Helps model learn *relative* expression
        sc.pp.scale(adata)

        print(adata)
        print(f'{label:<8} has min: {adata.X.min():.2f} and max: {adata.X.max():.2f}')


def draw_umaps(datasets: dict) -> None:
    user = os.environ.get('USER') or os.environ.get('USERNAME')
    fig_path = Path("/data/users") / user / "kand/data/figures/"
    # create the path if it doesn't exist
    fig_path.mkdir(parents=True, exist_ok=True)
    fig_path = str(fig_path)
    
    for label in list(datasets.keys()):
        adata = datasets[label]['orig']
        adata_proc = datasets[label]['pseudo']
        print(f'Drawing umaps for {label}')
        
        fname1 = f'{label}_orig'
        fname1 = os.path.join(fig_path,fname1)
        fname2 = f'{label}_proc'
        fname2 = os.path.join(fig_path,fname2)

        # UMAP for original
        sc.tl.pca(adata, svd_solver='arpack')
        sc.pp.neighbors(adata, n_neighbors=20, n_pcs = 30)
        sc.tl.umap(adata)
        sc.pl.umap(adata, 
                   title=f"{label} original",
                   color=['cell_type_high_resolution'], 
                   show=False).figure.savefig(fname1)

        # UMAP for preprocessed
        sc.tl.pca(adata_proc, svd_solver='arpack')
        sc.pp.neighbors(adata_proc, n_neighbors=20, n_pcs = 30)
        sc.tl.umap(adata_proc)
        sc.pl.umap(adata_proc, 
                   title=f"{label} pre-processed",
                   color=['cell_type_high_resolution'], 
                   show=False).figure.savefig(fname2)
        
        print("Drawing completed")


def add_metadata(datasets: dict, filepath: str, is_float=True) -> None:
    # Add subject disease status metadata
    directory = 'supplementary_data/'
    file = 'individual_metadata_deidentified.tsv'
    p = os.path.join(filepath,directory,file)

    metadata = pd.read_csv(p,sep='\t')

    AD_status_lbl = 'Pathologic_diagnosis_of_AD'
    md_sel = metadata[['subject', AD_status_lbl]]

    # Annotate the data with the disease status of the subject

    # floats for logits loss
    # int for BCE loss
    v_yes, v_no = (1.0, 0.0) if is_float else (1, 0)
    
    md_sel[AD_status_lbl].replace(to_replace='yes', value=v_yes, inplace=True)
    md_sel[AD_status_lbl].replace(to_replace='no', value=v_no, inplace=True)

    # create map with subject as key, AD status as value
    status_map = dict(zip(md_sel['subject'], md_sel[AD_status_lbl]))

    for label in list(datasets.keys()):
        pseudo = datasets[label]['pseudo']
        subjects = pseudo.obs['subject'].astype(str).values

        pseudo.obs['AD_status'] = [str(status_map.get(s)) for s in subjects]
        pseudo.obs['AD_status'] = pseudo.obs['AD_status'].astype('category')


def save_files(datasets: dict, filepath: str) -> None:
    # Save pseudobatched data
    for label in list(datasets.keys()):
        print(f'Writing "{label}" to file.')
        p = os.path.join(filepath, 'processed_data/')
        f = label + '.h5ad'
        to = os.path.join(p,f)

        datasets[label]['pseudo'].write_h5ad(to)

    print('Writing to file(s) completed')


def pipeline(to_include: list):
    ### To read from and write to current users folders
    user = os.environ.get('USER') or os.environ.get('USERNAME')
    base_path = Path("/data/users") / user / "kand/data/"
    filepath = str(base_path)

    included_labels = get_labels(to_include)

    datasets = get_datasets(included_labels)

    read_files(filepath, datasets)

    prep_for_hvg_sel(datasets)

    extract_common_hvgs(datasets, n_top_genes=2000)

    pseudobulk(datasets)
    
    normalize(datasets)

    add_metadata(datasets, filepath)

    save_files(datasets, filepath)

    draw_umaps(datasets)

    print('Pipeline completed')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description = "Pre-process anndata objects and save as .h5ad"
    )

    parser.add_argument(
        "to_include",
        help="indices to include: \n0=astro \n1=exc1 \n2=exc2 \n3=exc3 \n4=immune \n5=inhi \n6=oligo \n7=opcs \n8=vasc",
        nargs='+')

    args = parser.parse_args()
    to_include = [int(arg) for arg in args.to_include]

    pipeline(to_include)


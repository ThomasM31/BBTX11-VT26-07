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

import scipy.misc

def get_labels(to_include: list) -> list:
    labels = ['astro', 'exc1', 'exc2', 'exc3', 'immune', 'inhi', 'oligo', 'opcs', 'vasc']
    included_labels = [labels[i] for i in to_include]
    print(f'Labels to include: {included_labels}')

    return included_labels

def get_datasets(included_labels: list) -> dict:
    # inner dict holds the original data, hvg subset, and pseudobatched data
    # keeping all of them helps us compare them later to see the effects of pre-processing
    datasets = {
        label: None 
        for label in included_labels
    }
    return datasets

def read_files(datasets: dict, filepath: str) -> None:

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
        datasets[label] = ad.read_h5ad(f)
        print(datasets[label])

    return datasets

def read_hvg_adata(datasets: dict, filepath: str) -> None:
    p = os.path.join(filepath, 'processed_data/hvg')

    for label in list(datasets.keys()):
        print(f'Reading: {label}')
        f = os.path.join(p,f'{label}.h5ad')
        datasets[label] = ad.read_h5ad(f)
        print(datasets[label])

def filter_cells(datasets: dict, min_genes: int=200) -> None:
    for label, adata in datasets.items():
        print(f'Filtering cells for {label}')
        old_cells = adata.n_obs
        
        print(f'Original nr cells {label}: {old_cells}')
        
        print(f"Filtering cells with <{min_genes} genes...")
        sc.pp.filter_cells(adata, min_genes=min_genes)
        
        print(f"Filtered data, shape={adata.shape}\n")

        print("Calculating QC metrics")
        adata.var["mt"] = np.array(adata.var_names.str.startswith("MT-"), dtype=bool)
        print("MT genes detected:", adata.var["mt"].sum())

        sc.pp.calculate_qc_metrics(adata, qc_vars=["mt"], percent_top=None, log1p=False, inplace=True)

        print("Dropping cells with >5% mitochondrial counts...")
        datasets[label] = adata[adata.obs.pct_counts_mt < 5, :].copy()

        print(f"{label}: {old_cells} -> {adata.n_obs} cells (removed {old_cells - adata.n_obs})")

def write_gene_expr_count(datasets: dict, filepath: str):

    for label, adata in datasets.items():
        # counts how many cells expresses each gene
        sc.pp.filter_genes(adata, min_cells=0)
        counts = adata.var['n_cells'].tolist()

        to = os.path.join(filepath, f'processed_data/expr_counts/{label}.csv')
        adata.var['n_cells'].to_csv(to)

def read_gene_expr_count(datasets: dict, filepath: str):
    
    p = os.path.join(filepath, 'processed_data/expr_counts')

    files = [f.name for f in Path(p).iterdir() if f.is_file()]
    
    # correct cell type for this run
    rel_files = []
    for label in datasets.keys():
        for file in files:
            is_correct_label = file.startswith(label)
            
            if is_correct_label:
                rel_files.append(file)
    
    counts_list = []

    for i, file in enumerate(rel_files):
        to_read = os.path.join(p, file)
        counts = pd.read_csv(to_read, index_col=0).iloc[:, 0]
        counts_list.append(counts)
        
    return counts_list

def sum_gene_expr_counts(datasets, filepath, min_cells):
    '''
    Sum the number of cell where each gene is expressed for all data sets.
    Writes a file of all genes expressed in at least 'min_cells' cells.
    '''
    counts_list = read_gene_expr_count(datasets, filepath)

    # sum counts and list all that are expressed in more than at least min_cells
    total_counts = pd.concat(counts_list, axis = 1).sum(axis = 1)
    genes_to_keep = total_counts[total_counts >= min_cells].index

    to = os.path.join(
        f'{filepath}/processed_data/filter_genes', 
        f'genes_to_keep_{min_cells}.csv')
        
    pd.Series(genes_to_keep, name='gene_ids').to_csv(to, index=False)


def filter_genes(datasets: dict, filepath: str, min_cells: int=200) -> None:
    '''
    Filter genes that appear in less than 'min_cells' cells cumulatively 
    for all data sets.
    Doing this over all datasets ensures that we don't accidentally 
    remove genes that are lowly expressed in some cell type, 
    but may be highly expressed in another.
    '''
    print(f'Filtering out genes expressed in <{min_cells} cells.')

    to_read = os.path.join(filepath, f'processed_data/filter_genes/genes_to_keep_{min_cells}.csv')
    genes_to_keep = pd.read_csv(to_read)
    genes_to_keep = genes_to_keep['gene_ids'].tolist()

    # do the filtering
    for label, adata in datasets.items():
        print(f'{label}: removing genes expressed in <{min_cells} cells (cumulative)')
        old_genes = adata.n_vars
        datasets[label] = adata[:, adata.var_names.isin(genes_to_keep)].copy()
        new_genes = datasets[label].n_vars
        print(f"{label}: {old_genes} -> {new_genes} genes (removed {old_genes - new_genes})")


def prep_for_hvg_sel(datasets: dict) -> None:
    for label in list(datasets.keys()):
        dataset = datasets[label]
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


def extract_per_cell_type_hvgs(datasets: dict, filepath: str, n_top_genes: int) -> None:
    '''
    Writes n most variable genes to text file, per submitted dataset.
    '''
    filepath = os.path.join(filepath, 'processed_data/hvg_lists')
    # Keep only highly variable genes (HVGs)
    hvgs = []

    # extract HVGs (NOTE: expects normalized data)
    for label in list(datasets.keys()):
        print(f'Extracting the {n_top_genes} most variable genes from {label}')

        # use log1p layer
        adata = datasets[label].uns['log1p']

        sc.pp.highly_variable_genes(adata, n_top_genes=n_top_genes)
        hvg_per_type = adata.var_names[adata.var.highly_variable].tolist()

        # save list of hvgs as textfile
        # will allow us to find common hvgs later
        fname = f'{label}_ntop_{n_top_genes}.txt'
        to = os.path.join(filepath, fname)
        with open(to, 'w') as output:
            for gene in hvg_per_type:
                output.write(str(gene) + '\n')

        # save HVGs to dataset
        datasets[label].uns['hvg'] = datasets[label][:, hvg_per_type].copy()
        print(f'Writing HVGs for {label}')
        print(datasets[label].uns['hvg'])

def extract_hvgs_full_list(datasets: dict, filepath: str):
    '''
    Writes all genes to text file in order of descending variability.
    '''
    
    filepath = os.path.join(filepath, 'processed_data/hvg_lists')

    for label, adata in datasets.items():
        # seurat v3 flavor uses raw counts, normalized data not needed
        sc.pp.highly_variable_genes(adata, flavor='seurat_v3')

        sorted_genes = adata.var.sort_values('variances_norm', ascending=False).index.tolist()

        # save list of hvgs as textfile
        # will allow us to find common hvgs later
        fname = f'{label}.txt'
        to = os.path.join(filepath, fname)
        with open(to, 'w') as output:
            for gene in sorted_genes:
                output.write(str(gene) + '\n')


def find_common_hvgs(
    datasets: dict, 
    filepath: str,  
    n_top_genes: int, 
    min_common: int = 1000, 
    inc_val: int = 3000
    ) -> None:
    
    p = os.path.join(filepath, 'processed_data/hvg_lists')

    files = [f.name for f in Path(p).iterdir() if f.is_file()]

    # retain only files with n_top_genes and 
    # correct cell type for this run
    rel_files = []
    for label in datasets.keys():
        for file in files:
            if file.startswith(label):
                rel_files.append(file)
    
    hvgs = []  
    for i, file in enumerate(rel_files):
        path = os.path.join(p, file)
        with open(path, 'r') as input:
            genes = input.readlines()
            hvgs.append([g.strip('\n') for g in genes])

    # remove any genes that have already been filtered out
    for i, label in enumerate(datasets.keys()):
        hvgs[i] = [g for g in hvgs[i] if g in datasets[label].var_names]

    def get_common(hvgs: list, n_top_genes: int) -> list:
        common_dict = {g: None for g in hvgs[0][:n_top_genes]}

        for hvg_list in hvgs[1:]:
            print(f'ntop: {n_top_genes}')
            current_top = set(hvg_list[:n_top_genes])
            common_dict = {g: None for g in common_dict if g in current_top}

        return list(common_dict.keys())

    common_hvgs = []
    n = n_top_genes
    # run the get_common with larger numbers of variables, 
    # until we get a a value that is at least min_common
    while len(common_hvgs) < min_common:
        common_hvgs = get_common(hvgs, n)
        n += inc_val

    # just take the min_common most variable
    common_hvgs = common_hvgs[:min_common]

    print(f'Found {len(common_hvgs)} common HVGs with n_top_genes={n}')
    
    included = "_".join(datasets.keys())

    # create text file of all common HVGs
    fname = f'{included}_common_{len(common_hvgs)}.txt'
    to = os.path.join(f'{filepath}/processed_data/hvg_common', fname)
    with open(to, 'w') as output:
        for gene in common_hvgs:
            output.write(str(gene) + '\n')

    
def filter_common_hvgs(datasets: dict, filepath: str, hvg_file:str) -> None:
    p = os.path.join(filepath, 'processed_data/hvg_common')
    
    path = os.path.join(p, hvg_file)
    with open(path, 'r') as input:
        genes = input.readlines()
        common_hvgs = [g.strip('\n') for g in genes]

    ## save common hvgs in adata object
    for label, adata in datasets.items():
        common_hvgs = [g for g in common_hvgs if g in adata.var_names]
        adata.uns['common_hvgs'] = adata[:, common_hvgs].copy()
        print(f'Writing common hvgs to datset for {label}.')
        print(adata.uns['common_hvgs'])

def pseudobulk(datasets: dict) -> None:
    # Perform pseudobulk by test subject and high res celltype

    for label in list(datasets.keys()):
        print(f'Pseudobulking "{label}"')

        pseudo = sc.get.aggregate(
            datasets[label].uns['common_hvgs'], 
            by=['subject', 'cell_type_high_resolution'], 
            func='sum'
            )

        # moves pseudobulk to the main layer
        pseudo.X = pseudo.layers['sum'].copy()
        datasets[label].uns['pseudo'] = pseudo
        print(datasets[label].uns['pseudo'])


def normalize(datasets: dict) -> None:
    for label in list(datasets.keys()):
        print(f'Normalizing: "{label}"')
        
        adata = datasets[label].uns['pseudo']

        # Normalizes counts per pseudobulk sample
        # There are possible options, e.g. exclude highly expressed genes from computation

        sc.pp.normalize_total(adata, exclude_highly_expressed=False, target_sum=1e4)

        # Especially useful if range is large
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
        adata = datasets[label]
        adata_proc = datasets[label].uns['pseudo']
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
        
        print(f"Drawing completed for {label}")


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
        pseudo = datasets[label].uns['pseudo']
        subjects = pseudo.obs['subject'].astype(str).values

        pseudo.obs['AD_status'] = [str(status_map.get(s)) for s in subjects]
        pseudo.obs['AD_status'] = pseudo.obs['AD_status'].astype('category')


def save_files(datasets: dict, filepath: str, stage:str) -> None:
    print(f'Writing files at stage {stage}')
    
    for label in list(datasets.keys()):
        print(f'Writing "{label}" to file.')
        p = os.path.join(filepath, f'processed_data/{stage}')
        f = label + '.h5ad'
        to = os.path.join(p,f)

        datasets[label].write_h5ad(to)

    print(f'Writing to file(s) at stage {stage} completed')
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
    datasets = {
        label: None 
        for label in included_labels
    }
    return datasets

def read_files(datasets: dict, filepath: Path) -> None:

    files = {'astro'  : 'Astrocytes.h5ad',
             'exc1'   : 'Excitatory_neurons_set1.h5ad',
             'exc2'   : 'Excitatory_neurons_set2.h5ad',
             'exc3'   : 'Excitatory_neurons_set3.h5ad',   
             'immune' : 'Immune_cells.h5ad',   
             'inhi'   : 'Inhibitory_neurons.h5ad',   
             'oligo'  : 'Oligodendrocytes.h5ad',   
             'opcs'   : 'OPCs.h5ad',
             'vasc'   : 'Vasculature_cells.h5ad'}

    for label in datasets.keys():
        print(f'Reading: {label}')
        f = filepath / files[label]
        datasets[label] = ad.read_h5ad(f)
        print(datasets[label])

    return datasets

def read_intermediate(datasets: dict, filepath: Path) -> None:
    '''
    To read files saved at any stage of processing ('label.h5ad').
    '''
    for label, dataset in datasets.items():
        print(f'Reading: {label}')
        f = filepath / f'{label}.h5ad'
        datasets[label] = ad.read_h5ad(f)
        print(datasets[label])


def filter_cells_by_min_genes(datasets: dict, min_genes: int=200) -> None:
    for label, adata in datasets.items():
        print(f"Filtering {label} cells with <{min_genes} genes...")
        old_cells = adata.n_obs
        
        sc.pp.filter_cells(adata, min_genes=min_genes)
        
        print(f"Filtered data, shape={adata.shape}")
        print(f"{label}: {old_cells} -> {adata.n_obs} cells (removed {old_cells - adata.n_obs})\n")


def filter_cells_by_mitochondrial_content(datasets: dict, mt_threshold: float=5.0) -> None:
    for label, adata in datasets.items():
        print(f'Filtering cells for {label} by mitochondrial content')
        old_cells = adata.n_obs

        adata.var["mt"] = np.array(adata.var_names.str.startswith("MT-"), dtype=bool)
        print("MT genes detected:", adata.var["mt"].sum())

        sc.pp.calculate_qc_metrics(adata, qc_vars=["mt"], percent_top=None, log1p=False, inplace=True)

        print(f"Dropping cells with >{mt_threshold}% mitochondrial counts...")
        datasets[label] = adata[adata.obs.pct_counts_mt < mt_threshold, :].copy()

        print(f"{label}: {old_cells} -> {adata.n_obs} cells (removed {old_cells - adata.n_obs})")


def write_gene_expr_count(datasets: dict, filepath: Path) -> None:
    for label, adata in datasets.items():
        # counts how many cells express each gene
        sc.pp.filter_genes(adata, min_cells=0)
        counts = adata.var['n_cells'].tolist()

        f = filepath / f'{label}.csv'
        adata.var['n_cells'].to_csv(f)
    print(f'Gene expression counts saved to: {f}')

def read_gene_expr_count(included_labels: list[str], filepath: Path) -> list:
    files = []
    for label in included_labels:
        f = f'{label}.csv'
        if (filepath / f).exists():
            files.append(f)

    counts_list = []
    for file in files:
        to_read = filepath / file
        counts = pd.read_csv(to_read, index_col=0).iloc[:, 0]
        counts_list.append(counts)
    return counts_list

def sum_gene_expr_counts(
        included_labels: list[str], 
        filepath_from: Path, 
        filepath_to: Path, 
        min_cells: int
        ) -> None:
    '''
    Sum the number of cell where each gene is expressed for all data sets.
    Writes a file of all genes expressed in at least 'min_cells' cells.
    '''
    counts_list = read_gene_expr_count(included_labels, filepath_from)

    # sum counts and list all that are expressed in more than at least min_cells
    total_counts = pd.concat(counts_list, axis = 1).sum(axis = 1)
    genes_to_keep = total_counts[total_counts >= min_cells].index

    pd.Series(genes_to_keep, name='gene_ids').to_csv(filepath_to, index=False)

    print(f'{len(genes_to_keep)} genes expressed in over {min_cells} cells, saved to file.')


def get_expressed_genes(
        datasets: dict, 
        filepath: Path, 
        min_cells: int=200
        ) -> None:
    '''
    Returns genes expressed in more than min_cells cells.
    '''
    to_read = filepath / f'genes_to_keep_{min_cells}.csv'
    expressed_genes = pd.read_csv(to_read)
    return expressed_genes['gene_ids'].tolist()


def save_reactome_genes(
        filepath: Path,
        filename_read: str,
        filename_save: str
        ) -> list[str]:
    '''
    Exclude genes that are not present in the Reactome database.
    This so that we will not have genes in our final training data 
    that are not connected to a pathway or process in the neural network.
    '''
    # Read GMT-files to extract all genes in reactome database
    # GMT files contain entries with:
    # pathway name | pathway ID | list of associated genes

    print(f'Filtering out genes not present in Reactome.')
    
    reactome_genes = set()
    with open(filepath / filename_read, 'r') as lines:
        for line in lines:
            parts = line.strip().split('\t')
            pathway_id = parts[1]

            # only keep entries related to human biology
            if pathway_id.startswith('R-HSA'):
                pathway_genes = parts[2:]
                for gene in pathway_genes:
                    reactome_genes.add(gene)
    
    with open(filepath / filename_save, 'w') as output:
        for gene in reactome_genes:
            output.write(str(gene) + '\n')

    print(f'{len(reactome_genes)} Reactome genes written to {filepath / filename_save}')
    
def get_reactome_genes(
        datasets: dict,
        filepath: Path,
        filename: Path
        ) -> list[str]:

    with open(filepath / filename, 'r') as input:
        reactome_genes = [g.strip('\n') for g in input.readlines()]

    return reactome_genes


def filter_genes(datasets: dict, genes_to_keep: list[list[str]]) -> None:

    for label, adata in datasets.items():
        old_genes = adata.n_vars
        rel_genes = set(adata.var_names)
        for gene_list in genes_to_keep:
            rel_genes &= set(gene_list)
        
        datasets[label] = adata[:, list(rel_genes)]
        new_genes = datasets[label].n_vars
        print('Gene filtering completed.')
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


def extract_per_cell_type_hvgs(
        datasets: dict, 
        filepath: Path, 
        n_top_genes: int
        ) -> None:
    '''
    Writes n most variable genes to text file, per submitted dataset.
    '''
    
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
        to = filepath / fname
        with open(to, 'w') as output:
            for gene in hvg_per_type:
                output.write(str(gene) + '\n')

        # save HVGs to dataset
        datasets[label].uns['hvg'] = datasets[label][:, hvg_per_type].copy()
        print(f'Writing HVGs for {label}')
        print(datasets[label].uns['hvg'])

def extract_hvgs_full_list(datasets: dict, filepath: Path):
    '''
    Writes all genes to text file in order of descending variability.
    '''

    for label, adata in datasets.items():
        # seurat v3 flavor uses raw counts, normalized data not needed
        sc.pp.highly_variable_genes(adata, flavor='seurat_v3')

        sorted_genes = adata.var.sort_values('variances_norm', ascending=False).index.tolist()

        # save list of hvgs as textfile
        # will allow us to find common hvgs later
        fname = f'{label}.txt'
        to = filepath / fname
        with open(to, 'w') as output:
            for gene in sorted_genes:
                output.write(str(gene) + '\n')


def find_common_hvgs(
        datasets: dict, 
        filepath_from: Path, 
        filepath_to: Path,  
        n_top_genes: int,
        genes_to_keep: set[str],
        nr_common_hvgs: int = 1000, 
        inc_val: int = 3000
        ) -> None:

    rel_files = []
    for label in datasets:
        f = f'{label}.txt'
        if (filepath_from / f).exists():
            rel_files.append(f)
    
    hvgs = []  
    for i, file in enumerate(rel_files):
        path = filepath_from / file
        with open(path, 'r') as input:
            genes = input.readlines()
            genes = set([g.strip('\n') for g in genes])
            # only keep genes that are also in our list of "acceptable" genes
            hvgs.append([g for g in genes if g in genes_to_keep])

    def _get_common(hvgs: list, n_top_genes: int) -> list:
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
    while len(common_hvgs) < nr_common_hvgs:
        common_hvgs = _get_common(hvgs, n)
        n += inc_val

    # just take the min_common most variable
    common_hvgs = common_hvgs[:nr_common_hvgs]
    if len(common_hvgs) < nr_common_hvgs:
        print(f'WARNING: only found {len(common_hvgs)} hvgs')

    print(f'Found {len(common_hvgs)} common HVGs with n_top_genes={n}')

    filename = f'{"_".join(datasets.keys())}_common_{nr_common_hvgs}.txt' 
    # create text file of all common HVGs
    with open(filepath_to / filename, 'w') as output:
        for gene in common_hvgs:
            output.write(str(gene) + '\n')

    
def filter_common_hvgs(
        datasets: dict, 
        filepath: Path, 
        filename: str
        ) -> None:
    
    with open(filepath / filename, 'r') as input:
        genes = input.readlines()
        common_hvgs = [g.strip('\n') for g in genes]

    ## save common hvgs in adata object
    for label, adata in datasets.items():
        common_hvgs = [g for g in common_hvgs if g in adata.var_names]
        datasets[label].uns['common_hvgs'] = adata[:, common_hvgs].copy()
        print(f'Writing common hvgs to datset for {label}.')
        print(datasets[label].uns['common_hvgs'])


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

def pseudobulk_non_hvg(datasets: dict) -> None:
    # Perform pseudobulk by test subject and high res celltype

    for label, adata in datasets.items():
        print(f'Pseudobulking "{label}"')

        pseudo = sc.get.aggregate(
            adata, 
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


def draw_umaps(datasets: dict, filepath: Path) -> None:
    '''
    To draw UMAPS while both the original data and 
    processed data are in the same ad object
    '''
    for label, adata in datasets.items():
        adata_orig = adata
        adata_proc = adata.uns['pseudo']
        draw_umaps_logic(filepath, adata_orig, adata_proc, label)
        

def draw_umaps(datasets_orig: dict, datasets_proc: dict, filepath: Path) -> None:
    '''
    To draw UMAPS when original data and processed data are in the main 
    layers of two separate ad objects
    '''
    for label in datasets_orig.keys():
        adata_orig = datasets_orig[label]
        adata_proc = datasets_proc[label]
        draw_umaps_logic(adata_orig, adata_proc, label, filepath)


def add_metadata(datasets: dict, filepath: Path, is_float=True) -> None:
    # Add subject disease status metadata
    file = 'individual_metadata_deidentified.tsv'
    p = filepath / file
    metadata = pd.read_csv(p,sep='\t')

    AD_status_lbl = 'Pathologic_diagnosis_of_AD'
    md_sel = metadata[['subject', AD_status_lbl]]

    # Annotate the data with the disease status of the subject

    # floats for logits loss
    # int for BCE loss
    v_yes, v_no = (1.0, 0.0) if is_float else (1, 0)
    
    md_sel = md_sel.copy()
    md_sel[AD_status_lbl] = md_sel[AD_status_lbl].replace({'yes': v_yes, 'no': v_no})

    # create map with subject as key, AD status as value
    status_map = dict(zip(md_sel['subject'], md_sel[AD_status_lbl]))

    for label in list(datasets.keys()):
        pseudo = datasets[label].uns['pseudo']
        subjects = pseudo.obs['subject'].astype(str).values

        pseudo.obs['AD_status'] = [str(status_map.get(s)) for s in subjects]
        pseudo.obs['AD_status'] = pseudo.obs['AD_status'].astype('category')

def add_metadata_non_pseudo(datasets: dict, filepath: Path, is_float=True) -> None:
    # Add subject disease status metadata
    file = 'individual_metadata_deidentified.tsv'
    p = filepath / file
    metadata = pd.read_csv(p,sep='\t')

    AD_status_lbl = 'Pathologic_diagnosis_of_AD'
    md_sel = metadata[['subject', AD_status_lbl]]

    # Annotate the data with the disease status of the subject

    # floats for logits loss
    # int for BCE loss
    v_yes, v_no = (1.0, 0.0) if is_float else (1, 0)
    
    md_sel = md_sel.copy()
    md_sel[AD_status_lbl] = md_sel[AD_status_lbl].replace({'yes': v_yes, 'no': v_no})

    # create map with subject as key, AD status as value
    status_map = dict(zip(md_sel['subject'], md_sel[AD_status_lbl]))

    for label, adata in datasets.items():
        #pseudo = datasets[label].uns['pseudo']
        subjects = adata.obs['subject'].astype(str).values

        adata.obs['AD_status'] = [str(status_map.get(s)) for s in subjects]
        adata.obs['AD_status'] = adata.obs['AD_status'].astype('category')


def move_pseudo_main(datasets: dict):
    for label, adata in datasets.items():
        pseudo = adata.uns['pseudo']

        datasets[label] = ad.AnnData(pseudo)

        print(f'Dataset {label} after moving pseudo to main:')
        print(datasets[label])

def move_hvgs_main(datasets):
    for label, adata in datasets.items():
        hvgs = datasets[label].uns['common_hvgs']
        datasets[label] = ad.AnnData(hvgs)

def save_files(datasets: dict, filepath: Path, stage:str) -> None:
    print(f'Writing files at stage "{stage}"')
    
    for label in list(datasets.keys()):
        print(f'Writing "{label}" to file.')
        to = filepath / f'{label}.h5ad'

        datasets[label].write_h5ad(to)

    print(f'Writing to file(s) at stage {stage} done! Path:')
    print(to)
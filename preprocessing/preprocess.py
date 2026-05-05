from pathlib import Path
from typing import Literal, Optional

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

def read_files(datasets: dict, filepath: Path) -> dict:

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
        ) -> list[str]:
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
        ) -> None:
    '''
    Save a list of genes that are present in the Reactome database.
    For later filtering of genes, so that we will not have genes in our 
    final training data that are not connected to a pathway or process 
    in the neural network.
    '''
    # Read GMT-files to extract all genes in reactome database
    # GMT files contain entries with:
    # pathway name | pathway ID | list of associated genes

    print(f'Saving list of genes that are in Reactome')
    
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

    # get the genes that are in the intersection of all filter lists
    rel_genes = set(genes_to_keep[0])
    for gene_list in genes_to_keep:
        rel_genes &= set(gene_list)

    for label, adata in datasets.items():
        nr_old_genes = adata.n_vars
        adata_genes = adata.var_names

        # final filtering of adata genes is done on a list to keep the order
        keep = [g for g in adata_genes if g in rel_genes]
        
        datasets[label] = adata[:, keep]
        nr_new_genes = datasets[label].n_vars
        print('Gene filtering completed.')
        print(f"{label}: {nr_old_genes} -> {nr_new_genes} genes (removed {nr_old_genes - nr_new_genes})")



def extract_hvgs_per_cell_type(datasets: dict, filepath: Path):
    '''
    Writes all genes to file in order of descending variability, 
    along with variability value.
    '''
    for label, adata in datasets.items():
        # identify hvgs
        # seurat v3 flavor uses raw counts, normalized data not needed
        sc.pp.highly_variable_genes(adata, flavor='seurat_v3')

        hvg_df = adata.var[['variances_norm']].sort_values('variances_norm', ascending=False)
        
        hvg_df.to_csv(filepath / f'{label}.csv', index_label='gene')


def find_common_hvgs(
        datasets: dict, 
        filepath_from: Path, 
        filepath_to: Path
        ) -> None:
    '''
    Saves all genes present in input files in order of average variability.
    '''
    rel_files = []
    for label in datasets:
        f = filepath_from / f'{label}.csv'
        if (f).exists():
            rel_files.append(f)
    
    dataframes: list[pd.DataFrame] = []
    for file in rel_files:
        temp_df = pd.read_csv(file, index_col='gene')
        temp_df.columns = [file.stem]
        dataframes.append(temp_df)
    
    combined_df = pd.concat(dataframes, axis=1)
    combined_df['avg_variance'] = combined_df.mean(axis=1)
    combined_df = combined_df[['avg_variance']]

    combined_df = combined_df.sort_values(by='avg_variance', ascending=False)

    filename = f'{"_".join(datasets.keys())}_common.csv' 
    combined_df.to_csv(filepath_to / filename)

    print(f'Saved {combined_df.size} common HVGs to:')
    print(str(filepath_to / filename))
    

def filter_common_hvgs(
        datasets: dict, 
        filepath: Path,
        n: int
        ) -> None:
    '''
    Filters data to the n most highly variable genes (avg across cell types).
    Assumes that all other gene filtering has already been completed. 

    Parameters
    --------
    datasets
        Datasets for filtering.
    filepath
        Path to file where common HVGs for included cell type is located. 
        Assumes that genes are listed in oreder of average variability.
    n
        How many common HVGs are to be included in the final dataset.
    '''
    hvg_df = pd.read_csv(filepath, index_col='gene')

    for label, adata in datasets.items():
        # filter to only include genes present in adata
        hvg_filtered_df = hvg_df[hvg_df.index.isin(adata.var_names)]
        
        # n most variable
        top_genes = hvg_filtered_df.head(n).index

        datasets[label].uns['common_hvgs'] = adata[:, top_genes].copy()

        print(f'Writing common hvgs to datset for {label}.')
        print(datasets[label].uns['common_hvgs'])


def verify_gene_order(datasets: dict, master_gene_order: list) -> None:
    """
    Ensures the genes in each dataset follow the relative order 
    of the provided master_gene_order, 
    to verify that order has not been changed in any filtering step.
    """
    gene_to_index = {gene: i for i, gene in enumerate(master_gene_order)}
    
    for label, adata in datasets.items():
        current_genes = adata.var_names.tolist()
        
        # Get the original indices for the filtered genes
        try:
            current_indices = [gene_to_index[g] for g in current_genes]
        except KeyError as e:
            raise KeyError(f"Gene {e} in dataset '{label}' was not found in the master order.")

        # Check if indices are strictly increasing
        if current_indices != sorted(current_indices):
            raise ValueError(
                f"ORDER MISMATCH: Genes in dataset '{label}' are not in the original relative order."
            )
            
    print("SUCCESS: All datasets maintain original relative gene order.")

def pseudobulk(datasets: dict[str, ad.AnnData], 
               bulk_by: Literal['count_nonzero', 'mean', 'sum', 'var', 'median'],
               layer_key: Optional[str] = None
               ) -> None:
    """
    Perform pseudobulk by test subject and high res celltype.

    Parameters
    ----------
    datasets
        Dictionary containing AnnData objects to be processed.
    bulk_by
        Aggregation function name; must be a valid scanpy.get.aggregate method.
    layer_key
        Key in `adata.uns` to use as the data source. If None, uses the main object.
    """

    for label, adata in datasets.items():
        print(f'Pseudobulking "{label}"')

        source = adata.uns[layer_key] if layer_key else adata

        if 'cell_type_high_resolution' in adata.obs.columns:  
            pseudo = sc.get.aggregate(
                source, 
                by=['subject', 'cell_type_high_resolution'], 
                func=bulk_by
                )
        else:
            pseudo = sc.get.aggregate(
                source, 
                by=['subject'], 
                func=bulk_by
                )

        pseudo.X = pseudo.layers[bulk_by].copy()
        adata.uns['pseudo'] = pseudo 
        print(adata.uns['pseudo'])


def normalize(datasets: dict) -> None:
    ''' Normalizes pseudobulked data. '''
    for label, adata in datasets.items():
        print(f'Normalizing: "{label}"')
        
        source = adata.uns['pseudo']

        # Normalizes counts per pseudobulk sample
        # There are possible options, e.g. exclude highly expressed genes from computation
        sc.pp.normalize_total(source, exclude_highly_expressed=False, target_sum=1e4)

        # Especially useful if range is large
        sc.pp.log1p(source)

        # Scaling centers genes at 0
        # Helps model learn *relative* expression
        sc.pp.scale(source)

        print(source)
        print(f'{label:<8} has min: {source.X.min():.2f} and max: {source.X.max():.2f}')


def add_metadata(
        datasets: dict, 
        filepath: Path, 
        layer_key: Optional[str] = None, 
        is_float=True
        ) -> None:
    '''
    Add subject disease status metadata
    
    Parameters
    --------
    datasets
        Dictionary containing AnnData objects to be processed.
    filepath
        Path to metadata csv file containing per subject disease status
    layer_key
        Key in `adata.uns` to use as the data source. If None, uses the main object.
    is_float
        Set disease status to float or int (default float). 
        Float recommended for logit loss, int recommended for BCE loss
    '''
    metadata = pd.read_csv(filepath, sep='\t')

    subj_lbl = 'subject'
    AD_status_lbl = 'Pathologic_diagnosis_of_AD'
    md_sel = metadata[[subj_lbl, AD_status_lbl]]

    v_yes, v_no = (1.0, 0.0) if is_float else (1, 0)
    
    md_sel = md_sel.copy()
    md_sel[AD_status_lbl] = md_sel[AD_status_lbl].replace({'yes': v_yes, 'no': v_no})

    # create map with subject as key, AD status as value
    status_map = dict(zip(md_sel[subj_lbl], md_sel[AD_status_lbl]))

    for label, adata in datasets.items():
        print(f'Adding metadata to {label}.')
        source = adata.uns[layer_key] if layer_key else adata
        subjects = source.obs[subj_lbl].astype(str).values

        AD_status_lbl = 'AD_status'
        source.obs[AD_status_lbl] = [str(status_map.get(s)) for s in subjects]
        source.obs[AD_status_lbl] = source.obs[AD_status_lbl].astype('category')

    for label, adata in datasets.items():
        #pseudo = datasets[label].uns['pseudo']
        subjects = adata.obs[subj_lbl].astype(str).values

        adata.obs[AD_status_lbl] = [str(status_map.get(s)) for s in subjects]
        adata.obs[AD_status_lbl] = adata.obs[AD_status_lbl].astype('category')

def move_to_main(datasets: dict, layer_key: str):
    '''
    Move layer from `adata.uns` to main object
    '''
    for label, adata in datasets.items():
        from_layer = adata.uns[layer_key]

        datasets[label] = ad.AnnData(from_layer)

        print(f'Dataset {label} after moving {layer_key} to main:')
        print(datasets[label])

def save_files(datasets: dict, filepath: Path, stage:str) -> None:
    print(f'Writing files at stage "{stage}"')
    filepath.mkdir(parents=True, exist_ok=True)
    
    for label in list(datasets.keys()):
        print(f'Writing "{label}" to file.')
        to = filepath / f'{label}.h5ad'
        datasets[label].write_h5ad(to)

    print(f'Writing to file(s) at stage {stage} done! Path:')
    print(to)
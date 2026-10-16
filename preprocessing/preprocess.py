from pathlib import Path
from typing import Literal, Optional, Union

import scanpy as sc
import pandas as pd
import numpy as np

import anndata as ad
from anndata.experimental import AnnCollection
from anndata.experimental.pytorch import AnnLoader
ad.settings.allow_write_nullable_strings = True

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import os
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
             'exc_combined' : 'Exc_combined.h5ad',  
             'immune' : 'Immune_cells.h5ad',   
             'inhi'   : 'Inhibitory_neurons.h5ad',
             'inh_combined' : 'Inh_combined.h5ad',   
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

def create_new_labels(
        adata: sc.AnnData, 
        drop_unknown: bool = True
        ) -> sc.AnnData:
    """
    Stratify the subjects in the AnnData object into categories based on
    cognitive diagnosis (cogdx), Braak stage (braaksc), and CERAD score (ceradsc)
    into three categories: 
        - AD (Alzheimer's Disease)
        - NCI (No Cognitive Impairment)
        - MCIP (Mild Cognitive Impairment with Plaques)
    """   
    print(f"Stratifying subjects based on cogdx, braaksc, and ceradsc...") 
    # Define conditions for each category as a boolean mask
    stratification = { 
        "AD": (
            (adata.obs["cogdx"] == 4) &
            (adata.obs["braaksc"] >= 4) &
            (adata.obs["ceradsc"] <= 2)
        ),
        "NCI": (
            (adata.obs["cogdx"] == 1) &
            (adata.obs["braaksc"] <= 3) &
            (adata.obs["ceradsc"] >= 3)
        ),
        "MCIP": (
            (adata.obs["cogdx"] == 2) &
            (adata.obs["braaksc"] >= 4) &
            (adata.obs["ceradsc"] <= 2)
        )
    }

    adata.obs["label"] = "unknown" # default label for subjects that don't fit any category

    # Assign labels based on conditions
    for label, mask in stratification.items():
        adata.obs.loc[mask, "label"] = label

    # If drop_unknown is True, filter out subjects with the "unknown" label
    if drop_unknown:
        print(f"Dropping subjects with 'unknown' label...")
        old_subject_count = adata.obs['subject'].nunique()
        adata = adata[adata.obs["label"] != "unknown"].copy()
        print(f"Removed {old_subject_count - adata.obs['subject'].nunique()} subjects. ({old_subject_count} -> {adata.obs['subject'].nunique()})\n")

    print(f"""
    Stratification summary:
    AD: {adata.obs.loc[adata.obs["label"] == "AD", "subject"].nunique()}
    NCI: {adata.obs.loc[adata.obs["label"] == "NCI", "subject"].nunique()}
    MCIP: {adata.obs.loc[adata.obs["label"] == "MCIP", "subject"].nunique()}
          """)
    
    return adata

def separate_stratified_data(
        adata: sc.AnnData,
        save_dir: str,
        file_prefix: list[str]
        ) -> tuple[sc.AnnData, sc.AnnData]:
    """
    Separate the AnnData object into two stratified datasets based on the 'label' column in adata.obs.
    Returns two AnnData objects: AD+NCI and MCIP.
    """
    save_dir = Path(save_dir)
    print(f"Separating AnnData object into stratified datasets...")
    main_adata = adata[adata.obs['label'].isin(['AD', 'NCI'])].copy()
    mcip_adata = adata[adata.obs['label'] == 'MCIP'].copy()

    print(f"AD+NCI dataset: {main_adata.n_obs} cells from {main_adata.obs['subject'].nunique()} subjects")
    print(f"MCIP dataset: {mcip_adata.n_obs} cells from {mcip_adata.obs['subject'].nunique()} subjects\n")

    # save the two datasets to disk
    main_adata.write_h5ad(save_dir / f"{file_prefix[0]}_adata.h5ad")
    mcip_adata.write_h5ad(save_dir / f"{file_prefix[1]}_adata.h5ad")
    print(f"Saved AD+NCI dataset to {save_dir / f'{file_prefix[0]}_adata.h5ad'}")
    print(f"Saved MCIP dataset to {save_dir / f'{file_prefix[1]}_adata.h5ad'}\n")

    # Save the metadata to CSV files
    main_adata.obs.drop_duplicates().to_csv(save_dir / f'{file_prefix[0]}_metadata.csv', index=False)
    mcip_adata.obs.drop_duplicates().to_csv(save_dir / f'{file_prefix[1]}_metadata.csv', index=False)
    print(f"Saved AD+NCI metadata to {save_dir / f'{file_prefix[0]}_metadata.csv'}")
    print(f"Saved MCIP metadata to {save_dir / f'{file_prefix[1]}_metadata.csv'}\n")

    return main_adata, mcip_adata


def train_test_split_subjects(
        adata: sc.AnnData, 
        test_size: float = 0.2, 
        random_state: int = 17,
        stratify_key: str = 'label',
        save_dir: str = None,
        file_prefix: str = None,
        ) -> tuple[list, list]:
    """
    Split the AnnData object into training and testing sets based on subjects.
    The split is stratified based on the 'label' column in adata.obs.
    """
    save_dir = Path(save_dir)

    metadata = adata.obs[['subject', stratify_key]].drop_duplicates()
    metadata.reset_index(inplace=True, drop=True)

    print(f"Splitting the AD and NCI subjects into training and testing sets...")
    train_subjects, test_subjects = train_test_split(
        metadata['subject'], 
        test_size=test_size, 
        random_state=random_state, 
        stratify=metadata[stratify_key]
    )
    print(f"Training set: {len(train_subjects)} subjects")
    print(f"Testing set: {len(test_subjects)} subjects\n")

    with open(save_dir / f'{file_prefix}_train_subjects.txt', 'w') as f:
        for subject in train_subjects:
            f.write(f"{subject}\n")
    print(f"Saved training subjects to {save_dir / f'{file_prefix}_train_subjects.txt'}")

    with open(save_dir / f'{file_prefix}_test_subjects.txt', 'w') as f:
        for subject in test_subjects:
            f.write(f"{subject}\n")
    print(f"Saved testing subjects to {save_dir / f'{file_prefix}_test_subjects.txt'}\n")

    metadata_train = metadata[metadata['subject'].isin(train_subjects)]
    metadata_test = metadata[metadata['subject'].isin(test_subjects)]
    print(f"Training set stratification summary:")
    print(metadata_train[stratify_key].value_counts())
    print(f"\nTesting set stratification summary:")
    print(metadata_test[stratify_key].value_counts())

    # Save train and test metadata to CSV
    metadata_train.to_csv(save_dir / f'{file_prefix}_train_subjects_label.csv', index=False, header=False)
    metadata_test.to_csv(save_dir / f'{file_prefix}_test_subjects_label.csv', index=False, header=False)
    print(f"\nSaved training metadata to {save_dir /f'{file_prefix}_train_subjects_label.csv'}")
    print(f"Saved testing metadata to {save_dir / f'{file_prefix}_test_subjects_label.csv'}\n")
    
    return train_subjects.tolist(), test_subjects.tolist()

def train_test_split_adata(
        adata: sc.AnnData, 
        train_subjects: list, 
        test_subjects: list,
        save_dir: str = None,
        file_prefix: str = None
        ) -> tuple[sc.AnnData, sc.AnnData]:
    """
    Split the AnnData object into training and testing sets based on the provided subject lists.
    """
    print(f"Splitting AnnData object into training and testing sets...")
    train_adata = adata[adata.obs['subject'].isin(train_subjects)].copy()
    test_adata = adata[adata.obs['subject'].isin(test_subjects)].copy()

    print(f"Training set: {train_adata.n_obs} cells from {train_adata.obs['subject'].nunique()} subjects")
    print(f"Testing set: {test_adata.n_obs} cells from {test_adata.obs['subject'].nunique()} subjects\n")

    if save_dir is not None:
        train_adata.write_h5ad(os.path.join(save_dir, f'{file_prefix}_train_adata.h5ad'))
        test_adata.write_h5ad(os.path.join(save_dir, f'{file_prefix}_test_adata.h5ad'))
        print(f"Saved training AnnData to {os.path.join(save_dir, f'{file_prefix}_train_adata.h5ad')}")
        print(f"Saved testing AnnData to {os.path.join(save_dir, f'{file_prefix}_test_adata.h5ad')}\n")

    return train_adata, test_adata

def filter_cells_by_min_genes(adata: ad.AnnData, min_genes: int=200) -> None:
    print(f"Filtering cells with <{min_genes} genes...")
    old_cells = adata.n_obs
    
    sc.pp.filter_cells(adata, min_genes=min_genes, inplace=True)
    
    print(f"Filtered data, shape={adata.shape}")
    print(f"{old_cells} -> {adata.n_obs} cells (removed {old_cells - adata.n_obs})\n")
    return adata

def filter_cells_by_mitochondrial_content(adata: ad.AnnData, mt_threshold: float=5.0) -> None:
    # Filter cells by mitochondrial gene content
    print(f'Filtering cells with > {mt_threshold}% mitochondrial genes...')
    old_cell_count = adata.n_obs
    adata.var["mito"] = adata.var_names.str.startswith("MT-")
    sc.pp.calculate_qc_metrics(adata, qc_vars=["mito"], inplace=True)
    adata = adata[adata.obs['pct_counts_mito'] < mt_threshold, :].copy()
    print(f"Removed {old_cell_count - adata.n_obs} cells. ({old_cell_count} -> {adata.n_obs})\n")
    return adata


def write_gene_expr_count(datasets: dict, filepath: Path) -> dict[str:pd.DataFrame]:
    counts = {}
    for label, adata in datasets.items():
        # counts how many cells express each gene
        sc.pp.filter_genes(adata, min_cells=0)

        f = filepath / f'gene_count_{label}.csv'
        adata.var['n_cells'].to_csv(f)
        print(f'Gene expression counts saved to: {f}')
        counts[label] = adata.var['n_cells']
    return counts
        

def read_gene_expr_count(included_labels: list[str], filepath: Path) -> list:
    files = []
    for label in included_labels:
        f = f'gene_count_{label}.csv'
        if (filepath / f).exists():
            files.append(f)
            print(f'{f} is to be read')
        else:
            print(f'Cannot read gene expr count for {label}, file not found')

    counts_list = []
    print(files)
    for file in files:
        to_read = filepath / file
        counts = pd.read_csv(to_read, index_col=0).iloc[:, 0]
        counts_list.append(counts)
        #print(f'counts for {label}: {counts_list}')
    return counts_list

def sum_gene_expr_counts(
        included_labels: list[str], 
        data: Union[dict[str:pd.Series], Path], 
        filepath_to: Path, 
        min_cells: int
        ) -> None:
    '''
    Sum the number of cell where each gene is expressed for all data sets.
    Writes a file of all genes expressed in at least 'min_cells' cells.
    '''

    if isinstance(data, Path):
        counts_list = read_gene_expr_count(included_labels, data)
    else:
        counts_list = []
        for series in data.values():
            counts_list.append(series)
    
    # sum counts and list all that are expressed in more than at least min_cells
    total_counts = pd.concat(counts_list, axis = 1).sum(axis = 1)
    genes_to_keep = total_counts[total_counts >= min_cells].index

    pd.Series(genes_to_keep, name='gene_ids').to_csv(filepath_to, index=False)

    print(f'{len(genes_to_keep)} genes expressed in over {min_cells} cells, saved to file {filepath_to}.')

    return genes_to_keep

def filter_genes_by_min_expression(
    adata: ad.AnnData,  
    min_cells: int
    ) -> None:
    print(f"Filtering genes expressed in <{min_cells} cells...")
    old_gene_count = adata.n_vars
    sc.pp.filter_genes(adata, min_cells=min_cells, inplace=True)
    print(f"Removed {old_gene_count - adata.n_vars} genes. ({old_gene_count} -> {adata.n_vars})\n")
    return adata

def filter_low_cell_subjects(adata: ad.AnnData, min_cells_per_subject: int = 30):
    # Filter subjects by minimum number of cells
    print(f"Filtering subjects with < {min_cells_per_subject} cells...")
    old_subject_count = adata.obs['subject'].nunique()
    subject_cells_counts = adata.obs['subject'].value_counts()
    valid_subjects = subject_cells_counts[subject_cells_counts >= min_cells_per_subject].index
    adata = adata[adata.obs['subject'].isin(valid_subjects), :].copy()
    print(f"Removed {old_subject_count - adata.obs['subject'].nunique()} subjects. ({old_subject_count} -> {adata.obs['subject'].nunique()})\n")
    return adata

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
        ) -> set:
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

    return reactome_genes
    
def get_reactome_genes(
        datasets: dict,
        filepath: Path,
        filename: Path
        ) -> list[str]:

    with open(filepath / filename, 'r') as input:
        reactome_genes = [g.strip('\n') for g in input.readlines()]

    return reactome_genes

def filter_reactome_genes(adata: sc.AnnData, reactome_genes: set[str]) -> sc.AnnData:
    """
    Filter the AnnData object to only include genes present in the Reactome database.
    """
    print(f"Filtering AnnData object to only include genes present in the Reactome database...")
    old_gene_count = adata.n_vars

    keep = [gene for gene in adata.var_names if gene in reactome_genes]

    adata = adata[:, keep].copy()
    print(f"Removed {old_gene_count - adata.n_vars} genes. ({old_gene_count} -> {adata.n_vars})\n")
    
    return adata

def filter_genes(datasets: dict, genes_to_keep: list[list[str]]) -> None:
    # TODO: make it possible to send just a list[str]
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

def select_hvgs(adata: ad.AnnData, n_top_genes: int):
    adata_hvgs = sc.pp.highly_variable_genes(adata, n_top_genes=n_top_genes, flavor='seurat_v3', inplace=False)
    return adata_hvgs.index[adata_hvgs['highly_variable']]

def filter_hvgs(dataset: ad.AnnData, hvgs:pd.Series):

    old = dataset.n_vars
    dataset = dataset[:, hvgs]
    print(f'Nr genes in dataset {old} -> {dataset.n_vars}')
    return dataset

def find_common_hvgs(
        datasets: dict, 
        filepath_from: Path, 
        filepath_to: Path
        ) -> pd.DataFrame:
    '''
    Saves all genes present in input files in order of average variability.
    '''
    rel_files = []
    for label in datasets:
        f = filepath_from / f'{label}.csv'
        if (f).exists():
            rel_files.append(f)

    print(f'rel files: {rel_files}')
    
    dataframes: list[pd.DataFrame] = []
    for file in rel_files:
        temp_df = pd.read_csv(file, index_col='gene')
        temp_df.columns = [file.stem]
        dataframes.append(temp_df)

    print(f'len of dataframes is {len(dataframes)}')
    
    combined_df = pd.concat(dataframes, axis=1)
    combined_df['avg_variance'] = combined_df.mean(axis=1)
    combined_df = combined_df[['avg_variance']]

    combined_df = combined_df.sort_values(by='avg_variance', ascending=False)

    filename = f'{"_".join(datasets.keys())}_common.csv' 
    combined_df.to_csv(filepath_to / filename)

    print(f'Saved {combined_df.size} common HVGs to:')
    print(str(filepath_to / filename))

    return combined_df

def filter_common_hvgs(
        datasets: dict, 
        data: Union[pd.DataFrame, Path],
        n: int
        ) -> None:
    '''
    Filters data to the n most highly variable genes (avg across cell types).
    Assumes that all other gene filtering has already been completed. 

    Parameters
    --------
    datasets
        Datasets for filtering.
    data
        Path to file where common HVGs for included cell type is located. 
        Assumes that genes are listed in order of average variability.
        OR
        DataFrame with genes listed in order of average variability
    n
        How many common HVGs are to be included in the final dataset.
    '''
    if isinstance(data, Path):
        data = pd.read_csv(data, index_col='gene')

    for label, adata in datasets.items():
        # filter to only include genes present in adata
        hvg_filtered_df = data[data.index.isin(adata.var_names)]
        
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

def aggregate_to_pseudobulk(
        adata: sc.AnnData,
        bulk_by: Literal['count_nonzero', 'mean', 'sum', 'var', 'median'],
        save_dir: str = None,
        file_prefix: str = None
        ) -> sc.AnnData:
    """
    Aggregate single-cell data to pseudobulk.
    """
    print(f"Aggregating single-cell data to pseudobulk by {bulk_by}...")
    # Get one row of metadata for each subject to merge later
    subject_metadata = (adata.obs.groupby("subject", as_index=False, observed=True).first())

    pseudobulk = sc.get.aggregate(
                adata, 
                by=['subject'], 
                func=bulk_by
                )
    pseudobulk.X = pseudobulk.layers[bulk_by].copy() # copy the aggregated data to the main layer
   
    pseudobulk.obs = pseudobulk.obs.merge(
        subject_metadata,
        on="subject",
        how="left"
    )
    # Set the index of pseudobulk.obs to be the subject names
    pseudobulk.obs.set_index('subject', inplace=True)
    print(f"Shape of pseudobulk AnnData: {pseudobulk.shape}\n")
    if save_dir is not None:
        pseudobulk.write_h5ad(os.path.join(save_dir, f'{file_prefix}_pseudobulk_{bulk_by}.h5ad'))
        print(f"Saved pseudobulk AnnData to {os.path.join(save_dir, f'{file_prefix}_pseudobulk_{bulk_by}.h5ad')}\n")

    return pseudobulk

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

def standardize(adata: ad.AnnData, scaler: Union[StandardScaler, None] = None):
    """ 
    Standardizes an anndata object. 
    If a scaler is included its transformation is applied to scale, 
    else StandardScaler is used. 
    """
    if scaler is None:
        print(f"Standardizing data (zero mean, unit variance)...")
        scaler = StandardScaler(with_mean=True, with_std=True)
        adata.X = scaler.fit_transform(adata.X)
        return adata, scaler
    else:
        print("Standardizing test data using training statistics...")
        adata.X = scaler.transform(adata.X)
        return adata

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
import custom_train_test_split as ctts
import anndata as ad
from anndata.experimental import AnnCollection, AnnLoader
from binn_training import *
import os
from scipy.sparse import csr_matrix
import pandas as pd, numpy as np

# GLOBALS
ALL_CELLTYPES = [0,1,2,3,4,5,6,7,8]
MASK_PATHS = [f"/data/shared/alzgene26/PathwayData/MaskMatrixLayers/mg_200_mc_200_mhvg1000/oligo_exc3_exc2_vasc_immune_astro_inhi_opcs_exc1_layer_{i}_mask.csv" 
              for i in range(5)]

def read_adata(indices: list, 
               filepath: str, 
               train_size=0.8) -> tuple[ad.AnnData, ad.AnnData, AnnCollection]:
    """
    Reads the training anndata, testing anndata and the collection they come from.
    Indicies indicate celltype. 
    """
    train_adata, test_adata, acollection = ctts.pipeline(indices, filepath, train_size)
    return train_adata, test_adata, acollection

def data_concatenate(acollection : AnnCollection) -> ad.AnnData:
    """
    Concatenate the AnnCollection to a single AnnData object
    """
    adata_conc = ad.concat(acollection, label="cell_type_low_res")
    return adata_conc
    
    
def transpose_datasets(datasets:dict) -> dict:
    """
    Transposition necessary for feeding into network
    """
    datasets_t = {}
    for celltype in datasets.keys():
        adata_t = datasets[celltype].T
        datasets_t[celltype] = adata_t
        
    return datasets_t

def train_test_adatasplit(train_adata: ad.AnnData, 
                          test_adata: ad.AnnData):
    """
    Creates the train/test split for the input anndata
    """
    X_train = train_adata.X
    y_train = train_adata.obs["AD_status"]
    X_test = test_adata.X
    y_test = test_adata.obs["AD_status"]

    return X_train, y_train, X_test, y_test

def process_completed_data(datasets: dict) -> dict:
    """
    Fetch large preprocessed datafiles and extract preprocessed layer
    """
    datasets_proc = {}
    for label, dataset in datasets.items():
        print(f"fetching pseudo from {label}")
        # fetch pseudo-batched & preprocessed data
        datasets_proc[label] = dataset.uns['pseudo'].copy()
    return datasets_proc

def create_dataloaders(train_adata: ad.AnnData,
                       test_adata: ad.AnnData,
                       batch_size=16) -> tuple[AnnLoader, AnnLoader]:
    """
    Create dataloaders using built in AnnLoader type
    """
    train_loader = AnnLoader(train_adata, batch_size=batch_size)
    test_loader = AnnLoader(test_adata, batch_size=batch_size)
    return train_loader, test_loader

def save_data(datasets: dict, filepath: str) -> None:
    """
    Save data to expressed path
    """
    for label in list(datasets.keys()):
        print(f'Writing "{label}" to file.')
        to = os.path.join(filepath, f'{label}.h5ad')

        datasets[label].write_h5ad(to)

def read_masks(mask_paths, print_shapes=False) -> dict:
    """
    Reads all available masks into dict
    """
    mask_dict = {}
    for i, mask_path in enumerate(mask_paths):
        df = pd.read_csv(mask_path, index_col=0)
        mask_dict.update({f"df{i}": df})
        if print_shapes:
            print(f"Matrix {i} shape: {df.shape}")
    return mask_dict

def align_genes(datasets: dict, input_masks: pd.DataFrame) -> dict:
    """
    Subsets the adatas to only include genes present in BINN.
    Drops any HVGs that are not part of the BINN's pathways.
    """
    target_genes = list(input_masks.index)
    datasets_aligned = {}

    for label in list(datasets.keys()):
        adata = datasets[label]
        # Find the intersection of genes
        overlapping_genes = list(set(adata.var_names) & set(target_genes))
        
        # Subset the dataset
        adata_aligned = adata[:, overlapping_genes].copy()
        # Update dictionary
        datasets_aligned.update({label: adata_aligned})

        #print(f"Overlapping genes kept: {adata_aligned.n_vars} for {label}")
        #print(f"Genes dropped: {adata.n_vars - adata_aligned.n_vars}\n")

    return datasets_aligned

def pad_data(datasets: dict, input_masks: pd.DataFrame) -> dict:
    """
    Pads the adatas with sparse zero vectors for missing target genes,
    returns: adatas sorted in alphabetical order. 
    """
    target_genes = list(input_masks.index)
    sorted_target_genes = sorted(list(target_genes))
    datasets_padded = {}

    for label in list(datasets.keys()):
        adata = datasets[label]
        missing_genes = list(set(sorted_target_genes) - set(adata.var_names)) 

        # Create the sparse zero matrix for missing genes
        if len(missing_genes) > 0:
            zero_data = csr_matrix((adata.n_obs, len(missing_genes)), dtype=np.float32)
            
            # convert back to adata
            adata_missing = ad.AnnData(X=zero_data, obs=adata.obs)
            adata_missing.var_names = missing_genes
            
            # Concatenate the existing data with the missing zero-padded data
            adata_padded = ad.concat([adata, adata_missing], axis=1, merge='same')
        else:
            adata_padded = adata.copy()
            
        # Reorder the columns to match alphabetical target list
        adata_ordered = adata_padded[:, sorted_target_genes].copy()

        # add padded adata to datasets
        datasets_padded.update({label: adata_padded})
        
        print(f"Final tensor-ready shape: {adata_ordered.shape}\n")
    return datasets_padded

# TESTING
base_path = "/data/shared/alzgene26/data"
data_path = base_path + "/processed_data/completed/mg_200_mc_200_mhvg1000/"
comp_proc_data_path = "/data/users/thomath/kand/data/processed_data/extracted_from_completed/"

def pipeline() -> None:
    print("Reading masks...")
    masks = read_masks(MASK_PATHS)

    print("Reading data into datasets...")
    datasets = ctts.read_files(to_include=ALL_CELLTYPES, filepath=comp_proc_data_path)

    print("Aligning adatas to BINN...")
    datasets_aligend = align_genes(datasets, masks['df0'])

    print("Padding adatas to BINN-ready shape...")
    datasets_padded = pad_data(datasets_aligend, masks["df0"])

    # ONLY RUN ONCE ON LARGE FILES
    #print("Processing datasets...")
    #datasets_proc = process_completed_data(datasets)
    #print("Saving data...")
    #save_data(datasets_proc, comp_proc_data_path)
    
    print("Pipeline completed!")


if __name__ == "__main__":
    pipeline()
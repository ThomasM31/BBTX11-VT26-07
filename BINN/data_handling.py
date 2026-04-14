import custom_train_test_split as ctts
import anndata as ad
from anndata.experimental import AnnCollection, AnnLoader
from binn_training import *
import os
import pandas as pd

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

def pad_data(datasets: dict):
    """
    Pads missing genes with zeros
    """

    return datasets_padded

def align_data(datasets: dict):
    """
    Sort adata in datasets alphabetically for aligning with BINN
    """
    return datasets_aligned

# TESTING
base_path = "/data/shared/alzgene26/data"
data_path = base_path + "/processed_data/completed/mg_200_mc_200_mhvg1000/"
save_path = "/data/users/thomath/kand/data/processed_data/extracted_from_completed/"

def pipeline() -> None:
    """
    Run steps to load large datafiles and fetch important information for training/testing
    """
    print("Reading data into datasets...")
    datasets = ctts.read_files(to_include=ALL_CELLTYPES, filepath=data_path)

    print("Processing datasets...")
    datasets_proc = process_completed_data(datasets)

    print("Saving data...")
    save_data(datasets_proc, save_path)

    print("Pipeline completed!")


if __name__ == "__main__":
    pipeline()
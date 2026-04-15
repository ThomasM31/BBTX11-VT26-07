from torch.utils.data import DataLoader
import custom_train_test_split as ctts
import anndata as ad
from anndata.experimental import AnnCollection, AnnLoader
import os
import pandas as pd, numpy as np
from scipy.sparse import csr_matrix
import torch.nn as nn
import torch
from Binn import BINN
import binn_training as bt

def data_concatenate(acollection : AnnCollection):
    """
    Concatenate the AnnCollection to a single AnnData object
    """
    adata_conc = ad.concat(acollection, label="cell_type_low_res")
    return adata_conc
    
def train_test_adatasplit(train_adata: ad.AnnData, test_adata: ad.AnnData):
    """
    Creates the train/test split for the input anndata
    """
    X_train = train_adata.X
    y_train = train_adata.obs["AD_status"]
    X_test = test_adata.X
    y_test = test_adata.obs["AD_status"]

    return X_train, y_train, X_test, y_test

def process_completed_data(datasets: dict) -> dict:
    datasets_proc = {}
    for label, dataset in datasets.items():
        print(f"fetching pseudo from {label}")
        # fetch pseudo-batched & preprocessed data
        datasets_proc[label] = dataset.uns['pseudo'].copy()
    return datasets_proc

def transpose_datasets(datasets:dict) -> dict:
    """
    Transposition necessary for feeding into network
    """
    datasets_t = {}
    for celltype in datasets.keys():
        adata_t = datasets[celltype].T
        datasets_t[celltype] = adata_t
        
    return datasets_t

def create_dataloaders(train_adata: ad.AnnData,
                       test_adata: ad.AnnData,
                       batch_size=16) -> tuple[AnnLoader, AnnLoader]:
    """
    Create dataloaders using built in loader for AnnData
    """
    train_loader = AnnLoader(train_adata, batch_size=batch_size, shuffle=True)
    test_loader = AnnLoader(test_adata, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader

def save_data(datasets: dict, filepath: str) -> None:
    """
    Saves anndata dataset to filepath
    """
    for label in list(datasets.keys()):
        print(f'Writing "{label}" to file.')
        to = os.path.join(filepath, f'{label}.h5ad')
        # Save individual files
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

def compute_features(masks:dict, device) -> tuple[int, list, list]:
    #Extracting pure number representation matrices
    mask_matrix_list = [masks[mask].to_numpy() for mask in masks]
    # Starting amount of features
    in_features = masks["df0"].shape[0]
    # Extract layer dimensions
    layers_list = [masks[mask].shape[1] for mask in masks]

    # Conversion for mask matrix list, creates tensors for BINN, transposed
    tensor_masks = [torch.tensor(mask).float().t() for mask in mask_matrix_list]
    # Put on device
    tensor_masks = [mask.to(device) for mask in tensor_masks]

    return in_features, layers_list, tensor_masks

def subset_genes(datasets: dict, input_masks: pd.DataFrame) -> dict:
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

def pad_align_data(datasets: dict, input_masks: pd.DataFrame) -> dict:
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
        datasets_padded.update({label: adata_ordered})
        
        #print(f"Final shape: {adata_ordered.shape}\n")
    return datasets_padded

def create_model(in_features:int, layers_list:list, tensor_masks:list, device, opt_learning_rate=0.001):
    model = BINN(in_features=in_features,
                  layers_list=layers_list,
                  mask_list=tensor_masks).to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=opt_learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3)

    return model, criterion, optimizer, scheduler

def training_loop(model: BINN, 
                  train_loader: AnnLoader, 
                  test_loader: AnnLoader, 
                  criterion, 
                  optimizer, 
                  device, 
                  scheduler, 
                  epochs:int) -> None:
    
    for epoch in range(epochs):
        train_loss, train_acc = bt.train_one_epoch(model, train_loader, criterion, optimizer, device)
        test_loss, test_acc = bt.test_one_epoch(model, test_loader, criterion, device)
        
        scheduler.step(test_loss)
        
        print(f"Epoch {epoch+1} / {epochs}")
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"Test Loss:  {test_loss:.4f} | Test Acc:  {test_acc:.4f}")
        print("-" * 30)

# TESTING
base_path = "/data/shared/alzgene26/data"
data_path = base_path + "/processed_data/completed/mg_200_mc_200_mhvg1000/"
comp_proc_data_path = "/data/users/thomath/kand/data/processed_data/extracted_from_completed/"

def pipeline() -> None:
    # GLOBALS
    EPOCHS = 15
    TRAIN_SIZE = 0.8
    ALL_CELLTYPES = [0,1,2,3,4,5,6,7,8]
    MASK_PATHS = [f"/data/shared/alzgene26/PathwayData/MaskMatrixLayers/mg_200_mc_200_mhvg1000/oligo_exc3_exc2_vasc_immune_astro_inhi_opcs_exc1_layer_{i}_mask.csv" 
                for i in range(5)]

    print("Fetching device...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Reading masks...")
    masks = read_masks(MASK_PATHS)

    print("Computing BINN features...")
    in_features, layers_list, tensor_masks = compute_features(masks, device)

    print("Creating BINN...")
    model, criterion, optimizer, scheduler = create_model(in_features, layers_list, tensor_masks, device, opt_learning_rate=0.001)

    print("Reading data into datasets...")
    datasets = ctts.read_files(to_include=ALL_CELLTYPES, filepath=comp_proc_data_path)

    print("Aligning adatas to BINN...")
    datasets_aligend = subset_genes(datasets, masks['df0'])

    print("Padding adatas to BINN-ready shape...")
    datasets_padded = pad_align_data(datasets_aligend, masks["df0"])

    print("Creating AnnCollection...")
    acollection = ctts.create_encoded_collection(datasets_padded)

    print("Creating train/test split...")
    train_adata, test_adata = ctts.custom_train_test_split(acollection, train_size=TRAIN_SIZE)

    print("Getting dataloaders...")
    train_loader, test_loader = create_dataloaders(train_adata, test_adata)

    print("Running train/test loop")
    training_loop(model, train_loader, test_loader, criterion, optimizer, device, scheduler, EPOCHS)

    # ONLY RUN ONCE ON LARGE FILES
    #print("Processing datasets...")
    #datasets_proc = process_completed_data(datasets)
    #print("Saving data...")
    #save_data(datasets_proc, comp_proc_data_path)
    
    print("Pipeline completed!")

if __name__ == "__main__":
    pipeline()


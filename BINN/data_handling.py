# Own files
import custom_train_test_split as ctts
import optuna
from Binn import BINN
import binn_training as bt

import anndata as ad
from anndata.experimental import AnnCollection, AnnLoader
from sklearn.metrics import roc_auc_score
import os
import pandas as pd, numpy as np
from scipy.sparse import csr_matrix
import torch.nn as nn
import torch
import scanpy as sc
from sklearn.model_selection import StratifiedKFold

# TESTING
base_path = "/data/shared/alzgene26/data"
data_path = base_path + "/processed_data/completed/full_pipeline/mg_200_mc_200_mhvg1000/"

# GLOBALS
EPOCHS = 40
TRAIN_SIZE = 0.8
ALL_CELLTYPES = [0,1,2,3,4,5,6,7,8]
MASK_PATHS = [f"/data/shared/alzgene26/PathwayData/MaskMatrixLayers/full_pipeline/mg_200_mc_200_mhvg1000/oligo_exc3_exc2_vasc_immune_astro_inhi_opcs_exc1_layer_{i}_mask.csv" 
            for i in range(5)]


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
    """
    Copy pseudobulked layer from large anndata files
    """
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

def compute_features(masks:dict, device) -> tuple[list, int, list, list]:
    """
    Extract biological layer information for the BINN. 
    Returns:
        mask_matrix_list(list): List of mask matrices in order

        in_features(integer): Number of input featues (e.g. number of relevant genes).

        layer_list(list of integers): A list defining the number of neurons in each
            biological layer. The length of the list determines the amount of layers.

        mask_list(list of torch.Tensor): A list of binary tensors used to restrict the 
            connectivity between layers.
            Mask 0 shape: (layer_list)
    """
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

    print(f"input features: {in_features}")
    print(f"layer list: {layers_list}")

    return mask_matrix_list, in_features, layers_list, tensor_masks

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
        overlapping_genes = [g for g in target_genes if g in adata.var_names]
        
        # Subset the dataset
        adata_aligned = adata[:, overlapping_genes].copy()
        # Update dictionary
        datasets_aligned.update({label: adata_aligned})

        print(f"Overlapping genes kept: {adata_aligned.n_vars} for {label}")
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

def create_global_with_missing_patients(datasets_dict: dict) -> dict:
    """
    Concatenates adata to desired shape (427,945) across celltypes
    Args:
        datasets_dict: dict of adatas per low res celltype (preprocessed + aligned)

    """    
    all_subjects = set()
    for adata in datasets_dict.values():
        all_subjects.update(adata.obs_names)
    
    master_subjects = sorted(list(all_subjects)) 
    # Let's ensure gene_names are clean strings
    gene_names = list(datasets_dict[list(datasets_dict.keys())[0]].var_names.astype(str))
    print(f"  Total unique subjects found: {len(master_subjects)}")

    global_df = pd.DataFrame(0.0, index=master_subjects, columns=gene_names)
    total_cells_series = pd.Series(0, index=master_subjects)

    for cell_type, adata in datasets_dict.items():
        # Ensure var_names are strings to match gene_names
        adata.var_names = adata.var_names.astype(str)
        
        current_df = pd.DataFrame(
            adata.X.toarray() if hasattr(adata.X, 'toarray') else adata.X,
            index=adata.obs_names,
            columns=adata.var_names
        )
        
        # SAFE ALIGNMENT: Force both rows AND columns to match exactly
        current_df_aligned = current_df.reindex(index=master_subjects, columns=gene_names, fill_value=0.0)
        
        # SAFE ADDITION: .add() is much safer than += 
        global_df = global_df.add(current_df_aligned, fill_value=0.0)
        
        cell_counts = adata.obs['n_obs_aggregated'].reindex(master_subjects, fill_value=0)
        total_cells_series = total_cells_series.add(cell_counts, fill_value=0)
        
        print(f"  + Added {cell_type} ({len(adata)} subjects)")

    # SAFETY CHECK: Did Pandas introduce NaNs?
    if global_df.isna().sum().sum() > 0:
        print("  [!] WARNING: NaNs detected immediately after Pandas aggregation.")
        global_df = global_df.fillna(0.0)

    global_adata = ad.AnnData(X=csr_matrix(global_df.values))
    global_adata.obs_names = global_df.index.astype(str)
    global_adata.var_names = global_df.columns.astype(str)

    # Restore AD_status
    master_label_map = {}
    for adata in datasets_dict.values():
        current_map = adata.obs['AD_status'].to_dict()
        master_label_map.update(current_map)

    global_adata.obs['subject'] = global_adata.obs_names
    global_adata.obs['AD_status'] = global_adata.obs_names.map(master_label_map)
    global_adata.obs['total_cells_all_types'] = total_cells_series.values

    # SAFETY CHECK: The "Dead Patient" Check
    # If a patient has 0 counts, normalize_total will crash the matrix.
    row_sums = np.array(global_adata.X.sum(axis=1)).flatten()
    dead_patients = np.where(row_sums == 0)[0]
    
    if len(dead_patients) > 0:
        print(f"Removing {len(dead_patients)} patients with absolutely ZERO counts to prevent NaN Cascade!")
        global_adata = global_adata[row_sums > 0].copy()

    # fetch sparse matrix into memory
    dense_matrix = global_adata.X.toarray() if hasattr(global_adata.X, 'toarray') else global_adata.X
    
    # SAFETY CHECK: Negative Values
    if np.any(dense_matrix < 0):
         print("CRITICAL WARNING: Negative values found BEFORE log1p! Input datasets already scaled?")

    # Calculate variance
    variance = np.var(dense_matrix, axis=0)
    zero_var_genes = np.where(variance == 0)[0]

    # Remove genes with no variance (no information)
    if len(zero_var_genes) > 0:
        print(f"  [!] Removing {len(zero_var_genes)} genes with zero variance...")
        global_adata = global_adata[:, variance > 0].copy()

    # Normalization Pipeline
    print("Running Scanpy normalization...")
    sc.pp.normalize_total(global_adata, target_sum=1e4)
    sc.pp.log1p(global_adata)
    sc.pp.scale(global_adata, max_value=10)

    # Final verification
    if np.isnan(global_adata.X).any():
        print("FAILURE: Matrix still contains NaNs after scaling.")
    else:
        print("Matrix is clean.")

    print(f"Done! Final Global shape: {global_adata.shape}")
    return global_adata

def rollup_to_patient_level(datasets: dict) -> dict:
    """
    Pseudobulk per patient only
    """
    patient_level_datasets = {}
    
    for label in (datasets.keys()):
        adata = datasets[label]
        print(f"Rolling up '{label}'...")
        
        # Convert the sums and the subject names to a DataFrame
        # .layers['sum'] gives raw counts
        df = pd.DataFrame(
            adata.layers['sum'], 
            index=adata.obs['subject'], 
            columns=adata.var_names
        )
        
        # Group by subject and sum
        summed_df = df.groupby(level=0, observed=False).sum()
        
        # Create the new AnnData from the summed DataFrame
        patient_pseudo = ad.AnnData(X=summed_df.values)
        patient_pseudo.obs_names = summed_df.index.astype(str)
        patient_pseudo.var_names = summed_df.columns.astype(str)
        
        # Re-attach metadata (AD_status, etc.)
        meta = adata.obs.groupby('subject', observed=False).agg({
            'AD_status': 'first',
            'n_obs_aggregated': 'sum'
        })
        
        # Align metadata with the new rows
        patient_pseudo.obs['subject'] = patient_pseudo.obs_names
        patient_pseudo.obs['n_obs_aggregated'] = meta.loc[patient_pseudo.obs_names, 'n_obs_aggregated']
        patient_pseudo.obs["cell_type_high_resolution"] = label
        patient_pseudo.obs['AD_status'] = meta.loc[patient_pseudo.obs_names, 'AD_status']

        # Convert to sparse matrix (Scanpy prefers this for memory)
        patient_pseudo.X = csr_matrix(patient_pseudo.X)
        
        patient_level_datasets[label] = patient_pseudo
        
    return patient_level_datasets

def poison_scanner(dataloader: AnnLoader, device) -> None:
    """
    Checks for NaN, extreme values
    """
    for i, batch in enumerate(dataloader):
        inputs = batch.X.float().to(device)
        if type(batch.obs["AD_status"]) is pd.Series:
            labels = torch.tensor(batch.obs['AD_status'].values.astype(float)).float().reshape(-1, 1).to(device)
        else:
            labels = batch.obs['AD_status'].detach().clone().float().reshape(-1, 1).to(device)
        # Check for NaNs
        if torch.isnan(inputs).any():
            print(f"CRITICAL ERROR: NaN found in inputs at batch {i}!")
            break
        
        # Check for extreme outliers
        max_val = inputs.max().item()
        min_val = inputs.min().item()
        if max_val > 50.0 or min_val < -50.0:
            print(f"WARNING: Extreme values in batch {i} -> Max: {max_val:.2f}, Min: {min_val:.2f}")
            
        # Check for label NaNs
        if torch.isnan(labels).any():
            print(f"CRITICAL ERROR: NaN found in labels at batch {i}!")
            break
    
    print("No poison found!")

def renormalize(datasets:dict) -> dict:
    """
    Renormalize data after summing raw counts per subject
    """
    for label, adata in datasets.items():
        # Verification of the matrix range
        raw_max = adata.X.max()
        
        # Run the pipeline (Ignore the warning)
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)
        sc.pp.scale(adata, max_value=10)
        
        scaled_max = adata.X.max()
        scaled_mean = adata.X.mean()
        
        print(f"--- {label} ---")
        print(f"Pre-scaling Max: {raw_max:.2f}")
        print(f"Post-scaling Max: {scaled_max:.2f} (Should be near 10)")
        print(f"Post-scaling Mean: {scaled_mean:.4f} (Should be near 0)")
    return datasets

def create_model(in_features:int, 
                 layers_list:list, 
                 tensor_masks:list, 
                 device, 
                 lr=1e-4, 
                 weight_decay=1e-4): 
                 #dropout_opt=0.3):
    """
    Instantiate BINN and accompanying criterion, optimizer and scheduler
    """
    model = BINN(in_features=in_features,
                  layers_list=layers_list,
                  mask_list=tensor_masks).to(device)
                  #dropout_p=dropout_opt)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=75, eta_min=1e-6)

    return model, criterion, optimizer, scheduler

def training_loop(model: BINN, 
                  train_loader: AnnLoader, 
                  test_loader: AnnLoader, 
                  criterion, 
                  optimizer, 
                  device, 
                  scheduler, 
                  epochs:int) -> dict:
    """
    Performs the entire training & testing loop over the epochs
    """
    # Metrics dictionary
    history = {'train_loss': [], 'train_acc': [], 'test_loss': [], 'test_acc': []}

    for epoch in range(epochs):
        train_loss, train_acc = bt.train_one_epoch(model, train_loader, criterion, optimizer, device)
        test_loss, test_acc = bt.test_one_epoch(model, test_loader, criterion, device)
        
        scheduler.step() #test_loss som argument?
        
        print(f"Epoch {epoch:3d} | "
                f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} || "
                f"Val Loss: {test_loss:.4f} | Val Acc: {test_acc:.4f}")
        
        # Save metrics
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['test_loss'].append(test_loss)
        history['test_acc'].append(test_acc)
    
    return history

def dead_output_check(model: BINN, mask_matrix_list: list) -> None:
    """
    Check the mask matrix list for dead outputs in all layers
    """
    for i in range(len(mask_matrix_list)):
        mask = getattr(model, f'mask_{i}')
        # Check for rows (output neurons) with no active inputs
        dead_outputs = (mask.sum(dim=1 if mask.shape[1] > mask.shape[0] else 0) == 0).sum()
        if dead_outputs > 0:
            print(f"Layer {i} has {dead_outputs} dead nodes with zero biological connections!")
        else:
            print(f"Layer {i} has no dead outputs")
    
def evaluate_model_roc(model, test_loader: AnnLoader, device) -> tuple[np.array, np.array]:
    """
    Evaluate model with ROC-AUC
    """
    model.eval()
    all_probs = []
    all_labels = []
    
    with torch.no_grad():
        for batch in test_loader:
            inputs = batch.X.float().to(device)

            if type(batch.obs["AD_status"]) is pd.Series:
                labels = torch.tensor(batch.obs['AD_status'].values.astype(float)).float().reshape(-1, 1).to(device)
            else:
                labels = batch.obs['AD_status'].detach().clone().float().reshape(-1, 1).to(device)
            
            logits = model(inputs)
            # Apply sigmoid to get probabilities [0, 1]
            probs = torch.sigmoid(logits).cpu().numpy()
            
            all_probs.extend(probs)
            all_labels.extend(labels)
            
    # fetch probabilites and target labels
    probs, targets = np.array(all_probs).flatten(), np.array(all_labels).flatten()

    # Calculate AUC
    auc_score = roc_auc_score(targets, probs)

    return probs, targets, auc_score

def fetch_best_metrics(history:list) -> None:
    """
    Best metrics from Training & Testing
    Args:
        history(list): list of train losses, train accuracies, test losses, test accuracies over epochs
    """
    best_train_acc = sorted(history["train_acc"],reverse=True)[0]
    best_test_acc = sorted(history["test_acc"],reverse=True)[0]

    best_train_loss = sorted(history["train_loss"])[0]
    best_test_loss = sorted(history["test_loss"])[0]

    print(f"Best train Loss: {best_train_loss:.4f} | Best train acc: {best_train_acc:.4f} || "
                    f"Best test Loss: {best_test_loss:.4f} | Best test acc: {best_test_acc:.4f}")

def run_cross_validation(adata, 
                        in_features:int, 
                        layers_list:list, 
                        tensor_masks:list, 
                        device, 
                        lr=1e-4,
                        weight_decay=1e-2,
                        k=5, 
                        epochs=70) -> list:
    """
    Cross validate BINN
    Args:
        adata(ad.Anndata): global concatenated anndata
        k(int): number of cross-val splits
        epochs(int): 

    Returns:
        AUC scores for each fold
    """
    # Prepare indices and labels
    X_indices = np.arange(adata.n_obs)
    y_labels = adata.obs['AD_status'].values.astype(float)
    
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
    fold_aucs = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X_indices, y_labels)):
        print(f"\n--- Starting Fold {fold+1}/{k} ---")
        
        # Create subsets
        train_sub = adata[train_idx].copy()
        val_sub = adata[val_idx].copy()
        
        # Initialize fresh loaders
        train_loader = AnnLoader(train_sub, batch_size=32, shuffle=True)
        val_loader = AnnLoader(val_sub, batch_size=32, shuffle=False)
        
        # RE-INITIALIZE MODEL for each fold
        binn, criterion, optimizer, _ = create_model(in_features, layers_list, tensor_masks, device, 
                                                        lr=lr, weight_decay=weight_decay)
        
        best_fold_auc = 0
        
        # Training Loop for this fold
        for epoch in range(epochs):
            train_loss, train_acc = bt.train_one_epoch(binn, train_loader, criterion, optimizer, device)
            #test_loss, test_acc = bt.test_one_epoch(binn, val_loader, criterion, device)

            # Evaluate AUC at end of epoch
            probs, targets, auc_score = evaluate_model_roc(binn, val_loader, device)
            current_auc = roc_auc_score(targets, probs)
            
            if current_auc > best_fold_auc:
                best_fold_auc = current_auc
        
        print(f"Fold {fold+1} Best Test AUC: {best_fold_auc:.4f}")
        fold_aucs.append(best_fold_auc)

    # Final Results
    print("\n" + "="*30)
    print(f"Mean ROC-AUC: {np.mean(fold_aucs):.4f} +/- {np.std(fold_aucs):.4f}")
    print("="*30)
    return fold_aucs

def hyperparameter_tuning_optuna(adata, in_features, layers_list, tensor_masks, device) -> dict:
    """
    Hyperparameter tune the BINN model in regard to learning rate and weight decay
    """
    def objective(trial) -> float:
        # Let Optuna suggest the hyperparameters for this run
        lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
        weight_decay = trial.suggest_float("weight_decay", 1e-4, 1e-1, log=True)
        
        # Run your existing CV function, passing the suggested params
        fold_aucs = run_cross_validation(adata, in_features, layers_list, tensor_masks, 
                                            device, k=5, epochs=75, lr=lr, weight_decay=weight_decay)
        
        mean_auc = np.mean(fold_aucs)
        
        # Optuna will try to MAXIMIZE this returned value
        return mean_auc

    # Create the study and optimize!
    print("Starting Optuna Hyperparameter Search...")
    study = optuna.create_study(direction="maximize")

    # Run 30 trials (takes time, but worth it)
    study.optimize(objective, n_trials=30)

    # View the results
    print("\n=== Best Hyperparameters ===")
    print(study.best_params)
    print(f"Best Mean CV ROC-AUC: {study.best_value:.4f}")

    return study.best_params
    # Best recorded: {'lr': 0.0017772099~1.778e-3, 'weight_decay': 0.025921115~2.592e-2}

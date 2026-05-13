from BINN import data_handling as dh
from BINN import custom_train_test_split as ctts
from BINN import shap_explainer
import torch
import torch.nn as nn
import pandas as pd
import anndata as ad
import pickle
from pathlib import Path
import importlib.util
import argparse
from datetime import datetime as dt

# import module for consistent paths
path = Path(__file__).resolve().parent.parent / "pipeline_paths.py"
spec = importlib.util.spec_from_file_location("ppaths", path)
ppaths = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ppaths)

path = Path(__file__).resolve().parent.parent / "pipeline_paths_generalize.py"
spec = importlib.util.spec_from_file_location("gpaths", path)
gpaths = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gpaths)

# PATHS
# paths for train / test data
ttpaths = ppaths.PipelinePaths(True, 'mg_200_mc_200_mhvg1000')
tt_data_path = ttpaths.compl_full_pipe_path
mask_path = ttpaths.mask_full_pipe_path

# paths for generalizability data
gen_paths = gpaths.PipelinePaths(True, 'mg_200_mc_200_mhvg1000')
gen_data_path = ttpaths.compl_full_pipe_path

# GLOBALS
MASK_PATHS = [mask_path / f"oligo_exc3_exc2_vasc_immune_astro_inhi_opcs_exc1_layer_{i}_mask.csv" 
            for i in range(5)]
LR = 5e-4
WEIGHT_DECAY = 0.3
DROPOUT = 1.699e-1
BATCH_SIZE = 32
ACTIVATION_FN = nn.Tanh()


def pipeline(date: str, 
             m_paths=MASK_PATHS
             ):
    """
    Performs the full shap_pipeline(), including 
    """
    print("Starting pipeline...")
    print("Fetching device...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Reading masks & model files...")
    masks = dh.read_masks(m_paths)
    
    model_file        = ttpaths.binn_model_path / f"binn_model_{date}.pth"
    train_tensor_file = ttpaths.binn_model_path / f"X_train_tensor_{date}.pt"
    test_tensor_file  = ttpaths.binn_model_path / f"X_test_tensor_{date}.pt"
    gen_tensor_file   = ttpaths.binn_model_path / f"X_gen_tensor_{date}.pt"

    print("Loading tensors...")
    X_train_tensor = torch.load(train_tensor_file).to(device)
    X_test_tensor  = torch.load(test_tensor_file).to(device)
    X_gen_tensor   = torch.load(gen_tensor_file).to(device)

    # Recreate the architecture 
    print("Computing BINN features...")
    mask_matrix_list, in_features, layers_list, tensor_masks = dh.compute_features(masks, device)

    model, criterion, optimizer, scheduler = dh.create_model(in_features, layers_list, tensor_masks, 
                                                             device, lr=LR, weight_decay=WEIGHT_DECAY,
                                                             dropout=DROPOUT, activation_fn=ACTIVATION_FN)

    # Load the model weights
    model.load_state_dict(torch.load(model_file, map_location=device))
    model.eval() # Set to evaluation mode for SHAP

    print("Runnning shap for ROSMAP data set & generalizability data set")
    gene_names = masks['df0'].index.tolist()
    shap_explanation = shap_explainer.perform_shap(model, X_train_tensor, X_test_tensor, gene_names, ttpaths.figures_path_shap, 'ROSMAP', date)

    shap_explanation = shap_explainer.perform_shap(model, X_train_tensor, X_gen_tensor, gene_names, ttpaths.figures_path_shap, 'gen', date)

    # Save the shap explanation object
    shap_filename = ttpaths.binn_model_path / f"shap_explanation_{date}.pkl"
    with open(shap_filename, 'wb') as f:
        pickle.dump(shap_explanation, f)
    
    print("Creating layered SHAP...")
    shap_df = shap_explainer.layerwise_shap(model, X_train_tensor, X_test_tensor, masks, device)
    shap_filename = ttpaths.binn_model_path / f"shap_explanation_layered_{date}.pkl"
    with open(shap_filename, 'wb') as f:
        pickle.dump(shap_df, f)

    print("Pipeline completed!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run annData objects through BINN pipeline")

    parser.add_argument(
        "date", 
        type=str,
        help="format YYMMDD_HHMM"
    )

    # Optional argument tune_hyperparameters
    parser.add_argument(
        "--tune_hyperparameters", 
        type=bool,
        default=False,
        help="Tune hyperparameters or not, takes some time"
    )

    # Optional argument epochs
    parser.add_argument(
        "--epochs", 
        type=int,
        default=250,
        help="Amount of epochs to run network for"
    )

    args = parser.parse_args()

    # Call pipeline with terminal arguments
    pipeline(date=args.date)
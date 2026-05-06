import argparse
import data_handling as dh
import custom_train_test_split as ctts
import torch
import torch.nn as nn
import shap_explainer
import pandas as pd
from pathlib import Path
import importlib.util
import anndata as ad

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


def pipeline(to_include=list, 
             epochs=int, 
             d_path=tt_data_path, 
             m_paths=MASK_PATHS,
             tune_hyperparameters=False):
    """
    Performs the full binn_pipeline(), including 
    """
    print("Starting pipeline...")
    print("Fetching device...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Reading masks...")
    masks = dh.read_masks(m_paths)

    print("Computing BINN features...")
    mask_matrix_list, in_features, layers_list, tensor_masks = dh.compute_features(masks, device)

    print("Creating BINN...")
    model, criterion, optimizer, scheduler = dh.create_model(in_features, layers_list, tensor_masks, 
                                                             device, lr=LR, weight_decay=WEIGHT_DECAY,
                                                             dropout=DROPOUT, activation_fn=ACTIVATION_FN)

    print("Showing BINN...")
    print(model)

    print("Searching for dead biological nodes...")
    dead_nodes_dict = dh.find_dead_outputs(model, tensor_masks)

    print("Showing dead nodes...")
    dh.show_dead_nodes(dead_nodes_dict, masks)

    print("Reading data into datasets...")
    datasets = ctts.read_files(to_include=to_include, filepath=d_path)

    print("Rolling up adata to patient level...")
    patient_datasets = dh.rollup_to_patient_level(datasets)

    print("Aligning adatas to BINN...")
    datasets_aligend = dh.subset_genes(patient_datasets, masks['df0'])

    print("Padding adatas to BINN-ready shape...")
    datasets_padded = dh.pad_align_data(datasets_aligend, masks["df0"])

    print("Starting Global Rollup with missing subject handling...")
    adata_global = dh.create_global_with_missing_patients(datasets_padded)

    print("Creating train/test split...")
    train_subjects = pd.read_csv(ttpaths.metadata_path / 'train_subjects.tsv')
    train_adata, test_adata = ctts.predefined_subject_train_test_split(adata_global, train_subjects)

    print("Getting dataloaders...")
    train_loader, test_loader = dh.create_dataloaders(train_adata, test_adata, batch_size=BATCH_SIZE)

    print("Running train/test loop...")
    history = dh.training_loop(model, train_loader, test_loader, criterion, optimizer, device, scheduler, epochs)

    print("Fetching metrics...")
    metrics = dh.fetch_best_metrics(history)
    print(f"BINN Metrics: {metrics}")

    print("Generating and saving test predictions for visualization...")
    df_res = dh.save_test_results(model, test_loader, device)

    print("Calculating F1-score...")
    score = dh.f1_calculator(df_res)

    if tune_hyperparameters:
        best_params = dh.hyperparameter_tuning_optuna(adata_global, in_features, layers_list, tensor_masks, 
                                                device, batch_size=BATCH_SIZE, activation_fn=ACTIVATION_FN,
                                                k=5, epochs=epochs)
        print(f"Best parameters for BINN: {best_params}")
    
    print("Perform SHAP analysis...")
    X_train_tensor = torch.tensor(train_adata.X.toarray(), dtype=torch.float32).to(device)
    X_test_tensor = torch.tensor(test_adata.X.toarray(), dtype=torch.float32).to(device)
    gene_names = masks['df0'].index.tolist()
    shap_explainer.perform_shap(model, X_train_tensor, X_test_tensor, gene_names, ttpaths.figures_path_shap)

    # generalizability data
    gen_adata = ad.read_h5ad(gen_paths.compl_full_pipe_path / 'all.h5ad')
    gen_datasets = {'all' : gen_adata}
    
    gen_datasets = dh.rollup_generalizability_data(gen_datasets)

    gen_datasets_aligned = dh.subset_genes(gen_datasets, masks['df0'])

    gen_datasets_padded = dh.pad_align_data(gen_datasets_aligned, masks['df0'])
    
    gen_adata_global = dh.create_global_from_gen_data(gen_datasets_padded['all'])

    gen_loader = dh.create_generalizability_dataloader(gen_adata_global['all'], BATCH_SIZE)
    
    # test the model with the generalizability data 
    gen_loss, gen_acc = dh.generalizability_test(model, gen_loader, criterion, device)

    print(f"Generalizability Test - Loss: {gen_loss:.4f}, Accuracy: {gen_acc:.4f}")

    print("Pipeline completed!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run annData objects through BINN pipeline")
    parser.add_argument("to_include", type=int, nargs='+', help="indices to include")

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
    pipeline(to_include=args.to_include, 
             tune_hyperparameters=args.tune_hyperparameters,
             epochs=args.epochs)
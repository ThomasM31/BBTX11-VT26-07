import argparse
import data_handling as dh
import custom_train_test_split as ctts
import torch
import torch.nn as nn

# GLOBALS
EPOCHS = 200
TRAIN_SIZE = 0.8
ALL_CELLTYPES = [0,1,2,3,4,5,6,7,8]
MASK_PATHS = [f"/data/shared/alzgene26/PathwayData/MaskMatrixLayers/full_pipeline/mg_200_mc_200_mhvg1000/oligo_exc3_exc2_vasc_immune_astro_inhi_opcs_exc1_layer_{i}_mask.csv" 
            for i in range(5)]
LR = 9.76e-3
WEIGHT_DECAY = 9.96e-2
DROPOUT = 1.699e-1
BATCH_SIZE = 32
ACTIVATION_FN = nn.Tanh()

# PATHS
base_path = "/data/shared/alzgene26/data"
data_path = base_path + "/processed_data/completed/full_pipeline/mg_200_mc_200_mhvg1000/"

def pipeline(to_include=ALL_CELLTYPES, 
             epochs=EPOCHS, 
             train_size=TRAIN_SIZE, 
             d_path=data_path, 
             m_paths=MASK_PATHS):
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
    train_adata, test_adata = ctts.custom_train_test_split(adata_global, train_size=train_size)

    print("Getting dataloaders...")
    train_loader, test_loader = dh.create_dataloaders(train_adata, test_adata, batch_size=BATCH_SIZE)

    print("Running train/test loop...")
    history = dh.training_loop(model, train_loader, test_loader, criterion, optimizer, device, scheduler, epochs)

    print("Fetching metrics...")
    best_train_acc_i, best_test_acc_i, best_train_loss_i, best_test_loss_i = dh.fetch_best_metrics(history)

    print("Generating and saving test predictions for visualization...")
    df_res = dh.save_test_results(model, test_loader, device)

    print("Pipeline completed!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run annData objects through BINN pipeline")
    parser.add_argument("to_include", type=int, nargs='+', help="indices to include")
    
    args = parser.parse_args()

    # Call pipeline with terminal arguments
    pipeline(to_include=args.to_include)
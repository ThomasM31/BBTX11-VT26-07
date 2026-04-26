import argparse
import os
from Binn import BINN
import binn_training as bt, data_handling as dh, custom_train_test_split as ctts
import torch

def pipeline(MASK_PATHS: list[str],
            TRAIN_SIZE: float,
            EPOCHS: int,
            to_include: list, 
            data_path: str,
            in_features: int,
            layers_list: list
            ) -> None:
    """
    TODO: UPDATE !!!
    - Load completed+preprocessed .h5ad data
    - Create dataloaders
    - Read pathways + create mask matrices
    - Init BINN
    - Feed data:
        # Train BINN
        # Test BINN
    - Evaluate + Interpret BINN
    - Compare to SVM
    """
    print("Starting pipeline...")

    print("Fetching device...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Reading masks...")
    masks = dh.read_masks(MASK_PATHS)

    print("Computing BINN features...")
    in_features, layers_list, tensor_masks = dh.compute_features(masks, device)

    print("Creating BINN...")
    model, criterion, optimizer, scheduler = dh.create_model(in_features, layers_list, tensor_masks, device, opt_learning_rate=1e-4, weight_decay=1e-3)
    #model = ShallowMLP(in_features=945, hidden_size=128).to(device)
    #optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=0)
    #criterion = nn.BCEWithLogitsLoss()
    print(model)
    scheduler = 0

    print("Reading data into datasets...")
    datasets = ctts.read_files(to_include=to_include, filepath=data_path)

    print("Rolling up adata to patient level...")
    patient_datasets = dh.rollup_to_patient_level(datasets)

    print("Aligning adatas to BINN...")
    datasets_aligend = dh.subset_genes(patient_datasets, masks['df0'])

    # TODO: Delete? Should not be needed?
    print("Padding adatas to BINN-ready shape...")
    datasets_padded = dh.pad_align_data(datasets_aligend, masks["df0"])

    print("Starting Global Rollup with missing subject handling...")
    adata_global = dh.create_global_with_missing_patients(datasets_padded)

    #print("Creating AnnCollection...")
    #acollection = ctts.create_encoded_collection(datasets_aligend)

    print("Creating train/test split...")
    train_adata, test_adata = ctts.custom_train_test_split(adata_global, train_size=TRAIN_SIZE)

    print("Getting dataloaders...")
    train_loader, test_loader = dh.create_dataloaders(train_adata, test_adata)

    print("Running train/test loop")
    history = dh.training_loop(model, train_loader, test_loader, criterion, optimizer, device, scheduler, EPOCHS)

    print("Fetching metrics...")
    dh.fetch_best_metrics(history)

    print("Pipeline completed!")


if __name__ == "__main__":
    # TODO: UPDATE
    parser = argparse.ArgumentParser(
        description = "Run annData objects through BINN pipeline"
    )

    # Positional argument: accepts one or more integers
    parser.add_argument(
        "to_include",
        type=int,        
        nargs='+',       
        help="indices to include: \n0=astro \n1=exc1 \n2=exc2 \n3=exc3 \n4=immune \n5=inhi \n6=oligo \n7=opcs \n8=vasc",
    )

    pipeline()

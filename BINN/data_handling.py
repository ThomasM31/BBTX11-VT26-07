from torch.utils.data import DataLoader
import custom_train_test_split
import torch
import torch.nn as nn
import anndata as ad
from anndata.experimental import AnnCollection
from anndata.experimental.pytorch import AnnLoader
import SingleCellDataset
from BINN import Binn
from training_testing import *

ALL_CELLTYPES = [0,1,2,3,4,5,6,7,8]

def read_adata(indices: list, train_size=0.8):
    """
    Reads the training anndata, testing anndata and the collection they come from.
    Indicies indicate celltype. 
    """
    train_adata, test_adata, acollection = custom_train_test_split.pipeline(indices, train_size)
    return train_adata, test_adata, acollection

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

def get_dataloaders(train_adata : ad.AnnData, test_adata : ad.AnnData, batch_size=64):
    """
    Extracts data from AnnData and returns train and test DataLoaders.
    """
    X_train, X_test, y_train, y_test = train_test_adatasplit(train_adata, test_adata)

    train_dataset = SingleCellDataset(X_train, y_train)
    test_dataset = SingleCellDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader


## TESTING
print("Reading adata...")
train_adata, test_adata, acollection = read_adata(ALL_CELLTYPES, train_size=0.8)
print("Getting dataloaders...")
train_loader, test_loader = get_dataloaders(train_adata, test_adata)
print("Concatenating data...")
adata_conc = data_concatenate(acollection)

# ?????????
in_features = 2000

# In this example: 500 pathways, 50 processes, classify AD
layers_list = [500, 50, 2]

# What here??????
mask_list = []

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = Binn.BINN(in_features=in_features,
                  layers_list=layers_list,
                  mask_list=mask_list).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

EPOCHS = 20

for epoch in range(EPOCHS):
    train_loss, train_acc = train_binn(model, train_loader, criterion, optimizer, device)
    test_loss, test_acc = test_binn(model, test_loader, criterion, device)
    
    print(f"Epoch {epoch+1} / {EPOCHS}")
    print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
    print(f"Test Loss:  {test_loss:.4f} | Test Acc:  {test_acc:.4f}")
    print("-" * 30)


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
import numpy as np
import scipy.sparse as sp
from preprocessing import custom_train_test_split
import anndata as ad

def read_adata(indices: list, train_size=0.8):
    train_adata, test_adata, acollection = custom_train_test_split.pipeline(indices, train_size)
    return train_adata, test_adata, acollection
    
def train_test_adatasplit(train_adata: ad.AnnData, test_adata: ad.AnnData):
    X_train = train_adata.X
    y_train = train_adata.obs["AD_status"]
    X_test = test_adata.X
    y_test = test_adata.obs["AD_status"]

    return X_train, y_train, X_test, y_test

ALL_CELLTYPES = [0,1,2,3,4,5,6,7,8]

class SingleCellDataset(ad.AnnData):
    """
    Custom Pytorch dataset for our anndata objects, for feeding into loaders
    """
    def __init__(self, X, y):
        self.X = X
        self.y = torch.LongTensor(y)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        # Convert sparse row to dense tensor on the fly to save memory
        row = self.X[idx]
        if sp.issparse(row):
            row = row.toarray().squeeze()
        return torch.FloatTensor(row), self.y[idx]
    

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import numpy as np
import scipy.sparse as sp
import anndata as ad

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
    

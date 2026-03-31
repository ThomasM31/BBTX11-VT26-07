import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# BINN 
from BINN.model import BINN
from BINN.trainer import train_binn, test_binn

from preprocessing import custom_train_test_split_modified
import anndata as ad

in_features=99999 # Integers of how many genes we have as input
layer_list=[] # List of integers, which describes how many layers and how many nodes each layer has
mask_list=[] # List of binary tensors (binary matrices) that restricts each layer

binn=BINN(in_features, layer_list, mask_list)

def read_adata(indices: list, train_size=0.8):
    train_adata, test_adata, collection = custom_train_test_split_modified.pipeline(indices, train_size)
    return train_adata, test_adata, collection
    
def xy_datasplit(train_adata: ad.AnnData, test_adata: ad.AnnData):
    X_train = train_adata.X
    y_train = train_adata.obs["AD_status"]
    X_test = test_adata.X
    y_test = test_adata.obs["AD_status"]

    return X_train, y_train, X_test, y_test

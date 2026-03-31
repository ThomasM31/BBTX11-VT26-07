import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# BINN 
from BINN.model import BINN
from BINN.trainer import train_binn, test_binn

import anndata as ad

# Internal
import BINN.reading_data as reading_data

ALL_CELLTYPES = [0,1,2,3,4,5,6,7,8]

def train_loader(train_adata,test_adata, in_features,layer_list, mask_list, training_epochs):

    binn=BINN(in_features,layer_list,mask_list)

    train_adata, test_adata, acollection = reading_data.read_adata(ALL_CELLTYPES, train_size=0.8)

    X_train, y_train, X_test, y_test = reading_data.xy_datasplit(train_adata,test_adata)
    
    # Make into FloatTensors 
    X_train = torch.FloatTensor(X_train)
    X_test = torch.FloatTensor(X_test)

    # This tensor stores 1 or 0 but as floats
    y_train=torch.FloatTensor(y_train)
    y_test=torch.FloatTensor(y_test)


    optimizer=torch.optim.Adam(binn.parameters(),lr=0.01) # Adam optimizer is a type of optimization algorithm
    # Used to minimize loss

    criterion=nn.BCEWithLogitsLoss() # Binary cross entropy loss function

    epochs=training_epochs
    losses=[]

    # Losses
    for i in range(epochs):
        y_pred=binn.forward(X_train) # Predicts y with training data
        loss=criterion(y_pred,y_train) # Measures loss by comparing predicted vs training y data

        # Keep track of our losses 
        losses.append(loss.detach().numpy())
        # print losses
        print(f'Epoch: {i} and loss: {loss}')

        optimizer.zero_grad() # Clears gradients
        loss.backward() # backward is a method from torch that computes the gradients for tensors 
        optimizer.step() # Uses the gradients from backward and updates weights. This is were binn is learning
    
    with torch.no_grad(): # Do not need to store gradients anymore
        logits = binn.forward(X_test) # Go through the neural network with the updated trained weights 
        probs = torch.sigmoid(logits) # Turn logits intp probabilities

        for i, p in enumerate(probs):
            print(f"{i+1}.) Probability: {p.item():.4f}") # Print the probabilities
            

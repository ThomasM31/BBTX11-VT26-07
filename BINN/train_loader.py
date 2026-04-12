import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from model import BINN
import custom_train_test_split
import anndata

############################################################

# Copied from Thomas baselinemodel.py  

def read_adata(indices: list, filepath: str, train_size=0.8):
    #train_adata, test_adata, collection = custom_train_test_split_modified.pipeline(indices, train_size)
    train_adata, test_adata, collection = custom_train_test_split.pipeline(indices, filepath, train_size)
    return train_adata, test_adata, collection
    
def xy_datasplit(train_adata: anndata.AnnData, test_adata: anndata.AnnData):
    X_train = train_adata.X
    y_train = train_adata.uns['pseudo'].obs["AD_status"]
    X_test = test_adata.X
    y_test = test_adata.uns['pseudo'].obs["AD_status"]

    return X_train, y_train, X_test, y_test

#############################################################



def train_loader(X_train,y_train,X_test,y_test,in_features,layer_list,mask_list,training_epochs):

    binn=BINN(in_features,layer_list,mask_list)

    #train_adata, test_adata, acollection = read_adata([0,1,2,3,4,5,6,7,8], train_size=0.8)

    #X_train, y_train, X_test, y_test = xy_datasplit(train_adata,test_adata)
    # We can  use same train test split as Thomas to get X and y training and test data
    # X will be the gene expression data
    # y will be the AD status data


    
    # Using the data to train

    # Make into FloatTensors 
    X_train = torch.FloatTensor(X_train)
    X_test = torch.FloatTensor(X_test)

    # This tensor stores 1 or 0 but as floats
    y_train=torch.FloatTensor(y_train)
    y_test=torch.FloatTensor(y_test)


    optimizer=torch.optim.Adam(binn.parameters(),lr=0.01) # Adam optimizer is a type of optimization algorithm
    # Used to minimize loss

    #optimizer=torch.optim.SGD(binn.parameters(),lr=0.01)
    # Another common optimizer we can try

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




def run():
    print("HELLO")


if __name__ == "__main__":
    run()
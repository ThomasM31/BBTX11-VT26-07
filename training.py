import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from binn_copy import BINN
from preprocessing import custom_train_test_split

in_features=99999# Integers of how many genes we have as input
layer_list=[] # List of integers, which describes how many layers and how many nodes each layer has
mask_list=[] # List of binary tensors (binary matrices) that restricts each layer

binn=BINN(in_features,layer_list,mask_list)

# In the example flower model they have imported train_test_split
# We have the custom_train_test_split which returns train_adata, test_adata

train_adata, test_adata = custom_train_test_split(0.8,0) # SPlit and celltype



# We can maybe use same train test split as Thomas to get X and y training and test data
# X will be the gene expression data
# y will be the AD status data

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

epochs=100
losses=[]

for i in range(epochs):
    y_pred=binn.forward(X_train) # Predicts y with training data
    loss=criterion(y_pred,y_train) # Measures loss by comparing predicted vs training y data

    # Keep track of our losses 
    losses.append(loss.detach().numpy())
    # print losses
    print(f'Epoch: {i} and loss: {loss}')

    optimizer.zero_grad() # Clears gradients
    loss.backward() # backward is a method from torch that computes the gradients for tensors 
    optimizer.step() # Uses the gradients from backward and updates 

# Ask: Should we add sigmoid activation function for the last layer so our output becomes something between 0 and 1
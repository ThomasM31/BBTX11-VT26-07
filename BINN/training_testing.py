import numpy as np
import torch
from torch import optim
import torch.nn as nn
import torch.nn.functional as F

from model import BINN

def train_binn(model, train_loader, epochs = 10, lr = 0.001):
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.train()

    for epoch in range(epochs):
        running_loss = 0.0
        correct = 0
        total = 0

        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device).float().view(-1, 1)

            optimizer.zero_grad()
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # vectorizde calculation from pyTorch to compare prediced with actual
            predicted = (outputs > 0.0).float()
            correct += (predicted == labels).sum().item()

            # .size(0) gets the length of inputs
            total += inputs.size(0)
            # add to the total loss
            running_loss += loss.item()

        print(f"Epoch [{epoch+1} / {epochs}] "
              f"Train Loss: {running_loss / len(train_loader):.3f} "
              f"Train Acc: {100*correct / total:.2f} % ")

    print("finished training")


def test_binn(model, test_loader, criterion):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()
    
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device).float().view(-1, 1)
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
    
    epoch_loss = running_loss / total
    epoch_acc = correct / total

    print(f"Test Acc: {100*correct / total:.2f} % ")

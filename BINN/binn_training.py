import torch
from anndata.experimental import AnnLoader
from Binn import BINN
import torch.nn as nn
import torch.optim

def train_one_epoch(model: BINN, 
                    train_loader: AnnLoader, 
                    criterion: nn.Module, 
                    optimizer: torch.optim.Optimizer, 
                    device):
    """
    Trains the BINN one epoch
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for batch in train_loader:
        # fetch inputs and convert to tensor
        inputs = batch.X.float().to(device)
        # fetch labels and convert to tensor
        labels = torch.tensor(batch.obs['AD_status']).float().reshape(-1, 1).to(device)

        # zero parameter gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        # Backward pass + optimize
        loss.backward()
        optimizer.step()

        # Metrics
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum().item()
        # .size(0) gets the length of inputs
        running_loss += loss.item() * inputs.size(0)
        total += labels.size(0)

        epoch_loss = running_loss / total
        epoch_acc = correct / total

    return epoch_loss, epoch_acc

def test_one_epoch(model: BINN, 
                    test_loader: AnnLoader, 
                    criterion: nn.Module, 
                    device):
    """
    Tests the BINN one epoch
    """
    model.eval()
    model.to(device)
    running_loss = 0.0

    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch in test_loader:
            # fetch inputs and convert to tensor
            inputs = batch.X.float().to(device)
            # fetch labels and convert to tensor
            labels = torch.tensor(batch.obs['AD_status']).float().reshape(-1, 1).to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
    
    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc

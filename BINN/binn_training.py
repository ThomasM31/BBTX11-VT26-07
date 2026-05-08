import torch
from anndata.experimental import AnnLoader
from binn import BINN
import torch.nn as nn
import torch.optim
import pandas as pd

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
        # fetch labels and convert to tensor (Convert in different ways depending on of Series)
        if type(batch.obs["AD_status"]) is pd.Series:
            labels = torch.tensor(batch.obs['AD_status'].values.astype(float)).float().reshape(-1, 1).to(device)
        else:
            labels = batch.obs['AD_status'].detach().clone().float().reshape(-1, 1).to(device)

        # zero parameter gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(inputs)
            
        loss = criterion(outputs, labels)

        # Backward pass + optimize
        loss.backward()

        optimizer.step()
        # Metrics
        predicted = (outputs > 0.0).float()
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
                    device,
                    save: bool = False):
    """
    Tests the BINN one epoch
    """
    model.eval()
    model.to(device)
    correct = 0
    total = 0
    running_loss = 0.0

    all_results: list[pd.DataFrame] = []
    
    with torch.no_grad():
        for batch in test_loader:
            # fetch inputs and convert to tensor
            inputs = batch.X.float().to(device)
            # fetch labels and convert to tensor (Convert in different ways depending on of Series)
            if type(batch.obs["AD_status"]) is pd.Series:
                labels = torch.tensor(batch.obs['AD_status'].values.astype(float)).float().reshape(-1, 1).to(device)
            else:
                labels = batch.obs['AD_status'].detach().clone().float().reshape(-1, 1).to(device)

            outputs = model(inputs)

            if save:
                batch_data = {
                    "actual_label": labels.cpu().numpy().flatten(),
                    "prediction_logic": outputs.cpu().numpy().flatten(),
                    "binary_pred": (outputs > 0.0).float().cpu().numpy().flatten()
                }

                all_results.append(pd.DataFrame(batch_data))

            loss = criterion(outputs, labels)
            
            running_loss += loss.item() * inputs.size(0)
            predicted = (outputs > 0.0).float()
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
    
    if save:
        df_final = pd.concat(all_results, ignore_index=True)
        df_final.to_csv("gen_results.csv", index=False)

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


def train_one_epoch_with_debug(model: BINN, 
                    train_loader: AnnLoader, 
                    criterion: nn.Module, 
                    optimizer: torch.optim.Optimizer, 
                    device):
    """
    Trains the BINN one epoch, print loads of metrics for debugging along the way
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for batch in train_loader:
        # fetch inputs and convert to tensor
        inputs = batch.X.float().to(device)
        # fetch labels and convert to tensor (Convert in different ways depending on of Series)
        if type(batch.obs["AD_status"]) is pd.Series:
            labels = torch.tensor(batch.obs['AD_status'].values.astype(float)).float().reshape(-1, 1).to(device)
        else:
            labels = batch.obs['AD_status'].detach().clone().float().reshape(-1, 1).to(device)

        # Check for normalized values
        print(f"Input batch mean: {inputs.mean().item():.4f}")
        print(f"Input batch std: {inputs.std().item():.4f}")
        print(f"Labels in batch: {labels.tolist()}")

        # zero parameter gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(inputs)

        # Check for reasonable values      
        print(f"Raw logits (first 5): {outputs[:5].detach().cpu().squeeze().tolist()}")

        # Check for output mismatch
        print(f"Outputs shape: {outputs.shape}")
        print(f"Labels shape: {labels.shape}")
            
        loss = criterion(outputs, labels)

        # Backward pass + optimize
        loss.backward()

        # Search for optimizer effects
        for name, p in model.named_parameters():
            print("before: ")
            print(name, torch.mean(p.data))


        # Debug gradients
        total_norm = 0
        for p in model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** 0.5
        print("Grad norm: ", total_norm)
        
        optimizer.step()

        # optimizer cont.
        for name, p in model.named_parameters():
            print("after: ")
            print(name, torch.mean(p.data))

        # Metrics
        predicted = (outputs > 0.0).float()
        correct += (predicted == labels).sum().item()

        # .size(0) gets the length of inputs
        running_loss += loss.item() * inputs.size(0)
        total += labels.size(0)

        epoch_loss = running_loss / total
        epoch_acc = correct / total

    return epoch_loss, epoch_acc
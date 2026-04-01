import torch

def train_binn(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels in train_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)# .float().view(-1, 1) # ??????????????????????????

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

def test_binn(model, test_loader, criterion, device):
    model.eval()
    model.to(device)
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device) #.float().view(-1, 1) # ???????????????????????????
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
    
    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc

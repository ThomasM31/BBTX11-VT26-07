import torch.nn as nn
import torch.nn.functional as F

class BINN(nn.Module):
    """
    Biologically informed neural network (BINN) with binary masked weights.

    Args:
        in_features(integer): Number of input featues (e.g. number of relevant genes).

        layer_list(list of integers): A list defining the number of neurons in each
            biological layer. The length of the list determines the amount of layers.

        mask_list(list of torch.Tensor): A list of binary tensors used to restrict the 
            connectivity between layers.
            Mask 0 shape: (layer_list)
    """
    def __init__(self, 
                 in_features: int, 
                 layers_list: list, 
                 mask_list: list,
                 activation_fn = nn.LeakyReLU(0.1)):
        super(BINN, self).__init__()

        self.in_features = in_features
        self.layers_list = layers_list
        self.mask_list = mask_list
        self.n_masks = len(mask_list)
        self.activation_fn = activation_fn
        
        # Move masks to the same device as the model and store them in a list
        for i, m in enumerate(mask_list):
            self.register_buffer(f'mask_{i}', m)

        # Create the linear layers
        self.model_layers = nn.ModuleList()
        # self.batch_norms = nn.ModuleList()

        current_in_features = in_features
        for layer_size in layers_list:
            # Create Linear layer
            self.model_layers.append(nn.Linear(current_in_features, layer_size))

            # Create BatchNorm, except for final layer
            #if layer_size > 1:
            #    self.batch_norms.append(nn.BatchNorm1d(layer_size, eps=1e-3, momentum=0.01))

            current_in_features = layer_size

        #print(f"Created {len(self.model_layers)} layers and {len(self.batch_norms)} batch norms")
        
    def forward(self, x):
        for i, layer in enumerate(self.model_layers):
            # retrieve the correct mask from buffers
            mask = getattr(self, f'mask_{i}')

            # apply masks to the respective layer weights, transpose if necessary to fit BINN
            if mask.shape != layer.weight.shape:
                masked_weight = layer.weight * mask.t()
            else:
                # If it already matches, just multiply
                masked_weight = layer.weight * mask

            # linear pass :  x * (masked_weights) + bias
            x = F.linear(x, masked_weight, layer.bias)

            # Activation function (except for the last layer)
            #if i < len(self.batch_norms):
                # Re-scale signal
                # x = self.batch_norms[i](x) 
            # Activation function
            x = self.activation_fn(x)
    
        return x
    

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
                 activation_fn = nn.LeakyReLU(0.1), 
                 dropout: float = 0.5):
        super(BINN, self).__init__()

        self.in_features = in_features
        self.layers_list = layers_list
        self.mask_list = mask_list
        self.n_masks = len(mask_list)
        self.activation_fn = activation_fn
        self.dropout = nn.Dropout(p=dropout)
        
        # Move masks to the same device as the model and store them in a list
        for i, m in enumerate(mask_list):
            self.register_buffer(f'mask_{i}', m)

        # Create the linear layers & batch normalizations dynamically
        self.model_layers = nn.ModuleList()
        self.layer_norms = nn.ModuleList()

        current_in_features = in_features

        for layer_size in layers_list:
            self.model_layers.append(nn.Linear(current_in_features, layer_size))
            
            # Add LayerNorm for every layer except the last one
            if layer_size > 1:
                self.layer_norms.append(nn.LayerNorm(layer_size))

            current_in_features = layer_size
        
        # Weight init
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize Linear layers with Xavier uniform."""
        for m in self.model_layers:
            if isinstance(m, nn.Linear):
                # Xavier Uniform prevents early gradient explosions
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        
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
            
            # Apply LayerNorm and Activation only if it's not the last layer
            if i < len(self.model_layers) - 1:
                x = self.layer_norms[i](x)
                x = self.activation_fn(x)
                x = self.dropout(x)
        return x
    

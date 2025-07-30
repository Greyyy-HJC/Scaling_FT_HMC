import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from dataclasses import dataclass

@dataclass
class NetConfig:
    plaq_input_channels: int = 2
    rect_input_channels: int = 4
    plaq_output_channels: int = 4
    rect_output_channels: int = 8
    hidden_channels: int = 12
    kernel_size: tuple = (3, 3)
    
    
    
class LocalNet(nn.Module):
    """
    Simple 2-layer CNN model for local gauge field updates.
    
    Architecture:
    - Input: Concatenated plaquette and rectangle features (6 channels total)
    - Conv1: 6 → 12 channels, 3x3 kernel, circular padding, GELU activation
    - Conv2: 12 → 12 channels, 3x3 kernel, circular padding, GELU activation
    - Output: tanh scaling to [-1/4, 1/4] range
    
    Locality Properties:
    - Receptive field: 5x5 lattice sites (two 3x3 convolutions)
    
    Total parameters: ~ 2,000 (very lightweight)
    """
    def __init__(self):
        super().__init__()
        config = NetConfig()
            
        # Combined input channels for plaq and rect features
        combined_input_channels = config.plaq_input_channels + config.rect_input_channels

        # First conv layer to process combined features
        # Parameters = input_channels x output_channels x kernel_height x kernel_width + bias_terms
        # Parameters: 6 * 12 * 3 * 3 + 12 = 660
        self.conv1 = nn.Conv2d(
            combined_input_channels,
            config.hidden_channels,  # Double the channels
            config.kernel_size,
            padding='same',
            padding_mode='circular'
        )
        self.activation1 = nn.GELU()  # 0 parameters
        
        # Second conv layer to generate final outputs
        # Parameters: 12 * 12 * 3 * 3 + 12 = 1,308
        self.conv2 = nn.Conv2d(
            config.hidden_channels,
            config.plaq_output_channels + config.rect_output_channels,  # Combined output channels
            config.kernel_size,
            padding='same',
            padding_mode='circular'
        )
        self.activation2 = nn.GELU()  # 0 parameters

    def forward(self, plaq_features, rect_features):
        config = NetConfig()
        # plaq_features shape: [batch_size, plaq_input_channels, L, L]
        # rect_features shape: [batch_size, rect_input_channels, L, L]
        
        # Combine input features (0 parameters - tensor operation)
        x = torch.cat([plaq_features, rect_features], dim=1)
        
        # First conv layer (660 parameters used)
        x = self.conv1(x)
        x = self.activation1(x)  # 0 parameters
        
        # Second conv layer (1,308 parameters used)
        x = self.conv2(x)
        x = self.activation2(x)  # 0 parameters
        x = torch.tanh(x) * 0.25  # 0 parameters - tensor operation, range [-1/4, 1/4]
        
        # Split output into plaq and rect coefficients (0 parameters - tensor slicing)
        plaq_coeffs = x[:, :config.plaq_output_channels, :, :]  # [batch_size, 4, L, L]
        rect_coeffs = x[:, config.plaq_output_channels:, :, :]  # [batch_size, 8, L, L]
        
        return plaq_coeffs, rect_coeffs    





def choose_cnn_model(model_tag):
    if model_tag == 'base':
        return LocalNet
    else:
        raise ValueError(f"Invalid model tag: {model_tag}")
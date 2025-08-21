from turtle import update
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
    - Output: tanh scaling to [-1/6, 1/6] range
    
    Locality Properties:
    - Receptive field: 5x5 lattice sites (two 3x3 convolutions)
    
    Total parameters: ~ 2,000
    """
    def __init__(self):
        super().__init__()
        config = NetConfig()
            
        # Combined input channels for plaq and rect features
        combined_input_channels = config.plaq_input_channels + config.rect_input_channels

        # First conv layer to process combined features
        # Parameters = input_channels x output_channels x kernel_height x kernel_width + bias_terms
        # Parameters: 6 * 12 * 3 * 3 + 12 = 660
        self.conv_input = nn.Conv2d(
            combined_input_channels,
            config.hidden_channels,  # Double the channels
            config.kernel_size,
            padding='same',
            padding_mode='circular'
        )
        self.activation = nn.GELU()  # 0 parameters
        
        # Second conv layer to generate final outputs
        # Parameters: 12 * 12 * 3 * 3 + 12 = 1,308
        self.conv_output = nn.Conv2d(
            config.hidden_channels,
            config.plaq_output_channels + config.rect_output_channels,  # Combined output channels
            config.kernel_size,
            padding='same',
            padding_mode='circular'
        )

    def forward(self, plaq_features, rect_features):
        config = NetConfig()
        # plaq_features shape: [batch_size, plaq_input_channels, L, L]
        # rect_features shape: [batch_size, rect_input_channels, L, L]
        
        # Combine input features (0 parameters - tensor operation)
        x = torch.cat([plaq_features, rect_features], dim=1)
        
        # First conv layer (660 parameters used)
        x = self.conv_input(x)
        x = self.activation(x)  # 0 parameters
        
        # Second conv layer (1,308 parameters used)
        x = self.conv_output(x)
        x = torch.arctan(x) / math.pi / 3  # 0 parameters - tensor operation, range [-1/6, 1/6] 
        
        # Split output into plaq and rect coefficients (0 parameters - tensor slicing)
        plaq_coeffs = x[:, :config.plaq_output_channels, :, :]  # [batch_size, 4, L, L]
        rect_coeffs = x[:, config.plaq_output_channels:, :, :]  # [batch_size, 8, L, L]
        
        return plaq_coeffs, rect_coeffs    
    
    
class LocalNetTanh(nn.Module):
    """
    Simple 2-layer CNN model with learnable alpha scaling for local gauge field updates.
    
    Architecture:
    - Input: Concatenated plaquette and rectangle features (6 channels total)
    - Conv1: 6 → 12 channels, 3x3 kernel, circular padding, GELU activation
    - Conv2: 12 → 12 channels, 3x3 kernel, circular padding
    - Output: tanh scaling with larger coefficient on plaq
    
    Locality Properties:
    - Receptive field: 5x5 lattice sites (two 3x3 convolutions)
    
    Total parameters: ~ 2,000
    """
    def __init__(self):
        super().__init__()
        config = NetConfig()
            
        # Combined input channels for plaq and rect features
        combined_input_channels = config.plaq_input_channels + config.rect_input_channels

        # First conv layer to process combined features
        # Parameters = input_channels x output_channels x kernel_height x kernel_width + bias_terms
        # Parameters: 6 * 12 * 3 * 3 + 12 = 660
        self.conv_input = nn.Conv2d(
            combined_input_channels,
            config.hidden_channels,  # Double the channels
            config.kernel_size,
            padding='same',
            padding_mode='circular'
        )
        self.activation = nn.GELU()  # 0 parameters
        
        # Second conv layer to generate final outputs
        # Parameters: 12 * 12 * 3 * 3 + 12 = 1,308
        self.conv_output = nn.Conv2d(
            config.hidden_channels,
            config.plaq_output_channels + config.rect_output_channels,  # Combined output channels
            config.kernel_size,
            padding='same',
            padding_mode='circular'
        )
        

    def forward(self, plaq_features, rect_features):
        config = NetConfig()
        # plaq_features shape: [batch_size, plaq_input_channels, L, L]
        # rect_features shape: [batch_size, rect_input_channels, L, L]
        
        # Combine input features (0 parameters - tensor operation)
        x = torch.cat([plaq_features, rect_features], dim=1)
        
        # First conv layer (660 parameters used)
        x = self.conv_input(x)
        x = self.activation(x)  # 0 parameters
        
        # Second conv layer (1,308 parameters used)
        x = self.conv_output(x)
        
        # Output scaling
        plaq_coeffs = torch.tanh(x[:, :config.plaq_output_channels, :, :]) * 2 / 5  # [batch_size, 4, L, L] in range [-2/5, 2/5]
        rect_coeffs = torch.tanh(x[:, config.plaq_output_channels:, :, :]) / 20  # [batch_size, 8, L, L] in range [-1/20, 1/20]
        
        return plaq_coeffs, rect_coeffs 
    
    

class LocalResNet(nn.Module):
    """
    CNN model with residual blocks for local gauge field updates.
    
    Architecture:
    - Input: Concatenated plaquette and rectangle features (6 channels total)
    - Conv1: 6 → 12 channels, 3x3 kernel, circular padding, GELU activation
    - ResBlock1: 12 → 12 channels, 3x3 kernels, GELU activation, scaled residual connection (0.3x)
    - ResBlock2: 12 → 12 channels, 3x3 kernels, GELU activation, scaled residual connection (0.3x)
    - Conv_output: 12 → 12 channels, 1x1 kernel
    - Output: arctan scaling to [-1/6, 1/6] range
    
    Locality Properties:
    - Receptive field: 11x11 lattice sites (one input conv + two residual blocks, each with two layers of 3x3 kernels)
    
    Total parameters: ~ 660 (input) + 2,616x2 (res blocks) + 156 (output) = ~ 6,000  
    """
    def __init__(self):
        super().__init__()
        config = NetConfig()
        combined_input_channels = config.plaq_input_channels + config.rect_input_channels
        hidden_channels = config.hidden_channels 

        # First conv layer to process combined features
        # Parameters = input_channels x output_channels x kernel_height x kernel_width + bias_terms
        # Parameters: 6 * 12 * 3 * 3 + 12 = 660
        self.conv_input = nn.Conv2d(
            combined_input_channels,
            config.hidden_channels,  # Double the channels
            config.kernel_size,
            padding='same',
            padding_mode='circular'
        )
        self.activation = nn.GELU()  # 0 parameters

        # One standard ResidualBlock
        # Each has ~1,300 parameters
        self.res_block1 = ResidualBlock(hidden_channels, config.kernel_size)
        self.res_block2 = ResidualBlock(hidden_channels, config.kernel_size)

        # Output layer
        # Parameters: 12 * 12 * 1 * 1 + 12 = 156
        self.conv_output = nn.Conv2d(
            config.hidden_channels,
            config.plaq_output_channels + config.rect_output_channels,
            1,
            bias=True
        )

    def forward(self, plaq_features, rect_features):
        config = NetConfig()
        # plaq_features shape: [batch_size, plaq_input_channels, L, L]
        # rect_features shape: [batch_size, rect_input_channels, L, L]
        
        # Combine input features (0 parameters - tensor operation)
        x = torch.cat([plaq_features, rect_features], dim=1)
        
        # First conv layer (660 parameters used)
        x = self.conv_input(x)
        x = self.activation(x) 
        
        # ResNet blocks - add scaling factor for improved stability
        # 1,300 parameters used
        identity1 = x
        x = self.res_block1(x) * 0.3 + identity1  # Moderate scaling
        
        identity2 = x
        x = self.res_block2(x) * 0.3 + identity2
        
        # Second conv layer (1,308 parameters used)
        x = self.conv_output(x)
        x = torch.arctan(x) / math.pi / 3  # 0 parameters - tensor operation, range [-1/6, 1/6] 
        
        # Split output into plaq and rect coefficients (0 parameters - tensor slicing)
        plaq_coeffs = x[:, :config.plaq_output_channels, :, :]  # [batch_size, 4, L, L]
        rect_coeffs = x[:, config.plaq_output_channels:, :, :]  # [batch_size, 8, L, L]
        
        return plaq_coeffs, rect_coeffs
    
    
class LocalAttnNet(nn.Module):
    """
    CNN model with channel attention for local gauge field updates.
    
    Architecture:
    - Input: Concatenated plaquette and rectangle features (6 channels total)
    - Conv1: 6 → 12 channels, 3x3 kernel, circular padding, GELU activation
    - Channel attention: Squeeze-and-Excitation style attention mechanism
    - Conv2: 12 → 12 channels, 3x3 kernel, circular padding
    - Output: arctan scaling to [-1/6, 1/6] range
    
    Locality Properties:
    - Receptive field: 5x5 lattice sites (two 3x3 convolutions)
    - Global context via channel attention
    
    Total parameters: ~ 660 (conv1) + 1,308 (conv2) + ~100 (attention) = ~ 2,000
    """
    def __init__(self):
        super().__init__()
        config = NetConfig()
        
        combined_input_channels = config.plaq_input_channels + config.rect_input_channels
        hidden_channels = config.hidden_channels
        
        # First conv layer to process combined features
        # Parameters: 6 * 12 * 3 * 3 + 12 = 660
        self.conv_input = nn.Conv2d(
            combined_input_channels, 
            config.hidden_channels, 
            config.kernel_size,
            padding='same', 
            padding_mode='circular'
        )
        self.activation = nn.GELU()
        
        # Channel attention
        # Parameters: 12*3 + 3 + 3*12 + 12 = 87
        self.channel_attention = ChannelAttention(hidden_channels)
        
        # Second conv layer to generate final outputs
        # Parameters: 12 * 12 * 3 * 3 + 12 = 1,308
        self.conv_output = nn.Conv2d(
            config.hidden_channels,
            config.plaq_output_channels + config.rect_output_channels,  # Combined output channels
            config.kernel_size,
            padding='same',
            padding_mode='circular'
        )
        
    def forward(self, plaq_features, rect_features):
        config = NetConfig()
        # plaq_features shape: [batch_size, plaq_input_channels, L, L]
        # rect_features shape: [batch_size, rect_input_channels, L, L]
        
        # Merge inputs
        x = torch.cat([plaq_features, rect_features], dim=1)
        
        # First conv layer (660 parameters used)
        x = self.conv_input(x)
        x = self.activation(x)
        
        # Channel attention
        x = self.channel_attention(x) # 87 parameters used
        
        # Second conv layer (1,308 parameters used)
        x = self.conv_output(x)
        x = torch.arctan(x) / math.pi / 3  # 0 parameters - tensor operation, range [-1/6, 1/6] 
        
        # Split output into plaq and rect coefficients (0 parameters - tensor slicing)
        plaq_coeffs = x[:, :config.plaq_output_channels, :, :]  # [batch_size, 4, L, L]
        rect_coeffs = x[:, config.plaq_output_channels:, :, :]  # [batch_size, 8, L, L]
        
        return plaq_coeffs, rect_coeffs  
   
    
class LocalCombinedNet(nn.Module):
    """
    Hybrid CNN model combining residual blocks and channel attention for local gauge field updates.
    
    Architecture:
    - Input: Concatenated plaquette and rectangle features (6 channels total)
    - Conv1: 6 → 12 channels, 3x3 kernel, circular padding, GELU activation
    - ResBlock1: 12 → 12 channels, 3x3 kernels, GELU activation, scaled residual connection (0.3x)
    - ResBlock2: 12 → 12 channels, 3x3 kernels, GELU activation, scaled residual connection (0.3x)
    - Channel attention: Squeeze-and-Excitation style attention mechanism
    - Conv_output: 12 → 12 channels, 1x1 kernel
    - Output: tanh scaling with larger coefficient on plaq
    
    Locality Properties:
    - Receptive field: 11x11 lattice sites (one input conv + two residual blocks, each with two layers of 3x3 kernels)
    - Global context via channel attention
    
    Total parameters: ~ 660 (conv1) + 2,616x2 (res blocks) + ~100 (attention) + 156 (output) = ~ 6,000
    """
    def __init__(self):
        super().__init__()
        config = NetConfig()
        
        combined_input_channels = config.plaq_input_channels + config.rect_input_channels
        hidden_channels = config.hidden_channels
        
        # Simplified input projection
        self.conv_input = nn.Conv2d(
            combined_input_channels, 
            config.hidden_channels, 
            config.kernel_size,
            padding='same', 
            padding_mode='circular'
        )
        self.activation = nn.GELU()
        
        # Only use 2 ResNet blocks, but with more stable design
        self.res_block1 = ResidualBlock(config.hidden_channels, config.kernel_size)
        self.res_block2 = ResidualBlock(config.hidden_channels, config.kernel_size)
        
        # Simplified channel attention
        self.channel_attention = ChannelAttention(hidden_channels)
        
        # Output layer
        self.conv_output = nn.Conv2d(
            config.hidden_channels,
            config.plaq_output_channels + config.rect_output_channels,
            1,
            bias=True
        )
        
        
    def forward(self, plaq_features, rect_features):
        config = NetConfig()
        # plaq_features shape: [batch_size, plaq_input_channels, L, L]
        # rect_features shape: [batch_size, rect_input_channels, L, L]
        
        # Merge inputs
        x = torch.cat([plaq_features, rect_features], dim=1)
        
        # Input processing
        x = self.conv_input(x)
        x = self.activation(x)
        
        # ResNet blocks - add scaling factor for improved stability
        identity1 = x
        x = self.res_block1(x) * 0.3 + identity1  # Moderate scaling
        
        identity2 = x
        x = self.res_block2(x) * 0.3 + identity2
        
        # Channel attention
        x = self.channel_attention(x)
        
        # Output scaling
        x = self.conv_output(x)
        
        plaq_coeffs = torch.tanh(x[:, :config.plaq_output_channels, :, :]) * 2 / 5  # [batch_size, 4, L, L] in range [-2/5, 2/5]
        rect_coeffs = torch.tanh(x[:, config.plaq_output_channels:, :, :]) / 20  # [batch_size, 8, L, L] in range [-1/20, 1/20]
        
        return plaq_coeffs, rect_coeffs 
    
   
class ResidualBlock(nn.Module):
    """
    Residual block for feature processing in neural networks.
    
    Architecture:
    - Input: Feature tensor with arbitrary number of channels
    - Conv1: channels → channels, 3x3 kernel, circular padding, GELU activation
    - Conv2: channels → channels, 3x3 kernel, circular padding
    - Residual connection: output = conv_output + input
    - Final GELU activation
    
    Parameters: For N channels: NxNx3x3x2 + Nx2 = 18N² + 2N
    Example: For 12 channels: ~ 2,616 parameters
    """
    def __init__(self, channels, kernel_size=(3, 3)):
        super().__init__()
        self.conv_input = nn.Conv2d(channels, channels, kernel_size, 
                               padding='same', padding_mode='circular')
        
        self.conv_output = nn.Conv2d(channels, channels, kernel_size, 
                               padding='same', padding_mode='circular')

        self.activation = nn.GELU()

    def forward(self, x):
        identity = x

        out = self.conv_input(x)
        out = self.activation(out)

        out = self.conv_output(out)
        
        out += identity
        out = self.activation(out)

        return out


class ChannelAttention(nn.Module):
    """
    Squeeze-and-Excitation style channel attention mechanism.
    
    Architecture:
    - Global average pooling (squeeze): HxW → 1x1 per channel
    - Conv1: channels → channels//reduction, 1x1 kernel, ReLU activation
    - Conv2: channels//reduction → channels, 1x1 kernel, Sigmoid activation
    - Element-wise multiplication with input features
    
    Parameters: For N channels with reduction=4: Nx(N//4) + (N//4) + (N//4)xN + N = N²/2 + N/2 + N
    Example: For 12 channels: 12x3 + 3 + 3x12 + 12 = 87 parameters
    """
    def __init__(self, channels, reduction=4):
        super().__init__()
        mid = max(2, channels // reduction)
        # Total parameters: channels * mid + mid + mid * channels + channels
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # 0 parameters
            nn.Conv2d(channels, mid, 1),  # channels * mid + mid parameters
            nn.ReLU(),  # 0 parameters
            nn.Conv2d(mid, channels, 1),  # mid * channels + channels parameters
            nn.Sigmoid()  # 0 parameters
        )

    def forward(self, x):
        # Element-wise multiplication (0 parameters)
        return x * self.attention(x)  # attention module parameters used above
    


def choose_cnn_model(model_tag):
    if model_tag == 'base':
        return LocalNet
    elif model_tag == 'tanh':
        return LocalNetTanh
    elif model_tag == 'resn':
        return LocalResNet # base + 2 conv of resnet
    elif model_tag == 'attn':
        return LocalAttnNet # base + channel attn
    elif model_tag == 'combined':
        return LocalCombinedNet
    else:
        raise ValueError(f"Invalid model tag: {model_tag}")
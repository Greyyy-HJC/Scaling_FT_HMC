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
    
    
class LocalNetAlpha(nn.Module):
    """
    Simple 2-layer CNN model for local gauge field updates.
    
    Architecture:
    - Input: Concatenated plaquette and rectangle features (6 channels total)
    - Conv1: 6 → 12 channels, 3x3 kernel, circular padding, GELU activation
    - Conv2: 12 → 12 channels, 3x3 kernel, circular padding, GELU activation
    - Output: tanh scaling to [-1/4, 1/4] range
    
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
        
        self.alpha = nn.Parameter(torch.ones(1) * 0.25)  # 1 parameter

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
        x = torch.tanh(x) * self.alpha  # 1 parameter - learnable scaling
        
        # Split output into plaq and rect coefficients (0 parameters - tensor slicing)
        plaq_coeffs = x[:, :config.plaq_output_channels, :, :]  # [batch_size, 4, L, L]
        rect_coeffs = x[:, config.plaq_output_channels:, :, :]  # [batch_size, 8, L, L]
        
        return plaq_coeffs, rect_coeffs   
    
    
class LocalNetArcTan(nn.Module):
    """
    Simple 2-layer CNN model for local gauge field updates.
    
    Architecture:
    - Input: Concatenated plaquette and rectangle features (6 channels total)
    - Conv1: 6 → 12 channels, 3x3 kernel, circular padding, GELU activation
    - Conv2: 12 → 12 channels, 3x3 kernel, circular padding, GELU activation
    - Output: arctan scaling to [-1/4, 1/4] range
    
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
        x = torch.arctan(x) / math.pi / 2  # 0 parameters - tensor operation, range [-1/4, 1/4]
        
        # Split output into plaq and rect coefficients (0 parameters - tensor slicing)
        plaq_coeffs = x[:, :config.plaq_output_channels, :, :]  # [batch_size, 4, L, L]
        rect_coeffs = x[:, config.plaq_output_channels:, :, :]  # [batch_size, 8, L, L]
        
        return plaq_coeffs, rect_coeffs  



class LocalResNet(nn.Module):
    """
    Simple 2-layer CNN model with residual block for local gauge field updates.
    
    Architecture:
    - Input: Concatenated plaquette and rectangle features (6 channels total)
    - Conv1: 6 → 12 channels, 3x3 kernel, circular padding, GELU activation
    - Residual block: 12 → 12 channels, 3x3 kernel, circular padding, SiLU activation
    - Conv2: 12 → 12 channels, 3x3 kernel, circular padding, GELU activation
    - Output: tanh scaling to [-1/4, 1/4] range
    
    Locality Properties:
    - Receptive field: 7x7 lattice sites (three 3x3 kernels)
    
    Total parameters: ~ 3,300
    """
    def __init__(self):
        super().__init__()
        config = NetConfig()
        combined_input_channels = config.plaq_input_channels + config.rect_input_channels
        hidden_channels = config.hidden_channels 

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

        # One standard ResidualBlock
        # Each has ~1,300 parameters
        self.res_block = ResidualBlock(hidden_channels, config.kernel_size)

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
        
        # Residual block (~1,300 parameters used)
        x = self.res_block(x)
        
        # Second conv layer (1,308 parameters used)
        x = self.conv2(x)
        x = self.activation2(x)  # 0 parameters
        x = torch.tanh(x) * 0.25  # 0 parameters - tensor operation, range [-1/4, 1/4]
        
        # Split output into plaq and rect coefficients (0 parameters - tensor slicing)
        plaq_coeffs = x[:, :config.plaq_output_channels, :, :]  # [batch_size, 4, L, L]
        rect_coeffs = x[:, config.plaq_output_channels:, :, :]  # [batch_size, 8, L, L]
        
        return plaq_coeffs, rect_coeffs 


class Local2ResNet(nn.Module):
    """
    Simple 2-layer CNN model with double residual block for local gauge field updates.
    
    Architecture:
    - Input: Concatenated plaquette and rectangle features (6 channels total)
    - Conv1: 6 → 12 channels, 3x3 kernel, circular padding, GELU activation
    - Double residual block: 12 → 12 channels, two 3x3 kernels, circular padding, SiLU activation
    - Conv2: 12 → 12 channels, 3x3 kernel, circular padding, GELU activation
    - Output: tanh scaling to [-1/4, 1/4] range
    
    Locality Properties:
    - Receptive field: 9x9 lattice sites (four 3x3 kernels)
    
    Total parameters: ~ 4,600
    """
    def __init__(self):
        super().__init__()
        config = NetConfig()
        combined_input_channels = config.plaq_input_channels + config.rect_input_channels
        hidden_channels = config.hidden_channels 

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

        # One double ResidualBlock
        # Each has ~2,600 parameters
        self.double_res_block = DoubleResidualBlock(hidden_channels, config.kernel_size)

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
        
        # Double residual block (~2,600 parameters used)
        x = self.double_res_block(x)
        
        # Second conv layer (1,308 parameters used)
        x = self.conv2(x)
        x = self.activation2(x)  # 0 parameters
        x = torch.tanh(x) * 0.25  # 0 parameters - tensor operation, range [-1/4, 1/4]
        
        # Split output into plaq and rect coefficients (0 parameters - tensor slicing)
        plaq_coeffs = x[:, :config.plaq_output_channels, :, :]  # [batch_size, 4, L, L]
        rect_coeffs = x[:, config.plaq_output_channels:, :, :]  # [batch_size, 8, L, L]
        
        return plaq_coeffs, rect_coeffs 
    
    
class LocalAttnNet(nn.Module):
    """
    Simple 2-layer CNN model for local gauge field updates.
    
    Architecture:
    - Input: Concatenated plaquette and rectangle features (6 channels total)
    - Conv1: 6 → 12 channels, 3x3 kernel, circular padding, GELU activation
    - Channel attention: 12 → 12 channels, 1x1 kernel, global average pooling, ReLU activation, 1x1 kernel, Sigmoid activation
    - Conv2: 12 → 12 channels, 3x3 kernel, circular padding, GELU activation
    - Output: tanh scaling to [-1/4, 1/4] range
    
    Locality Properties:
    - Receptive field: 5x5 lattice sites (two 3x3 convolutions)
    
    Total parameters: ~ 2,100
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
        
        # Channel attention using standard ChannelAttention module
        # Parameters: ~100
        self.channel_attention = ChannelAttention(config.hidden_channels)
        
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
        
        # Channel attention (~100 parameters used)
        x = self.channel_attention(x)
        
        # Second conv layer (1,308 parameters used)
        x = self.conv2(x)
        x = self.activation2(x)  # 0 parameters
        x = torch.tanh(x) * 0.25  # 0 parameters - tensor operation, range [-1/4, 1/4]
        
        # Split output into plaq and rect coefficients (0 parameters - tensor slicing)
        plaq_coeffs = x[:, :config.plaq_output_channels, :, :]  # [batch_size, 4, L, L]
        rect_coeffs = x[:, config.plaq_output_channels:, :, :]  # [batch_size, 8, L, L]
        
        return plaq_coeffs, rect_coeffs    
    
    
class Local2ResAttnNet(nn.Module):
    """
    Simple 2-layer CNN model with double residual block and channel attention.
    
    Architecture:
    - Input: Concatenated plaquette and rectangle features (6 channels total)
    - Conv1: 6 → 12 channels, 3x3 kernel, circular padding, GELU activation
    - Double residual block: 12 → 12 channels, two 3x3 kernels, circular padding, GELU activation
    - Channel attention: 12 → 12 channels, 1x1 kernel, global average pooling, ReLU activation, 1x1 kernel, Sigmoid activation
    - Conv2: 12 → 12 channels, 3x3 kernel, circular padding, GELU activation
    - Output: tanh scaling to [-1/4, 1/4] range
    
    Locality Properties:
    - Receptive field: 9x9 lattice sites (four 3x3 kernels)
    
    Total parameters: ~ 4,700
    """
    def __init__(self):
        super().__init__()
        config = NetConfig()
        combined_input_channels = config.plaq_input_channels + config.rect_input_channels
        hidden_channels = config.hidden_channels

        # First conv layer to process combined features
        # Parameters = input_channels x output_channels x kernel_height x kernel_width + bias_terms
        self.conv1 = nn.Conv2d(
            combined_input_channels,
            hidden_channels,
            config.kernel_size,
            padding='same',
            padding_mode='circular'
        )
        self.activation1 = nn.GELU()

        # Double residual block
        # Each has ~2,600 parameters
        self.double_res_block = DoubleResidualBlock(hidden_channels, config.kernel_size, alpha=0.1, group_norm_groups=4) # smaller alpha, more groups

        # Channel attention (on hidden channels)
        # Parameters: ~100
        self.channel_attention = ChannelAttention(hidden_channels)

        # Output conv to generate final outputs
        # Parameters: 12 * 12 * 3 * 3 + 12 = 1,308
        self.conv2 = nn.Conv2d(
            hidden_channels,
            config.plaq_output_channels + config.rect_output_channels,  # Combined output channels
            config.kernel_size,
            padding='same',
            padding_mode='circular'
        )
        self.activation2 = nn.GELU()

    def forward(self, plaq_features, rect_features):
        config = NetConfig()
        # plaq_features shape: [batch_size, plaq_input_channels, L, L]
        # rect_features shape: [batch_size, rect_input_channels, L, L]
        
        # Combine input features (0 parameters - tensor operation)
        x = torch.cat([plaq_features, rect_features], dim=1)

        # First conv layer (660 parameters used)
        x = self.conv1(x)
        x = self.activation1(x)

        # Double residual block (~2,600 parameters used)
        x = self.double_res_block(x)

        # Channel attention (~100 parameters used)
        x = self.channel_attention(x)

        # Output conv to generate final outputs (1,308 parameters used)
        x = self.conv2(x)
        # x = self.activation2(x)  # 0 parameters 
        x = torch.tanh(x) * 0.25  # 0 parameters - tensor operation, range [-1/4, 1/4]

        # Split output into plaq and rect coefficients (0 parameters - tensor slicing)
        plaq_coeffs = x[:, :config.plaq_output_channels, :, :]  # [batch_size, 4, L, L]
        rect_coeffs = x[:, config.plaq_output_channels:, :, :]  # [batch_size, 8, L, L]
        
        return plaq_coeffs, rect_coeffs
    


class LocalCoorConvNet(nn.Module):
    """
    Simple 2-layer CNN model for local gauge field updates.
    
    Architecture:
    - Input: Concatenated plaquette and rectangle features (6 channels total)
    - Conv1: 6 → 12 channels, 3x3 kernel, circular padding, GELU activation
    - Conv2: 12 → 12 channels, 3x3 kernel, circular padding, GELU activation
    - Output: tanh scaling to [-1/4, 1/4] range
    
    Locality Properties:
    - Receptive field: 5x5 lattice sites (two 3x3 convolutions)
    
    Total parameters: ~ 2,200
    """
    def __init__(self):
        super().__init__()
        config = NetConfig()
            
        # Combined input channels for plaq and rect features, plus 2 for CoordConv
        combined_input_channels = config.plaq_input_channels + config.rect_input_channels + 2

        # First conv layer to process combined features
        # Parameters = input_channels x output_channels x kernel_height x kernel_width + bias_terms
        # Parameters: 8 * 12 * 3 * 3 + 12 = 876
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
        
        B, _, H, W = plaq_features.shape
        device = plaq_features.device
        # CoordConv: add two channels of normalized coords (0 parameters - computed), sin cos
        xs = torch.linspace(-1, 1, W, device=device).view(1,1,1,W).expand(B,1,H,W)  # 0 parameters
        ys = torch.linspace(-1, 1, H, device=device).view(1,1,H,1).expand(B,1,H,W)  # 0 parameters
        coords = torch.cat([xs, ys], dim=1)  # 0 parameters
        
        # Combine input features (0 parameters - tensor operation)
        x = torch.cat([plaq_features, rect_features, coords], dim=1)
        
        # First conv layer (876 parameters used)
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
    

class LocalUNet(nn.Module):
    """
    Fast locality-preserving U-Net style architecture for local gauge field updates.
    
    Architecture:
    - Sequential multi-scale feature extraction (not parallel)
    - Lightweight dilated convolutions for efficiency
    - Reduced channel numbers
    - Output: tanh scaling to [-1/4, 1/4] range
    
    Multi-scale Features:
    - Progressive dilation: 1→2→4 (sequential, not parallel)
    - Skip connections preserve information from each scale
    - Efficient computation with shared feature maps
    
    Locality Properties:
    - Receptive field: 15x15 lattice sites (through progressive dilation)
    - Multiple effective receptive fields: 3x3, 7x7, 15x15
    - True locality preserved: RF << L for typical L values
    - Fast computation: ~3x faster than parallel version
    
    Total parameters: ~ 8,000
    """
    def __init__(self):
        super().__init__()
        config = NetConfig()
        combined_input_channels = config.plaq_input_channels + config.rect_input_channels
        
        # Sequential multi-scale processing (much faster than parallel)
        # Parameters: 6 * 16 * 3 * 3 + 16 = 880
        self.conv1 = nn.Conv2d(combined_input_channels, 16, 3, padding=1, padding_mode='circular')  # RF=3
        # Parameters: 16 * 16 * 3 * 3 + 16 = 2,320
        self.conv2 = nn.Conv2d(16, 16, 3, padding=2, dilation=2, padding_mode='circular')  # RF=7
        # Parameters: 16 * 16 * 3 * 3 + 16 = 2,320
        self.conv3 = nn.Conv2d(16, 16, 3, padding=4, dilation=4, padding_mode='circular')  # RF=15
        
        # Lightweight skip connection processing
        # Parameters: 16 * 8 * 1 * 1 + 8 = 136
        self.skip_conv1 = nn.Conv2d(16, 8, 1)  # 1x1 conv to reduce channels
        # Parameters: 16 * 8 * 1 * 1 + 8 = 136
        self.skip_conv2 = nn.Conv2d(16, 8, 1)  # 1x1 conv to reduce channels
        
        # Final fusion (much smaller than 48→24)
        # Parameters: 32 * 16 * 1 * 1 + 16 = 528
        self.fusion_conv = nn.Conv2d(16 + 8 + 8, 16, 1)  # 32→16
        
        # Output layers
        # Parameters: 16 * 12 * 3 * 3 + 12 = 1,740
        self.output_conv = nn.Conv2d(16, config.plaq_output_channels + config.rect_output_channels, 3, padding=1, padding_mode='circular')
        
        self.activation = nn.GELU()  # 0 parameters
        
    def forward(self, plaq_features, rect_features):
        config = NetConfig()
        # Combine input features
        x = torch.cat([plaq_features, rect_features], dim=1)  # [B, 6, L, L]
        
        # Sequential multi-scale feature extraction with skip connections
        x1 = self.activation(self.conv1(x))      # [B, 16, L, L], RF=3x3 (880 parameters used)
        skip1 = self.skip_conv1(x1)              # [B, 8, L, L] - compressed skip (136 parameters used)
        
        x2 = self.activation(self.conv2(x1))     # [B, 16, L, L], RF=7x7 (2,320 parameters used)
        skip2 = self.skip_conv2(x2)              # [B, 8, L, L] - compressed skip (136 parameters used)
        
        x3 = self.activation(self.conv3(x2))     # [B, 16, L, L], RF=15x15 (2,320 parameters used)
        
        # Lightweight fusion
        fused = torch.cat([x3, skip2, skip1], dim=1)  # [B, 32, L, L] (0 parameters - tensor operation)
        fused = self.activation(self.fusion_conv(fused))  # [B, 16, L, L] (528 parameters used)
        
        # Output
        output = self.output_conv(fused)  # [B, 12, L, L] (1,740 parameters used)
        output = torch.tanh(output) * 0.25  # [-1/4, 1/4] range (0 parameters - tensor operation)
        
        # Split output
        plaq_coeffs = output[:, :config.plaq_output_channels, :, :]
        rect_coeffs = output[:, config.plaq_output_channels:, :, :]
        
        return plaq_coeffs, rect_coeffs


class LocalTransformerNet(nn.Module):
    """
    Fast local Vision Transformer for local gauge field updates.
    
    Architecture:
    - Small patch embedding (2x2 patches for speed)
    - Single transformer layer with limited attention window
    - Lightweight MLP head
    - Output: tanh scaling to [-1/4, 1/4] range
    
    Locality Properties:
    - Receptive field: 16x16 lattice sites (8x8 patches × 2 patch_size)  
    - Each patch attends to 5x5 neighborhood patches (local attention)
    - True locality: each position sees at most 10x10 lattice region
    - Fast computation: O(L²) scaling, much faster than global attention
    
    Speed Optimizations:
    - Small patch size (2x2) reduces sequence length by 4x
    - Small embedding dimension (32) for efficiency
    - Single transformer layer
    - Local attention window (5x5 patches)
    
    Total parameters: ~ 11,000
    """
    def __init__(self, patch_size=2, embed_dim=32, num_heads=4, window_size=5):
        super().__init__()
        config = NetConfig()
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.window_size = window_size  # 5x5 patch attention window
        
        combined_input_channels = config.plaq_input_channels + config.rect_input_channels
        combined_output_channels = config.plaq_output_channels + config.rect_output_channels
        
        # Patch embedding (small patches for speed)
        patch_dim = combined_input_channels * patch_size * patch_size  # 6 * 4 = 24
        # Parameters: 24 * 32 + 32 = 800
        self.patch_embed = nn.Linear(patch_dim, embed_dim)  # 24 → 32
        
        # Single lightweight transformer layer
        # Parameters: 32 * 2 = 64 (LayerNorm: weight + bias)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attention = LocalWindowAttention(embed_dim, num_heads, window_size)  # ~4,200 parameters
        # Parameters: 32 * 2 = 64 (LayerNorm: weight + bias)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),  # Parameters: 32 * 64 + 64 = 2,112
            nn.GELU(),  # 0 parameters
            nn.Linear(embed_dim * 2, embed_dim)   # Parameters: 64 * 32 + 32 = 2,080
        )
        
        # Output projection
        output_patch_dim = combined_output_channels * patch_size * patch_size  # 12 * 4 = 48
        # Parameters: 32 * 48 + 48 = 1,584
        self.output_proj = nn.Linear(embed_dim, output_patch_dim)  # 32 → 48
        
        self.activation = nn.GELU()  # 0 parameters
        
    def forward(self, plaq_features, rect_features):
        config = NetConfig()
        B, _, L, L = plaq_features.shape
        
        # Combine input features
        x = torch.cat([plaq_features, rect_features], dim=1)  # [B, 6, L, L]
        
        # Create patches: [B, 6, L, L] → [B, H_p, W_p, patch_dim]
        patches = self.create_patches(x)
        H_p, W_p = patches.shape[1], patches.shape[2]
        
        # Patch embedding: [B, H_p, W_p, patch_dim] → [B, H_p, W_p, embed_dim]
        patches = self.patch_embed(patches)  # (800 parameters used)
        
        # Apply local transformer layer
        # Pre-norm
        normed = self.norm1(patches)  # (64 parameters used)
        
        # Local attention
        attn_out = self.attention(normed, H_p, W_p)  # (~4,200 parameters used)
        patches = patches + attn_out  # Residual connection (0 parameters - tensor operation)
        
        # MLP with residual
        normed = self.norm2(patches)  # (64 parameters used)
        mlp_out = self.mlp(normed)  # (2,112 + 2,080 = 4,192 parameters used)
        patches = patches + mlp_out  # Residual connection (0 parameters - tensor operation)
        
        # Output projection
        patches = self.output_proj(patches)  # [B, H_p, W_p, output_patch_dim] (1,584 parameters used)
        
        # Reconstruct to spatial layout
        output = self.reconstruct_patches(patches, L)  # [B, 12, L, L] (0 parameters - tensor operation)
        
        # Output scaling
        output = torch.tanh(output) * 0.25  # (0 parameters - tensor operation)
        
        # Split output
        plaq_coeffs = output[:, :config.plaq_output_channels, :, :]
        rect_coeffs = output[:, config.plaq_output_channels:, :, :]
        
        return plaq_coeffs, rect_coeffs
    
    def create_patches(self, x):
        """Convert [B, C, L, L] to [B, H_p, W_p, patch_dim]"""
        B, C, L, L = x.shape
        p = self.patch_size
        
        # Unfold: [B, C, L, L] → [B, C, L//p, L//p, p, p]
        patches = x.unfold(2, p, p).unfold(3, p, p)
        # Rearrange: [B, L//p, L//p, C, p, p]
        patches = patches.permute(0, 2, 3, 1, 4, 5).contiguous()
        # Flatten patch: [B, L//p, L//p, C*p*p]
        patches = patches.view(B, L//p, L//p, C * p * p)
        
        return patches
    
    def reconstruct_patches(self, patches, L):
        """Convert [B, H_p, W_p, output_patch_dim] back to [B, C_out, L, L]"""
        B, H_p, W_p, output_patch_dim = patches.shape
        p = self.patch_size
        
        config = NetConfig()
        C_out = config.plaq_output_channels + config.rect_output_channels
        
        # Reshape: [B, H_p, W_p, C_out, p, p]
        patches = patches.view(B, H_p, W_p, C_out, p, p)
        # Rearrange: [B, C_out, H_p, p, W_p, p]
        patches = patches.permute(0, 3, 1, 4, 2, 5).contiguous()
        # Reconstruct: [B, C_out, L, L]
        output = patches.view(B, C_out, L, L)
        
        return output


class LocalNetMultiFreq(nn.Module):
    """
    Multi-frequency branch network for gauge field updates.
    
    Architecture:
    - High-freq branch: small kernels for local fluctuations
    - Mid-freq branch: medium kernels for intermediate correlations  
    - Low-freq branch: dilated kernels for long-range correlations
    - Adaptive fusion based on input characteristics
    - Output: tanh scaling to [-1/4, 1/4] range
    
    Physics Motivation:
    - Different β values have different correlation lengths
    - High β: long-range correlations (need low-freq branch)
    - Low β: local fluctuations dominate (need high-freq branch)
    - Network learns to weight branches appropriately
    
    Locality Properties:
    - High-freq: RF=5x5 (local patterns)
    - Mid-freq: RF=9x9 (intermediate scale)
    - Low-freq: RF=17x17 (broader correlations but still local)
    - Adaptive combination preserves locality
    
    Total parameters: ~ 7,500
    """
    def __init__(self):
        super().__init__()
        config = NetConfig()
        combined_input_channels = config.plaq_input_channels + config.rect_input_channels
        branch_channels = 8  # Reduced channels per branch for efficiency
        
        # High-frequency branch: small kernels for local features
        # Parameters: 6 * 8 * 1 * 1 + 8 = 56
        self.high_conv1 = nn.Conv2d(combined_input_channels, branch_channels, 1, padding=0, padding_mode='circular')
        # Parameters: 8 * 8 * 3 * 3 + 8 = 584
        self.high_conv2 = nn.Conv2d(branch_channels, branch_channels, 3, padding=1, padding_mode='circular')  # RF=3→5
        
        # Mid-frequency branch: standard kernels
        # Parameters: 6 * 8 * 3 * 3 + 8 = 440
        self.mid_conv1 = nn.Conv2d(combined_input_channels, branch_channels, 3, padding=1, padding_mode='circular')
        # Parameters: 8 * 8 * 3 * 3 + 8 = 584
        self.mid_conv2 = nn.Conv2d(branch_channels, branch_channels, 3, padding=1, padding_mode='circular')  # RF=3→5→7→9
        # Parameters: 8 * 8 * 3 * 3 + 8 = 584
        self.mid_conv3 = nn.Conv2d(branch_channels, branch_channels, 3, padding=1, padding_mode='circular')
        
        # Low-frequency branch: dilated kernels for long-range
        # Parameters: 6 * 8 * 3 * 3 + 8 = 440
        self.low_conv1 = nn.Conv2d(combined_input_channels, branch_channels, 3, padding=2, dilation=2, padding_mode='circular')
        # Parameters: 8 * 8 * 3 * 3 + 8 = 584
        self.low_conv2 = nn.Conv2d(branch_channels, branch_channels, 3, padding=4, dilation=4, padding_mode='circular')  # RF=5→13→17
        
        # Adaptive fusion weights based on input statistics
        # Parameters: 24 * 16 + 16 = 400 (global avg pool → fc → weights)
        self.adaptive_pool = nn.AdaptiveAvgPool2d(1)
        self.fusion_weights = nn.Sequential(
            nn.Linear(branch_channels * 3, 16),  # 24 → 16
            nn.ReLU(),  # 0 parameters
            nn.Linear(16, 3),  # Parameters: 16 * 3 + 3 = 51
            nn.Softmax(dim=1)  # 0 parameters
        )
        
        # Final processing
        # Parameters: 24 * 12 * 3 * 3 + 12 = 2,604
        self.output_conv = nn.Conv2d(branch_channels * 3, config.plaq_output_channels + config.rect_output_channels, 3, padding=1, padding_mode='circular')
        
        self.activation = nn.GELU()  # 0 parameters
        
    def forward(self, plaq_features, rect_features):
        config = NetConfig()
        # Combine input features
        x = torch.cat([plaq_features, rect_features], dim=1)  # [B, 6, L, L]
        B, C, H, W = x.shape
        
        # High-frequency branch (local patterns)
        high_feat = self.activation(self.high_conv1(x))  # [B, 8, L, L] (56 parameters used)
        high_feat = self.activation(self.high_conv2(high_feat))  # RF=5x5 (584 parameters used)
        
        # Mid-frequency branch (intermediate scale)
        mid_feat = self.activation(self.mid_conv1(x))  # [B, 8, L, L] (440 parameters used)
        mid_feat = self.activation(self.mid_conv2(mid_feat))  # (584 parameters used)
        mid_feat = self.activation(self.mid_conv3(mid_feat))  # RF=9x9 (584 parameters used)
        
        # Low-frequency branch (long-range correlations)
        low_feat = self.activation(self.low_conv1(x))  # [B, 8, L, L] (440 parameters used)
        low_feat = self.activation(self.low_conv2(low_feat))  # RF=17x17 (584 parameters used)
        
        # Adaptive fusion based on feature statistics
        concat_feat = torch.cat([high_feat, mid_feat, low_feat], dim=1)  # [B, 24, L, L]
        
        # Compute adaptive weights based on global statistics
        pooled = self.adaptive_pool(concat_feat).view(B, -1)  # [B, 24] (0 parameters - tensor operation)
        weights = self.fusion_weights(pooled)  # [B, 3] (400 + 51 = 451 parameters used)
        
        # Apply adaptive weighting to each branch
        weighted_high = high_feat * weights[:, 0:1, None, None]  # [B, 8, L, L]
        weighted_mid = mid_feat * weights[:, 1:2, None, None]   # [B, 8, L, L]
        weighted_low = low_feat * weights[:, 2:3, None, None]   # [B, 8, L, L]
        
        # Combine weighted features
        fused = torch.cat([weighted_high, weighted_mid, weighted_low], dim=1)  # [B, 24, L, L]
        
        # Final output
        output = self.output_conv(fused)  # [B, 12, L, L] (2,604 parameters used)
        output = torch.tanh(output) * 0.25  # [-1/4, 1/4] range (0 parameters - tensor operation)
        
        # Split output
        plaq_coeffs = output[:, :config.plaq_output_channels, :, :]
        rect_coeffs = output[:, config.plaq_output_channels:, :, :]
        
        return plaq_coeffs, rect_coeffs


class LocalNetAdaptiveRF(nn.Module):
    """
    Adaptive receptive field network for gauge field updates.
    
    Architecture:
    - Multiple parallel convolution paths with different receptive fields
    - Spatial attention to select appropriate RF for each location
    - Dynamic feature aggregation based on local field characteristics
    - Output: tanh scaling to [-1/4, 1/4] range
    
    Physics Motivation:
    - Different regions may need different correlation lengths
    - Strong coupling regions need larger RF, weak coupling need smaller RF
    - Network learns spatially-varying optimal scale selection
    - More flexible than fixed RF approaches
    
    Locality Properties:
    - Path 1: RF=3x3 (very local)
    - Path 2: RF=7x7 (medium local)  
    - Path 3: RF=15x15 (broader local)
    - Spatial attention preserves locality (learns local RF selection)
    - Maximum effective RF = 15x15, still local for large L
    
    Total parameters: ~ 6,500
    """
    def __init__(self):
        super().__init__()
        config = NetConfig()
        combined_input_channels = config.plaq_input_channels + config.rect_input_channels
        path_channels = 8  # Channels per RF path
        
        # Multi-scale convolution paths
        # Path 1: Small RF (3x3)
        # Parameters: 6 * 8 * 3 * 3 + 8 = 440
        self.path1_conv = nn.Conv2d(combined_input_channels, path_channels, 3, padding=1, padding_mode='circular')
        
        # Path 2: Medium RF (7x7 through dilated)
        # Parameters: 6 * 8 * 3 * 3 + 8 = 440
        self.path2_conv1 = nn.Conv2d(combined_input_channels, path_channels, 3, padding=1, padding_mode='circular')
        # Parameters: 8 * 8 * 3 * 3 + 8 = 584
        self.path2_conv2 = nn.Conv2d(path_channels, path_channels, 3, padding=2, dilation=2, padding_mode='circular')
        
        # Path 3: Large RF (15x15 through progressive dilation)
        # Parameters: 6 * 8 * 3 * 3 + 8 = 440
        self.path3_conv1 = nn.Conv2d(combined_input_channels, path_channels, 3, padding=2, dilation=2, padding_mode='circular')
        # Parameters: 8 * 8 * 3 * 3 + 8 = 584
        self.path3_conv2 = nn.Conv2d(path_channels, path_channels, 3, padding=4, dilation=4, padding_mode='circular')
        
        # Spatial attention for RF selection
        attention_input_channels = path_channels * 3  # 24 channels from 3 paths
        # Parameters: 24 * 12 * 1 * 1 + 12 = 300
        self.attention_conv1 = nn.Conv2d(attention_input_channels, 12, 1, padding=0)
        # Parameters: 12 * 3 * 1 * 1 + 3 = 39
        self.attention_conv2 = nn.Conv2d(12, 3, 1, padding=0)  # 3 attention maps for 3 paths
        
        # Feature fusion
        # Parameters: 24 * 16 * 1 * 1 + 16 = 400
        self.fusion_conv = nn.Conv2d(attention_input_channels, 16, 1, padding=0)
        
        # Output projection
        # Parameters: 16 * 12 * 3 * 3 + 12 = 1,740
        self.output_conv = nn.Conv2d(16, config.plaq_output_channels + config.rect_output_channels, 3, padding=1, padding_mode='circular')
        
        self.activation = nn.GELU()  # 0 parameters
        
    def forward(self, plaq_features, rect_features):
        config = NetConfig()
        # Combine input features
        x = torch.cat([plaq_features, rect_features], dim=1)  # [B, 6, L, L]
        
        # Multi-scale feature extraction paths
        # Path 1: Small receptive field (3x3)
        path1 = self.activation(self.path1_conv(x))  # [B, 8, L, L] (440 parameters used)
        
        # Path 2: Medium receptive field (3→7x7)
        path2 = self.activation(self.path2_conv1(x))  # [B, 8, L, L] (440 parameters used)
        path2 = self.activation(self.path2_conv2(path2))  # RF=7x7 (584 parameters used)
        
        # Path 3: Large receptive field (5→15x15)
        path3 = self.activation(self.path3_conv1(x))  # [B, 8, L, L] (440 parameters used)
        path3 = self.activation(self.path3_conv2(path3))  # RF=15x15 (584 parameters used)
        
        # Concatenate all paths for attention computation
        all_paths = torch.cat([path1, path2, path3], dim=1)  # [B, 24, L, L]
        
        # Spatial attention for adaptive RF selection
        attention_features = self.activation(self.attention_conv1(all_paths))  # [B, 12, L, L] (300 parameters used)
        attention_weights = torch.softmax(self.attention_conv2(attention_features), dim=1)  # [B, 3, L, L] (39 parameters used)
        
        # Apply spatially-adaptive weighting
        weighted_path1 = path1 * attention_weights[:, 0:1, :, :]  # [B, 8, L, L]
        weighted_path2 = path2 * attention_weights[:, 1:2, :, :]  # [B, 8, L, L]
        weighted_path3 = path3 * attention_weights[:, 2:3, :, :]  # [B, 8, L, L]
        
        # Combine weighted features
        weighted_features = torch.cat([weighted_path1, weighted_path2, weighted_path3], dim=1)  # [B, 24, L, L]
        
        # Feature fusion
        fused = self.activation(self.fusion_conv(weighted_features))  # [B, 16, L, L] (400 parameters used)
        
        # Final output
        output = self.output_conv(fused)  # [B, 12, L, L] (1,740 parameters used)
        output = torch.tanh(output) * 0.25  # [-1/4, 1/4] range (0 parameters - tensor operation)
        
        # Split output
        plaq_coeffs = output[:, :config.plaq_output_channels, :, :]
        rect_coeffs = output[:, config.plaq_output_channels:, :, :]
        
        return plaq_coeffs, rect_coeffs


class LocalWindowAttention(nn.Module):
    """
    Efficient local window attention for transformer.
    Each patch only attends to patches within a fixed window.
    
    Parameters: ~ 4,200 for embed_dim=32
    """
    def __init__(self, embed_dim, num_heads, window_size):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.head_dim = embed_dim // num_heads
        
        # Standard attention projections
        # Parameters: 32 * 96 + 96 = 3,168
        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        # Parameters: 32 * 32 + 32 = 1,056
        self.proj = nn.Linear(embed_dim, embed_dim)
        
    def forward(self, x, H, W):
        """
        x: [B, H, W, embed_dim]
        H, W: patch grid dimensions
        """
        B, H, W, C = x.shape
        
        # Generate QKV
        qkv = self.qkv(x)  # [B, H, W, 3*embed_dim]
        qkv = qkv.reshape(B, H, W, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.permute(3, 0, 4, 1, 2, 5)  # [3, B, num_heads, H, W, head_dim]
        
        # Apply local attention efficiently by processing patches in windows
        output = torch.zeros_like(v[0])  # [B, num_heads, H, W, head_dim]
        
        w = self.window_size
        for i in range(0, H, w//2):  # Sliding window with overlap
            for j in range(0, W, w//2):
                # Extract window
                end_i, end_j = min(i + w, H), min(j + w, W)
                
                q_win = q[:, :, i:end_i, j:end_j, :]  # [B, num_heads, win_h, win_w, head_dim]
                k_win = k[:, :, i:end_i, j:end_j, :]
                v_win = v[:, :, i:end_i, j:end_j, :]
                
                # Reshape for attention
                win_h, win_w = q_win.shape[2], q_win.shape[3]
                q_win = q_win.reshape(B, self.num_heads, win_h * win_w, self.head_dim)
                k_win = k_win.reshape(B, self.num_heads, win_h * win_w, self.head_dim)
                v_win = v_win.reshape(B, self.num_heads, win_h * win_w, self.head_dim)
                
                # Attention within window
                attn = torch.matmul(q_win, k_win.transpose(-2, -1)) / (self.head_dim ** 0.5)
                attn = torch.softmax(attn, dim=-1)
                out_win = torch.matmul(attn, v_win)  # [B, num_heads, win_h*win_w, head_dim]
                
                # Reshape back and accumulate
                out_win = out_win.reshape(B, self.num_heads, win_h, win_w, self.head_dim)
                output[:, :, i:end_i, j:end_j, :] += out_win
        
        # Reshape and project
        output = output.permute(0, 2, 3, 1, 4).reshape(B, H, W, C)  # [B, H, W, embed_dim]
        output = self.proj(output)
        
        return output


class ResidualBlock(nn.Module):
    """
    Pre-norm residual block with learnable scaling.
    
    Architecture:
    - Pre-norm → SiLU → Conv
    - Residual connection with learnable scaling factor α
    - GroupNorm for stable training
    - Circular padding for lattice boundary conditions
    
    Parameters per block: ~ 1,300 for 12 hidden channels
    """
    def __init__(self, channels, kernel_size=(3, 3)):
        super().__init__()
        # Parameters: 2 * channels each
        self.norm = nn.GroupNorm(2, channels)  # 2 * channels
        self.conv = nn.Conv2d(channels, channels, kernel_size, padding='same', padding_mode='circular')  # channels² * 9 + channels
        
        self.activation = nn.SiLU()  # 0 parameters
        self.alpha = nn.Parameter(torch.tensor(0.3))  # 1 parameter - learnable residual scaling

    def forward(self, x):
        identity = x  # 0 parameters

        # First block (norm + conv parameters used)
        out = self.norm(x)  # 2 * channels parameters
        out = self.activation(out)  # 0 parameters
        out = self.conv(out)  # channels² * 9 + channels parameters

        # Residual connection with learnable scaling (1 parameter used)
        return identity + self.alpha * out  # 1 parameter for alpha


class DoubleResidualBlock(nn.Module):
    """
    Pre-norm residual block with learnable scaling.
    
    Architecture:
    - Pre-norm → SiLU → Conv → Pre-norm → SiLU → Conv
    - Residual connection with learnable scaling factor α
    - GroupNorm for stable training
    - Circular padding for lattice boundary conditions
    
    Parameters per block: ~ 2,600 for 12 hidden channels
    """
    def __init__(self, channels, kernel_size=(3, 3), alpha=None, group_norm_groups=2):
        super().__init__()
        # Parameters: 2 * channels each
        self.norm1 = nn.GroupNorm(group_norm_groups, channels)  # 2 * channels
        self.conv1 = nn.Conv2d(channels, channels, kernel_size, padding='same', padding_mode='circular')  # channels² * 9 + channels
        self.norm2 = nn.GroupNorm(group_norm_groups, channels)  # 2 * channels
        self.conv2 = nn.Conv2d(channels, channels, kernel_size, padding='same', padding_mode='circular')  # channels² * 9 + channels
        
        self.activation = nn.SiLU()  # 0 parameters
        if alpha is not None:
            self.alpha = nn.Parameter(torch.tensor(alpha))  # 1 parameter - learnable residual scaling
        else:
            self.alpha = nn.Parameter(torch.tensor(0.3))  # 1 parameter - learnable residual scaling

    def forward(self, x):
        identity = x  # 0 parameters

        # First block (norm + conv parameters used)
        out = self.norm1(x)  # 2 * channels parameters
        out = self.activation(out)  # 0 parameters
        out = self.conv1(out)  # channels² * 9 + channels parameters
        
        # Second block (norm + conv parameters used)
        out = self.norm2(out)  # 2 * channels parameters
        out = self.activation(out)  # 0 parameters
        out = self.conv2(out)  # channels² * 9 + channels parameters

        # Residual connection with learnable scaling (1 parameter used)
        return identity + self.alpha * out  # 1 parameter for alpha


class ChannelAttention(nn.Module):
    """
    Squeeze-and-Excitation style channel attention.
    
    Architecture:
    - Global average pooling (squeeze)
    - Channel dimension reduction by factor of 4
    - ReLU activation
    - Channel dimension expansion back to original
    - Sigmoid gating
    
    Parameters: ~ 100 for 12 input channels
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
    elif model_tag == 'alpha':
        return LocalNetAlpha
    elif model_tag == 'arctan':
        return LocalNetArcTan
    elif model_tag == 'resn':
        return LocalResNet # base + 1 conv of resnet
    elif model_tag == 'res2n':
        return Local2ResNet # base + 2 conv of resnet
    elif model_tag == 'attn':
        return LocalAttnNet # base + channel attn
    elif model_tag == 'res2attn':
        return Local2ResAttnNet # base + 2 conv of resnet + channel attn
    elif model_tag == 'coorconv':
        return LocalCoorConvNet
    elif model_tag == 'unet':
        return LocalUNet
    elif model_tag == 'trans':
        return LocalTransformerNet
    elif model_tag == 'multif':
        return LocalNetMultiFreq
    elif model_tag == 'adrf':
        return LocalNetAdaptiveRF
    else:
        raise ValueError(f"Invalid model tag: {model_tag}")
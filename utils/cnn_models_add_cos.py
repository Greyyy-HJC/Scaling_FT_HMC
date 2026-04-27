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


class LocalNetAddCosWeight(nn.Module):
    """
    Add weighted threasholds to plaq and rect channels
    """
    def __init__(self):
        super().__init__()
        config = NetConfig()
            
        # Combined input channels for plaq and rect features
        combined_input_channels = config.plaq_input_channels + config.rect_input_channels
        combined_output_channels = 2 * (config.plaq_output_channels + config.rect_output_channels) #! add cos terms

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
            combined_output_channels,  # Combined output channels
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
        plaq_coeffs = torch.tanh(x[:, :2 * config.plaq_output_channels, :, :]) / 5  # [batch_size, 8, L, L] in range [-1/5, 1/5]
        rect_coeffs = torch.tanh(x[:, 2 * config.plaq_output_channels:, :, :]) / 40  # [batch_size, 16, L, L] in range [-1/40, 1/40]
        
        return plaq_coeffs, rect_coeffs 
    

class LocalNetAddCosEqual(nn.Module):
    """
    Add equal threasholds to plaq and rect channels
    """
    def __init__(self):
        super().__init__()
        config = NetConfig()
            
        # Combined input channels for plaq and rect features
        combined_input_channels = config.plaq_input_channels + config.rect_input_channels
        combined_output_channels = 2 * (config.plaq_output_channels + config.rect_output_channels) #! add cos terms

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
            combined_output_channels,  # Combined output channels
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
        plaq_coeffs = torch.tanh(x[:, :2 * config.plaq_output_channels, :, :]) / 12  # [batch_size, 8, L, L] in range [-1/12, 1/12]
        rect_coeffs = torch.tanh(x[:, 2 * config.plaq_output_channels:, :, :]) / 12  # [batch_size, 16, L, L] in range [-1/12, 1/12]
        
        return plaq_coeffs, rect_coeffs 
    


class LocalNetAddCosAllPlaq(nn.Module):
    """
    Add most weight to plaq channels
    """
    def __init__(self):
        super().__init__()
        config = NetConfig()
            
        # Combined input channels for plaq and rect features
        combined_input_channels = config.plaq_input_channels + config.rect_input_channels
        combined_output_channels = 2 * (config.plaq_output_channels + config.rect_output_channels) #! add cos terms

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
            combined_output_channels,  # Combined output channels
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
        plaq_coeffs = torch.tanh(x[:, :2 * config.plaq_output_channels, :, :]) / 4.00001  # [batch_size, 8, L, L]
        rect_coeffs = torch.tanh(x[:, 2 * config.plaq_output_channels:, :, :]) / 3200010  # [batch_size, 16, L, L]

        # 4 / 4.00001 + 8 / 3200010 = 0.9999999999984376
        
        return plaq_coeffs, rect_coeffs 



class LocalNetAddCosAllRect(nn.Module):
    """
    Add most weight to rect channels
    """
    def __init__(self):
        super().__init__()
        config = NetConfig()
            
        # Combined input channels for plaq and rect features
        combined_input_channels = config.plaq_input_channels + config.rect_input_channels
        combined_output_channels = 2 * (config.plaq_output_channels + config.rect_output_channels) #! add cos terms

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
            combined_output_channels,  # Combined output channels
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
        plaq_coeffs = torch.tanh(x[:, :2 * config.plaq_output_channels, :, :]) / 3200010  # [batch_size, 8, L, L] 
        rect_coeffs = torch.tanh(x[:, 2 * config.plaq_output_channels:, :, :]) / 8.00001  # [batch_size, 16, L, L]
        
        return plaq_coeffs, rect_coeffs 



class LocalNetAddCosNoRect(nn.Module):
    """
    Totally remove the rect channels
    """
    def __init__(self):
        super().__init__()
        config = NetConfig()
            
        # Combined input channels for plaq and rect features
        combined_input_channels = config.plaq_input_channels + config.rect_input_channels
        combined_output_channels = 2 * config.plaq_output_channels #! add cos terms

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
            combined_output_channels,  # Combined output channels
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
        plaq_coeffs = torch.tanh(x[:, :, :, :]) / 4  # [batch_size, 8, L, L]
        
        return plaq_coeffs 
    

class LocalNetAddCosNoRectMoreCos(nn.Module):
    """
    Totally remove the rect channels
    """
    def __init__(self):
        super().__init__()
        config = NetConfig()
            
        # Combined input channels for plaq and rect features
        combined_input_channels = config.plaq_input_channels + config.rect_input_channels
        combined_output_channels = 2 * config.plaq_output_channels #! add cos terms

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
            combined_output_channels,  # Combined output channels
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
        x_tanh = torch.tanh(x) # [batch_size, 8, L, L]
        plaq_coeffs = torch.cat([x_tanh[:, 0:4, :, :] * (1/8), x_tanh[:, 4:8, :, :] * (3/8)], dim=1) # 0:4 for sin terms, 4:8 for cos terms.
        
        return plaq_coeffs 
    

class LocalNetAddCosNoRectMoreSin(nn.Module):
    """
    Totally remove the rect channels
    """
    def __init__(self):
        super().__init__()
        config = NetConfig()
            
        # Combined input channels for plaq and rect features
        combined_input_channels = config.plaq_input_channels + config.rect_input_channels
        combined_output_channels = 2 * config.plaq_output_channels #! add cos terms

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
            combined_output_channels,  # Combined output channels
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
        x_tanh = torch.tanh(x) # [batch_size, 8, L, L]
        plaq_coeffs = torch.cat([x_tanh[:, 0:4, :, :] * (3/8), x_tanh[:, 4:8, :, :] * (1/8)], dim=1) # 0:4 for sin terms, 4:8 for cos terms.
        
        return plaq_coeffs 


def choose_cnn_model(model_tag):
    if model_tag == 'weight':
        return LocalNetAddCosWeight
    elif model_tag == 'equal':
        return LocalNetAddCosEqual
    elif model_tag == 'allp':
        return LocalNetAddCosAllPlaq
    elif model_tag == 'allr':
        return LocalNetAddCosAllRect
    elif model_tag == 'norect':
        return LocalNetAddCosNoRect
    elif model_tag == 'morecos':
        return LocalNetAddCosNoRectMoreCos
    elif model_tag == 'moresin':
        return LocalNetAddCosNoRectMoreSin
    else:
        raise ValueError(f"Invalid model tag: {model_tag}")
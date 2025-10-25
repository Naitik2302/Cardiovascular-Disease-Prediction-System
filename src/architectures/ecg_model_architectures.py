"""
ECG Model Architectures for Cardiovascular Disease Prediction
============================================================

This module contains various 1D CNN and LSTM architectures for ECG signal classification,
including custom architectures and transfer learning support.

Features:
- 1D CNN architectures for ECG signals
- LSTM and BiLSTM architectures
- Transformer-based architectures
- Attention mechanisms
- Model compilation utilities
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, Any, List
import logging
import math

logger = logging.getLogger(__name__)

class ECG1DCNN(nn.Module):
    """
    1D CNN architecture for ECG signal classification.
    """
    
    def __init__(self, 
                 input_length: int = 1000,
                 num_classes: int = 2,
                 dropout_rate: float = 0.3,
                 num_filters: List[int] = [32, 64, 128, 256],
                 kernel_sizes: List[int] = [7, 5, 3, 3],
                 pool_sizes: List[int] = [2, 2, 2, 2]):
        """
        Initialize the 1D CNN model with residual connections.
        
        Args:
            input_length (int): Length of input ECG signal
            num_classes (int): Number of output classes
            dropout_rate (float): Dropout rate
            num_filters (list): Number of filters in each layer
            kernel_sizes (list): Kernel sizes for each layer
            pool_sizes (list): Pooling sizes for each layer
        """
        super(ECG1DCNN, self).__init__()
        
        self.input_length = input_length
        self.num_classes = num_classes
        self.dropout_rate = dropout_rate
        
        # Convolutional layers with residual connections
        self.conv_layers = nn.ModuleList()
        self.bn_layers = nn.ModuleList()
        self.pool_layers = nn.ModuleList()
        self.skip_layers = nn.ModuleList()
        
        in_channels = 1
        current_length = input_length
        
        for i, (out_channels, kernel_size, pool_size) in enumerate(zip(num_filters, kernel_sizes, pool_sizes)):
            # Convolution
            conv = nn.Conv1d(in_channels, out_channels, kernel_size, padding=kernel_size//2)
            self.conv_layers.append(conv)
            
            # Skip connection
            skip = nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=1, padding=0) if in_channels != out_channels else nn.Identity()
            self.skip_layers.append(skip)
            
            # Batch normalization
            bn = nn.BatchNorm1d(out_channels)
            self.bn_layers.append(bn)
            
            # Pooling
            pool = nn.MaxPool1d(pool_size)
            self.pool_layers.append(pool)
            
            # Update dimensions
            current_length = current_length // pool_size
            in_channels = out_channels
        
        # Calculate final feature size
        self.final_length = current_length
        self.final_channels = num_filters[-1]
        
        # Global average pooling
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        
        # Enhanced classifier
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(self.final_channels, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )
        
    def forward(self, x):
        """Forward pass through the network."""
        # Reshape input if needed (batch_size, 1, signal_length)
        if x.dim() == 2:
            x = x.unsqueeze(1)
        
        # Convolutional layers with residual connections
        for conv, bn, pool, skip in zip(self.conv_layers, self.bn_layers, self.pool_layers, self.skip_layers):
            identity = skip(x)
            x = conv(x)
            x = bn(x)
            x = F.relu(x)
            x = x + identity  # Residual connection
            x = pool(x)
        
        # Global average pooling
        x = self.global_avg_pool(x)
        x = x.view(x.size(0), -1)
        
        # Classifier
        x = self.classifier(x)
        
        return x


class ECGLSTM(nn.Module):
    """
    LSTM architecture for ECG signal classification.
    """
    
    def __init__(self, 
                 input_length: int = 1000,
                 num_classes: int = 2,
                 hidden_size: int = 128,
                 num_layers: int = 2,
                 dropout_rate: float = 0.5,
                 bidirectional: bool = True):
        """
        Initialize the LSTM model.
        
        Args:
            input_length (int): Length of input ECG signal
            num_classes (int): Number of output classes
            hidden_size (int): Hidden size of LSTM
            num_layers (int): Number of LSTM layers
            dropout_rate (float): Dropout rate
            bidirectional (bool): Whether to use bidirectional LSTM
        """
        super(ECGLSTM, self).__init__()
        
        self.input_length = input_length
        self.num_classes = num_classes
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=1,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout_rate if num_layers > 1 else 0,
            bidirectional=bidirectional,
            batch_first=True
        )
        
        # Calculate LSTM output size
        lstm_output_size = hidden_size * 2 if bidirectional else hidden_size
        
        # Attention mechanism
        self.attention = nn.MultiheadAttention(
            embed_dim=lstm_output_size,
            num_heads=8,
            dropout=dropout_rate,
            batch_first=True
        )
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(lstm_output_size, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(64, num_classes)
        )
        
    def forward(self, x):
        """Forward pass through the network."""
        # Reshape input if needed (batch_size, signal_length, 1)
        if x.dim() == 2:
            x = x.unsqueeze(-1)
        
        # LSTM forward pass
        lstm_out, (hidden, cell) = self.lstm(x)
        
        # Apply attention
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
        
        # Global average pooling
        x = torch.mean(attn_out, dim=1)
        
        # Classifier
        x = self.classifier(x)
        
        return x


class ECGBiLSTM(nn.Module):
    """
    Bidirectional LSTM architecture for ECG signal classification.
    """
    
    def __init__(self, 
                 input_length: int = 1000,
                 num_classes: int = 2,
                 hidden_size: int = 128,
                 num_layers: int = 2,
                 dropout_rate: float = 0.5):
        """
        Initialize the BiLSTM model.
        
        Args:
            input_length (int): Length of input ECG signal
            num_classes (int): Number of output classes
            hidden_size (int): Hidden size of LSTM
            num_layers (int): Number of LSTM layers
            dropout_rate (float): Dropout rate
        """
        super(ECGBiLSTM, self).__init__()
        
        self.input_length = input_length
        self.num_classes = num_classes
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Bidirectional LSTM
        self.bilstm = nn.LSTM(
            input_size=1,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout_rate if num_layers > 1 else 0,
            bidirectional=True,
            batch_first=True
        )
        
        # Attention mechanism
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_size * 2,
            num_heads=8,
            dropout=dropout_rate,
            batch_first=True
        )
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size * 2, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(64, num_classes)
        )
        
    def forward(self, x):
        """Forward pass through the network."""
        # Reshape input if needed
        if x.dim() == 2:
            x = x.unsqueeze(-1)
        
        # BiLSTM forward pass
        lstm_out, _ = self.bilstm(x)
        
        # Apply attention
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
        
        # Global average pooling
        x = torch.mean(attn_out, dim=1)
        
        # Classifier
        x = self.classifier(x)
        
        return x


class ECGTransformer(nn.Module):
    """
    Transformer architecture for ECG signal classification.
    """
    
    def __init__(self, 
                 input_length: int = 1000,
                 num_classes: int = 2,
                 d_model: int = 128,
                 nhead: int = 8,
                 num_layers: int = 6,
                 dropout_rate: float = 0.1):
        """
        Initialize the Transformer model.
        
        Args:
            input_length (int): Length of input ECG signal
            num_classes (int): Number of output classes
            d_model (int): Model dimension
            nhead (int): Number of attention heads
            num_layers (int): Number of transformer layers
            dropout_rate (float): Dropout rate
        """
        super(ECGTransformer, self).__init__()
        
        self.input_length = input_length
        self.num_classes = num_classes
        self.d_model = d_model
        
        # Input projection
        self.input_projection = nn.Linear(1, d_model)
        
        # Positional encoding
        self.pos_encoding = self._create_positional_encoding(input_length, d_model)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout_rate,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Global pooling
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(d_model, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(64, num_classes)
        )
        
    def _create_positional_encoding(self, seq_len: int, d_model: int) -> torch.Tensor:
        """Create positional encoding."""
        pe = torch.zeros(seq_len, d_model)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)
        
    def forward(self, x):
        """Forward pass through the network."""
        # Reshape input if needed
        if x.dim() == 2:
            x = x.unsqueeze(-1)
        
        # Input projection
        x = self.input_projection(x)
        
        # Add positional encoding
        x = x + self.pos_encoding.to(x.device)
        
        # Transformer forward pass
        x = self.transformer(x)
        
        # Global pooling
        x = x.transpose(1, 2)  # (batch, d_model, seq_len)
        x = self.global_pool(x)  # (batch, d_model, 1)
        x = x.squeeze(-1)  # (batch, d_model)
        
        # Classifier
        x = self.classifier(x)
        
        return x


class ECGHybrid(nn.Module):
    """
    Hybrid CNN-LSTM architecture for ECG signal classification.
    """
    
    def __init__(self, 
                 input_length: int = 1000,
                 num_classes: int = 2,
                 cnn_filters: List[int] = [64, 128, 256],
                 cnn_kernels: List[int] = [3, 3, 3],
                 lstm_hidden: int = 128,
                 lstm_layers: int = 2,
                 dropout_rate: float = 0.5):
        """
        Initialize the hybrid model.
        
        Args:
            input_length (int): Length of input ECG signal
            num_classes (int): Number of output classes
            cnn_filters (list): CNN filter sizes
            cnn_kernels (list): CNN kernel sizes
            lstm_hidden (int): LSTM hidden size
            lstm_layers (int): Number of LSTM layers
            dropout_rate (float): Dropout rate
        """
        super(ECGHybrid, self).__init__()
        
        self.input_length = input_length
        self.num_classes = num_classes
        
        # CNN layers
        self.cnn_layers = nn.ModuleList()
        self.cnn_bn = nn.ModuleList()
        self.cnn_pool = nn.ModuleList()
        
        in_channels = 1
        current_length = input_length
        
        for out_channels, kernel_size in zip(cnn_filters, cnn_kernels):
            # Convolution
            conv = nn.Conv1d(in_channels, out_channels, kernel_size, padding=kernel_size//2)
            self.cnn_layers.append(conv)
            
            # Batch normalization
            bn = nn.BatchNorm1d(out_channels)
            self.cnn_bn.append(bn)
            
            # Pooling
            pool = nn.MaxPool1d(2)
            self.cnn_pool.append(pool)
            
            current_length = current_length // 2
            in_channels = out_channels
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=cnn_filters[-1],
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            dropout=dropout_rate if lstm_layers > 1 else 0,
            bidirectional=True,
            batch_first=True
        )
        
        # Attention
        self.attention = nn.MultiheadAttention(
            embed_dim=lstm_hidden * 2,
            num_heads=8,
            dropout=dropout_rate,
            batch_first=True
        )
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(lstm_hidden * 2, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(64, num_classes)
        )
        
    def forward(self, x):
        """Forward pass through the network."""
        # Reshape input if needed
        if x.dim() == 2:
            x = x.unsqueeze(1)
        
        # CNN layers
        for conv, bn, pool in zip(self.cnn_layers, self.cnn_bn, self.cnn_pool):
            x = conv(x)
            x = bn(x)
            x = F.relu(x)
            x = pool(x)
        
        # Reshape for LSTM (batch, seq, features)
        x = x.transpose(1, 2)
        
        # LSTM forward pass
        lstm_out, _ = self.lstm(x)
        
        # Apply attention
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
        
        # Global average pooling
        x = torch.mean(attn_out, dim=1)
        
        # Classifier
        x = self.classifier(x)
        
        return x


def create_ecg_model(model_config: Dict[str, Any]) -> nn.Module:
    """
    Create an ECG model based on configuration.
    
    Args:
        model_config (dict): Model configuration
        
    Returns:
        nn.Module: Created model
    """
    model_type = model_config.get('type', '1d_cnn')
    input_length = model_config.get('input_length', 1000)
    num_classes = model_config.get('num_classes', 2)
    dropout_rate = model_config.get('dropout_rate', 0.5)
    
    if model_type == '1d_cnn':
        model = ECG1DCNN(
            input_length=input_length,
            num_classes=num_classes,
            dropout_rate=dropout_rate,
            num_filters=model_config.get('num_filters', [64, 128, 256, 512]),
            kernel_sizes=model_config.get('kernel_sizes', [3, 3, 3, 3]),
            pool_sizes=model_config.get('pool_sizes', [2, 2, 2, 2])
        )
    elif model_type == 'lstm':
        model = ECGLSTM(
            input_length=input_length,
            num_classes=num_classes,
            hidden_size=model_config.get('hidden_size', 128),
            num_layers=model_config.get('num_layers', 2),
            dropout_rate=dropout_rate,
            bidirectional=model_config.get('bidirectional', True)
        )
    elif model_type == 'bilstm':
        model = ECGBiLSTM(
            input_length=input_length,
            num_classes=num_classes,
            hidden_size=model_config.get('hidden_size', 128),
            num_layers=model_config.get('num_layers', 2),
            dropout_rate=dropout_rate
        )
    elif model_type == 'transformer':
        model = ECGTransformer(
            input_length=input_length,
            num_classes=num_classes,
            d_model=model_config.get('d_model', 128),
            nhead=model_config.get('nhead', 8),
            num_layers=model_config.get('num_layers', 6),
            dropout_rate=dropout_rate
        )
    elif model_type == 'hybrid':
        model = ECGHybrid(
            input_length=input_length,
            num_classes=num_classes,
            cnn_filters=model_config.get('cnn_filters', [64, 128, 256]),
            cnn_kernels=model_config.get('cnn_kernels', [3, 3, 3]),
            lstm_hidden=model_config.get('lstm_hidden', 128),
            lstm_layers=model_config.get('lstm_layers', 2),
            dropout_rate=dropout_rate
        )
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    
    logger.info(f"Created {model_type} model with {count_parameters(model):,} parameters")
    return model


def count_parameters(model: nn.Module) -> int:
    """
    Count the number of trainable parameters in a model.
    
    Args:
        model (nn.Module): PyTorch model
        
    Returns:
        int: Number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_model_summary(model: nn.Module, input_shape: Tuple[int, int] = (1000,)) -> str:
    """
    Get a summary of the model architecture.
    
    Args:
        model (nn.Module): PyTorch model
        input_shape (tuple): Input shape (signal_length,)
        
    Returns:
        str: Model summary
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    summary = f"""
ECG Model Summary:
=================
Total parameters: {total_params:,}
Trainable parameters: {trainable_params:,}
Non-trainable parameters: {total_params - trainable_params:,}
Input shape: {input_shape}
Model type: {model.__class__.__name__}
"""
    
    return summary


def main():
    """
    Example usage of the ECG model architectures.
    """
    # Example configurations
    configs = [
        {'type': '1d_cnn', 'input_length': 1000, 'num_classes': 2},
        {'type': 'lstm', 'input_length': 1000, 'num_classes': 2, 'hidden_size': 128},
        {'type': 'bilstm', 'input_length': 1000, 'num_classes': 2, 'hidden_size': 128},
        {'type': 'transformer', 'input_length': 1000, 'num_classes': 2, 'd_model': 128},
        {'type': 'hybrid', 'input_length': 1000, 'num_classes': 2}
    ]
    
    for config in configs:
        print(f"\nCreating {config['type']} model...")
        model = create_ecg_model(config)
        
        # Count parameters
        total_params = count_parameters(model)
        print(f"Total trainable parameters: {total_params:,}")
        
        # Get model summary
        summary = get_model_summary(model)
        print(summary)
        
        # Test forward pass
        dummy_input = torch.randn(1, config['input_length'])
        with torch.no_grad():
            output = model(dummy_input)
            print(f"Output shape: {output.shape}")


if __name__ == "__main__":
    main()

"""
CNN Model Architectures for Cardiovascular Disease Prediction
============================================================

This module contains various CNN architectures for fundus image classification,
including ResNet, EfficientNet, and custom architectures.

Features:
- ResNet variants (ResNet18, ResNet50, ResNet101)
- EfficientNet variants (B0-B7)
- Custom CNN architectures
- Transfer learning support
- Model compilation utilities
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision.models import ResNet18_Weights, ResNet50_Weights, EfficientNet_B0_Weights
import timm
from typing import Optional, Tuple, Dict, Any
import logging

logger = logging.getLogger(__name__)

class FundusCNN(nn.Module):
    """
    Custom CNN architecture for fundus image classification.
    """
    
    def __init__(self, 
                 input_channels: int = 3,
                 num_classes: int = 2,
                 dropout_rate: float = 0.5):
        """
        Initialize the custom CNN model.
        
        Args:
            input_channels (int): Number of input channels
            num_classes (int): Number of output classes
            dropout_rate (float): Dropout rate
        """
        super(FundusCNN, self).__init__()
        
        self.input_channels = input_channels
        self.num_classes = num_classes
        self.dropout_rate = dropout_rate
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # Block 1
        self.conv2_1 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn2_1 = nn.BatchNorm2d(64)
        self.conv2_2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn2_2 = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Block 2
        self.conv3_1 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn3_1 = nn.BatchNorm2d(128)
        self.conv3_2 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.bn3_2 = nn.BatchNorm2d(128)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Block 3
        self.conv4_1 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.bn4_1 = nn.BatchNorm2d(256)
        self.conv4_2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.bn4_2 = nn.BatchNorm2d(256)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Block 4
        self.conv5_1 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
        self.bn5_1 = nn.BatchNorm2d(512)
        self.conv5_2 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.bn5_2 = nn.BatchNorm2d(512)
        self.pool4 = nn.AdaptiveAvgPool2d((1, 1))
        
        # Classifier
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(512, num_classes)
        
    def forward(self, x):
        """Forward pass through the network."""
        # Initial convolution
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        # Block 1
        residual = x
        x = self.conv2_1(x)
        x = self.bn2_1(x)
        x = self.relu(x)
        x = self.conv2_2(x)
        x = self.bn2_2(x)
        x = self.relu(x)
        x = self.pool1(x)
        
        # Block 2
        x = self.conv3_1(x)
        x = self.bn3_1(x)
        x = self.relu(x)
        x = self.conv3_2(x)
        x = self.bn3_2(x)
        x = self.relu(x)
        x = self.pool2(x)
        
        # Block 3
        x = self.conv4_1(x)
        x = self.bn4_1(x)
        x = self.relu(x)
        x = self.conv4_2(x)
        x = self.bn4_2(x)
        x = self.relu(x)
        x = self.pool3(x)
        
        # Block 4
        x = self.conv5_1(x)
        x = self.bn5_1(x)
        x = self.relu(x)
        x = self.conv5_2(x)
        x = self.bn5_2(x)
        x = self.relu(x)
        x = self.pool4(x)
        
        # Flatten and classify
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)
        
        return x


class ResNetFundus(nn.Module):
    """
    ResNet-based architecture for fundus image classification.
    """
    
    def __init__(self, 
                 model_name: str = 'resnet50',
                 num_classes: int = 2,
                 pretrained: bool = True,
                 freeze_backbone: bool = False,
                 dropout_rate: float = 0.5):
        """
        Initialize the ResNet model.
        
        Args:
            model_name (str): ResNet model name ('resnet18', 'resnet50', 'resnet101')
            num_classes (int): Number of output classes
            pretrained (bool): Whether to use pretrained weights
            freeze_backbone (bool): Whether to freeze backbone layers
            dropout_rate (float): Dropout rate
        """
        super(ResNetFundus, self).__init__()
        
        self.model_name = model_name
        self.num_classes = num_classes
        self.pretrained = pretrained
        self.freeze_backbone = freeze_backbone
        
        # Load pretrained ResNet
        if model_name == 'resnet18':
            if pretrained:
                self.backbone = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
            else:
                self.backbone = models.resnet18(weights=None)
            num_features = 512
        elif model_name == 'resnet50':
            if pretrained:
                self.backbone = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
            else:
                self.backbone = models.resnet50(weights=None)
            num_features = 2048
        elif model_name == 'resnet101':
            if pretrained:
                self.backbone = models.resnet101(weights=models.ResNet101_Weights.IMAGENET1K_V1)
            else:
                self.backbone = models.resnet101(weights=None)
            num_features = 2048
        else:
            raise ValueError(f"Unsupported ResNet model: {model_name}")
        
        # Freeze backbone if specified
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
        
        # Replace the final layer
        self.backbone.fc = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(num_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(512, num_classes)
        )
        
    def forward(self, x):
        """Forward pass through the network."""
        return self.backbone(x)


class EfficientNetFundus(nn.Module):
    """
    EfficientNet-based architecture for fundus image classification.
    """
    
    def __init__(self, 
                 model_name: str = 'efficientnet_b0',
                 num_classes: int = 2,
                 pretrained: bool = True,
                 freeze_backbone: bool = False,
                 dropout_rate: float = 0.5):
        """
        Initialize the EfficientNet model.
        
        Args:
            model_name (str): EfficientNet model name ('efficientnet_b0' to 'efficientnet_b7')
            num_classes (int): Number of output classes
            pretrained (bool): Whether to use pretrained weights
            freeze_backbone (bool): Whether to freeze backbone layers
            dropout_rate (float): Dropout rate
        """
        super(EfficientNetFundus, self).__init__()
        
        self.model_name = model_name
        self.num_classes = num_classes
        self.pretrained = pretrained
        self.freeze_backbone = freeze_backbone
        
        # Load pretrained EfficientNet
        if pretrained:
            self.backbone = timm.create_model(model_name, pretrained=True)
        else:
            self.backbone = timm.create_model(model_name, pretrained=False)
        
        # Get number of features
        num_features = self.backbone.classifier.in_features
        
        # Freeze backbone if specified
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
        
        # Replace the classifier
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(num_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(512, num_classes)
        )
        
    def forward(self, x):
        """Forward pass through the network."""
        return self.backbone(x)


class VGGFundus(nn.Module):
    """
    VGG-based architecture for fundus image classification.
    """
    
    def __init__(self, 
                 model_name: str = 'vgg16',
                 num_classes: int = 2,
                 pretrained: bool = True,
                 freeze_backbone: bool = False,
                 dropout_rate: float = 0.5):
        """
        Initialize the VGG model.
        
        Args:
            model_name (str): VGG model name ('vgg16', 'vgg19')
            num_classes (int): Number of output classes
            pretrained (bool): Whether to use pretrained weights
            freeze_backbone (bool): Whether to freeze backbone layers
            dropout_rate (float): Dropout rate
        """
        super(VGGFundus, self).__init__()
        
        self.model_name = model_name
        self.num_classes = num_classes
        self.pretrained = pretrained
        self.freeze_backbone = freeze_backbone
        
        # Load pretrained VGG
        if model_name == 'vgg16':
            if pretrained:
                self.backbone = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
            else:
                self.backbone = models.vgg16(weights=None)
        elif model_name == 'vgg19':
            if pretrained:
                self.backbone = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1)
            else:
                self.backbone = models.vgg19(weights=None)
        else:
            raise ValueError(f"Unsupported VGG model: {model_name}")
        
        # Freeze backbone if specified
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
        
        # Replace the classifier
        num_features = self.backbone.classifier[0].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Linear(num_features, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(4096, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(512, num_classes)
        )
        
    def forward(self, x):
        """Forward pass through the network."""
        return self.backbone(x)


class DenseNetFundus(nn.Module):
    """
    DenseNet-based architecture for fundus image classification.
    """
    
    def __init__(self, 
                 model_name: str = 'densenet121',
                 num_classes: int = 2,
                 pretrained: bool = True,
                 freeze_backbone: bool = False,
                 dropout_rate: float = 0.5):
        """
        Initialize the DenseNet model.
        
        Args:
            model_name (str): DenseNet model name ('densenet121', 'densenet161', 'densenet201')
            num_classes (int): Number of output classes
            pretrained (bool): Whether to use pretrained weights
            freeze_backbone (bool): Whether to freeze backbone layers
            dropout_rate (float): Dropout rate
        """
        super(DenseNetFundus, self).__init__()
        
        self.model_name = model_name
        self.num_classes = num_classes
        self.pretrained = pretrained
        self.freeze_backbone = freeze_backbone
        
        # Load pretrained DenseNet
        if model_name == 'densenet121':
            if pretrained:
                self.backbone = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1)
            else:
                self.backbone = models.densenet121(weights=None)
            num_features = 1024
        elif model_name == 'densenet161':
            if pretrained:
                self.backbone = models.densenet161(weights=models.DenseNet161_Weights.IMAGENET1K_V1)
            else:
                self.backbone = models.densenet161(weights=None)
            num_features = 2208
        elif model_name == 'densenet201':
            if pretrained:
                self.backbone = models.densenet201(weights=models.DenseNet201_Weights.IMAGENET1K_V1)
            else:
                self.backbone = models.densenet201(weights=None)
            num_features = 1920
        else:
            raise ValueError(f"Unsupported DenseNet model: {model_name}")
        
        # Freeze backbone if specified
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
        
        # Replace the classifier
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(num_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(512, num_classes)
        )
        
    def forward(self, x):
        """Forward pass through the network."""
        return self.backbone(x)


def create_model(model_config: Dict[str, Any]) -> nn.Module:
    """
    Create a model based on configuration.
    
    Args:
        model_config (dict): Model configuration
        
    Returns:
        nn.Module: Created model
    """
    model_type = model_config.get('type', 'resnet50')
    num_classes = model_config.get('num_classes', 2)
    pretrained = model_config.get('pretrained', True)
    freeze_backbone = model_config.get('freeze_backbone', False)
    dropout_rate = model_config.get('dropout_rate', 0.5)
    
    if model_type.startswith('resnet'):
        model = ResNetFundus(
            model_name=model_type,
            num_classes=num_classes,
            pretrained=pretrained,
            freeze_backbone=freeze_backbone,
            dropout_rate=dropout_rate
        )
    elif model_type.startswith('efficientnet'):
        model = EfficientNetFundus(
            model_name=model_type,
            num_classes=num_classes,
            pretrained=pretrained,
            freeze_backbone=freeze_backbone,
            dropout_rate=dropout_rate
        )
    elif model_type.startswith('vgg'):
        model = VGGFundus(
            model_name=model_type,
            num_classes=num_classes,
            pretrained=pretrained,
            freeze_backbone=freeze_backbone,
            dropout_rate=dropout_rate
        )
    elif model_type.startswith('densenet'):
        model = DenseNetFundus(
            model_name=model_type,
            num_classes=num_classes,
            pretrained=pretrained,
            freeze_backbone=freeze_backbone,
            dropout_rate=dropout_rate
        )
    elif model_type == 'custom_cnn':
        model = FundusCNN(
            input_channels=model_config.get('input_channels', 3),
            num_classes=num_classes,
            dropout_rate=dropout_rate
        )
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    
    logger.info(f"Created {model_type} model with {num_classes} classes")
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


def get_model_summary(model: nn.Module, input_size: Tuple[int, int, int] = (3, 224, 224)) -> str:
    """
    Get a summary of the model architecture.
    
    Args:
        model (nn.Module): PyTorch model
        input_size (tuple): Input size (channels, height, width)
        
    Returns:
        str: Model summary
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    summary = f"""
Model Summary:
==============
Total parameters: {total_params:,}
Trainable parameters: {trainable_params:,}
Non-trainable parameters: {total_params - trainable_params:,}
Input size: {input_size}
"""
    
    return summary


def freeze_layers(model: nn.Module, freeze_until: str = 'fc') -> None:
    """
    Freeze layers up to a specified layer.
    
    Args:
        model (nn.Module): PyTorch model
        freeze_until (str): Layer name to freeze until
    """
    freeze = True
    for name, param in model.named_parameters():
        if freeze_until in name:
            freeze = False
        param.requires_grad = not freeze


def unfreeze_all_layers(model: nn.Module) -> None:
    """
    Unfreeze all layers in the model.
    
    Args:
        model (nn.Module): PyTorch model
    """
    for param in model.parameters():
        param.requires_grad = True


def main():
    """
    Example usage of the model architectures.
    """
    # Example configurations
    configs = [
        {'type': 'resnet50', 'num_classes': 2, 'pretrained': True},
        {'type': 'efficientnet_b0', 'num_classes': 2, 'pretrained': True},
        {'type': 'custom_cnn', 'num_classes': 2, 'input_channels': 3}
    ]
    
    for config in configs:
        print(f"\nCreating {config['type']} model...")
        model = create_model(config)
        
        # Count parameters
        total_params = count_parameters(model)
        print(f"Total trainable parameters: {total_params:,}")
        
        # Get model summary
        summary = get_model_summary(model)
        print(summary)
        
        # Test forward pass
        dummy_input = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            output = model(dummy_input)
            print(f"Output shape: {output.shape}")


if __name__ == "__main__":
    main()

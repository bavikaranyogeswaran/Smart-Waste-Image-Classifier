"""
ResNet18 transfer-learning model builder for the Waste Image Classifier.

This module provides a utility to build and configure a ResNet18 model
pretrained on ImageNet, adapted for the specific number of waste classes.
"""
import torch.nn as nn
from torchvision import models


def build_model(num_classes):
    """
    Builds a pretrained ResNet18 model configured for transfer learning.

    This function loads a ResNet18 model with default ImageNet weights,
    freezes all base layers to prevent them from updating during training,
    and replaces the final fully connected layer to match the specified
    number of output classes.

    Args:
        num_classes (int): The number of classes to predict.

    Returns:
        torch.nn.Module: The configured ResNet18 PyTorch model.
    """
    # 1. Load the default ImageNet weights for ResNet18
    weights = models.ResNet18_Weights.DEFAULT
    
    # 2. Initialize the ResNet18 model with the loaded weights
    model = models.resnet18(weights=weights)

    # 3. Freeze all parameters in the base model so they are not updated during training
    for param in model.parameters():
        param.requires_grad = False

    # 4. Get the number of input features for the original fully connected layer
    in_features = model.fc.in_features
    
    # 5. Replace the fully connected layer with a new one matching the number of classes
    model.fc = nn.Linear(in_features, num_classes)

    # 6. Return the modified model
    return model

"""ResNet18 transfer-learning model builder.

Placeholder created in Step 1. Implemented in Step 6.
"""
import torch.nn as nn
from torchvision import models


def build_model(num_classes):
    weights = models.ResNet18_Weights.DEFAULT
    model = models.resnet18(weights=weights)

    for param in model.parameters():
        param.requires_grad = False

    return model

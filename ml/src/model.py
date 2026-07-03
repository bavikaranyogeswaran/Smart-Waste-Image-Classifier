"""ResNet18 transfer-learning model builder.

"""
import torch.nn as nn
from torchvision import models


def build_model(num_classes):
    weights = models.ResNet18_Weights.DEFAULT
    model = models.resnet18(weights=weights)

    for param in model.parameters():
        param.requires_grad = False

    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)

    return model

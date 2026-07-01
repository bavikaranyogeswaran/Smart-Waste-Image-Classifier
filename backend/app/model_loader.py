import os
import torch
import torch.nn as nn
from torchvision import models

MODEL_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..", "models", "waste_resnet18.pth"
)


def load_model():
    checkpoint = torch.load(MODEL_PATH, map_location="cpu")
    classes = checkpoint["classes"]

    # weights=None: skip downloading ImageNet weights at startup since the
    # checkpoint's state_dict is loaded below and overwrites them anyway.
    model = models.resnet18(weights=None)

    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, len(classes))

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return model, classes

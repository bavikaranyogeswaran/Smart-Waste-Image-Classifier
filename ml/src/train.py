"""Training loop for the waste classifier.

Placeholder created in Step 1. Implemented in Step 7.
"""
import torch

from dataset import get_dataloaders
from model import build_model

DATA_DIR   = "ml/data/splits"
MODEL_PATH = "ml/models/waste_resnet18.pth"
BATCH_SIZE = 32
EPOCHS     = 10
LR         = 0.001

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

train_loader, val_loader, test_loader, classes = get_dataloaders(DATA_DIR, BATCH_SIZE)

model = build_model(num_classes=len(classes))
model = model.to(device)
print(f"Classes: {classes}")

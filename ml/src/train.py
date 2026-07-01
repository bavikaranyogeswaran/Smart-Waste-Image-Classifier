"""Training loop for the waste classifier.

Placeholder created in Step 1. Implemented in Step 7.
"""
import torch

DATA_DIR   = "ml/data/splits"
MODEL_PATH = "ml/models/waste_resnet18.pth"
BATCH_SIZE = 32
EPOCHS     = 10
LR         = 0.001

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

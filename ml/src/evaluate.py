"""Evaluate the trained model: precision, recall, F1, confusion matrix.

Placeholder created in Step 1. Implemented in Step 8.
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
from sklearn.metrics import classification_report, confusion_matrix

from dataset import get_dataloaders
from model import build_model

DATA_DIR   = "ml/data/splits"
MODEL_PATH = "ml/models/waste_resnet18.pth"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

_, _, test_loader, _ = get_dataloaders(DATA_DIR)

checkpoint = torch.load(MODEL_PATH, map_location=device)
classes = checkpoint["classes"]

model = build_model(num_classes=len(classes))
model.load_state_dict(checkpoint["model_state_dict"])
model = model.to(device)
model.eval()

all_preds  = []
all_labels = []

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.numpy())

print("Classification Report:")
print(classification_report(all_labels, all_preds, target_names=classes))

print("Confusion Matrix:")
print(confusion_matrix(all_labels, all_preds))

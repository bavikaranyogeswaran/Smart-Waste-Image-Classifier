"""Evaluate the trained model: precision, recall, F1, confusion matrix.

"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns  # type: ignore
from sklearn.metrics import classification_report, confusion_matrix

from dataset import get_dataloaders
from model import build_model

DATA_DIR    = "ml/data/splits"
MODEL_PATH  = "ml/models/waste_resnet18.pth"
REPORTS_DIR = "ml/reports"

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

report = classification_report(all_labels, all_preds, target_names=classes)
cm     = confusion_matrix(all_labels, all_preds)

print("Classification Report:")
print(report)
print("Confusion Matrix:")
print(cm)

os.makedirs(REPORTS_DIR, exist_ok=True)

with open(f"{REPORTS_DIR}/classification_report.txt", "w") as f:
    f.write(str(report))

fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=classes, yticklabels=classes, ax=ax)  # type: ignore
ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")
ax.set_title("Confusion Matrix")
plt.tight_layout()
plt.savefig(f"{REPORTS_DIR}/confusion_matrix.png", dpi=150)
plt.close()

print(f"\nReports saved to {REPORTS_DIR}/")

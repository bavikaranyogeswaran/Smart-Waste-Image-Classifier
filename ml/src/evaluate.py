"""
Evaluation script for the Smart Waste Image Classifier.

This script loads the trained PyTorch ResNet18 model and evaluates it
on the test dataset. It calculates precision, recall, F1-score, and generates
a confusion matrix, saving the results to a reports directory.
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

# 1. Define paths for data, the trained model, and the output reports
DATA_DIR    = "ml/data/splits"
MODEL_PATH  = "ml/models/waste_resnet18.pth"
REPORTS_DIR = "ml/reports"

# 2. Determine the device to run on
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 3. Load the test DataLoader (ignoring train/val loaders)
_, _, test_loader, _ = get_dataloaders(DATA_DIR)

# 4. Load the saved model checkpoint
checkpoint = torch.load(MODEL_PATH, map_location=device)
classes = checkpoint["classes"]

# 5. Initialize the model architecture and load the trained weights
model = build_model(num_classes=len(classes))
model.load_state_dict(checkpoint["model_state_dict"])
model = model.to(device)

# 6. Set the model to evaluation mode
model.eval()

all_preds  = []
all_labels = []

# 7. Disable gradient calculation for inference
with torch.no_grad():
    # 8. Iterate over the test dataset
    for images, labels in test_loader:
        images = images.to(device)
        
        # 9. Get predictions from the model
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        
        # 10. Store predictions and actual labels
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.numpy())

# 11. Generate classification metrics (precision, recall, f1-score)
report = classification_report(all_labels, all_preds, target_names=classes)

# 12. Generate the confusion matrix
cm     = confusion_matrix(all_labels, all_preds)

print("Classification Report:")
print(report)
print("Confusion Matrix:")
print(cm)

# 13. Create the reports directory if it doesn't exist
os.makedirs(REPORTS_DIR, exist_ok=True)

# 14. Save the classification report to a text file
with open(f"{REPORTS_DIR}/classification_report.txt", "w") as f:
    f.write(str(report))

# 15. Plot the confusion matrix using seaborn and matplotlib
fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=classes, yticklabels=classes, ax=ax)  # type: ignore
ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")
ax.set_title("Confusion Matrix")
plt.tight_layout()

# 16. Save the confusion matrix plot as a PNG image
plt.savefig(f"{REPORTS_DIR}/confusion_matrix.png", dpi=150)
plt.close()

print(f"\nReports saved to {REPORTS_DIR}/")

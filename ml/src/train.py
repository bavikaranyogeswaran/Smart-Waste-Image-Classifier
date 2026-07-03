"""
Training script for the Smart Waste Image Classifier.

This script implements the main training loop for fine-tuning the
pretrained ResNet18 model on the waste dataset. It handles data loading,
model initialization, optimization, evaluation on a validation set,
and saving the best performing model based on validation accuracy.
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn as nn
import torch.optim as optim

from dataset import get_dataloaders
from model import build_model

# 1. Define hyperparameters and paths
DATA_DIR   = "ml/data/splits"
MODEL_PATH = "ml/models/waste_resnet18.pth"
BATCH_SIZE = 32
EPOCHS     = 10
LR         = 0.001

# 2. Determine the device to run on (GPU if available, else CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 3. Load the data using DataLoaders
train_loader, val_loader, test_loader, classes = get_dataloaders(DATA_DIR, BATCH_SIZE)

# 4. Initialize the model configured for the correct number of classes
model = build_model(num_classes=len(classes))
model = model.to(device)
print(f"Classes: {classes}")

# 5. Define the loss function (CrossEntropy for classification) and optimizer (Adam)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.fc.parameters(), lr=LR)

best_val_acc = 0.0

# 6. Start the training loop over the specified number of epochs
for epoch in range(EPOCHS):
    # 7. Set model to training mode (enables gradients, dropout, etc.)
    model.train()
    train_loss    = 0.0
    train_correct = 0
    train_total   = 0

    # 8. Iterate over batches of training data
    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        # 9. Zero the parameter gradients to prevent accumulation
        optimizer.zero_grad()
        
        # 10. Forward pass to get predictions
        outputs = model(images)
        
        # 11. Calculate the loss
        loss = criterion(outputs, labels)
        
        # 12. Backward pass to compute gradients
        loss.backward()
        
        # 13. Update weights
        optimizer.step()

        train_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        train_total   += labels.size(0)
        train_correct += (predicted == labels).sum().item()

    # 14. Calculate average training accuracy for the epoch
    train_acc = train_correct / train_total

    # 15. Set model to evaluation mode (disables dropout, affects batch norm)
    model.eval()
    val_correct = 0
    val_total   = 0

    # 16. Disable gradient calculation for validation phase to save memory and compute
    with torch.no_grad():
        # 17. Iterate over batches of validation data
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            val_total   += labels.size(0)
            val_correct += (predicted == labels).sum().item()

    # 18. Calculate average validation accuracy for the epoch
    val_acc = val_correct / val_total

    # 19. Print epoch statistics
    print(
        f"Epoch [{epoch+1}/{EPOCHS}] "
        f"Loss: {train_loss/len(train_loader):.4f} "
        f"Train Acc: {train_acc:.4f} "
        f"Val Acc: {val_acc:.4f}"
    )

    # 20. Save the model if validation accuracy improved
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save({"model_state_dict": model.state_dict(), "classes": classes}, MODEL_PATH)
        print(f"  -> Saved best model (val_acc={val_acc:.4f})")

print(f"\nTraining complete. Best val acc: {best_val_acc:.4f}")

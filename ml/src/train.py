"""Training loop for the waste classifier.

Placeholder created in Step 1. Implemented in Step 7.
"""
import torch
import torch.nn as nn
import torch.optim as optim

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

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.fc.parameters(), lr=LR)

best_val_acc = 0.0

for epoch in range(EPOCHS):
    model.train()
    train_loss    = 0.0
    train_correct = 0
    train_total   = 0

    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        train_total   += labels.size(0)
        train_correct += (predicted == labels).sum().item()

    train_acc = train_correct / train_total

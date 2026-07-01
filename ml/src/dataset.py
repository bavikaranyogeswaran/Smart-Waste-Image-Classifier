"""PyTorch DataLoaders and image transforms (ImageFolder based).

Placeholder created in Step 1. Implemented in Step 5.
"""
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

train_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

eval_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


def get_dataloaders(data_dir, batch_size=32):
    train_dataset = datasets.ImageFolder(
        root=f"{data_dir}/train",
        transform=train_transforms
    )

    val_dataset = datasets.ImageFolder(
        root=f"{data_dir}/val",
        transform=eval_transforms
    )

    test_dataset = datasets.ImageFolder(
        root=f"{data_dir}/test",
        transform=eval_transforms
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False)
    test_loader  = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader, train_dataset.classes

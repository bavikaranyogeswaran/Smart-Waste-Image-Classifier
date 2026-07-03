"""
PyTorch DataLoaders and image transforms for the Waste Image Classifier.

This module defines the data augmentation transforms and provides a utility
function to create PyTorch DataLoaders for training, validation, and testing.
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
    """
    Creates DataLoaders for the train, validation, and test datasets.

    Args:
        data_dir (str): The root directory containing the 'train', 'val', and 'test' folders.
        batch_size (int, optional): The number of samples per batch. Defaults to 32.

    Returns:
        tuple: A tuple containing:
            - train_loader (DataLoader): DataLoader for the training set.
            - val_loader (DataLoader): DataLoader for the validation set.
            - test_loader (DataLoader): DataLoader for the test set.
            - classes (list of str): List of class names derived from the directory structure.
    """
    # 1. Initialize the training dataset with data augmentation transforms
    train_dataset = datasets.ImageFolder(
        root=f"{data_dir}/train",
        transform=train_transforms
    )

    # 2. Initialize the validation dataset with evaluation transforms
    val_dataset = datasets.ImageFolder(
        root=f"{data_dir}/val",
        transform=eval_transforms
    )

    # 3. Initialize the testing dataset with evaluation transforms
    test_dataset = datasets.ImageFolder(
        root=f"{data_dir}/test",
        transform=eval_transforms
    )

    # 4. Create DataLoaders to batch and optionally shuffle the datasets
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False)
    test_loader  = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False)

    # 5. Return the loaders along with the list of classes inferred from folder names
    return train_loader, val_loader, test_loader, train_dataset.classes

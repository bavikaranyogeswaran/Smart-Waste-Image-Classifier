"""
Dataset splitting script for the Smart Waste Image Classifier.

This script takes the processed image dataset and splits it into
training, validation, and test sets according to defined ratios.
It preserves the directory structure for each class in the output splits.
"""
import os
import shutil
import random
from pathlib import Path

# 1. Define paths for the source data and the output splits
SOURCE_DIR = Path("ml/data/processed")
OUTPUT_DIR = Path("ml/data/splits")

# 2. Define the ratios for splitting the dataset
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# 3. Set a random seed for reproducible splits
random.seed(42)

# 4. Iterate over each class directory in the source folder
for class_dir in sorted(SOURCE_DIR.iterdir()):
    if not class_dir.is_dir():
        continue

    # 5. Collect all valid image files for the current class
    images = [
        f for f in class_dir.glob("*")
        if f.is_file() and f.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    ]
    
    # 6. Randomly shuffle the list of images
    random.shuffle(images)

    # 7. Calculate the split indices based on the total number of images
    total = len(images)
    train_end = int(total * TRAIN_RATIO)
    val_end = train_end + int(total * VAL_RATIO)

    # 8. Create a dictionary to hold the split lists
    split_data = {
        "train": images[:train_end],
        "val":   images[train_end:val_end],
        "test":  images[val_end:]
    }

    # 9. Iterate over the splits and copy files to their respective destinations
    for split, files in split_data.items():
        # 10. Construct the target directory path and create it if necessary
        target_dir = OUTPUT_DIR / split / class_dir.name
        target_dir.mkdir(parents=True, exist_ok=True)

        # 11. Copy each file to the target directory
        for file in files:
            shutil.copy(file, target_dir / file.name)

    # 12. Print the split statistics for the current class
    print(f"{class_dir.name}: train={len(split_data['train'])} val={len(split_data['val'])} test={len(split_data['test'])}")

print("\nSplit complete.")

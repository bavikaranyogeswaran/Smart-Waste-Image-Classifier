import os
import shutil
import random
from pathlib import Path

SOURCE_DIR = Path("ml/data/processed")
OUTPUT_DIR = Path("ml/data/splits")

TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15

random.seed(42)

for class_dir in sorted(SOURCE_DIR.iterdir()):
    if not class_dir.is_dir():
        continue

    images = [
        f for f in class_dir.glob("*")
        if f.is_file() and f.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    ]
    random.shuffle(images)

    total = len(images)
    train_end = int(total * TRAIN_RATIO)
    val_end = train_end + int(total * VAL_RATIO)

    split_data = {
        "train": images[:train_end],
        "val":   images[train_end:val_end],
        "test":  images[val_end:]
    }

    for split, files in split_data.items():
        target_dir = OUTPUT_DIR / split / class_dir.name
        target_dir.mkdir(parents=True, exist_ok=True)

        for file in files:
            shutil.copy(file, target_dir / file.name)

    print(f"{class_dir.name}: train={len(split_data['train'])} val={len(split_data['val'])} test={len(split_data['test'])}")

print("\nSplit complete.")

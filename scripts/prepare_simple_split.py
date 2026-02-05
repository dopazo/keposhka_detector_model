"""
Prepara un split simple 80/20 sin k-fold cross validation.

Uso:
    uv run python scripts/prepare_simple_split.py
    uv run python scripts/prepare_simple_split.py --split 0.9  # 90/10 split
"""

import argparse
import random
import shutil
from pathlib import Path

import yaml


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", type=float, default=0.8, help="Train split ratio (default: 0.8)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    random.seed(args.seed)

    # Paths
    data_dir = Path("data")
    images_dir = data_dir / "images"
    labels_dir = data_dir / "labels"
    output_dir = Path("data_split")

    # Get all images (case-insensitive deduplication for Windows)
    image_files = []
    for ext in ["*.jpg", "*.jpeg", "*.png"]:
        image_files.extend(images_dir.glob(ext))
    image_files = list({f.name.lower(): f for f in image_files}.values())

    # Filter to only images with labels
    valid_images = []
    for img in image_files:
        label_path = labels_dir / f"{img.stem}.txt"
        if label_path.exists():
            valid_images.append(img)

    print(f"Found {len(valid_images)} images with labels")

    # Shuffle and split
    random.shuffle(valid_images)
    split_idx = int(len(valid_images) * args.split)
    train_images = valid_images[:split_idx]
    val_images = valid_images[split_idx:]

    print(f"Train: {len(train_images)} images")
    print(f"Val: {len(val_images)} images")

    # Create output directories
    train_img_dir = output_dir / "train" / "images"
    train_lbl_dir = output_dir / "train" / "labels"
    val_img_dir = output_dir / "val" / "images"
    val_lbl_dir = output_dir / "val" / "labels"

    for d in [train_img_dir, train_lbl_dir, val_img_dir, val_lbl_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # Copy files
    for img in train_images:
        shutil.copy2(img, train_img_dir / img.name)
        shutil.copy2(labels_dir / f"{img.stem}.txt", train_lbl_dir / f"{img.stem}.txt")

    for img in val_images:
        shutil.copy2(img, val_img_dir / img.name)
        shutil.copy2(labels_dir / f"{img.stem}.txt", val_lbl_dir / f"{img.stem}.txt")

    # Read class names
    classes_file = data_dir / "classes.txt"
    with open(classes_file) as f:
        class_names = [line.strip() for line in f if line.strip()]

    # Create data.yaml
    data_yaml = {
        "path": str(output_dir.absolute()),
        "train": "train/images",
        "val": "val/images",
        "nc": len(class_names),
        "names": class_names,
    }

    yaml_path = output_dir / "data.yaml"
    with open(yaml_path, "w") as f:
        yaml.dump(data_yaml, f, default_flow_style=False, allow_unicode=True)

    print(f"\nCreated {yaml_path}")
    print(f"To train: uv run python scripts/train.py --data {yaml_path}")


if __name__ == "__main__":
    main()

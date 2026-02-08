"""
Prepares directory structure for k-fold cross validation.

Usage:
    uv run python scripts/prepare_folds.py

Outputs:
    folds/fold_{0-2}/train/images/
    folds/fold_{0-2}/train/labels/
    folds/fold_{0-2}/val/images/
    folds/fold_{0-2}/val/labels/
    folds/fold_{0-2}/data.yaml
"""

import logging
import random
import shutil
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# Configuration
K = 3
SEED = 42
DATA_DIR = Path("data")
FOLDS_DIR = Path("folds")


def main():
    # Get list of images with annotations
    images_dir = DATA_DIR / "images"
    labels_dir = DATA_DIR / "labels"

    if not images_dir.exists():
        logger.error(f"Images directory does not exist: {images_dir}")
        return

    if not labels_dir.exists():
        logger.error(f"Labels directory does not exist: {labels_dir}")
        return

    image_files = []
    for ext in ["*.jpg", "*.jpeg", "*.png"]:
        image_files.extend(images_dir.glob(ext))
    # Remove duplicates (Windows is case-insensitive)
    image_files = list({f.name.lower(): f for f in image_files}.values())

    # Filter to only images with corresponding annotations
    valid_pairs = []
    for img_path in image_files:
        label_path = labels_dir / f"{img_path.stem}.txt"
        if label_path.exists():
            valid_pairs.append((img_path, label_path))

    if not valid_pairs:
        logger.error("No images with valid annotations found")
        return

    logger.info(f"Found {len(valid_pairs)} images with annotations")

    # Deterministic shuffle
    random.seed(SEED)
    random.shuffle(valid_pairs)

    # Split into k partitions
    fold_size = len(valid_pairs) // K
    partitions = []
    for i in range(K):
        start = i * fold_size
        end = start + fold_size if i < K - 1 else len(valid_pairs)
        partitions.append(valid_pairs[start:end])

    # Create fold structure
    for fold_idx in range(K):
        fold_dir = FOLDS_DIR / f"fold_{fold_idx}"

        # Clean up if exists
        if fold_dir.exists():
            shutil.rmtree(fold_dir)

        # Create directories
        for split in ["train", "val"]:
            (fold_dir / split / "images").mkdir(parents=True)
            (fold_dir / split / "labels").mkdir(parents=True)

        # Validation = current partition
        val_pairs = partitions[fold_idx]
        # Training = remaining partitions
        train_pairs = []
        for j in range(K):
            if j != fold_idx:
                train_pairs.extend(partitions[j])

        # Copy files
        for split, pairs in [("train", train_pairs), ("val", val_pairs)]:
            for img_path, label_path in pairs:
                img_dest = fold_dir / split / "images" / img_path.name
                label_dest = fold_dir / split / "labels" / label_path.name

                shutil.copy2(img_path, img_dest)
                shutil.copy2(label_path, label_dest)

        # Create data.yaml for this fold
        create_data_yaml(fold_dir, fold_idx)

        logger.info(f"Fold {fold_idx}: {len(train_pairs)} train, {len(val_pairs)} val")


def create_data_yaml(fold_dir: Path, fold_idx: int):
    """Generates data.yaml for a specific fold."""
    classes = load_classes()

    yaml_content = f"""# YOLO Dataset Configuration - Fold {fold_idx}
path: {fold_dir.resolve()}
train: train/images
val: val/images

nc: {len(classes)}
names: {classes}
"""

    (fold_dir / "data.yaml").write_text(yaml_content)


def load_classes() -> list:
    """Loads class list from classes.txt"""
    classes_file = DATA_DIR / "classes.txt"
    if not classes_file.exists():
        logger.error(f"Classes file not found: {classes_file}")
        return []
    return [line.strip() for line in classes_file.read_text().splitlines() if line.strip()]


if __name__ == "__main__":
    main()

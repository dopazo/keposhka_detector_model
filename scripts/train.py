"""
Trains YOLO v11 model with k-fold cross validation.

Usage:
    uv run python scripts/train.py                    # Train all folds
    uv run python scripts/train.py --fold 0          # Train only fold 0
    uv run python scripts/train.py --config custom.yaml
"""

# Disable MLflow BEFORE importing ultralytics (must be first)
import os
os.environ["MLFLOW_TRACKING_URI"] = ""

import argparse
import logging
from pathlib import Path

import yaml
from ultralytics import YOLO
from ultralytics.utils import callbacks

# Remove MLflow callbacks from Ultralytics
if "mlflow" in callbacks.default_callbacks:
    del callbacks.default_callbacks["mlflow"]

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

K = 3  # Number of folds


def load_config(config_path: str = "configs/train_config.yaml") -> dict:
    with open(config_path, encoding='utf-8') as f:
        return yaml.safe_load(f)


def train_fold(fold_idx: int, config: dict):
    """Trains a specific fold."""
    fold_dir = Path(f"folds/fold_{fold_idx}")
    data_yaml = fold_dir / "data.yaml"

    if not data_yaml.exists():
        logger.error(f"{data_yaml} does not exist. Run prepare_folds.py first.")
        return None

    # Load base model
    model = YOLO(config["model"])

    # Train
    results = model.train(
        data=str(data_yaml),
        epochs=config["epochs"],
        patience=config["patience"],
        batch=config["batch"],
        imgsz=config["imgsz"],
        cos_lr=config.get("cos_lr", False),
        freeze=config.get("freeze", None),
        optimizer=config["optimizer"],
        lr0=config["lr0"],
        lrf=config["lrf"],
        momentum=config["momentum"],
        weight_decay=config["weight_decay"],
        hsv_h=config["hsv_h"],
        hsv_s=config["hsv_s"],
        hsv_v=config["hsv_v"],
        degrees=config["degrees"],
        translate=config["translate"],
        scale=config["scale"],
        shear=config["shear"],
        perspective=config.get("perspective", 0.0),
        flipud=config["flipud"],
        fliplr=config["fliplr"],
        mosaic=config["mosaic"],
        close_mosaic=config.get("close_mosaic", 10),
        copy_paste=config.get("copy_paste", 0.0),
        mixup=config["mixup"],
        erasing=config.get("erasing", 0.0),
        workers=config["workers"],
        device=config["device"],
        project=config["project"],
        name=f"fold_{fold_idx}",
        exist_ok=config["exist_ok"],
        verbose=config["verbose"],
    )

    if results:
        logger.info(f"Fold {fold_idx} completed")
        logger.info(f"  mAP50: {results.results_dict.get('metrics/mAP50(B)', 0):.4f}")
        logger.info(f"  mAP50-95: {results.results_dict.get('metrics/mAP50-95(B)', 0):.4f}")

    return results


def train_single(data_yaml: str, config: dict, name: str = "full"):
    """Trains with a specific data.yaml (without folds)."""
    data_path = Path(data_yaml)

    if not data_path.exists():
        logger.error(f"{data_yaml} does not exist")
        return None

    model = YOLO(config["model"])

    results = model.train(
        data=str(data_path),
        epochs=config["epochs"],
        patience=config["patience"],
        batch=config["batch"],
        imgsz=config["imgsz"],
        cos_lr=config.get("cos_lr", False),
        freeze=config.get("freeze", None),
        optimizer=config["optimizer"],
        lr0=config["lr0"],
        lrf=config["lrf"],
        momentum=config["momentum"],
        weight_decay=config["weight_decay"],
        hsv_h=config["hsv_h"],
        hsv_s=config["hsv_s"],
        hsv_v=config["hsv_v"],
        degrees=config["degrees"],
        translate=config["translate"],
        scale=config["scale"],
        shear=config["shear"],
        perspective=config.get("perspective", 0.0),
        flipud=config["flipud"],
        fliplr=config["fliplr"],
        mosaic=config["mosaic"],
        close_mosaic=config.get("close_mosaic", 10),
        copy_paste=config.get("copy_paste", 0.0),
        mixup=config["mixup"],
        erasing=config.get("erasing", 0.0),
        workers=config["workers"],
        device=config["device"],
        project=config["project"],
        name=name,
        exist_ok=config["exist_ok"],
        verbose=config["verbose"],
    )

    if results:
        logger.info(f"Training completed")
        logger.info(f"  mAP50: {results.results_dict.get('metrics/mAP50(B)', 0):.4f}")
        logger.info(f"  mAP50-95: {results.results_dict.get('metrics/mAP50-95(B)', 0):.4f}")

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", type=int, default=None, help="Specific fold (0-2)")
    parser.add_argument("--data", type=str, default=None, help="Path to data.yaml (without folds)")
    parser.add_argument("--name", type=str, default="full", help="Experiment name")
    parser.add_argument("--config", type=str, default="configs/train_config.yaml")
    args = parser.parse_args()

    config = load_config(args.config)

    # No-folds mode: use --data directly
    if args.data is not None:
        logger.info(f"Training with {args.data}")
        train_single(args.data, config, args.name)
        return

    # Folds mode
    if args.fold is not None:
        if args.fold < 0 or args.fold >= K:
            logger.error(f"Fold must be between 0 and {K-1}")
            return
        logger.info(f"Training fold {args.fold}")
        train_fold(args.fold, config)
    else:
        logger.info(f"Training all folds (k={K})")
        for fold_idx in range(K):
            print(f"\n{'='*50}")
            print(f"FOLD {fold_idx}")
            print(f"{'='*50}\n")
            train_fold(fold_idx, config)


if __name__ == "__main__":
    main()

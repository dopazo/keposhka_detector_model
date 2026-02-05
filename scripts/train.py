"""
Entrena modelo YOLO v11 con k-fold cross validation.

Uso:
    uv run python scripts/train.py                    # Entrena todos los folds
    uv run python scripts/train.py --fold 0          # Entrena solo fold 0
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

K = 3  # Número de folds


def load_config(config_path: str = "configs/train_config.yaml") -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def train_fold(fold_idx: int, config: dict):
    """Entrena un fold específico."""
    fold_dir = Path(f"folds/fold_{fold_idx}")
    data_yaml = fold_dir / "data.yaml"

    if not data_yaml.exists():
        logger.error(f"No existe {data_yaml}. Ejecuta prepare_folds.py primero.")
        return None

    # Cargar modelo base
    model = YOLO(config["model"])

    # Entrenar
    results = model.train(
        data=str(data_yaml),
        epochs=config["epochs"],
        patience=config["patience"],
        batch=config["batch"],
        imgsz=config["imgsz"],
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
        mixup=config["mixup"],
        workers=config["workers"],
        device=config["device"],
        project=config["project"],
        name=f"fold_{fold_idx}",
        exist_ok=config["exist_ok"],
        verbose=config["verbose"],
    )

    if results:
        logger.info(f"Fold {fold_idx} completado")
        logger.info(f"  mAP50: {results.results_dict.get('metrics/mAP50(B)', 0):.4f}")
        logger.info(f"  mAP50-95: {results.results_dict.get('metrics/mAP50-95(B)', 0):.4f}")

    return results


def train_single(data_yaml: str, config: dict, name: str = "full"):
    """Entrena con un data.yaml específico (sin folds)."""
    data_path = Path(data_yaml)

    if not data_path.exists():
        logger.error(f"No existe {data_yaml}")
        return None

    model = YOLO(config["model"])

    results = model.train(
        data=str(data_path),
        epochs=config["epochs"],
        patience=config["patience"],
        batch=config["batch"],
        imgsz=config["imgsz"],
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
        mixup=config["mixup"],
        workers=config["workers"],
        device=config["device"],
        project=config["project"],
        name=name,
        exist_ok=config["exist_ok"],
        verbose=config["verbose"],
    )

    if results:
        logger.info(f"Entrenamiento completado")
        logger.info(f"  mAP50: {results.results_dict.get('metrics/mAP50(B)', 0):.4f}")
        logger.info(f"  mAP50-95: {results.results_dict.get('metrics/mAP50-95(B)', 0):.4f}")

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", type=int, default=None, help="Fold específico (0-2)")
    parser.add_argument("--data", type=str, default=None, help="Path a data.yaml (sin folds)")
    parser.add_argument("--name", type=str, default="full", help="Nombre del experimento")
    parser.add_argument("--config", type=str, default="configs/train_config.yaml")
    args = parser.parse_args()

    config = load_config(args.config)

    # Modo sin folds: usar --data directamente
    if args.data is not None:
        logger.info(f"Entrenando con {args.data}")
        train_single(args.data, config, args.name)
        return

    # Modo con folds
    if args.fold is not None:
        if args.fold < 0 or args.fold >= K:
            logger.error(f"Fold debe estar entre 0 y {K-1}")
            return
        logger.info(f"Entrenando fold {args.fold}")
        train_fold(args.fold, config)
    else:
        logger.info(f"Entrenando todos los folds (k={K})")
        for fold_idx in range(K):
            print(f"\n{'='*50}")
            print(f"FOLD {fold_idx}")
            print(f"{'='*50}\n")
            train_fold(fold_idx, config)


if __name__ == "__main__":
    main()

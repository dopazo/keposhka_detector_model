"""
Runs inference on new images.

Usage:
    uv run python scripts/predict.py image.jpg
    uv run python scripts/predict.py images_dir/
    uv run python scripts/predict.py image.jpg --model runs/fold_0/weights/best.pt
    uv run python scripts/predict.py image.jpg --conf 0.5 --save
"""

import argparse
import logging
from pathlib import Path

from ultralytics import YOLO

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

K = 3  # Number of folds


def find_best_model() -> Path:
    """Finds the best available model (fold 0 by default)."""
    default = Path("runs/fold_0/weights/best.pt")
    if default.exists():
        return default

    # Search for any available model
    for fold in range(K):
        path = Path(f"runs/fold_{fold}/weights/best.pt")
        if path.exists():
            return path

    raise FileNotFoundError("No trained model found in runs/")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=str, help="Image or directory")
    parser.add_argument("--model", type=str, default=None, help="Path to .pt model")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.45, help="IoU threshold for NMS")
    parser.add_argument("--save", action="store_true", help="Save images with detections")
    parser.add_argument("--save-txt", action="store_true", help="Save .txt annotations")
    args = parser.parse_args()

    # Load model
    try:
        model_path = args.model if args.model else find_best_model()
    except FileNotFoundError as e:
        logger.error(str(e))
        return

    logger.info(f"Using model: {model_path}")
    model = YOLO(model_path)

    # Inferencia
    results = model.predict(
        source=args.source,
        conf=args.conf,
        iou=args.iou,
        save=args.save,
        save_txt=args.save_txt,
        verbose=True,
    )

    # Show results
    for result in results:
        print(f"\n{result.path}:")
        if result.boxes:
            for box in result.boxes:
                cls_id = int(box.cls)
                conf = float(box.conf)
                label = model.names[cls_id]
                print(f"  {label}: {conf:.2f}")
        else:
            print("  No characters detected")


if __name__ == "__main__":
    main()

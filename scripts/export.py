"""
Exports trained model to TFLite for Android deployment.

Usage:
    uv run python scripts/export.py                           # Export fold 0
    uv run python scripts/export.py --fold 1                  # Export specific fold
    uv run python scripts/export.py --model path/to/best.pt   # Export specific model (e.g. runs/detect/runs/final/weights/last.pt)
"""

import argparse
import logging
from pathlib import Path

from ultralytics import YOLO

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

K = 3  # Number of folds


def find_best_model(fold_idx: int = 0) -> Path:
    """Finds the best model for a fold."""
    path = Path(f"runs/fold_{fold_idx}/weights/best.pt")
    if path.exists():
        return path
    raise FileNotFoundError(f"Model not found at {path}")


def export_model(model_path: Path, imgsz: int = 1280):
    """Exports model to TFLite FP16."""
    logger.info(f"Loading model: {model_path}")
    model = YOLO(model_path)

    logger.info("Exporting to TFLite FP16...")
    export_path = model.export(
        format="tflite",
        half=True,  # FP16
        imgsz=imgsz,  # must match what the model was trained at (1280); the app has to letterbox to the same size
    )

    logger.info(f"Model exported: {export_path}")
    return export_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", type=int, default=0, help="Fold to export (0-2)")
    parser.add_argument("--model", type=str, default=None, help="Path to specific .pt model")
    parser.add_argument("--imgsz", type=int, default=1280, help="Export input size (default: 1280, the training size)")
    args = parser.parse_args()

    try:
        if args.model:
            model_path = Path(args.model)
            if not model_path.exists():
                logger.error(f"Model does not exist: {model_path}")
                return
        else:
            model_path = find_best_model(args.fold)

        export_model(model_path, args.imgsz)

    except FileNotFoundError as e:
        logger.error(str(e))
    except Exception as e:
        logger.error(f"Error during export: {e}")


if __name__ == "__main__":
    main()

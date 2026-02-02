"""
Exporta modelo entrenado a TFLite para deployment en Android.

Uso:
    uv run python scripts/export.py                           # Exporta fold 0
    uv run python scripts/export.py --fold 1                  # Exporta fold específico
    uv run python scripts/export.py --model path/to/best.pt   # Exporta modelo específico
"""

import argparse
import logging
from pathlib import Path

from ultralytics import YOLO

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

K = 3  # Número de folds


def find_best_model(fold_idx: int = 0) -> Path:
    """Busca el mejor modelo de un fold."""
    path = Path(f"runs/fold_{fold_idx}/weights/best.pt")
    if path.exists():
        return path
    raise FileNotFoundError(f"No se encontró modelo en {path}")


def export_model(model_path: Path, output_dir: Path = None):
    """Exporta modelo a TFLite FP16."""
    logger.info(f"Cargando modelo: {model_path}")
    model = YOLO(model_path)

    logger.info("Exportando a TFLite FP16...")
    export_path = model.export(
        format="tflite",
        half=True,  # FP16
        imgsz=640,
    )

    logger.info(f"Modelo exportado: {export_path}")
    return export_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", type=int, default=0, help="Fold a exportar (0-2)")
    parser.add_argument("--model", type=str, default=None, help="Ruta al modelo .pt específico")
    args = parser.parse_args()

    try:
        if args.model:
            model_path = Path(args.model)
            if not model_path.exists():
                logger.error(f"No existe el modelo: {model_path}")
                return
        else:
            model_path = find_best_model(args.fold)

        export_model(model_path)

    except FileNotFoundError as e:
        logger.error(str(e))
    except Exception as e:
        logger.error(f"Error durante exportación: {e}")


if __name__ == "__main__":
    main()

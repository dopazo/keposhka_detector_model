"""
Realiza inferencia en nuevas imágenes.

Uso:
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

K = 3  # Número de folds


def find_best_model() -> Path:
    """Busca el mejor modelo disponible (por mAP del fold 0 por defecto)."""
    default = Path("runs/fold_0/weights/best.pt")
    if default.exists():
        return default

    # Buscar cualquier modelo disponible
    for fold in range(K):
        path = Path(f"runs/fold_{fold}/weights/best.pt")
        if path.exists():
            return path

    raise FileNotFoundError("No se encontró ningún modelo entrenado en runs/")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=str, help="Imagen o directorio")
    parser.add_argument("--model", type=str, default=None, help="Ruta al modelo .pt")
    parser.add_argument("--conf", type=float, default=0.25, help="Umbral de confianza")
    parser.add_argument("--iou", type=float, default=0.45, help="Umbral IoU para NMS")
    parser.add_argument("--save", action="store_true", help="Guardar imágenes con detecciones")
    parser.add_argument("--save-txt", action="store_true", help="Guardar anotaciones .txt")
    args = parser.parse_args()

    # Cargar modelo
    try:
        model_path = args.model if args.model else find_best_model()
    except FileNotFoundError as e:
        logger.error(str(e))
        return

    logger.info(f"Usando modelo: {model_path}")
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

    # Mostrar resultados
    for result in results:
        print(f"\n{result.path}:")
        if result.boxes:
            for box in result.boxes:
                cls_id = int(box.cls)
                conf = float(box.conf)
                label = model.names[cls_id]
                print(f"  {label}: {conf:.2f}")
        else:
            print("  No se detectaron caracteres")


if __name__ == "__main__":
    main()

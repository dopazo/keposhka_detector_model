"""
Evalúa modelos entrenados y genera reporte de métricas.

Uso:
    uv run python scripts/evaluate.py                 # Evalúa todos los folds
    uv run python scripts/evaluate.py --fold 0       # Evalúa solo fold 0
"""

import argparse
import json
import logging
from pathlib import Path

import numpy as np
from ultralytics import YOLO

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

K = 3  # Número de folds


def evaluate_fold(fold_idx: int) -> dict:
    """Evalúa el modelo de un fold en su conjunto de validación."""
    # Buscar el mejor modelo
    weights_path = Path(f"runs/fold_{fold_idx}/weights/best.pt")
    if not weights_path.exists():
        logger.warning(f"No se encontró modelo para fold {fold_idx}")
        return None

    data_yaml = Path(f"folds/fold_{fold_idx}/data.yaml")

    model = YOLO(weights_path)
    metrics = model.val(data=str(data_yaml), verbose=False)

    return {
        "fold": fold_idx,
        "mAP50": float(metrics.box.map50),
        "mAP50-95": float(metrics.box.map),
        "precision": float(metrics.box.mp),
        "recall": float(metrics.box.mr),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", type=int, default=None)
    parser.add_argument("--output", type=str, default="evaluation_results.json")
    args = parser.parse_args()

    results = []

    if args.fold is not None:
        result = evaluate_fold(args.fold)
        if result:
            results.append(result)
    else:
        for fold_idx in range(K):
            result = evaluate_fold(fold_idx)
            if result:
                results.append(result)

    if not results:
        logger.error("No hay resultados para evaluar")
        return

    # Calcular promedios
    avg_metrics = {
        "mAP50": np.mean([r["mAP50"] for r in results]),
        "mAP50-95": np.mean([r["mAP50-95"] for r in results]),
        "precision": np.mean([r["precision"] for r in results]),
        "recall": np.mean([r["recall"] for r in results]),
        "std_mAP50": np.std([r["mAP50"] for r in results]),
        "std_mAP50-95": np.std([r["mAP50-95"] for r in results]),
    }

    # Imprimir reporte
    print("\n" + "="*60)
    print("RESULTADOS K-FOLD CROSS VALIDATION")
    print("="*60)

    for r in results:
        print(f"\nFold {r['fold']}:")
        print(f"  mAP50:    {r['mAP50']:.4f}")
        print(f"  mAP50-95: {r['mAP50-95']:.4f}")
        print(f"  Precision: {r['precision']:.4f}")
        print(f"  Recall:    {r['recall']:.4f}")

    print(f"\n{'='*60}")
    print("PROMEDIO (+/- std):")
    print(f"  mAP50:    {avg_metrics['mAP50']:.4f} +/- {avg_metrics['std_mAP50']:.4f}")
    print(f"  mAP50-95: {avg_metrics['mAP50-95']:.4f} +/- {avg_metrics['std_mAP50-95']:.4f}")
    print(f"  Precision: {avg_metrics['precision']:.4f}")
    print(f"  Recall:    {avg_metrics['recall']:.4f}")

    # Guardar resultados
    output = {
        "folds": results,
        "average": avg_metrics,
    }

    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)

    logger.info(f"Resultados guardados en {args.output}")


if __name__ == "__main__":
    main()

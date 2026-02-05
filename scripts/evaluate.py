"""
Evalua modelos entrenados y genera reporte de metricas.

Uso:
    uv run python scripts/evaluate.py                                    # Evalua todos los folds
    uv run python scripts/evaluate.py --fold 0                          # Evalua solo fold 0
    uv run python scripts/evaluate.py --model path/best.pt --data d.yaml # Evalua modelo especifico
"""

import argparse
import json
import logging
from pathlib import Path

import numpy as np
from ultralytics import YOLO

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

K = 3  # Numero de folds


def evaluate_model(model_path: str, data_yaml: str, name: str = "model") -> dict:
    """Evalua un modelo en un dataset dado."""
    model_path = Path(model_path)
    if not model_path.exists():
        logger.warning(f"No se encontro modelo: {model_path}")
        return None

    logger.info(f"Evaluando {name}: {model_path}")
    model = YOLO(model_path)
    metrics = model.val(data=str(data_yaml), verbose=True)

    # Per-class metrics
    per_class = {}
    for i, cls_name in metrics.names.items():
        ap50 = float(metrics.box.ap50[i]) if i < len(metrics.box.ap50) else 0.0
        per_class[cls_name] = {"ap50": ap50}

    return {
        "name": name,
        "model": str(model_path),
        "mAP50": float(metrics.box.map50),
        "mAP50-95": float(metrics.box.map),
        "precision": float(metrics.box.mp),
        "recall": float(metrics.box.mr),
        "per_class": per_class,
    }


def evaluate_fold(fold_idx: int) -> dict:
    """Evalua el modelo de un fold en su conjunto de validacion."""
    weights_path = Path(f"runs/fold_{fold_idx}/weights/best.pt")
    data_yaml = Path(f"folds/fold_{fold_idx}/data.yaml")
    result = evaluate_model(weights_path, data_yaml, name=f"fold_{fold_idx}")
    if result:
        result["fold"] = fold_idx
    return result


def print_results(results: list, mode: str = "single"):
    """Imprime reporte de resultados."""
    print("\n" + "=" * 60)
    if mode == "kfold":
        print("RESULTADOS K-FOLD CROSS VALIDATION")
    else:
        print("RESULTADOS DE EVALUACION")
    print("=" * 60)

    for r in results:
        label = r.get("name", f"Fold {r.get('fold', '?')}")
        print(f"\n{label}:")
        print(f"  mAP50:     {r['mAP50']:.4f}")
        print(f"  mAP50-95:  {r['mAP50-95']:.4f}")
        print(f"  Precision: {r['precision']:.4f}")
        print(f"  Recall:    {r['recall']:.4f}")

        # Per-class breakdown
        if "per_class" in r:
            # Sort by AP50 descending
            sorted_classes = sorted(
                r["per_class"].items(), key=lambda x: x[1]["ap50"], reverse=True
            )
            detected = [(name, m) for name, m in sorted_classes if m["ap50"] > 0]
            undetected = [(name, m) for name, m in sorted_classes if m["ap50"] == 0]

            print(f"\n  Clases detectadas ({len(detected)}/{len(sorted_classes)}):")
            for name, m in detected:
                bar = "#" * int(m["ap50"] * 20)
                print(f"    {name:4s}: AP50={m['ap50']:.3f} {bar}")

            if undetected:
                names = [name for name, _ in undetected]
                print(f"\n  Clases NO detectadas ({len(undetected)}):")
                print(f"    {', '.join(names)}")

    if len(results) > 1:
        avg_metrics = {
            "mAP50": np.mean([r["mAP50"] for r in results]),
            "mAP50-95": np.mean([r["mAP50-95"] for r in results]),
            "precision": np.mean([r["precision"] for r in results]),
            "recall": np.mean([r["recall"] for r in results]),
            "std_mAP50": np.std([r["mAP50"] for r in results]),
            "std_mAP50-95": np.std([r["mAP50-95"] for r in results]),
        }
        print(f"\n{'=' * 60}")
        print("PROMEDIO (+/- std):")
        print(f"  mAP50:     {avg_metrics['mAP50']:.4f} +/- {avg_metrics['std_mAP50']:.4f}")
        print(f"  mAP50-95:  {avg_metrics['mAP50-95']:.4f} +/- {avg_metrics['std_mAP50-95']:.4f}")
        print(f"  Precision: {avg_metrics['precision']:.4f}")
        print(f"  Recall:    {avg_metrics['recall']:.4f}")


def main():
    parser = argparse.ArgumentParser(description="Evaluar modelos YOLO entrenados")
    parser.add_argument("--fold", type=int, default=None, help="Fold especifico a evaluar")
    parser.add_argument("--model", type=str, default=None, help="Ruta al modelo .pt")
    parser.add_argument("--data", type=str, default=None, help="Ruta al data.yaml")
    parser.add_argument("--output", type=str, default="evaluation_results.json", help="Archivo de salida JSON")
    args = parser.parse_args()

    results = []

    if args.model and args.data:
        # Evaluate a specific model
        result = evaluate_model(args.model, args.data, name=Path(args.model).parent.parent.name)
        if result:
            results.append(result)
        print_results(results, mode="single")
    elif args.fold is not None:
        result = evaluate_fold(args.fold)
        if result:
            results.append(result)
        print_results(results, mode="kfold")
    else:
        for fold_idx in range(K):
            result = evaluate_fold(fold_idx)
            if result:
                results.append(result)
        print_results(results, mode="kfold")

    if not results:
        logger.error("No hay resultados para evaluar")
        return

    # Save results
    output = {"results": results}
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    logger.info(f"Resultados guardados en {args.output}")


if __name__ == "__main__":
    main()

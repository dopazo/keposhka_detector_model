"""
Búsqueda de hiperparámetros usando el tuner integrado de Ultralytics.

Uso:
    uv run python scripts/tune.py                     # Tuning con config por defecto
    uv run python scripts/tune.py --iterations 50    # Más iteraciones
    uv run python scripts/tune.py --epochs 30        # Menos epochs por iteración
    uv run python scripts/tune.py --fold 1           # Usar fold específico

El tuner usa un algoritmo genético para encontrar los mejores hiperparámetros.
Los resultados se guardan en runs/tune/ con el mejor best_hyperparameters.yaml
"""

import argparse
import logging
from pathlib import Path

import yaml
from ultralytics import YOLO

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def load_base_config(config_path: str = "configs/train_config.yaml") -> dict:
    """Carga configuración base para valores fijos durante tuning."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description="Hyperparameter tuning para YOLO")
    parser.add_argument("--fold", type=int, default=0, help="Fold a usar para tuning (default: 0)")
    parser.add_argument("--iterations", type=int, default=30, help="Número de iteraciones/experimentos (default: 30)")
    parser.add_argument("--epochs", type=int, default=30, help="Epochs por iteración (default: 30, menor que entrenamiento final)")
    parser.add_argument("--model", type=str, default="yolo11n.pt", help="Modelo base (default: yolo11n.pt)")
    parser.add_argument("--config", type=str, default="configs/train_config.yaml", help="Config base")
    args = parser.parse_args()

    # Verificar que existe el fold
    data_yaml = Path(f"folds/fold_{args.fold}/data.yaml")
    if not data_yaml.exists():
        logger.error(f"No existe {data_yaml}. Ejecuta prepare_folds.py primero.")
        return

    # Cargar config base para valores fijos
    base_config = load_base_config(args.config)

    logger.info(f"Iniciando hyperparameter tuning")
    logger.info(f"  Modelo: {args.model}")
    logger.info(f"  Fold: {args.fold}")
    logger.info(f"  Iteraciones: {args.iterations}")
    logger.info(f"  Epochs por iteración: {args.epochs}")

    # Cargar modelo
    model = YOLO(args.model)

    # Ejecutar tuning
    # El tuner optimiza: lr0, lrf, momentum, weight_decay, warmup_epochs,
    # box, cls, hsv_h, hsv_s, hsv_v, degrees, translate, scale, shear,
    # perspective, flipud, fliplr, mosaic, mixup, copy_paste
    result = model.tune(
        data=str(data_yaml),
        epochs=args.epochs,
        iterations=args.iterations,
        optimizer=base_config.get("optimizer", "AdamW"),
        imgsz=base_config.get("imgsz", 640),
        batch=base_config.get("batch", 16),
        workers=base_config.get("workers", 8),
        device=base_config.get("device", 0),
        project="runs",
        name="tune",
        exist_ok=True,
        plots=True,
        save=True,
        val=True,
    )

    logger.info("Tuning completado!")
    logger.info("Revisa runs/tune/ para:")
    logger.info("  - best_hyperparameters.yaml: mejores hiperparámetros encontrados")
    logger.info("  - tune_results.csv: resultados de todas las iteraciones")
    logger.info("  - tune_fitness.png: gráfico de evolución del fitness")

    # Mostrar mejores hiperparámetros
    best_hp_path = Path("runs/tune/best_hyperparameters.yaml")
    if best_hp_path.exists():
        print("\n" + "=" * 60)
        print("MEJORES HIPERPARÁMETROS ENCONTRADOS")
        print("=" * 60)
        with open(best_hp_path) as f:
            best_hp = yaml.safe_load(f)
        for key, value in best_hp.items():
            print(f"  {key}: {value}")

        print("\nPara usar estos hiperparámetros, actualiza configs/train_config.yaml")
        print("o crea un nuevo archivo de configuración con estos valores.")


if __name__ == "__main__":
    main()

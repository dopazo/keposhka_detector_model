"""
Hyperparameter search using the built-in Ultralytics tuner.

Usage:
    uv run python scripts/tune.py                     # Tuning with default config
    uv run python scripts/tune.py --iterations 50    # More iterations
    uv run python scripts/tune.py --epochs 30        # Fewer epochs per iteration
    uv run python scripts/tune.py --fold 1           # Use specific fold

The tuner uses a genetic algorithm to find the best hyperparameters.
Results are saved in runs/tune/ with the best best_hyperparameters.yaml
"""

import argparse
import logging
from pathlib import Path

import yaml
from ultralytics import YOLO

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def load_base_config(config_path: str = "configs/train_config.yaml") -> dict:
    """Loads base configuration for fixed values during tuning."""
    with open(config_path, encoding='utf-8') as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description="Hyperparameter tuning for YOLO")
    parser.add_argument("--fold", type=int, default=0, help="Fold to use for tuning (default: 0)")
    parser.add_argument("--iterations", type=int, default=30, help="Number of iterations/experiments (default: 30)")
    parser.add_argument("--epochs", type=int, default=30, help="Epochs per iteration (default: 30, less than final training)")
    parser.add_argument("--model", type=str, default="yolo11n.pt", help="Base model (default: yolo11n.pt)")
    parser.add_argument("--config", type=str, default="configs/train_config.yaml", help="Base config")
    args = parser.parse_args()

    # Verify fold exists
    data_yaml = Path(f"folds/fold_{args.fold}/data.yaml")
    if not data_yaml.exists():
        logger.error(f"{data_yaml} does not exist. Run prepare_folds.py first.")
        return

    # Load base config for fixed values
    base_config = load_base_config(args.config)

    logger.info(f"Starting hyperparameter tuning")
    logger.info(f"  Model: {args.model}")
    logger.info(f"  Fold: {args.fold}")
    logger.info(f"  Iterations: {args.iterations}")
    logger.info(f"  Epochs per iteration: {args.epochs}")

    # Load model
    model = YOLO(args.model)

    # Run tuning
    # The tuner optimizes: lr0, lrf, momentum, weight_decay, warmup_epochs,
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

    logger.info("Tuning completed!")
    logger.info("Check runs/tune/ for:")
    logger.info("  - best_hyperparameters.yaml: best hyperparameters found")
    logger.info("  - tune_results.csv: results from all iterations")
    logger.info("  - tune_fitness.png: fitness evolution plot")

    # Show best hyperparameters
    best_hp_path = Path("runs/tune/best_hyperparameters.yaml")
    if best_hp_path.exists():
        print("\n" + "=" * 60)
        print("BEST HYPERPARAMETERS FOUND")
        print("=" * 60)
        with open(best_hp_path) as f:
            best_hp = yaml.safe_load(f)
        for key, value in best_hp.items():
            print(f"  {key}: {value}")

        print("\nTo use these hyperparameters, update configs/train_config.yaml")
        print("or create a new configuration file with these values.")


if __name__ == "__main__":
    main()

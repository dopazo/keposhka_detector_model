# Keposhka Detector Model

Especificación técnica para entrenar un modelo YOLO v11 que detecte y clasifique caracteres del lenguaje ficticio Keposhka en imágenes.

## Objetivo

Crear un modelo de detección de objetos capaz de:
1. Localizar caracteres Keposhka en una imagen (bounding boxes)
2. Clasificar cada carácter detectado entre 70 clases posibles
3. Funcionar con imágenes de diferentes resoluciones, estilos, y orientaciones (incluyendo texto en arcos y con perspectiva)
4. Ser exportable a TFLite para deployment en Android

## Stack Tecnológico

- **Framework ML**: Ultralytics YOLO v11
- **Backend**: PyTorch
- **Python**: 3.13+
- **Gestión de dependencias**: `uv`
- **Tracking de experimentos**: MLflow (local)
- **Export**: TFLite FP16 para Android

## Clases (70 total)

IDs 0-69, mapeados desde `classes.txt`:

| ID | Label | ID | Label | ID | Label | ID | Label |
|----|-------|----|---------|----|-------|---------|-------|
|  0 | 0     | 18 | D       | 36 | L     | 54 | T     |
|  1 | 1     | 19 | d       | 37 | l     | 55 | t     |
|  2 | 2     | 20 | E       | 38 | M     | 56 | TH    |
|  3 | 3     | 21 | e       | 39 | m     | 57 | th    |
|  4 | 4     | 22 | F       | 40 | N     | 58 | U     |
|  5 | 5     | 23 | f       | 41 | n     | 59 | u     |
|  6 | 6     | 24 | G       | 42 | O     | 60 | V     |
|  7 | 7     | 25 | g       | 43 | o     | 61 | v     |
|  8 | 8     | 26 | GH      | 44 | P     | 62 | W     |
|  9 | 9     | 27 | gh      | 45 | p     | 63 | w     |
| 10 | A     | 28 | H       | 46 | Q     | 64 | X     |
| 11 | a     | 29 | h       | 47 | q     | 65 | x     |
| 12 | B     | 30 | I       | 48 | R     | 66 | Y     |
| 13 | b     | 31 | i       | 49 | r     | 67 | y     |
| 14 | C     | 32 | J       | 50 | S     | 68 | Z     |
| 15 | c     | 33 | j       | 51 | s     | 69 | z     |
| 16 | CH    | 34 | K       | 52 | SH    |    |       |
| 17 | ch    | 35 | k       | 53 | sh    |    |       |

## Estructura del Proyecto

```
keposhka_detector/
├── data/
│   ├── images/              # Todas las imágenes (.jpg, .jpeg, .png)
│   ├── labels/              # Anotaciones YOLO (.txt, mismo nombre que imagen)
│   └── classes.txt          # Lista de clases (una por línea)
│
├── folds/                   # Generado por prepare_folds.py
│   ├── fold_0/
│   │   ├── train/
│   │   │   ├── images/      # Copias de archivos
│   │   │   └── labels/
│   │   └── val/
│   │       ├── images/
│   │       └── labels/
│   ├── fold_1/
│   └── fold_2/
│
├── configs/
│   ├── data_template.yaml   # Template para data.yaml
│   └── train_config.yaml    # Hiperparámetros de entrenamiento
│
├── scripts/
│   ├── analyze_dataset.py   # Analiza distribución de clases
│   ├── prepare_folds.py     # Crea estructura k-fold
│   ├── tune.py              # Búsqueda de hiperparámetros
│   ├── train.py             # Entrena un fold o todos
│   ├── evaluate.py          # Evalúa modelo(s) entrenado(s)
│   ├── predict.py           # Inferencia en nuevas imágenes
│   └── export.py            # Exporta modelo a TFLite
│
├── runs/                    # Generado por Ultralytics (pesos, logs, métricas)
├── mlruns/                  # Generado por MLflow (tracking de experimentos)
│
├── pyproject.toml
└── README.md
```

## Formato de Datos

### Imágenes
- Formatos soportados: `.jpg`, `.jpeg`, `.png`
- Resolución variable (YOLO redimensiona automáticamente)
- Tipos: imágenes digitales y fotos de cámara
- Fondos: variados (texturas, colores, imágenes de fondo)

### Anotaciones YOLO (labels/*.txt)
Cada línea representa un bounding box:
```
<class_id> <x_center> <y_center> <width> <height>
```
- Todos los valores normalizados entre 0 y 1
- `x_center`, `y_center`: centro del box relativo al tamaño de imagen
- `width`, `height`: dimensiones del box relativas al tamaño de imagen

### classes.txt
```
0
1
2
...
z
```
Una clase por línea, en orden de class_id (0-69).

## K-Fold Cross Validation

### Configuración
- **k = 3** folds (optimizado para dataset pequeño <50 imágenes)
- División estratificada por imagen (no por anotación)
- Seed fijo para reproducibilidad

### Proceso
1. Listar todas las imágenes con anotaciones válidas
2. Shuffle con seed fijo
3. Dividir en 3 particiones iguales
4. Para cada fold i:
   - Validación: partición i
   - Entrenamiento: particiones restantes (j != i)

### Script: analyze_dataset.py

```python
"""
Analiza la distribución de clases en el dataset.

Uso:
    uv run python scripts/analyze_dataset.py

Genera:
    - Reporte en consola con conteos por clase
    - Archivo JSON con estadísticas detalladas
"""

import json
import logging
from collections import Counter
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

DATA_DIR = Path("data")
OUTPUT_FILE = "dataset_analysis.json"


def load_classes() -> list:
    """Carga lista de clases desde classes.txt"""
    classes_file = DATA_DIR / "classes.txt"
    if not classes_file.exists():
        logger.error(f"No se encontró {classes_file}")
        return []
    return [line.strip() for line in classes_file.read_text().splitlines() if line.strip()]


def analyze_labels(labels_dir: Path, classes: list) -> dict:
    """Analiza todos los archivos de anotaciones."""
    class_counts = Counter()
    boxes_per_image = []
    total_boxes = 0

    label_files = list(labels_dir.glob("*.txt"))

    for label_file in label_files:
        lines = label_file.read_text().strip().splitlines()
        boxes_per_image.append(len(lines))

        for line in lines:
            parts = line.strip().split()
            if len(parts) >= 5:
                class_id = int(parts[0])
                class_counts[class_id] += 1
                total_boxes += 1

    return {
        "total_images": len(label_files),
        "total_boxes": total_boxes,
        "boxes_per_image": {
            "min": min(boxes_per_image) if boxes_per_image else 0,
            "max": max(boxes_per_image) if boxes_per_image else 0,
            "avg": sum(boxes_per_image) / len(boxes_per_image) if boxes_per_image else 0,
        },
        "class_counts": dict(class_counts),
    }


def main():
    classes = load_classes()
    if not classes:
        return

    labels_dir = DATA_DIR / "labels"
    if not labels_dir.exists():
        logger.error(f"No se encontró directorio {labels_dir}")
        return

    stats = analyze_labels(labels_dir, classes)

    # Reporte en consola
    print("\n" + "=" * 60)
    print("ANÁLISIS DEL DATASET")
    print("=" * 60)
    print(f"\nImágenes anotadas: {stats['total_images']}")
    print(f"Total de boxes: {stats['total_boxes']}")
    print(f"Boxes por imagen: min={stats['boxes_per_image']['min']}, "
          f"max={stats['boxes_per_image']['max']}, "
          f"avg={stats['boxes_per_image']['avg']:.1f}")

    print("\n" + "-" * 60)
    print("DISTRIBUCIÓN DE CLASES")
    print("-" * 60)

    # Ordenar por conteo
    sorted_counts = sorted(
        [(classes[cid], count) for cid, count in stats["class_counts"].items()],
        key=lambda x: x[1],
        reverse=True
    )

    # Clases con ejemplos
    print("\nClases con ejemplos (ordenadas por frecuencia):")
    for label, count in sorted_counts:
        bar = "█" * min(count, 50)
        print(f"  {label:4s}: {count:4d} {bar}")

    # Clases sin ejemplos
    classes_with_data = set(stats["class_counts"].keys())
    missing_classes = [classes[i] for i in range(len(classes)) if i not in classes_with_data]

    if missing_classes:
        print(f"\n⚠️  Clases SIN ejemplos ({len(missing_classes)}):")
        print(f"    {', '.join(missing_classes)}")

    # Guardar JSON
    output = {
        "summary": stats,
        "classes": classes,
        "missing_classes": missing_classes,
    }

    with open(OUTPUT_FILE, "w") as f:
        json.dump(output, f, indent=2)

    logger.info(f"Estadísticas guardadas en {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
```

### Script: prepare_folds.py

```python
"""
Prepara la estructura de directorios para k-fold cross validation.

Uso:
    uv run python scripts/prepare_folds.py

Genera:
    folds/fold_{0-2}/train/images/
    folds/fold_{0-2}/train/labels/
    folds/fold_{0-2}/val/images/
    folds/fold_{0-2}/val/labels/
    folds/fold_{0-2}/data.yaml
"""

import logging
import random
import shutil
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# Configuración
K = 3
SEED = 42
DATA_DIR = Path("data")
FOLDS_DIR = Path("folds")


def main():
    # Obtener lista de imágenes con anotaciones
    images_dir = DATA_DIR / "images"
    labels_dir = DATA_DIR / "labels"

    if not images_dir.exists():
        logger.error(f"No existe directorio de imágenes: {images_dir}")
        return

    if not labels_dir.exists():
        logger.error(f"No existe directorio de labels: {labels_dir}")
        return

    image_files = []
    for ext in ["*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"]:
        image_files.extend(images_dir.glob(ext))

    # Filtrar solo imágenes con anotación correspondiente
    valid_pairs = []
    for img_path in image_files:
        label_path = labels_dir / f"{img_path.stem}.txt"
        if label_path.exists():
            valid_pairs.append((img_path, label_path))

    if not valid_pairs:
        logger.error("No se encontraron imágenes con anotaciones válidas")
        return

    logger.info(f"Encontradas {len(valid_pairs)} imágenes con anotaciones")

    # Shuffle determinístico
    random.seed(SEED)
    random.shuffle(valid_pairs)

    # Dividir en k particiones
    fold_size = len(valid_pairs) // K
    partitions = []
    for i in range(K):
        start = i * fold_size
        end = start + fold_size if i < K - 1 else len(valid_pairs)
        partitions.append(valid_pairs[start:end])

    # Crear estructura de folds
    for fold_idx in range(K):
        fold_dir = FOLDS_DIR / f"fold_{fold_idx}"

        # Limpiar si existe
        if fold_dir.exists():
            shutil.rmtree(fold_dir)

        # Crear directorios
        for split in ["train", "val"]:
            (fold_dir / split / "images").mkdir(parents=True)
            (fold_dir / split / "labels").mkdir(parents=True)

        # Validación = partición actual
        val_pairs = partitions[fold_idx]
        # Entrenamiento = resto
        train_pairs = []
        for j in range(K):
            if j != fold_idx:
                train_pairs.extend(partitions[j])

        # Copiar archivos
        for split, pairs in [("train", train_pairs), ("val", val_pairs)]:
            for img_path, label_path in pairs:
                img_dest = fold_dir / split / "images" / img_path.name
                label_dest = fold_dir / split / "labels" / label_path.name

                shutil.copy2(img_path, img_dest)
                shutil.copy2(label_path, label_dest)

        # Crear data.yaml para este fold
        create_data_yaml(fold_dir, fold_idx)

        logger.info(f"Fold {fold_idx}: {len(train_pairs)} train, {len(val_pairs)} val")


def create_data_yaml(fold_dir: Path, fold_idx: int):
    """Genera data.yaml para un fold específico."""
    classes = load_classes()

    yaml_content = f"""# YOLO Dataset Configuration - Fold {fold_idx}
path: {fold_dir.resolve()}
train: train/images
val: val/images

nc: {len(classes)}
names: {classes}
"""

    (fold_dir / "data.yaml").write_text(yaml_content)


def load_classes() -> list:
    """Carga lista de clases desde classes.txt"""
    classes_file = DATA_DIR / "classes.txt"
    if not classes_file.exists():
        logger.error(f"No se encontró {classes_file}")
        return []
    return [line.strip() for line in classes_file.read_text().splitlines() if line.strip()]


if __name__ == "__main__":
    main()
```

## Configuración de Entrenamiento

### configs/train_config.yaml

```yaml
# Modelo base
model: yolo11n.pt  # Opciones: yolo11n/s/m/l/x

# Entrenamiento
epochs: 100
patience: 20          # Early stopping
batch: 16             # Ajustar según VRAM
imgsz: 640            # Tamaño de imagen

# Optimizador
optimizer: AdamW
lr0: 0.001            # Learning rate inicial
lrf: 0.01             # Learning rate final (lr0 * lrf)
momentum: 0.937
weight_decay: 0.0005

# Augmentation
hsv_h: 0.015          # Hue
hsv_s: 0.7            # Saturation
hsv_v: 0.4            # Value
degrees: 20.0         # Rotación ±20° (para texto en arcos)
translate: 0.1
scale: 0.5
shear: 2.0
perspective: 0.0005   # Distorsión de perspectiva
flipud: 0.0           # Sin flip vertical (texto)
fliplr: 0.0           # Sin flip horizontal (texto direccional)
mosaic: 0.2           # Reducido para preservar caracteres pequeños
mixup: 0.1

# Otros
workers: 8
device: 0             # GPU index
project: runs
exist_ok: true
verbose: true
```

### Script: train.py

```python
"""
Entrena modelo YOLO v11 con k-fold cross validation.

Uso:
    uv run python scripts/train.py                    # Entrena todos los folds
    uv run python scripts/train.py --fold 0          # Entrena solo fold 0
    uv run python scripts/train.py --config custom.yaml
"""

import argparse
import logging
from pathlib import Path

import mlflow
import yaml
from ultralytics import YOLO

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

    # Configurar MLflow
    mlflow.set_experiment("keposhka_detector")

    with mlflow.start_run(run_name=f"fold_{fold_idx}"):
        # Log de parámetros
        mlflow.log_params({
            "fold": fold_idx,
            "model": config["model"],
            "epochs": config["epochs"],
            "batch": config["batch"],
            "imgsz": config["imgsz"],
            "lr0": config["lr0"],
        })

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

        # Log de métricas finales
        if results:
            mlflow.log_metrics({
                "mAP50": results.results_dict.get("metrics/mAP50(B)", 0),
                "mAP50-95": results.results_dict.get("metrics/mAP50-95(B)", 0),
            })

        logger.info(f"Fold {fold_idx} completado")
        return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", type=int, default=None, help="Fold específico (0-2)")
    parser.add_argument("--config", type=str, default="configs/train_config.yaml")
    args = parser.parse_args()

    config = load_config(args.config)

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
```

### Script: tune.py

```python
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
```

#### Hiperparámetros optimizados por tune.py

El tuner de Ultralytics optimiza automáticamente:

| Categoría | Hiperparámetros |
|-----------|-----------------|
| Optimizador | `lr0`, `lrf`, `momentum`, `weight_decay`, `warmup_epochs` |
| Loss weights | `box`, `cls` |
| Augmentation | `hsv_h`, `hsv_s`, `hsv_v`, `degrees`, `translate`, `scale`, `shear`, `perspective`, `flipud`, `fliplr`, `mosaic`, `mixup`, `copy_paste` |

#### Flujo recomendado para tuning

1. Preparar folds: `uv run python scripts/prepare_folds.py`
2. Ejecutar tuning: `uv run python scripts/tune.py --iterations 30`
3. Revisar `runs/tune/best_hyperparameters.yaml`
4. Actualizar `configs/train_config.yaml` con los mejores valores
5. Entrenar con config optimizada: `uv run python scripts/train.py`

### Script: evaluate.py

```python
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
    print("PROMEDIO (± std):")
    print(f"  mAP50:    {avg_metrics['mAP50']:.4f} ± {avg_metrics['std_mAP50']:.4f}")
    print(f"  mAP50-95: {avg_metrics['mAP50-95']:.4f} ± {avg_metrics['std_mAP50-95']:.4f}")
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
```

### Script: predict.py

```python
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
```

### Script: export.py

```python
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
```

## Dependencias

### pyproject.toml

```toml
[project]
name = "keposhka-detector-model"
version = "0.1.0"
description = "YOLO v11 model for Keposhka character detection"
requires-python = ">=3.13"
dependencies = [
    "ultralytics",
    "torch",
    "torchvision",
    "numpy",
    "pyyaml",
    "opencv-python",
    "mlflow",
]

# Dependencias adicionales (instalar con uv add cuando se necesiten):
# - tensorflow: requerido para exportar a TFLite
# - pytest, ruff: desarrollo y linting

[tool.ruff]
line-length = 100
```

## Requisitos de GPU

### Hardware
- GPU NVIDIA con Compute Capability 3.5+ (GTX 700 series o posterior)
- Mínimo 4GB VRAM (8GB+ recomendado)

### Software
1. **Drivers NVIDIA** actualizados
   ```bash
   nvidia-smi  # Verificar instalación y versión
   ```

2. **PyTorch con CUDA** - La instalación estándar de PyTorch es solo CPU. Para GPU:
   ```bash
   # Opción 1: CUDA 11.8 (más compatible)
   uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

   # Opción 2: CUDA 12.1 (más reciente)
   uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
   ```

3. **Verificar instalación:**
   ```python
   import torch
   print(torch.cuda.is_available())      # Debe ser True
   print(torch.cuda.get_device_name(0))  # Nombre de tu GPU
   print(torch.version.cuda)             # Versión de CUDA
   ```

### Notas
- El `pyproject.toml` incluye `torch` y `torchvision` pero uv instalará versión CPU por defecto
- Debes reinstalar PyTorch con CUDA manualmente después de instalar dependencias
- Si `torch.cuda.is_available()` retorna `False`, verifica drivers y reinstala PyTorch

## Flujo de Trabajo

### 1. Preparación inicial

```bash
# Crear proyecto
mkdir keposhka_detector
cd keposhka_detector

# Inicializar con uv
uv init

# Instalar dependencias
uv add ultralytics torch torchvision numpy pyyaml opencv-python mlflow

# Reinstalar PyTorch con CUDA (ver sección "Requisitos de GPU")
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### 2. Agregar datos

Copiar desde el proyecto de labeling:
```bash
# Estructura esperada
cp -r ../keposhka_labeling/backend/data/images data/images
cp ../keposhka_labeling/backend/data/annotations/*.txt data/labels/
cp ../keposhka_labeling/backend/data/annotations/classes.txt data/
```

### 3. Analizar dataset

```bash
uv run python scripts/analyze_dataset.py
```

### 4. Preparar folds

```bash
uv run python scripts/prepare_folds.py
```

### 5. (Opcional) Buscar hiperparámetros

```bash
# Ejecutar tuning (30 iteraciones por defecto)
uv run python scripts/tune.py

# O con más iteraciones para mejor búsqueda
uv run python scripts/tune.py --iterations 50 --epochs 20

# Revisar resultados en runs/tune/best_hyperparameters.yaml
# Actualizar configs/train_config.yaml con los mejores valores
```

### 6. Entrenar

```bash
# Todos los folds
uv run python scripts/train.py

# O un fold específico
uv run python scripts/train.py --fold 0
```

### 7. Evaluar

```bash
uv run python scripts/evaluate.py
```

### 8. Inferencia

```bash
uv run python scripts/predict.py nueva_imagen.jpg --save
```

### 9. Exportar para Android

```bash
# Instalar dependencias de exportación
uv add tensorflow

# Exportar a TFLite
uv run python scripts/export.py
```

## Métricas Esperadas

Con un dataset pequeño (<50 imágenes) y 70 clases, se esperan métricas iniciales conservadoras:

| Métrica | Objetivo Inicial | Objetivo con más datos |
|---------|------------------|------------------------|
| mAP50 | > 0.50 | > 0.75 |
| mAP50-95 | > 0.30 | > 0.50 |
| Precision | > 0.50 | > 0.70 |
| Recall | > 0.50 | > 0.70 |

Estas métricas mejorarán significativamente conforme crezca el dataset.

## Fases de Desarrollo

### Fase 1: Setup & Preparación de Datos
- [x] Crear repositorio `keposhka_detector`
- [x] Inicializar proyecto con `uv init`
- [x] Instalar dependencias con `uv add ultralytics torch torchvision numpy pyyaml opencv-python mlflow`
- [x] Verificar drivers NVIDIA (`nvidia-smi`)
- [x] Reinstalar PyTorch con CUDA (`uv pip install torch torchvision --torch-backend=auto`)
- [x] Verificar CUDA funcional (`torch.cuda.is_available()`)
- [x] Crear estructura de carpetas (`data/`, `configs/`, `scripts/`)
- [x] Copiar imágenes y anotaciones desde proyecto de labeling
- [x] Crear `classes.txt` con las 70 clases
- [x] Ejecutar `analyze_dataset.py` para verificar distribución
- [ ] Asegurar cobertura de todas las clases (anotar más si faltan)

### Fase 2: Entrenamiento Inicial
- [x] Crear `train_config.yaml` con hiperparámetros base
- [ ] Ejecutar `prepare_folds.py` para generar k=3 folds
- [ ] Entrenar fold 0 como prueba inicial
- [ ] Verificar que el entrenamiento completa sin errores
- [ ] Revisar curvas de pérdida en MLflow/TensorBoard
- [ ] Entrenar folds restantes (1 y 2)

### Fase 3: Evaluación & Análisis
- [ ] Ejecutar `evaluate.py` para métricas agregadas
- [ ] Analizar varianza entre folds (¿muy alta = datos insuficientes?)
- [ ] Identificar clases con peor rendimiento (confusion matrix)
- [ ] Ejecutar `predict.py` en imágenes de prueba para inspección visual
- [ ] Documentar errores comunes (falsos positivos/negativos típicos)

### Fase 4: Iteración & Optimización
- [ ] Ejecutar `tune.py` para búsqueda automática de hiperparámetros
- [ ] Actualizar `train_config.yaml` con mejores hiperparámetros encontrados
- [ ] Considerar modelo más grande (`yolo11s`) si nano es insuficiente
- [ ] Agregar más datos de entrenamiento si hay clases problemáticas
- [ ] Re-entrenar con configuración optimizada
- [ ] Comparar métricas con baseline inicial

### Fase 5: Exportación & Validación
- [ ] Seleccionar mejor modelo (mejor fold o ensemble)
- [ ] Instalar dependencias de exportación (`uv add tensorflow`)
- [ ] Ejecutar `export.py` para generar TFLite FP16
- [ ] Validar tamaño del modelo exportado (< 20MB ideal para móvil)
- [ ] Probar inferencia con modelo TFLite localmente
- [ ] Documentar métricas finales y características del modelo

### Fase 6: Preparación para Deployment
- [ ] Crear README con instrucciones de uso del modelo
- [ ] Documentar requisitos de entrada (tamaño imagen, formato)
- [ ] Crear script de ejemplo para integración Android
- [ ] Empaquetar modelo y assets para distribución
- [ ] Pruebas de integración en app Android de prueba

---

## Consideraciones

### Desbalance de clases
- Ejecutar `analyze_dataset.py` para identificar clases sub-representadas
- Considerar focal loss o class weights si hay desbalance severo
- Asegurar cobertura de las 70 clases en el dataset

### Data Augmentation
- **Flip deshabilitado**: El texto Keposhka es direccional
- **Rotación ±20°**: Para manejar texto en arcos y curvas
- **Perspectiva 0.0005**: Simula variación de ángulo de cámara
- **Mosaic reducido (0.2)**: Preserva tamaño de caracteres pequeños

### Tamaño del modelo
- `yolo11n`: Más rápido, ideal para móvil (recomendado para empezar)
- `yolo11s/m`: Balance velocidad/precisión
- `yolo11l/x`: Mayor precisión, más lento (no recomendado para móvil)

### GPU
- Mínimo: 4GB VRAM (batch=8, imgsz=640)
- Recomendado: 8GB+ VRAM (batch=16+)

### MLflow
- Los experimentos se guardan en `mlruns/`
- Para ver la UI: `mlflow ui` y abrir http://localhost:5000

## Troubleshooting

### CUDA out of memory
```
RuntimeError: CUDA out of memory
```
**Solución**: Reducir `batch` en `train_config.yaml` (ej: 16 → 8 → 4)

### CUDA not available
```
CUDA is not available
```
**Solución**:
1. Verificar drivers NVIDIA: `nvidia-smi`
2. Reinstalar PyTorch con CUDA: `uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118`

### No se encontró modelo
```
FileNotFoundError: No se encontró ningún modelo entrenado
```
**Solución**: Ejecutar `train.py` primero para entrenar el modelo

### TFLite export falla
```
ModuleNotFoundError: No module named 'tensorflow'
```
**Solución**: Instalar dependencias de exportación: `uv add tensorflow`

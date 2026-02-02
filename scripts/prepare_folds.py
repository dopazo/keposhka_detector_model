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

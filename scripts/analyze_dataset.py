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
        bar = "#" * min(count, 50)
        print(f"  {label:4s}: {count:4d} {bar}")

    # Clases sin ejemplos
    classes_with_data = set(stats["class_counts"].keys())
    missing_classes = [classes[i] for i in range(len(classes)) if i not in classes_with_data]

    if missing_classes:
        print(f"\n[!] Clases SIN ejemplos ({len(missing_classes)}):")
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

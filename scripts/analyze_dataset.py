"""
Analyzes class distribution in the dataset.

Usage:
    uv run python scripts/analyze_dataset.py

Outputs:
    - Console report with per-class counts
    - JSON file with detailed statistics
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
    """Loads class list from classes.txt"""
    classes_file = DATA_DIR / "classes.txt"
    if not classes_file.exists():
        logger.error(f"Classes file not found: {classes_file}")
        return []
    return [line.strip() for line in classes_file.read_text().splitlines() if line.strip()]


def analyze_labels(labels_dir: Path, classes: list) -> dict:
    """Analyzes all annotation files."""
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
        logger.error(f"Labels directory not found: {labels_dir}")
        return

    stats = analyze_labels(labels_dir, classes)

    # Console report
    print("\n" + "=" * 60)
    print("DATASET ANALYSIS")
    print("=" * 60)
    print(f"\nAnnotated images: {stats['total_images']}")
    print(f"Total boxes: {stats['total_boxes']}")
    print(f"Boxes per image: min={stats['boxes_per_image']['min']}, "
          f"max={stats['boxes_per_image']['max']}, "
          f"avg={stats['boxes_per_image']['avg']:.1f}")

    print("\n" + "-" * 60)
    print("CLASS DISTRIBUTION")
    print("-" * 60)

    # Sort by count
    sorted_counts = sorted(
        [(classes[cid], count) for cid, count in stats["class_counts"].items()],
        key=lambda x: x[1],
        reverse=True
    )

    # Classes with examples
    print("\nClasses with examples (sorted by frequency):")
    for label, count in sorted_counts:
        bar = "#" * min(count, 50)
        print(f"  {label:4s}: {count:4d} {bar}")

    # Classes without examples
    classes_with_data = set(stats["class_counts"].keys())
    missing_classes = [classes[i] for i in range(len(classes)) if i not in classes_with_data]

    if missing_classes:
        print(f"\n[!] Classes WITHOUT examples ({len(missing_classes)}):")
        print(f"    {', '.join(missing_classes)}")

    # Save JSON
    output = {
        "summary": stats,
        "classes": classes,
        "missing_classes": missing_classes,
    }

    with open(OUTPUT_FILE, "w") as f:
        json.dump(output, f, indent=2)

    logger.info(f"Statistics saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()

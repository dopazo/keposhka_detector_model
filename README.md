# Keposhka Detector Model

YOLO v11 object detection model for detecting and classifying characters from the Keposhka fictional language. Designed for export to TFLite for Android deployment.

## Features

- Detects and classifies 70 character classes (digits, letters, digraphs)
- Handles various image types: digital images, camera photos, arc text, perspective views
- 3-fold cross-validation for robust performance estimation
- MLflow experiment tracking
- TFLite FP16 export for mobile deployment

## Requirements

- Python 3.13+
- NVIDIA GPU with 4GB+ VRAM (8GB+ recommended)
- CUDA 11.8 or 12.1

## Installation

```bash
# Install dependencies
uv sync

# Reinstall PyTorch with CUDA support
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Verify CUDA is available
uv run python -c "import torch; print(torch.cuda.is_available())"
```

## Data Setup

Place your data in the following structure:

```
data/
├── images/       # .jpg, .jpeg, .png files
├── labels/       # YOLO format .txt files (same name as images)
└── classes.txt   # 70 class names, one per line
```

Label format (YOLO): `<class_id> <x_center> <y_center> <width> <height>` (normalized 0-1)

## Usage

### 1. Analyze Dataset

```bash
uv run python scripts/analyze_dataset.py
```

Outputs class distribution and identifies missing classes.

### 2. Prepare K-Fold Splits

```bash
uv run python scripts/prepare_folds.py
```

Creates `folds/fold_{0-2}/` with train/val splits.

### 3. Train

```bash
# Train all folds
uv run python scripts/train.py

# Train specific fold
uv run python scripts/train.py --fold 0

# Use custom config
uv run python scripts/train.py --config configs/custom.yaml
```

### 4. Evaluate

```bash
uv run python scripts/evaluate.py
```

Outputs mAP50, mAP50-95, precision, and recall with cross-fold statistics.

### 5. Predict

```bash
uv run python scripts/predict.py image.jpg --save
uv run python scripts/predict.py images_dir/ --conf 0.5
```

### 6. Export to TFLite

```bash
uv add tensorflow
uv run python scripts/export.py
```

## Configuration

Training hyperparameters are in `configs/train_config.yaml`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| model | yolo11n.pt | Base model (n/s/m/l/x) |
| epochs | 100 | Max training epochs |
| patience | 20 | Early stopping patience |
| batch | 16 | Batch size (reduce if OOM) |
| imgsz | 640 | Input image size |

## MLflow

View experiment results:

```bash
mlflow ui
```

Open http://localhost:5000

## Class Mapping

70 classes: 0-9, A-Z, a-z, CH, ch, GH, gh, SH, sh, TH, th

Full mapping available in `specs/keposhka_detector_model.md`.

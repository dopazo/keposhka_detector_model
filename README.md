# Keposhka Detector Model

YOLO v11 object detection model for detecting and classifying characters from the Keposhka fictional language. Designed for export to TFLite for Android deployment.

## Features

- Detects and classifies 70 character classes (digits, letters, digraphs)
- Handles various image types: digital images, camera photos, arc text, perspective views
- Two training modes: simple split (recommended for small datasets) and k-fold cross-validation
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

### 2. Training

There are two training modes. Choose based on your dataset size:

#### Option A: Simple Split (Recommended for small datasets)

For datasets with few images (e.g. <100), a simple train/val split is more practical than cross-validation. Cross-validation divides the data into multiple folds, which with very few images leaves too few samples per fold for meaningful training.

```bash
# Step 1: Create 90/10 train/val split
uv run python scripts/prepare_simple_split.py

# You can customize the split ratio and seed
uv run python scripts/prepare_simple_split.py --split 0.8 --seed 42

# Step 2: Train
uv run python scripts/train.py --data data_split/data.yaml --name my_experiment
```

This creates the split in `data_split/` with `train/` and `val/` subdirectories.

#### Option B: K-Fold Cross-Validation

For larger datasets, k-fold cross-validation (k=3) gives a more robust performance estimate by training on different data splits.

```bash
# Step 1: Create k-fold splits
uv run python scripts/prepare_folds.py

# Step 2: Train all folds
uv run python scripts/train.py

# Or train a specific fold
uv run python scripts/train.py --fold 0
```

Creates `folds/fold_{0-2}/` with train/val splits.

#### Custom config

Both modes support a custom config file:

```bash
uv run python scripts/train.py --config configs/custom.yaml --data data_split/data.yaml --name my_exp
uv run python scripts/train.py --config configs/custom.yaml --fold 0
```

### 3. Evaluate

```bash
uv run python scripts/evaluate.py
```

Outputs mAP50, mAP50-95, precision, and recall with cross-fold statistics.

### 4. Predict

```bash
uv run python scripts/predict.py image.jpg --save
uv run python scripts/predict.py images_dir/ --conf 0.5
```

### 5. Export to TFLite

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

## MLflow (Disabled)

MLflow experiment tracking is currently disabled due to a Windows compatibility issue. Ultralytics' MLflow callback fails because Windows paths with drive letters (e.g., `C:\...`) are incorrectly parsed as URI schemes.

Training metrics are logged to console and saved in the `runs/` directory by Ultralytics.

## Class Mapping

70 classes: 0-9, A-Z, a-z, CH, ch, GH, gh, SH, sh, TH, th

Full mapping available in `specs/keposhka_detector_model.md`.

# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

YOLO v11 object detection model for detecting and classifying 70 characters from the fictional Keposhka language. The model will be exported to TFLite FP16 for Android deployment.

## Tech Stack

- **ML Framework**: Ultralytics YOLO v11
- **Backend**: PyTorch (requires CUDA for GPU)
- **Python**: 3.13+
- **Package Manager**: uv
- **Experiment Tracking**: MLflow (local)

## Common Commands

```bash
# Install dependencies
uv sync

# Reinstall PyTorch with CUDA (required for GPU training)
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Run scripts
uv run python scripts/analyze_dataset.py
uv run python scripts/prepare_folds.py
uv run python scripts/train.py [--fold 0] [--config custom.yaml]
uv run python scripts/evaluate.py [--fold 0]
uv run python scripts/predict.py image.jpg [--model path] [--conf 0.5] [--save]
uv run python scripts/export.py [--fold 0]

# View MLflow experiments
mlflow ui  # Opens http://localhost:5000

# Verify CUDA is working
uv run python -c "import torch; print(torch.cuda.is_available())"
```

## Architecture

### Data Pipeline
1. `data/images/` and `data/labels/` contain source images and YOLO-format annotations
2. `scripts/prepare_folds.py` creates 3-fold cross-validation splits in `folds/fold_{0-2}/`
3. Each fold has `train/` and `val/` directories with `images/` and `labels/` subdirectories
4. Per-fold `data.yaml` files configure YOLO training paths

### Training Pipeline
- Uses k=3 fold cross-validation (optimized for small datasets <50 images)
- Config in `configs/train_config.yaml` controls hyperparameters
- MLflow tracks experiments under `keposhka_detector`
- Model weights saved to `runs/fold_{N}/weights/best.pt`

### Key Augmentation Settings (text-specific)
- Rotation ±20° (for arc/curved text)
- Perspective distortion 0.0005
- **No horizontal/vertical flips** (text is directional)
- Reduced mosaic (0.2) to preserve small characters

### Class Mapping
70 classes total: digits 0-9, uppercase A-Z, lowercase a-z, plus digraphs (CH/ch, GH/gh, SH/sh, TH/th). Full mapping in `specs/keposhka_detector_model.md`.

## Project Structure

```
scripts/           # Main Python scripts
  analyze_dataset.py  # Analyze class distribution
  prepare_folds.py    # Create k-fold splits
  train.py            # Train model(s)
  evaluate.py         # Evaluate trained models
  predict.py          # Run inference
  export.py           # Export to TFLite
configs/           # YAML config files
data/              # Source images and labels
folds/             # Generated k-fold splits
runs/              # YOLO training outputs
mlruns/            # MLflow experiment data
specs/             # Technical specification
```

## GPU Requirements

- NVIDIA GPU with 4GB+ VRAM (8GB+ recommended)
- CUDA 11.8 or 12.1 installed separately
- Must reinstall PyTorch with CUDA after `uv sync` (see commands above)

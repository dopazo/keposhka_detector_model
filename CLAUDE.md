# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

YOLO v11 object detection model for detecting and classifying 70 characters from the fictional Keposhka language. The model will be exported to TFLite FP16 for Android deployment.

## Tech Stack

- **ML Framework**: Ultralytics YOLO v11
- **Backend**: PyTorch (requires CUDA for GPU)
- **Python**: 3.13+
- **Package Manager**: uv
- **Experiment Tracking**: MLflow (DISABLED - see Known Issues)

## Common Commands

```bash
# Install dependencies
uv sync

# Reinstall PyTorch with CUDA (required for GPU training)
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Run scripts
uv run python scripts/analyze_dataset.py                                        # Analizar distribución de clases
uv run python scripts/prepare_folds.py                                          # Crear k-fold splits (k=3)
uv run python scripts/prepare_simple_split.py [--split 0.9] [--seed 42]         # Crear split simple 90/10 sin folds
uv run python scripts/train.py [--fold 0] [--config custom.yaml]                # Entrenar con k-fold
uv run python scripts/train.py --data data_split/data.yaml --name exp_name      # Entrenar con split simple
uv run python scripts/evaluate.py [--fold 0]                                    # Evaluar modelo entrenado
uv run python scripts/predict.py image.jpg [--model path] [--conf 0.5] [--save] # Inferencia sobre imagen
uv run python scripts/export.py [--fold 0]                                      # Exportar a TFLite

# View MLflow experiments (DISABLED - see Known Issues)
# mlflow ui  # Opens http://localhost:5000

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
- Two modes: k=3 fold cross-validation (`--fold`) or simple split (`--data`)
- `prepare_simple_split.py` creates 90/10 split in `data_split/` (no folds)
- Config in `configs/train_config.yaml` controls hyperparameters
- Model weights saved to `runs/detect/runs/{name}/weights/best.pt`

### Key Augmentation Settings (small dataset, text-specific)
- **Mosaic 1.0** (max): most impactful augmentation for 39-image dataset
- **Copy-paste 0.3**: compensates class imbalance by copying objects between images
- **Freeze backbone (10 layers)**: transfer learning to prevent overfitting
- **No mixup**: overlapping blended characters confuse text detection
- Rotation ±20° (for arc/curved text)
- Perspective distortion 0.0005
- **No horizontal/vertical flips** (text is directional)
- Close mosaic at 30 epochs before end for clean fine-tuning
- Random erasing 0.2 for occlusion robustness

### Class Mapping
70 classes total: digits 0-9, uppercase A-Z, lowercase a-z, plus digraphs (CH/ch, GH/gh, SH/sh, TH/th). Full mapping in `specs/keposhka_detector_model.md`.

## Project Structure

```
scripts/           # Main Python scripts
  analyze_dataset.py  # Analyze class distribution
  prepare_folds.py        # Create k-fold splits
  prepare_simple_split.py # Create simple 90/10 split (no folds)
  train.py                # Train model(s) (supports --fold or --data)
  evaluate.py         # Evaluate trained models
  predict.py          # Run inference
  export.py           # Export to TFLite
configs/           # YAML config files
data/              # Source images and labels
data_split/        # Simple train/val split (generated)
folds/             # Generated k-fold splits
runs/              # YOLO training outputs (runs/detect/runs/{name}/)
specs/             # Technical specification
```

## GPU Requirements

- NVIDIA GPU with 4GB+ VRAM (8GB+ recommended)
- CUDA 11.8 or 12.1 installed separately
- Must reinstall PyTorch with CUDA after `uv sync` (see commands above)

## Known Issues

### MLflow disabled on Windows
MLflow was removed (`uv remove mlflow`) because Ultralytics' MLflow callback fails on Windows. The error occurs because Windows paths with drive letters (e.g., `C:\...`) are incorrectly parsed as URI schemes by MLflow.

**Error**: `UnsupportedModelRegistryStoreURIException: got unsupported URI 'C:\...'`

**Workaround**: MLflow was uninstalled. Training metrics are logged to console and saved in `runs/` directory by Ultralytics. To re-enable MLflow later, would need to configure `MLFLOW_TRACKING_URI` with proper `file:///` prefix or run MLflow server.

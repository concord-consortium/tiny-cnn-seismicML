# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Lightweight PyTorch CNN for classifying seismic signals from Raspberry Shake seismometers into 3 classes (background noise, traffic/urban, earthquake). Designed for edge deployment (~20K params compact model) with browser-based inference option.

## Commands

```bash
pip install -r requirements.txt                    # Install dependencies
python examples/test_workflow.py                   # Verify setup (integration test with dummy data)
python train.py --config configs/standard_config.yaml --save-dir models
python predict.py --model-path trained_models/best_model.pth --config configs/standard_config.yaml
python scripts/export_to_browser.py --export_all --output_dir browser_demo/models  # ONNX export
```

There is no unit test suite. `examples/test_workflow.py` is the primary verification script — it exercises model creation, data preprocessing, training, save/load, and inference with synthetic data.

## Architecture

```
src/
├── models/cnn.py         # SeismicCNN (~100K params), CompactSeismicCNN (~20K params)
├── data/preprocessing.py # SeismicDataset, preprocess_seismogram, DataAugmentation
└── utils/trainer.py      # Trainer class, optimizer/scheduler factories
```

**Model I/O**: Input `(N, 3, 6000)` → Output `(N, 3)` logits. 3 channels = E-N-Z components, 6000 samples = 60s at 100Hz.

**Factory pattern**: `model = get_model(model_type='standard', num_classes=3, input_channels=3)`

**Configuration**: YAML files in `configs/` with sections: `model`, `training`, `data`. Use `configs/standard_config.yaml` as template.

**Checkpoint format**: Dict with keys `epoch`, `model_state_dict`, `optimizer_state_dict`, `train_losses`, `val_losses`, `val_accuracies`. Loading must handle both checkpoint dicts and raw state_dicts.

**Entry points**: CLI (`train.py`, `predict.py`), Jupyter notebooks (numbered workflow `01_` → `04_`), browser demo (`browser_demo/`).

## Code Style

- Google-style docstrings with Args/Returns — **no type hints**
- Imports: standard lib → third-party → local (`from src.models import get_model`)
- `PascalCase` classes, `snake_case` functions and config keys

## Data Pipeline

Seismic data sourced from Raspberry Shake network and IRIS FDSN. Preprocessing: bandpass filter 1-45 Hz → window extraction → normalize (zero mean, unit variance). Augmentation: Gaussian noise, time shifts, amplitude scaling. Classes: 0=Background, 1=Traffic, 2=Earthquake.

Notebooks workflow: `01_data_exploration/` → `02_labeling/` (generates `labeled_data/*.npy`) → `03_training/` → `04_inference/`.

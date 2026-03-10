# Project Guidelines

Seismic signal classifier using CNNs. Classifies 3-component waveforms into noise, traffic, or earthquake.

## Code Style

- **Imports**: Standard lib → third-party → local (`from src.models import get_model`)
- **Docstrings**: Google-style with Args/Returns ([src/data/preprocessing.py](../src/data/preprocessing.py) for examples)
- **No type hints** - document parameters in docstrings instead
- **Naming**: `PascalCase` classes, `snake_case` functions, `snake_case` config keys

## Architecture

```
src/
├── models/cnn.py        # SeismicCNN (standard, ~100K params), CompactSeismicCNN (~20K params)
├── data/preprocessing.py # SeismicDataset, preprocess_seismogram, DataAugmentation
└── utils/trainer.py     # Trainer class, optimizer/scheduler factories
```

**Model I/O**: Input `(N, 3, 6000)` → Output `(N, 3)` logits for 3 classes  
**Factory**: `model = get_model(model_type='standard', num_classes=3, input_channels=3)`

## Build and Test

```bash
pip install -r requirements.txt           # Install dependencies
python examples/test_workflow.py          # Verify setup works
python train.py --config configs/standard_config.yaml --save-dir models
python predict.py --model-path trained_models/best_model.pth --config configs/standard_config.yaml
```

## Project Conventions

### Config files (YAML)
Use [configs/standard_config.yaml](../configs/standard_config.yaml) as template. Key sections: `model`, `training`, `data`.

### Model checkpoints
```python
# Save format (Trainer.save_checkpoint)
{'epoch': N, 'model_state_dict': {...}, 'optimizer_state_dict': {...}, 'train_losses': [...]}

# Loading must handle both checkpoint dict and raw state_dict
checkpoint = torch.load(path, map_location=device)
model.load_state_dict(checkpoint.get('model_state_dict', checkpoint))
```

### Notebooks workflow
1. `01_data_exploration/` - EDA on seismic data
2. `02_labeling/` - Generate `labeled_data/*.npy` files
3. `03_training/` - Train and save models
4. `04_inference/` - Predict on new stations

### Browser export
```bash
python scripts/export_to_browser.py --export_all --output_dir browser_demo/models
```
Outputs ONNX model + metadata JSON. See [browser_demo/](../browser_demo/) for JS inference.

## Data Format

- **Input**: 3-component (E-N-Z) seismograms, 100 Hz sampling
- **Preprocessing**: bandpass 1-45 Hz → extract window → normalize (zero mean, unit variance)
- **Classes**: 0=Background, 1=Traffic, 2=Earthquake

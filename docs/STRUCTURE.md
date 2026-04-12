# Project Structure

This document describes the standardized structure of the EEG-BCI Experiment Framework.

## Directory Layout

```
EEGBCI-SEF/
├── core/                          # Core framework components
│   ├── __init__.py
│   ├── registry.py               # Component registry system
│   ├── config.py                 # Configuration management
│   ├── experiment.py             # Experiment orchestration
│   └── base/                     # Abstract base classes
│       ├── __init__.py
│       ├── base_trainer.py       # Base trainer interface
│       └── base_dataset.py       # Base dataset interface
│
├── data/                          # Data loading and processing
│   ├── __init__.py
│   ├── datasets.py               # Dataset implementations
│   └── transforms/               # Data transformations (future)
│
├── models/                        # Neural network models
│   ├── __init__.py               # Model registration
│   ├── EEGNet.py
│   ├── CNNLSTM.py
│   ├── GACLNet.py
│   ├── myFBCNet.py
│   ├── FBCNN.py
│   ├── myADFCNN.py
│   └── layers.py
│
├── trainers/                      # Training implementations
│   ├── __init__.py
│   └── supervised_trainer.py     # Supervised learning with CV
│
├── utils/                         # Utility functions
│   ├── __init__.py
│   ├── path_manager.py           # Path management
│   ├── logging.py                # Logging utilities
│   └── visualization.py          # Plotting functions
│
├── configs/                       # Configuration files
│   ├── experiment/               # Experiment configurations
│   │   ├── default.yaml
│   │   ├── xwstroke_eegnet.yaml
│   │   └── bciciv2a_gaclnet.yaml
│   └── dataset/                  # Dataset configurations
│       ├── BCICIV2a.yaml
│       ├── XWStroke.yaml
│       └── LowerStroke.yaml
│
├── experiments/                   # Experiment scripts
│   └── run_experiment.py         # Main entry point
│
├── scripts/                       # Utility scripts
│   ├── test_framework.py         # Framework tests
│   └── preprocess_data.py        # Data preprocessing
│
├── preprocessing/                 # Preprocessing modules
│   ├── slide_windows.py
│   ├── filter_banks.py
│   ├── resample.py
│   ├── data_augmentation.py
│   └── base_processor.py
│
├── train.py                       # Unified training script (NEW)
├── README.md                      # Main documentation
├── MIGRATION.md                   # Migration guide
├── STRUCTURE.md                   # This file
└── requirements.txt               # Dependencies

# Legacy files (kept for backward compatibility)
├── torch_fold_train.py            # Original training script
├── constant_value.py              # Original configuration
├── train.py (old)                 # Old training script
├── legacy_dataloader/             # Original data loaders (old framework)
├── legacy_config/                 # Original configs (old framework)
├── legacy_utils.py                # Old utils (old framework)
├── legacy_plot.py                 # Old plotting script (old framework)
└── legacy_preprocessing_XW.py     # Old preprocessing script (old framework)
```

## Key Components

### Core Module (`core/`)

The heart of the framework, providing:

- **Registry System**: Plugin architecture for models, datasets, and trainers
- **Configuration Management**: YAML-based configuration with dot notation access
- **Experiment Management**: Orchestrates the complete experiment lifecycle
- **Base Classes**: Abstract interfaces for trainers and datasets

### Data Module (`data/`)

Handles data loading:

- **Datasets**: Concrete implementations for each dataset (BCICIV2a, XWStroke, etc.)
- **BaseDataset**: Abstract base with common preprocessing pipeline

### Models Module (`models/`)

Neural network implementations:

- All models use standardized interface: `(num_channels, num_classes, num_bands, input_length)`
- Adapters provided for models with different parameter names
- Registration via `@MODELS.register('ModelName')` decorator

### Trainers Module (`trainers/`)

Training loop implementations:

- **SupervisedTrainer**: Standard supervised learning with k-fold CV
- Easy to extend with custom trainers via `@TRAINERS.register()`

### Utils Module (`utils/`)

Supporting functionality:

- **PathManager**: Organized directory structure for experiment outputs
- **Logging**: Configurable logging to file and console
- **Visualization**: Training curves and result plots

## Configuration System

### Experiment Config (`configs/experiment/*.yaml`)

```yaml
experiment:
  name: "MyExperiment"
  seed: 42

data:
  dataset: "XWStroke"           # Registered dataset name
  subjects: "all"               # or [1, 2, 3]
  info_path: "configs/dataset/XWStroke.yaml"

model:
  type: "EEGNet"                # Registered model name
  args: {}                      # Optional model arguments

training:
  device: "cuda"
  epochs: 200
  batch_size: 32
  k_folds: 5
  optimizer:
    lr: 0.001
    weight_decay: 0.5

trainer:
  type: "SupervisedTrainer"
```

### Dataset Config (`configs/dataset/*.yaml`)

```yaml
dataset:
  name: "XWStroke"
  data_dir: "src/datasets/21679035/sourcedata"
  original_sr: 500
  num_channels: 32
  num_classes: 2
  subjects: [1, 2, 3, ...]

preprocessing:
  windowing:
    window_length_sec: 2.0
    window_stride_sec: 0.5
  resample:
    target_sr: 250
  channel_selection:
    channels_to_select: [0, 1, 2, ...]
  filter_bank:
    filter_banks:
      - [4, 40]
  augmentation:
    - method: "add_noise"
      snr_db: 20
      probability: 0.3
```

## Adding New Components

### New Dataset

1. Create class in `data/datasets.py`:

```python
from core.base.base_dataset import BaseDataset
from core.registry import DATASETS

@DATASETS.register('MyDataset')
class MyDataset(BaseDataset):
    def _load_raw_data(self):
        # Load data from files
        return data, labels
```

2. Create config in `configs/dataset/MyDataset.yaml`

3. Use in experiment config:

```yaml
data:
  dataset: "MyDataset"
```

### New Model

1. Implement model in `models/my_model.py`:

```python
import torch.nn as nn
from core.registry import MODELS

@MODELS.register('MyModel')
class MyModel(nn.Module):
    def __init__(self, num_channels, num_classes, num_bands, input_length):
        super().__init__()
        # Your architecture
    
    def forward(self, x):
        # Forward pass
        return x
```

2. Import in `models/__init__.py`

3. Use in experiment config:

```yaml
model:
  type: "MyModel"
```

### New Trainer

1. Implement in `trainers/my_trainer.py`:

```python
from core.base.base_trainer import BaseTrainer
from core.registry import TRAINERS

@TRAINERS.register('MyTrainer')
class MyTrainer(BaseTrainer):
    def train(self, datasets):
        # Your training logic
        return results
```

2. Import in `trainers/__init__.py`

## Experiment Output Structure

Each experiment creates:

```
experiments/EXPERIMENT_NAME/
├── config.yaml              # Saved configuration
├── checkpoints/             # Model checkpoints
│   ├── subject_1_fold_0_acc_0.8500.pt
│   └── ...
├── logs/
│   └── training.log         # Detailed training log
├── visualizations/
│   ├── training_history.png
│   ├── subject_comparison_TIMESTAMP.png
│   └── subject_1/
│       └── training_history.png
└── results/
    └── results.json         # Final metrics
```

## Usage Examples

### Run Experiment

```bash
# With default config
python train.py

# With specific config
python train.py --config configs/experiment/xwstroke_eegnet.yaml

# With custom name
python train.py --config my_config.yaml --name my_exp

# On CPU
python train.py --device cpu

# List available options
python train.py --list-datasets
python train.py --list-models
python train.py --list-trainers
```

### Programmatic API

```python
from core import Config, ExperimentManager

# Load config
config = Config.fromfile('configs/experiment/my_exp.yaml')

# Create experiment
exp = ExperimentManager(config, exp_name='test')

# Run
exp.setup()
results = exp.run()
```

## Migration from Old Framework

See [MIGRATION.md](MIGRATION.md) for detailed migration instructions.

Quick reference:

| Old | New |
|-----|-----|
| `constant_value.SELECTED_DATASET` | `configs/experiment/*.yaml` → `data.dataset` |
| `constant_value.SELECTED_MODEL` | `configs/experiment/*.yaml` → `model.type` |
| `torch_fold_train.py` | `train.py` or `experiments/run_experiment.py` |
| `src/results/` | `experiments/EXPERIMENT_NAME/results/` |
| `src/checkpoints/` | `experiments/EXPERIMENT_NAME/checkpoints/` |

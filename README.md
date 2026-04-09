# EEG-BCI Standard Experiment Framework

A standardized, modular, and extensible platform for BCI algorithm experimentation. Supports various datasets, deep learning models, preprocessing pipelines, and training strategies.

## :sparkles: Features

- **Modular Design**: Clean separation of data, models, trainers, and evaluation
- **Configuration-Driven**: YAML-based experiment configuration for reproducibility
- **Registry Pattern**: Easy to extend with new datasets and models via decorators
- **Comprehensive Logging**: Tracks training metrics, console output, and generates visualizations
- **Cross-Validation**: Built-in k-fold cross-validation with group-aware splitting

## :open_file_folder: Project Structure

```
eeg_bci_experiment/
├── core/                       # Core framework components
│   ├── base/                   # Abstract base classes
│   │   ├── base_trainer.py     # Base trainer interface
│   │   └── base_dataset.py     # Base dataset interface
│   ├── registry.py             # Component registry system
│   ├── config.py               # Configuration management
│   └── experiment.py           # Experiment orchestration
├── data/                       # Data loading and processing
│   ├── datasets.py             # Dataset implementations
│   └── transforms/             # Data transformations
├── models/                     # Model definitions
│   ├── EEGNet.py
│   ├── CNNLSTM.py
│   ├── GACLNet.py
│   └── ...
├── trainers/                   # Training implementations
│   └── supervised_trainer.py   # Supervised learning with CV
├── utils/                      # Utility functions
│   ├── path_manager.py         # Path management
│   ├── logging.py              # Logging utilities
│   └── visualization.py        # Plotting functions
├── configs/                    # Configuration files
│   ├── experiment/             # Experiment configs
│   │   ├── default.yaml
│   │   ├── xwstroke_eegnet.yaml
│   │   └── bciciv2a_gaclnet.yaml
│   └── dataset/                # Dataset configs
│       ├── BCICIV2a.yaml
│       ├── XWStroke.yaml
│       └── LowerStroke.yaml
├── experiments/                # Experiment scripts
│   └── run_experiment.py       # Main entry point
└── scripts/                    # Utility scripts
    └── preprocess_data.py

# Legacy files (kept for reference)
├── torch_fold_train.py         # Original training script
├── constant_value.py           # Original configuration
└── train.py                    # Backward-compatible wrapper
```

## :rocket: Quick Start

### 1. Installation

```bash
# Clone the repository
git clone <repository-url>
cd eeg_bci_experiment

# Create virtual environment
conda create -n eeg_bci python=3.10
conda activate eeg_bci

# Install PyTorch (adjust for your CUDA version)
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

# Install other dependencies
pip install -r requirements.txt
```

### 2. Data Preparation

Download and prepare datasets according to the following structure:

```
src/datasets/
├── BCICIV_2a/                 # BCI Competition IV 2a
│   ├── A01T.mat
│   ├── A01E.mat
│   └── ...
├── 21679035/                  # XuanWu Stroke dataset
│   └── sourcedata/
│       └── sub-01/
│           └── sub-01_task-motor-imagery_eeg.mat
└── LowerStroke/               # Lower limb stroke dataset
    └── subject_1.mat
```

**Note**: Run preprocessing scripts before training if required:
```bash
python legacy_preprocessing_XW.py  # For XW Stroke dataset
```

### 3. Running Experiments

#### Using the New Framework

```bash
# Run with default configuration
python train.py

# Run with specific configuration
python train.py --config configs/experiment/xwstroke_eegnet.yaml

# Run with custom experiment name
python train.py --config configs/experiment/my_exp.yaml --name my_experiment

# List available options
python train.py --list-datasets
python train.py --list-models
python train.py --list-trainers

# Specify device
python train.py --device cuda
python train.py --device cpu
```

#### Legacy Mode - Using `torch_fold_train.py`

The original training script `torch_fold_train.py` is still supported for backward compatibility.

**Configuration**

Edit `constant_value.py` to select dataset and model:

```python
# Dataset selection (index in DATASETS list)
SELECTED_DATASET = 3  # 0:BCICIV_2a, 1:Croatian, 2:LowerStroke, 3:XWStroke

# Model selection (index in MODELS list)
SELECTED_MODEL = 1    # 0:EEGNet, 1:CNN_LSTM, 2:DeepConvNet, 3:ShallowConvNet, 4:GACLNet

# Target subjects for training
target_subjects = "all"  # or specify list like [1, 2, 3]
```

Then edit the training configuration file (default: `legacy_config/train_config/base_config.yaml`):

```yaml
dataset_id: 3              # Should match SELECTED_DATASET above
model_id: 1                # Should match SELECTED_MODEL above

Training:
  num_epochs: 200
  batch_size: 32
  learning_rate: 0.001
  weight_decay: 0.5
  k_folds: 5
  train_device: "cuda"
  target_subjects: "all"   # or list like [1, 2, 3]
```

**Running Training**

```bash
# Basic usage
python torch_fold_train.py

# The script will automatically:
# 1. Load configuration from constant_value.py and legacy_config/train_config/base_config.yaml
# 2. Train models for each subject with k-fold cross-validation
# 3. Save checkpoints to src/checkpoints/
# 4. Generate visualization plots to src/vis/
# 5. Log training details to training.log
```

**Output Structure**

```
src/
├── checkpoints/
│   └── {DATASET_NAME}/
│       └── {MODEL_NAME}/
│           └── {subject_id}/
│               └── best_fold{fold_idx}_acc{accuracy:.4f}.pt
└── vis/
    └── {DATASET_NAME}/
        └── {MODEL_NAME}/
            ├── subject_{subject_id}/
            │   └── training_history_per_fold.png
            └── subject_performance_comparison_{timestamp}.png
```

**Key Features**

- **Real-time Progress**: Single progress bar per fold showing epoch progress with live metrics:
  ```
  Sub1 Fold1 |███████         | 10/100 [00:15<02:15] LR:1.0e-03 T:0.523/0.812 V:0.489/0.845
  ```
  Where `T:loss/acc` = Train loss/accuracy, `V:loss/acc` = Validation loss/accuracy

- **Automatic Checkpoint Saving**: Best model for each fold is automatically saved

- **Comprehensive Logging**: Detailed metrics logged to `training.log`

**Note**: While `torch_fold_train.py` is maintained for compatibility, new projects are recommended to use the new `train.py` framework for better modularity and configuration management.

### 4. Results

Experiment outputs are organized as:

```
experiments/
└── EXPERIMENT_NAME/
    ├── config.yaml              # Saved configuration
    ├── checkpoints/             # Model checkpoints
    │   ├── subject_1_fold_0_acc_0.8500.pt
    │   └── ...
    ├── logs/
    │   └── training.log         # Training log
    ├── visualizations/
    │   ├── training_history.png
    │   ├── subject_comparison_TIMESTAMP.png
    │   └── subject_1/
    │       └── training_history.png
    └── results/
        └── results.json         # Final metrics
```

## :gear: Configuration

### Experiment Configuration

Create a YAML configuration file:

```yaml
experiment:
  name: "MyExperiment"
  seed: 42

data:
  dataset: "XWStroke"           # Dataset name
  subjects: "all"               # or [1, 2, 3]
  info_path: "configs/dataset/XWStroke.yaml"

model:
  type: "EEGNet"                # Model name
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

### Adding a New Dataset

1. Create a dataset class in `data/datasets.py`:

```python
from core.base.base_dataset import BaseDataset
from core.registry import DATASETS

@DATASETS.register('MyDataset')
class MyDataset(BaseDataset):
    def _load_raw_data(self):
        # Load your data
        data = ...  # Shape: (n_trials, n_channels, n_timepoints)
        labels = ...  # Shape: (n_trials,)
        return data, labels
```

2. Create a dataset configuration in `configs/dataset/MyDataset.yaml`

3. Create an experiment configuration using your dataset

### Adding a New Model

1. Implement your model in `models/my_model.py`:

```python
import torch.nn as nn
from core.registry import MODELS

@MODELS.register('MyModel')
class MyModel(nn.Module):
    def __init__(self, num_channels, num_classes, num_bands, input_length):
        super().__init__()
        # Your model architecture
        
    def forward(self, x):
        # Forward pass
        return x
```

2. Import and register in `models/__init__.py`

## :notebook: Development Guide

### Creating a Custom Trainer

```python
from core.base.base_trainer import BaseTrainer
from core.registry import TRAINERS

@TRAINERS.register('MyTrainer')
class MyTrainer(BaseTrainer):
    def train(self, datasets):
        # Your training logic
        results = {}
        return results
```

### Experiment Programmatic API

```python
from core import Config, ExperimentManager

# Load configuration
config = Config.fromfile('configs/experiment/my_exp.yaml')

# Create experiment
exp = ExperimentManager(config, exp_name='my_test')

# Setup and run
exp.setup()
results = exp.run()
```

## :warning: Migration from Old Framework

If you have been using the old `constant_value.py` based configuration:

1. Copy your settings to a new YAML file in `configs/experiment/`
2. Use `python train.py --config your_config.yaml` instead of editing `constant_value.py`
3. The old `torch_fold_train.py` is kept for backward compatibility

## :clipboard: Requirements

- Python >= 3.10
- PyTorch >= 2.0
- NumPy >= 1.20
- SciPy >= 1.7
- scikit-learn >= 1.0
- matplotlib >= 3.5
- PyYAML >= 6.0
- tqdm >= 4.60

See `requirements.txt` for complete list.

## :bug: Troubleshooting

**CUDA out of memory**: Reduce `batch_size` in configuration

**Data not found**: Check `data_dir` paths in dataset configuration files

**Model not found**: Ensure model is registered in `models/__init__.py`

## :page_facing_up: License

## :busts_in_silhouette: Contact


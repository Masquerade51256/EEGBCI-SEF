# Experiment Guide

This guide explains how to use the EEG-BCI Experiment Framework to run your experiments.

## Quick Start

### 1. Basic Experiment

Run a pre-configured experiment:

```bash
python train.py --config configs/experiment/xwstroke_eegnet.yaml
```

### 2. Custom Experiment

Create your own configuration:

```yaml
# configs/experiment/my_experiment.yaml
experiment:
  name: "MyFirstExperiment"
  seed: 42

data:
  dataset: "BCICIV2a"
  subjects: [1, 2, 3]  # Or "all" for all subjects
  info_path: "configs/dataset/BCICIV2a.yaml"

model:
  type: "EEGNet"
  args: {}  # Auto-populated from dataset

training:
  device: "cuda"
  epochs: 100
  batch_size: 32
  k_folds: 5
  optimizer:
    lr: 0.001
    weight_decay: 0.01

trainer:
  type: "SupervisedTrainer"
```

Then run:

```bash
python train.py --config configs/experiment/my_experiment.yaml --name my_first_run
```

## Configuration Reference

### Experiment Section

```yaml
experiment:
  name: "ExperimentName"     # Experiment identifier
  seed: 42                    # Random seed for reproducibility
  description: "..."          # Optional description
```

### Data Section

```yaml
data:
  dataset: "XWStroke"         # Dataset name (must be registered)
  subjects: "all"             # "all" or list [1, 2, 3]
  info_path: "..."            # Path to dataset config
```

### Model Section

```yaml
model:
  type: "EEGNet"              # Model name (must be registered)
  args:                       # Optional model arguments
    num_channels: 22          # Auto-populated if not specified
    num_classes: 4            # Auto-populated if not specified
    num_bands: 1              # Auto-populated if not specified
    input_length: 750         # Auto-populated if not specified
```

### Training Section

```yaml
training:
  device: "cuda"              # "cuda" or "cpu"
  epochs: 100                 # Training epochs
  batch_size: 32              # Batch size
  k_folds: 5                  # Number of CV folds
  deterministic: false        # True for full reproducibility
  
  optimizer:
    type: "Adam"              # Optimizer type
    lr: 0.001                 # Learning rate
    weight_decay: 0.01        # Weight decay
  
  scheduler:
    type: "cosine"            # "cosine", "step", "plateau", "none"
  
  early_stopping:
    enabled: false
    patience: 20
    min_delta: 0.001
```

### Trainer Section

```yaml
trainer:
  type: "SupervisedTrainer"   # Trainer name (must be registered)
  args: {}                    # Trainer-specific arguments
```

## Available Options

### Datasets

| Name | Description | Classes | Subjects |
|------|-------------|---------|----------|
| BCICIV2a | BCI Competition IV 2a | 4 (Left, Right, Foot, Tongue) | 9 |
| BCICIV2b | BCI Competition IV 2b | 2 (Left, Right) | 9 |
| XWStroke | XuanWu Stroke Dataset | 2 (Left, Right) | 50 |
| LowerStroke | Lower Limb Stroke | Variable | Variable |

### Models

| Name | Description | Paper |
|------|-------------|-------|
| EEGNet | Compact CNN for EEG | Lawhern et al., 2018 |
| CNNLSTM | CNN + LSTM architecture | Custom |
| MultiBand_CNNLSTM | Multi-band variant | Custom |
| Simplified_MultiBand_CNNLSTM | Lightweight variant | Custom |
| ADFCNN | Attention-based CNN | Custom |
| FilterBankCNN | Filter bank CNN | Custom |
| FBCNet_Standard | Filter-bank convolutional | Mane et al., 2021 |
| GACLNet | Graph Attention CNN+LSTM | Custom |

### Trainers

| Name | Description |
|------|-------------|
| SupervisedTrainer | Standard supervised learning with k-fold CV |

## Command Line Options

```bash
python train.py [OPTIONS]

Options:
  -c, --config PATH       Configuration file path
  -n, --name NAME         Experiment name
  -r, --resume PATH       Resume from checkpoint
  -d, --device [cuda|cpu] Device to use
  --list-datasets         List available datasets
  --list-models           List available models
  --list-trainers         List available trainers
  -h, --help              Show help message
```

## Experiment Workflow

### 1. Data Preparation

Ensure your data is in the correct location:

```
src/datasets/
├── BCICIV_2a/
│   ├── A01T.mat
│   ├── A01E.mat
│   └── ...
├── 21679035/
│   └── sourcedata/
│       └── sub-01/
│           └── sub-01_task-motor-imagery_eeg.mat
└── LowerStroke/
    └── subject_1.mat
```

Run preprocessing if needed:

```bash
python preprocessing_XW.py  # For XW Stroke dataset
```

### 2. Configuration

Create or modify a configuration file:

```yaml
# configs/experiment/my_experiment.yaml
experiment:
  name: "BCICIV2a_GACLNet_Test"
  seed: 42

data:
  dataset: "BCICIV2a"
  subjects: [1, 2, 3]  # Test on first 3 subjects
  info_path: "configs/dataset/BCICIV2a.yaml"

model:
  type: "GACLNet"

training:
  device: "cuda"
  epochs: 50
  batch_size: 64
  k_folds: 5
  optimizer:
    lr: 0.001
    weight_decay: 0.075
```

### 3. Run Experiment

```bash
python train.py --config configs/experiment/my_experiment.yaml
```

### 4. Monitor Progress

- Check console output for progress
- View detailed logs in `experiments/EXPERIMENT_NAME/logs/training.log`
- Check visualizations in `experiments/EXPERIMENT_NAME/visualizations/`

### 5. Analyze Results

Results are saved in `experiments/EXPERIMENT_NAME/results/results.json`:

```json
{
  "subjects": [
    {
      "subject_id": 1,
      "avg_val_acc": 0.8234,
      "std_val_acc": 0.0456,
      "fold_accuracies": [0.81, 0.83, 0.82, 0.83, 0.82]
    },
    ...
  ],
  "overall_mean": 0.8156,
  "overall_std": 0.0523
}
```

## Advanced Usage

### Hyperparameter Search

Create multiple config files and run them sequentially:

```bash
for lr in 0.1 0.01 0.001 0.0001; do
    python train.py --config configs/experiment/template.yaml \
                    --name "lr_${lr}"
done
```

Or use a script to generate configs:

```python
from core import Config

base_config = Config.fromfile('configs/experiment/template.yaml')

for lr in [0.1, 0.01, 0.001, 0.0001]:
    config = Config(base_config.to_dict())
    config.set('training.optimizer.lr', lr)
    config.dump(f'configs/experiment/lr_{lr}.yaml')
```

### Cross-Dataset Validation

Test a model trained on one dataset on another:

```bash
# Train on BCICIV2a
python train.py --config configs/experiment/bciciv2a_eegnet.yaml \
                --name bciciv2a_pretrained

# Fine-tune on XWStroke
python train.py --config configs/experiment/xwstroke_eegnet.yaml \
                --name xwstroke_finetuned \
                --resume experiments/bciciv2a_pretrained/checkpoints/...
```

### Custom Evaluation

Create a custom evaluation script:

```python
from core import Config, ExperimentManager
from core.registry import MODELS
import torch

# Load config
config = Config.fromfile('configs/experiment/my_exp.yaml')

# Build model
model_cls = MODELS.get(config.model.type)
model = model_cls(**config.model.args)

# Load checkpoint
checkpoint = torch.load('path/to/checkpoint.pt')
model.load_state_dict(checkpoint['model_state_dict'])

# Run custom evaluation
# ...
```

## Troubleshooting

### CUDA Out of Memory

Reduce batch size in config:

```yaml
training:
  batch_size: 16  # or 8
```

### Data Not Found

Check data paths in dataset config:

```yaml
dataset:
  data_dir: "correct/path/to/data"
```

### Model Not Found

Ensure model is registered in `models/__init__.py`:

```python
from .my_model import MyModel
MODELS.register('MyModel')(MyModel)
```

### Poor Performance

Try:
1. Different learning rates
2. More epochs
3. Different model architectures
4. Data augmentation
5. Different preprocessing parameters

## Best Practices

1. **Use descriptive experiment names**: `xwstroke_eegnet_lr0.001_wd0.01`
2. **Set random seeds**: For reproducibility
3. **Start small**: Test on 1-2 subjects before full run
4. **Monitor logs**: Check for warnings or errors
5. **Save checkpoints**: Enable resuming from interruptions
6. **Document configs**: Add descriptions to your configs
7. **Version control**: Track config changes with git

## Examples

See `configs/experiment/` for example configurations:

- `default.yaml`: Basic template
- `xwstroke_eegnet.yaml`: XWStroke with EEGNet
- `bciciv2a_gaclnet.yaml`: BCICIV2a with GACLNet

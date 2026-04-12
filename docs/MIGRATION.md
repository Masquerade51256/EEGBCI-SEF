# Migration Guide: Old Framework → New Framework

This guide helps you migrate from the old `constant_value.py` based configuration to the new YAML-based configuration system.

## Overview of Changes

| Old Framework | New Framework |
|--------------|---------------|
| `constant_value.py` | `configs/experiment/*.yaml` |
| `torch_fold_train.py` | `train.py` (unified entry) |
| `constant_value.SELECTED_DATASET` | `data.dataset` in YAML |
| `constant_value.SELECTED_MODEL` | `model.type` in YAML |
| Hardcoded paths | Configurable `paths` section |
| Limited extensibility | Registry pattern for easy extension |

## Quick Migration

### 1. Identify Your Current Settings

In your `constant_value.py`, find these settings:

```python
# Dataset selection
SELECTED_DATASET = 3  # Maps to: 1=BCICIV2a, 2=BCICIV2b, 3=XWStroke, 4=LowerStroke

# Model selection  
SELECTED_MODEL = 8    # Maps to: 1=EEGNet, 2=CNNLSTM, ..., 8=GACLNet

# Training parameters
target_subjects = [1, 2, 3]
batch_size = 16
num_epochs = 200
learning_rate = 1e-3
weight_decay = 0.075
k_folds = 5
```

### 2. Create New YAML Configuration

Create a file `configs/experiment/my_experiment.yaml`:

```yaml
experiment:
  name: "MyExperiment"
  seed: 42

data:
  dataset: "XWStroke"        # From SELECTED_DATASET mapping
  subjects: [1, 2, 3]        # From target_subjects
  info_path: "configs/dataset/XWStroke.yaml"

model:
  type: "GACLNet"            # From SELECTED_MODEL mapping
  args: {}

training:
  device: "cuda"
  epochs: 200                # From num_epochs
  batch_size: 16             # From batch_size
  k_folds: 5                 # From k_folds
  
  optimizer:
    type: "Adam"
    lr: 0.001                # From learning_rate
    weight_decay: 0.075      # From weight_decay

trainer:
  type: "SupervisedTrainer"
  args: {}

paths:
  root_dir: "./experiments"

logging:
  level: "INFO"
  console: true
```

### 3. Run with New Configuration

```bash
# Instead of editing constant_value.py and running torch_fold_train.py
python train.py --config configs/experiment/my_experiment.yaml
```

## Dataset Mapping

| constant_value ID | constant_value Name | New YAML Value |
|-------------------|--------------------|----------------|
| 1 | BCICIV2a | `"BCICIV2a"` |
| 2 | BCICIV2b | `"BCICIV2b"` |
| 3 | XWStroke | `"XWStroke"` |
| 4 | LowerStroke | `"LowerStroke"` |

## Model Mapping

| constant_value ID | constant_value Name | New YAML Value |
|-------------------|--------------------|----------------|
| 1 | EEGNet | `"EEGNet"` |
| 2 | CNNLSTM | `"CNNLSTM"` |
| 3 | ADFCNN | `"ADFCNN"` |
| 4 | MultiBand_CNNLSTM | `"MultiBand_CNNLSTM"` |
| 5 | Simplified_MultiBand_CNNLSTM | `"Simplified_MultiBand_CNNLSTM"` |
| 6 | FilterBankCNN | `"FilterBankCNN"` |
| 7 | FBCNet_Standard | `"FBCNet_Standard"` |
| 8 | GACLNet | `"GACLNet"` |

## What's Different

### 1. Configuration Management

**Old:**
```python
# Edit constant_value.py each time
SELECTED_DATASET = 3
SELECTED_MODEL = 8
```

**New:**
```yaml
# Create separate YAML files for different experiments
# configs/experiment/exp1.yaml
data:
  dataset: "XWStroke"
model:
  type: "EEGNet"

# configs/experiment/exp2.yaml
data:
  dataset: "BCICIV2a"
model:
  type: "GACLNet"
```

### 2. Running Experiments

**Old:**
```bash
# Edit constant_value.py, then:
python torch_fold_train.py
```

**New:**
```bash
# No code editing needed
python train.py --config configs/experiment/exp1.yaml
python train.py --config configs/experiment/exp2.yaml
```

### 3. Output Organization

**Old:**
```
src/results/VISUALIZATION/DATASET/MODEL/
src/checkpoints/DATASET/MODEL/SUBJECT/
training.log  # Single log file, overwritten
```

**New:**
```
experiments/EXPERIMENT_NAME/
├── config.yaml              # Copy of configuration used
├── checkpoints/             # Organized by experiment
├── logs/training.log        # Separate log per experiment
├── visualizations/          # Organized by experiment
└── results/results.json     # Structured results
```

### 4. Extending the Framework

**Old:**
- Edit `dataloader/_get_data.py` to add new dataset
- Edit `models/_get_model.py` to add new model
- Risk of breaking existing code

**New:**
- Use decorators to register components:
```python
from core.registry import DATASETS, MODELS

@DATASETS.register('MyDataset')
class MyDataset(BaseDataset):
    pass

@MODELS.register('MyModel')
class MyModel(nn.Module):
    pass
```

## Troubleshooting Migration

### Dataset Not Found

**Problem:** `KeyError: 'MyDataset' is not registered`

**Solution:** Ensure the dataset module is imported:
```python
# In your script or __init__.py
from data import MyDataset  # This triggers registration
```

### Model Arguments Missing

**Problem:** Model complains about missing arguments

**Solution:** The new framework auto-populates common arguments from dataset info:
- `num_channels` from `channel_selection.channels_to_select`
- `num_classes` from `dataset.num_classes`
- `input_length` from windowing and sampling rate
- `num_bands` from filter banks

If your model needs additional arguments, specify them:
```yaml
model:
  type: "MyModel"
  args:
    custom_param: value
```

### Paths Not Found

**Problem:** Data files not found

**Solution:** Update paths in dataset config:
```yaml
# configs/dataset/MyDataset.yaml
dataset:
  data_dir: "path/to/your/data"  # Update this
```

## Backward Compatibility

The old framework files are preserved:
- `torch_fold_train.py` - Original training script
- `constant_value.py` - Original configuration

You can still use them if needed, but they are deprecated and won't receive new features.

## Getting Help

If you encounter issues during migration:

1. Check the example configurations in `configs/experiment/`
2. Review the new README.md for detailed documentation
3. Use `--help` flag with the training script:
   ```bash
   python train.py --help
   ```

## Migration Checklist

- [ ] Identify current settings in `constant_value.py`
- [ ] Create new YAML configuration file
- [ ] Update data paths in dataset config if needed
- [ ] Test with a single subject first
- [ ] Verify output structure
- [ ] Update any custom scripts to use new imports
- [ ] Archive old configuration files if desired

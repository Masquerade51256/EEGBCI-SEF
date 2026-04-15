# Thwe et al. (2026) Reproduction Configurations

This directory contains experiment configurations to reproduce the artifact removal
and model architectures from:

> Y. Thwe, D. Maneetham, and P. N. Crisnapati,
> "Integrating EEG Artifact Removal with Deep Learning for Accurate Motor Imagery
> Classification in Acute Stroke,"
> *Engineering, Technology & Applied Science Research*, vol. 16, no. 1, 2026.

---

## Quick Start

### 1. List Available Models (Verify Registration)

```bash
python train.py --list-models
```

You should see `ThweCNN`, `ThweLSTM`, and `ThweCNNLSTM` in the output.

### 2. Run a Single Experiment

The paper's primary result is the **CNN-LSTM** model. Below are the three
artifact-removal variants:

```bash
# EMAR + CNN-LSTM
python train.py --config configs/reproductions/thwe2026/xwstroke_emar.yaml

# SASICA + CNN-LSTM  (best average accuracy reported in the paper)
python train.py --config configs/reproductions/thwe2026/xwstroke_sasica.yaml

# MARA + CNN-LSTM
python train.py --config configs/reproductions/thwe2026/xwstroke_mara.yaml
```

### 3. Run Baseline CNN / LSTM Controls

```bash
# SASICA + CNN
python train.py --config configs/reproductions/thwe2026/xwstroke_sasica_cnn.yaml

# SASICA + LSTM
python train.py --config configs/reproductions/thwe2026/xwstroke_sasica_lstm.yaml

# EMAR + CNN
python train.py --config configs/reproductions/thwe2026/xwstroke_emar_cnn.yaml

# EMAR + LSTM
python train.py --config configs/reproductions/thwe2026/xwstroke_emar_lstm.yaml

# MARA + CNN
python train.py --config configs/reproductions/thwe2026/xwstroke_mara_cnn.yaml

# MARA + LSTM
python train.py --config configs/reproductions/thwe2026/xwstroke_mara_lstm.yaml
```

---

## Faithful Replication Configs (Recommended)

If your initial runs (e.g., the original `xwstroke_sasica.yaml`) produced results
(~63%) far below the paper's reported ~95%, the main reasons are likely:

1. **SASICA over-removal**: original thresholds excluded ~9/13 ICA components,
   throwing away too much neural signal.
2. **Resampling to 250 Hz**: the paper explicitly uses the original 500 Hz.
3. **Sliding windows**: the paper states 40 trials per subject; the old config
   created 200 windows (2s / 0.5s stride) per subject.
4. **Narrow filter bank `[4, 40]`**: the paper describes a 0–500 Hz broadband.
5. **Weight decay + cosine annealing**: not mentioned in the paper.
6. **Channel dropping**: the paper does not mention dropping HEOL/HEOR before
   modeling.

We created **paper-faithful configs** that address all of these issues:

```bash
# Most faithful SASICA reproduction (recommended)
python train.py --config configs/reproductions/thwe2026/xwstroke_sasica_paper.yaml

# Paper-faithful EMAR reproduction
python train.py --config configs/reproductions/thwe2026/xwstroke_emar_paper.yaml

# Paper-faithful MARA reproduction
python train.py --config configs/reproductions/thwe2026/xwstroke_mara_paper.yaml
```

**What changed in the `_paper.yaml` configs:**

| Setting | Old Config | Paper-Faithful Config |
|---|---|---|
| Sampling rate | 250 Hz | **500 Hz** (no resampling) |
| Windowing | 2.0s / 0.5s stride (200 samples) | **4.0s / 4.0s stride** (40 samples) |
| Channels | 30 EEG | **32 channels** (keeps HEOL/HEOR) |
| Filter bank | `[4, 40]` | **`[0.5, 100]`** (broadband) |
| Normalization | z-score | **none** |
| Augmentation | noise/dropout/timewarp (p=0.3) | **none** (p=0) |
| SASICA muscle PSD threshold | 0.3 | **0.5** (conservative) |
| SASICA eye PSD threshold | 0.4 | **0.6** (conservative) |
| SASICA variance/kurtosis z | 3.0 | **4.0** (conservative) |
| SASICA EOG corr threshold | 0.6 | **0.7** (conservative) |
| Epochs | 200 | **500** |
| Batch size | 32 | **16** |
| Weight decay | 0.01 | **0.0** |
| LR scheduler | cosine | **none** (constant LR) |

> **Note:** The paper does not fully specify train/validation protocol (epochs,
> batch size, splits, etc.). The faithful config uses 5-fold CV by default.
> If you want a single 80/20 train-test split per subject (closer to what many
> simple DL papers implicitly use), set `training.k_folds: 1` in the config.
> The trainer now supports `k_folds=1` natively.

### Sanity-Check Command (1 subject, 5 epochs)

```bash
python -c "
from core import Config, ExperimentManager
from models import register_all_models
import data, trainers
register_all_models()

config = Config.fromfile('configs/reproductions/thwe2026/xwstroke_sasica_paper.yaml')
config.set('data.subjects', [1])
config.set('training.epochs', 5)

exp = ExperimentManager(config, exp_name='sanity_check_sasica')
exp.setup()
results = exp.run()
print('Sanity check completed:', results)
"
```

---

## Batch Run All 9 Original Experiments

You can copy the following into a shell script to run the full factorial
comparison overnight:

### Windows PowerShell

```powershell
$configs = @(
    'configs/reproductions/thwe2026/xwstroke_emar_cnn.yaml',
    'configs/reproductions/thwe2026/xwstroke_emar_lstm.yaml',
    'configs/reproductions/thwe2026/xwstroke_emar.yaml',
    'configs/reproductions/thwe2026/xwstroke_sasica_cnn.yaml',
    'configs/reproductions/thwe2026/xwstroke_sasica_lstm.yaml',
    'configs/reproductions/thwe2026/xwstroke_sasica.yaml',
    'configs/reproductions/thwe2026/xwstroke_mara_cnn.yaml',
    'configs/reproductions/thwe2026/xwstroke_mara_lstm.yaml',
    'configs/reproductions/thwe2026/xwstroke_mara.yaml'
)

foreach ($cfg in $configs) {
    Write-Host "Running $cfg ..."
    python train.py --config $cfg
    if ($LASTEXITCODE -ne 0) {
        Write-Host "ERROR: $cfg failed. Stopping batch."
        break
    }
}
```

### Linux / macOS / WSL Bash

```bash
#!/bin/bash
set -e

CONFIGS=(
    "configs/reproductions/thwe2026/xwstroke_emar_cnn.yaml"
    "configs/reproductions/thwe2026/xwstroke_emar_lstm.yaml"
    "configs/reproductions/thwe2026/xwstroke_emar.yaml"
    "configs/reproductions/thwe2026/xwstroke_sasica_cnn.yaml"
    "configs/reproductions/thwe2026/xwstroke_sasica_lstm.yaml"
    "configs/reproductions/thwe2026/xwstroke_sasica.yaml"
    "configs/reproductions/thwe2026/xwstroke_mara_cnn.yaml"
    "configs/reproductions/thwe2026/xwstroke_mara_lstm.yaml"
    "configs/reproductions/thwe2026/xwstroke_mara.yaml"
)

for cfg in "${CONFIGS[@]}"; do
    echo "Running $cfg ..."
    python train.py --config "$cfg"
done
```

---

## Dataset

- **XWStroke** (Xuanwu Hospital acute stroke motor imagery dataset)
- 50 subjects, 32 channels (paper mentions 31 EEG electrodes), 500 Hz sampling rate
- 2-class motor imagery (left hand / right hand)
- Single-band filter bank `[0.5, 100] Hz` in the faithful configs to match the
  paper's broadband preprocessing

## Artifact Removal Methods

The paper compares three ICA-based artifact removal techniques:

1. **EMAR** (`xwstroke_emar*.yaml`)
   - Eyes and Muscle Artifact Removal
   - Uses `mne-icalabel` (ICLabel) to classify independent components
   - Excludes components labeled as `muscle artifact` or `eye blink` with probability >= 0.9
   - Matches the MATLAB reference: `pop_icflag(EEG, [NaN NaN; 0.9 1; 0.9 1; ...])`

2. **SASICA** (`xwstroke_sasica*.yaml`)
   - Semi-Automated Selection of Independent Components for Artifact Correction
   - Multi-criteria thresholding based on:
     - EOG channel correlation
     - Temporal variance (z-score)
     - Temporal kurtosis (z-score)
     - Power spectrum ratios (muscle 30-100 Hz, eye 1-7 Hz)

3. **MARA** (`xwstroke_mara*.yaml`)
   - Multiple Artifact Rejection Algorithm
   - Extracts spatial/temporal features from each IC:
     - Spatial kurtosis, average deviation, variance
     - Temporal kurtosis, variance
   - Uses unsupervised One-Class SVM for outlier detection
   - Also provides `iclabel_proxy` strategy as fallback

## Models

The paper evaluates three architectures. All are implemented in `models/ThweCNNLSTM.py`:

- **ThweCNN**: `Conv1d(channels×bands → 64, k=3) → ReLU → MaxPool → Flatten → Dense(50) → ReLU → Linear(num_classes)`
- **ThweLSTM**: `LSTM(50) → ReLU → Linear(num_classes)`
- **ThweCNNLSTM**: `Conv1d(channels×bands → 64, k=3) → ReLU → MaxPool → reshape → LSTM(50) → ReLU → Linear(num_classes)`

All models output raw logits (softmax omitted) for compatibility with `CrossEntropyLoss`.

### Model Parameter Counts (single-band input)

| Model | Parameters @ 250 Hz (2 s) | Parameters @ 500 Hz (4 s) |
|---|---|---|
| ThweCNN | ~806K | ~3.2M |
| ThweLSTM | ~16.5K | ~16.5K |
| ThweCNNLSTM | ~29K | ~29.5K |

## Notes

- All three processors apply **Common Average Reference (CAR)** before ICA, as
  recommended by the ICLabel authors and the original paper.
- For XWStroke (32 channels), channel names and montage positions are
  auto-populated from the paper's Table II if not explicitly provided.
- ICA is fitted with **extended Infomax** (`method='infomax', extended=True`).
- The high-pass filter (1 Hz) is applied **only to the ICA-fitting copy**;
  the reconstructed data retains the original broadband signal content.
- `EOG_channels: [30,31]` from the dataset config is now **automatically**
  injected into the artifact-removal processors as `eog_channels: ["HEOL", "HEOR"]`.

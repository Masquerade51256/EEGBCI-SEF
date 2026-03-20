# EEGBCI Standard Experiment Framwork
A standardized, modular, and extensible platform for BCI algorithm experimentation. It supports various datasets, deep learning models, preprocessing pipelines, and training strategies.

## :sparkles: Objective
- **Modular Design**: Clean separation of data, models, trainers, and evaluation.
- **Reproducibility**: Experiment-specific YAML configuration files snapshot all hyperparameters.
- **Extensibility**: Easily add new datasets (inherit from `BaseDataset`) or models.
- **Comprehensive Logging**: Tracks training metrics, console output, and generates key figures.



## :rocket: Quick Start

### 1. Project Installation

- download or clone this repository
- create new virtual environment  with `Python=3.10`
- install Pytorch, CUDA and cudnn
- `pip install -r requirements.txt` (Not available yet)

### 2. Prepare Data

- project structure

  ```
  ├── dataloader
  ├── models
  ├── preprocessing
  └── src
      ├── checkpoints
      │   ├── BCICIV2a
      │   └── XWStroke
      ├── datasets
      │   ├── 21679035
      │   │   └── sourcedata
      │   │   │   ├── sub-01
      │   │   │   │   └── sub-01_task-motor-imagery_eeg.mat
      │   │   │   └── ...
      │   ├── 27130299
      │   └── BCICIV_2a
      │       ├── A01E.gdf
      │       ├── A01E.mat
      │       ├── A01T.gdf
      │       ├── A01T.mat
      │       └── ...
      └── results
          └── visualizations
  ```

- Please download **XuanWu** dataset from [this site](https://figshare.com/articles/dataset/EEG_datasets_of_stroke_patients/21679035/5), decompress and put all contents (especially `sourcedata`) into `src\datasets\21679035`.

- Please download **BCICompetetion IV 2a** dataset from [this site](https://www.bbci.de/competition/iv/#download), and the true lable from [this site](https://www.bbci.de/competition/iv/results/ds2a/true_labels.zip). Decompress and put all files (including `.gdf` & `.mat`) into dir `src\datasets\BCICIV_2a`

  

### 3. Run Experiment

- You can change Model, Dataset and other configuration in file `constant_value`. (It will be updated, and finally deprecated, in subsequent versions)

- Just run `python train.py`

### 4. Check Results

Training script will save figures to dir `src\result\DATASET\MODEL`.

:warning: For now, the figures could be **COVERED** by new training procedures.



## :open_file_folder: How to Extend














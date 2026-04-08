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
- `pip install -r requirements.txt`

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

  - :warning::warning::warning: Run `preprocessing_XW.py` ( to extract the specific 4 sec MI signal) before train​ing.

- Please download **BCICompetetion IV 2a** dataset from [this site](https://www.bbci.de/competition/iv/#download), and the true label from [this site](https://www.bbci.de/competition/iv/results/ds2a/true_labels.zip). 

  - :warning::warning::warning: Decompress and put all files (including `.gdf` & `.mat`) into direction `src\datasets\BCICIV_2a`

  


### 3. Run Experiment

- ~~Just run `python train.py`~~
  - ~~You can run `python _train.py` now. It is a more elegant implementation, but still under updating.~~
- ~~You can change **Model**, **Dataset** and **other training configuration** in file `constant_value`. (It will be updated, and finally deprecated, in subsequent versions)~~
  - For `_train.py`, you need change **these** configuration in the file `config/train_config/base_config.yaml`; while some other database relevant configuration in the file `config/data_config/DATASETNAME_config.yaml`
- Now the old version `train.py` has been deprecated. The new one is updated from `_train.py`, and **rename as `torch_fold_train.py`** to distinguish it from other ML training script.


### 4. Check Results

- Training script will save figures to direction `src\result\DATASETNAME\MODELNAME`.

:warning: For now, the figures would be **COVERED** by new training procedures.



## :open_file_folder: How to Extend

### 1. New Dataset

### 2. New Model



## :notebook: Dev Log

- April 2, 2026
  - :red_circle: Add a script "preprocessing_XW.py" to correctly cut the MI signal.








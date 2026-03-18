import os
# Selection
DATASETS = {
    1: 'BCICIV2a',
    2: 'BCICIV2b',
    3: 'XWStroke',
}
SELECTED_DATASET = 3
target_subjects = list(range(1,51))
# target_subjects = list(range(1,10))

MODELS = {
    1: 'EEGNet',
    2: 'CNNLSTM',
    3: 'ADFCNN',
    4: 'MultiBand_CNNLSTM',
    5: 'Simplified_MultiBand_CNNLSTM',
    6: 'MyADFCNN'
}
SELECTED_MODEL = 3

# Mode
is_test = False

# Preprocessing Config
window_length = 2
window_stride = 1
filter_banks = [
    [4, 40],
]
use_csp = False
need_resample = False
resample_rate = 250


# Training Config
train_device = 'cuda'

batch_size = 16
num_epochs = 300
learning_rate = 1e-3
weight_decay = 0.075
k_folds = 5

# Path
dataInfo_path = os.path.join('Dataloader', 'DatasetInfo')
log_path = 'Logs'
ckpt_path = 'Checkpoints'

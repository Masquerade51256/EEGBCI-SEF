import os
# Selection
DATASETS = {
    1: 'BCICIV2a',
    2: 'BCICIV2b',
    3: 'XWStroke',
}
SELECTED_DATASET = 3
target_subjects = list(range(1,51))

MODELS = {
    1: 'EEGNet',
    2: 'CNNLSTM',
}
SELECTED_MODEL = 2

# Mode
is_test = False

# Preprocessing Config
window_length = 2
window_stride = 1
use_filter_bank = False
bank = [[4, 16], [16, 40]]
band_filter = [8, 38]
use_csp = False
need_resample = False
resample_rate = 250


# Training Config
train_device = 'cuda'

batch_size = 16
num_epochs = 35
learning_rate = 1e-3
weight_decay = 0.01
k_folds = 5

# Path
dataInfo_path = os.path.join('Dataloader', 'DatasetInfo')
log_path = 'Logs'
ckpt_path = 'Checkpoints'

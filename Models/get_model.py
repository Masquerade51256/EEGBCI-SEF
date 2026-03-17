from Models.ADFCNN import ADFCNN
from Models.EEGNet import EEGNet
from Models.CNNLSTM import CNNLSTM
import config
import os, yaml

dataset_path = os.path.join(config.dataInfo_path,f"{config.DATASETS[config.SELECTED_DATASET]}.yaml")
with open(dataset_path, 'r') as f:
    dataset_info = yaml.safe_load(f)

CHANNELS_NUM = len(dataset_info['channels_selected'])
CLASS_NUM = dataset_info['num_classes']
INPUT_SIZE = config.window_length * dataset_info['sample_rate']
SR = dataset_info['sample_rate']

def get_model(model_id):
    print(model_id)
    model_name = config.MODELS[model_id]
    if model_name == 'EEGNet':
        return EEGNet(CHANNELS_NUM, CLASS_NUM)
    elif model_name == 'CNNLSTM':
        return CNNLSTM(CHANNELS_NUM, CLASS_NUM, INPUT_SIZE)
    elif model_name == 'ADFCNN':
        return ADFCNN(num_classes=CLASS_NUM, num_channels=CHANNELS_NUM)
    else:
        raise ValueError(f"Unknown model name: {model_name}")
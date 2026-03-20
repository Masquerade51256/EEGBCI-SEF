# from Models.ADFCNN import ADFCNN
from models.EEGNet import EEGNet
# from Models.EEGNet_old import EEGNet
from models.CNNLSTM import CNNLSTM, MultiBand_CNNLSTM, Simplified_MultiBand_CNNLSTM
from models.myADFCNN import ADFCNN
import constant_value
import os
import utils

dataset_path = os.path.join(constant_value.dataInfo_path,f"{constant_value.DATASETS[constant_value.SELECTED_DATASET]}.yaml")

dataset_info = utils.load_config(dataset_path)

CHANNELS_NUM = len(dataset_info['dataset']['channels_selected'])
CLASS_NUM = dataset_info['dataset']['num_classes']
INPUT_SIZE = constant_value.window_length * dataset_info['dataset']['sample_rate']
SR = dataset_info['dataset']['sample_rate']
BAND_NUM = len(constant_value.filter_banks)

def get_model(model_id):
    model_name = constant_value.MODELS[model_id]
    if model_name == 'EEGNet':
        return EEGNet(num_channels=CHANNELS_NUM, num_classes=CLASS_NUM, num_bands=BAND_NUM, input_length=INPUT_SIZE)
    elif model_name == 'CNNLSTM':
        return CNNLSTM(num_channels=CHANNELS_NUM, num_classes=CLASS_NUM, num_bands=BAND_NUM, input_length=INPUT_SIZE)
    elif model_name == 'ADFCNN':
        return ADFCNN(num_channels=CHANNELS_NUM, num_classes=CLASS_NUM, num_bands=BAND_NUM, input_length=INPUT_SIZE)
    elif model_name == 'MultiBand_CNNLSTM':
        return MultiBand_CNNLSTM(channels_num=CHANNELS_NUM, bands_num=BAND_NUM, class_num=CLASS_NUM, input_length=INPUT_SIZE)
    elif model_name == 'Simplified_MultiBand_CNNLSTM':
        return Simplified_MultiBand_CNNLSTM(channels_num=CHANNELS_NUM, bands_num=BAND_NUM, class_num=CLASS_NUM, input_length=INPUT_SIZE)
    else:
        raise ValueError(f"Unknown model name: {model_name}")
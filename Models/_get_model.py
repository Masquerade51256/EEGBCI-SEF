# from Models.ADFCNN import ADFCNN
from models.EEGNet import EEGNet
# from Models.EEGNet_old import EEGNet
from models.CNNLSTM import CNNLSTM, MultiBand_CNNLSTM, Simplified_MultiBand_CNNLSTM
from models.myADFCNN import ADFCNN
import constant_value
import os
import utils



def get_model(model_id, dataset_info):
    channels_num = len(dataset_info['preprocessing']['channel_selection']['channels_to_select'])
    class_num = dataset_info['dataset']['num_classes']
    sr = dataset_info['preprocessing']['resample']['target_sr']
    input_size = constant_value.window_length * sr
    band_num = len(constant_value.filter_banks)

    model_name = constant_value.MODELS[model_id]
    if model_name == 'EEGNet':
        return EEGNet(num_channels=channels_num, num_classes=class_num, num_bands=band_num, input_length=input_size)
    elif model_name == 'CNNLSTM':
        return CNNLSTM(num_channels=channels_num, num_classes=class_num, num_bands=band_num, input_length=input_size)
    elif model_name == 'ADFCNN':
        return ADFCNN(num_channels=channels_num, num_classes=class_num, num_bands=band_num, input_length=input_size)
    elif model_name == 'MultiBand_CNNLSTM':
        return MultiBand_CNNLSTM(channels_num=channels_num, bands_num=band_num, class_num=class_num, input_length=input_size)
    elif model_name == 'Simplified_MultiBand_CNNLSTM':
        return Simplified_MultiBand_CNNLSTM(channels_num=channels_num, bands_num=band_num, class_num=class_num, input_length=input_size)
    else:
        raise ValueError(f"Unknown model name: {model_name}")
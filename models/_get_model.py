# from Models.ADFCNN import ADFCNN
from models.EEGNet import EEGNet
# from Models.EEGNet_old import EEGNet
from models.CNNLSTM import CNNLSTM, MultiBand_CNNLSTM, Simplified_MultiBand_CNNLSTM
from models.GACLNet import GACLNet
from models.myADFCNN import ADFCNN
from models.FBCNN import FilterBankCNN
from models.myFBCNet import FBCNet_Standard
import constant_value
import os
import utils



def get_model(model_id, dataset_info):
    channels_num = len(dataset_info['preprocessing']['channel_selection']['channels_to_select'])
    class_num = dataset_info['dataset']['num_classes']
    sr = dataset_info['preprocessing']['resample']['target_sr']
    input_size = int(dataset_info['preprocessing']['windowing']['window_length_sec'] * sr)
    band_num = len(dataset_info['preprocessing']['filter_bank']['filter_banks'])

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
    elif model_name == 'FilterBankCNN':
        return FilterBankCNN(n_filterbanks=band_num, n_channels=channels_num, n_times=input_size, n_classes=class_num)
    elif model_name == 'FBCNet_Standard':
        return FBCNet_Standard(num_classes=class_num, num_channels=channels_num, input_time_length=input_size, n_band=band_num)
    elif model_name == 'GACLNet':
        return GACLNet(num_classes=class_num, num_channels=channels_num, num_bands=band_num, input_length=input_size)

    else:
        raise ValueError(f"Unknown model name: {model_name}")
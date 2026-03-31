from dataloader._dataset_BCICIV2a import BCICompet2aIV
from dataloader._dataset_XWStroke import XWStrokeDataset
from dataloader._dataset_LowerStroke import StrokeLowerLimbMI
import constant_value
import os, yaml



def load_single_subject_data(dataset_id, subject_id, dataset_info):
    dataset_name = constant_value.DATASETS[dataset_id]
    if dataset_name == 'BCICIV2a':
        return BCICompet2aIV(subject_id, dataset_info)
    if dataset_name == 'XWStroke':
        return XWStrokeDataset(subject_id, dataset_info)
    if dataset_name == 'LowerStroke':
        return StrokeLowerLimbMI(subject_id, dataset_info)
    else:
        raise ValueError("Unknown dataset")
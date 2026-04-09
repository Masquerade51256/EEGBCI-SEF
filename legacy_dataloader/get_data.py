from legacy_dataloader.dataset_BCICIV2a import BCICompet2aIV
from legacy_dataloader.dataset_XWStroke import XWStrokeDataset
import constant_value
import os, yaml



def load_single_subject_data(dataset_id, subject_id):
    info_path = os.path.join(constant_value.dataInfo_path,f"{constant_value.DATASETS[constant_value.SELECTED_DATASET]}.yaml")
    dataset_name = constant_value.DATASETS[dataset_id]
    if dataset_name == 'BCICIV2a':
        return BCICompet2aIV(subject_id, info_path)
    if dataset_name == 'XWStroke':
        return XWStrokeDataset(subject_id, info_path)
    else:
        raise ValueError("Unknown dataset")
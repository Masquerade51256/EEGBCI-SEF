from Dataloader.BCICIV2a import BCICompet2aIV
from Dataloader.XWStroke import XWStrokeDataset
import config
import os, yaml



def load_single_subject_data(dataset_id, subject_id):
    info_path = os.path.join(config.dataInfo_path,f"{config.DATASETS[config.SELECTED_DATASET]}.yaml")
    dataset_name = config.DATASETS[dataset_id]
    if dataset_name == 'BCICIV2a':
        return BCICompet2aIV(subject_id, info_path)
    if dataset_name == 'XWStroke':
        return XWStrokeDataset(subject_id, info_path)
    else:
        raise ValueError("Unknown dataset")
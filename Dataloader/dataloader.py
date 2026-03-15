from Dataloader.BCICIV2a import BCICompet2aIV
from Dataloader.XWStroke import XWStroke
import config

def load_data(dataset_id):
    dataset_name = config.DATASETS[dataset_id]
    if dataset_name == 'BCICIV2a':
        return BCICompet2aIV()
    if dataset_name == 'XWStroke':
        return XWStroke()
    else:
        raise ValueError("Unknown dataset")
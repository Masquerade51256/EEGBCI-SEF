from glob import glob
from tqdm import tqdm
import mne
import torch
import scipy
import numpy as np
import numpy as np
from torch.utils.data import Dataset
import config
import os,yaml

SUBJECTS = config.target_subjects
dataset_path = os.path.join(config.dataInfo_path,f"{config.DATASETS[config.SELECTED_DATASET]}.yaml")
with open(dataset_path, 'r') as f:
    dataset_info = yaml.safe_load(f)

DATA_DIR = dataset_info['data_dir']
CHANNELS = dataset_info['num_channels']
CHANNELS_SELECTED = dataset_info['channels_selected']
SAMPLE_RATE = dataset_info['sample_rate']
CLASS_NUM = dataset_info['num_classes']
is_test = config.is_test

def windowSlide(signal,label,win_len = config.window_length*SAMPLE_RATE,step = config.window_stride*SAMPLE_RATE):
    slices = []
    label_slices = []
    _,len = signal.shape
    for i in range(0,len-win_len+1,step):
        slices.append(signal[:,i:i+win_len])
        label_slices.append(label)
    return np.array(slices), np.array(label_slices)

class XWStroke(Dataset):
    def __init__(self):
        self.data, self.label = self.get_brain_data()
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        label = self.label[idx]
        data = np.array(self.data[idx]).astype(np.float32)
        # print(data.shape)
        return data, label

    def get_brain_data(self):
        all_sub_data = []
        all_labels = []
        for subject_id in SUBJECTS:
            file_path = os.path.join(DATA_DIR, f"sub-{subject_id:02d}", f"sub-{subject_id:02d}_task-motor-imagery_eeg.mat")
            try:
                data = scipy.io.loadmat(file_path)
                print(f"data file '{file_path}' loaded")
            except FileNotFoundError:
                print(f"ERROR: file '{file_path}' not found.")
                continue
            except Exception as e:
                print(f"ERROR: failed to load file: {e}")
                continue
            for key, value in data.items():
                if not key.startswith('__'):
                    eeg_data = value[0][0][0]
                    labels = value[0][0][1]
                    # print(f"EEG data shape: {eeg_data.shape}")
                    # print(f"label shape: {labels.shape}")
            eeg_data = eeg_data[:,CHANNELS_SELECTED,:]
            labels = labels.reshape(-1, 1)
            labels = labels-1
            print(f"labels shape after reshaping: {labels.shape}")
            # one_hot_labels = np.zeros((labels.size, CLASS_NUM))
            # one_hot_labels[np.arange(labels.size), labels.flatten()] = 1

            # print(f"EEG data shape after selection: {eeg_data.shape}")
            # print(f"label shape after reshaping: {labels.shape}")
            windowed_data = []
            windowed_labels = []
            for i, signal in enumerate(eeg_data):
                signal_slices, labels_sclices = windowSlide(signal, labels[i])
                # print(f"Windowed EEG data shape: {signal_slices.shape}")
                # print(f"Windowed label shape: {labels_sclices.shape}")
                windowed_data.append(signal_slices)
                windowed_labels.append(labels_sclices)

            all_sub_data.append(windowed_data)
            all_labels.append(windowed_labels)
        # all_sub_data[sub,sess,ch,time].shape = (n,40,33,4000)
        all_sub_data = np.array(all_sub_data)
        all_labels = np.array(all_labels)
        sub, sess, slices, ch, timepts = all_sub_data.shape
        all_sub_data = all_sub_data.reshape(sub*sess*slices, ch, timepts)
        all_labels = all_labels.reshape(-1)
        print(f"all_sub_data shape: {all_sub_data.shape}")
        print(f"all_labels shape: {all_labels.shape}")
        return all_sub_data, all_labels


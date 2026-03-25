import numpy as np
import scipy.io
import os
from torch.utils.data import Dataset
import constant_value
import yaml
from preprocessing import bandpass
from dataloader._dataset_Base import BaseDataset

class XWStrokeDataset(BaseDataset):
    
    def _load_mat_data(self, file_path):
        """Loads raw EEG data and labels from a .mat file."""
        try:
            data = scipy.io.loadmat(file_path)
            for key, value in data.items():
                if not key.startswith('__'):
                    eeg_data = value[0][0][0]  # Expected shape: (n_sessions, n_channels, n_timepoints)
                    labels = value[0][0][1]    # Expected shape: (n_sessions, 1)
                    break
            return eeg_data, labels.reshape(-1)  # Flatten labels to (n_sessions,)
        except FileNotFoundError:
            raise FileNotFoundError(f"Data file not found: {file_path}")
        except KeyError:
            raise KeyError(f"File structure does not match expectations: {file_path}")

    def _load_raw_data(self):
        """Constructs the file path for the subject's data based on the dataset info."""
        file_name = f"sub-{self.subject_id:02d}_task-motor-imagery_eeg.mat"
        file_path = os.path.join(self.data_dir, f"sub-{self.subject_id:02d}", file_name)
        data, labels = self._load_mat_data(file_path)
        return data, labels

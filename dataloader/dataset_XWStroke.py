import numpy as np
import scipy.io
import os
from torch.utils.data import Dataset
# from braindecode.preprocessing import Preprocessor, preprocess, PreprocessingPipeline
import constant_value
import utils
from preprocessing import bandpass

class XWStrokeDataset(Dataset):
    """
    An optimized PyTorch Dataset class for loading and preprocessing the XWStroke EEG motor imagery dataset.
    Core improvements: Utilizes braindecode's preprocessing pipeline for a standardized and efficient workflow.
    """
    def __init__(self, subject_id, dataset_info_path, is_test=False, transform=None):
        """
        Initializes the dataset for a specific subject.

        Args:
            subject_id: The ID of the subject.
            data_dir: The root directory for the data (read from dataset_info.yaml).
            dataset_info_path: Path to the `dataset_info.yaml` constant_valueuration file.
            is_test: Whether the dataset is for testing (currently not fully utilized in windowing logic).
            transform: Optional additional transforms (e.g., for data augmentation).
        """
        self.subject_id = subject_id
        self.is_test = is_test
        self.transform = transform

        # Load dataset metadata from the YAML constant_valueuration file
        self.info = utils.load_config(dataset_info_path)

        self.channels_selected = self.info['channels_selected']
        self.sample_rate = self.info['sample_rate']
        self.data_dir = self.info['data_dir']

        # Assuming constant_value.window_length and constant_value.window_stride are defined globally
        self.window_len_samples = int(constant_value.window_length * self.sample_rate)
        self.window_stride_samples = int(constant_value.window_stride * self.sample_rate)

        # Define the preprocessing pipeline
        self.filter_banks = constant_value.filter_banks
        self.n_bands = len(self.filter_banks)

        # Load and process the subject's data
        self.data, self.labels, self.group_ids = self._load_and_process_subject_data()

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

    def _apply_preprocessing(self, eeg_data):
        """
        Applies filter bank preprocessing to the EEG data.
        """
        print(eeg_data.shape)
        band_data_list = []
        for band_idx, (l_freq, h_freq) in enumerate(self.filter_banks):
            filtered_data = bandpass.bandpass_filter(eeg_data,
                                                 low_freq=l_freq,
                                                 high_freq=h_freq,
                                                 sfreq=self.sample_rate)
            mean = filtered_data.mean(axis=-1, keepdims=True)
            std = filtered_data.std(axis=-1, keepdims=True)
            std[std < 1e-8] = 1.0
            normalized_band_data = (filtered_data - mean) / std
            normalized_band_data = normalized_band_data[:, np.newaxis, :, :]
            band_data_list.append(normalized_band_data)
        multi_band_data = np.concatenate(band_data_list, axis=1)
        return multi_band_data


    def _apply_sliding_window(self, processed_data, labels):
        """
        Applies a sliding window to each session's data and generates corresponding window labels.
        """
        windowed_data_list = []
        windowed_labels_list = []
        all_group_ids = []

        n_sessions, n_bands, n_channels, n_timepoints = processed_data.shape

        for sess_idx in range(n_sessions):
            session_data = processed_data[sess_idx]  # Shape: (n_bands, n_channels, n_timepoints)
            session_label = labels[sess_idx]

            # Calculate the number of windows for this session
            n_windows = 1 + (n_timepoints - self.window_len_samples) // self.window_stride_samples

            for win_idx in range(n_windows):
                start = win_idx * self.window_stride_samples
                end = start + self.window_len_samples
                window = session_data[:, :, start:end]  # Shape: (n_bands, n_channels, window_len_samples)

                windowed_data_list.append(window)
                windowed_labels_list.append(session_label)

            all_group_ids.extend([sess_idx] * n_windows)

        # Stack all windows
        all_windowed_data = np.stack(windowed_data_list, axis=0).astype(np.float32)  # (n_total_windows, n_bands, n_channels, window_len)
        all_windowed_labels = np.array(windowed_labels_list, dtype=np.int64)
        all_group_ids = np.array(all_group_ids, dtype=np.int64)
        print(all_group_ids.shape)
        return all_windowed_data, all_windowed_labels, all_group_ids

    def _load_and_process_subject_data(self):
        """Main pipeline: loads, preprocesses, and windows a single subject's data."""
        # 1. Construct file path and load raw data
        file_name = f"sub-{self.subject_id:02d}_task-motor-imagery_eeg.mat"
        file_path = os.path.join(self.data_dir, f"sub-{self.subject_id:02d}", file_name)

        raw_eeg_data, raw_labels = self._load_mat_data(file_path)  # raw_eeg_data: (n_sessions, n_all_channels, n_timepoints)

        # 2. Channel selection
        selected_eeg_data = raw_eeg_data[:, self.channels_selected, :]  # (n_sessions, n_selected_channels, n_timepoints)

        # 3. Apply preprocessing pipeline (Filtering + Normalization)
        processed_eeg_data = self._apply_preprocessing(selected_eeg_data)  # (n_sessions, n_selected_channels, n_timepoints)

        # 4. Label shift (1,2 to 0,1)
        adjusted_labels = raw_labels - 1

        # 5. Sliding window segmentation
        final_data, final_labels, final_group_ids = self._apply_sliding_window(processed_eeg_data, adjusted_labels)

        print(f"[Subject {self.subject_id}] Data loaded. Final data shape: {final_data.shape}, Label shape: {final_labels.shape}")
        return final_data, final_labels, final_group_ids

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = self.data[idx]  # Shape: (n_channels, window_len_samples)
        label = self.labels[idx]
        group_id = self.group_ids[idx]

        if self.transform:
            data = self.transform(data)

        # Data is returned as a NumPy array. Conversion to PyTorch Tensor can be done here or via a collate_fn in the DataLoader.
        # Example: data = torch.from_numpy(data).float()
        return data, label
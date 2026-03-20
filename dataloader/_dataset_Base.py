from torch.utils.data import Dataset
import os,yaml

class BaseDataset(Dataset):
    def __init__(self, subject_id, dataset_info_path):
        # Load dataset metadata from the YAML constant_valueuration file
        with open(dataset_info_path, 'r') as f:
            self.info = yaml.safe_load(f)

        self.subject_id = subject_id
        self.channels_selected = self.info['channels_selected']
        self.sample_rate = self.info['sample_rate']
        self.data_dir = self.info['data_dir']

    
    def __len__(self):        # This should be implemented in the subclass based on the actual data
        raise NotImplementedError("Subclasses must implement __len__ method")
    
    def __getitem__(self, idx):  # This should be implemented in the subclass to return (data, label) for the given index
        raise NotImplementedError("Subclasses must implement __getitem__ method")
    
    def _load_raw_data(self, file_path):
        # This method should be implemented in the subclass to load raw data from the given file path
        raise NotImplementedError("Subclasses must implement _load_raw_data method")
    
    def _get_file_path(self):
        # This method should be implemented in the subclass to construct the file path for the subject's data
        raise NotImplementedError("Subclasses must implement _get_file_path method")

    def _load_and_process_subject_data(self,):
         # 1. Construct file path and load raw data
        file_path = self._get_file_path()
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
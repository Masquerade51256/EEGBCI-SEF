from torch.utils.data import Dataset

class BaseDataset(Dataset):
    def __init__(self, 
                 subject_id: int, dataset_info: dict):
        self.subject_id = subject_id

        # Load dataset metadata from the YAML configuration file
        self.info = dataset_info

        self.channels_selected = self.info['dataset']['channels_selected']
        self.sample_rate = self.info['dataset']['sample_rate']
        self.data_dir = self.info['dataset']['data_dir']

        # Assuming constant_value.window_length and constant_value.window_stride are defined globally
        self.window_len_samples = int(self.info['dataset']["window_length_sec"] * self.sample_rate)
        self.window_stride_samples = int(self.info['dataset']["window_stride_sec"] * self.sample_rate)

        # Define the preprocessing pipeline
        self.filter_banks = self.info['dataset']['filter_banks']
        self.n_bands = len(self.filter_banks)

        # Load and process the subject's data
        self.data, self.labels, self.group_ids = self._load_and_process_subject_data()
        print(f"[{self.info['dataset']['name']} Subject {self.subject_id}] Data loaded. Final shape: {self.data.shape}, Labels: {self.labels.shape}")

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
        raw_eeg_data, raw_labels = self._load_raw_data()  # raw_eeg_data: (n_sessions, n_all_channels, n_timepoints)

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